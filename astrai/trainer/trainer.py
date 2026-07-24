import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from astrai.config import TrainConfig
from astrai.parallel.setup import get_current_device, get_rank, spawn_parallel_fn
from astrai.trainer.train_callback import (
    CallbackFactory,
    TrainCallback,
)
from astrai.trainer.train_context import TrainContext, TrainContextBuilder

logger = logging.getLogger(__name__)


class _StepMetricAccumulator:
    """Batch scalar device metrics without synchronizing every microbatch."""

    def __init__(self):
        self._names: Optional[tuple[str, ...]] = None
        self._vectors: List[torch.Tensor] = []

    def add(self, loss: torch.Tensor, metrics: Dict[str, torch.Tensor]) -> None:
        values = {"loss": loss.detach(), **metrics}
        names = tuple(values)
        if self._names is None:
            self._names = names
        elif names != self._names:
            raise RuntimeError(
                "training metric names changed within a gradient accumulation step"
            )

        scalars = []
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"training metric {name!r} must be a scalar tensor")
            scalars.append(value.detach())
        self._vectors.append(torch.stack(scalars))

    def materialize(self, context: TrainContext) -> None:
        if not self._vectors or self._names is None:
            return

        # Only rank 0 owns metric/progress callbacks. One packed transfer
        # replaces O(metrics * grad_accum_steps * world_size) scalar .item()
        # synchronizations. Other ranks discard their detached metric vectors.
        if get_rank() == 0:
            averages = torch.stack(self._vectors).float().mean(dim=0)
            host_values = averages.cpu().tolist()
            for name, value in zip(self._names, host_values, strict=True):
                if hasattr(context, name):
                    setattr(context, name, value)

        self._names = None
        self._vectors.clear()


class Trainer:
    def __init__(
        self, train_config: TrainConfig, callbacks: Optional[List[TrainCallback]] = None
    ):
        self.train_config = train_config
        default_callbacks = self._get_default_callbacks()
        self.callbacks = (
            default_callbacks + callbacks if callbacks else default_callbacks
        )

    def _get_default_callbacks(self) -> List[TrainCallback]:
        cfg = self.train_config
        callbacks = [
            CallbackFactory.create(
                "gradient_checkpointing",
                modules=cfg.gradient_checkpointing_modules,
            ),
            CallbackFactory.create(
                "checkpoint",
                cfg.ckpt_dir,
                cfg.ckpt_interval,
                checkpoint_after_first_step=cfg.checkpoint_after_first_step,
            ),
            # Clip before metric/progress callbacks so the current step's
            # gradient norm is recorded instead of the previous step's value.
            CallbackFactory.create("gradient_clipping", cfg.max_grad_norm),
            CallbackFactory.create(
                "metric",
                log_dir=cfg.log_dir,
                save_interval=cfg.ckpt_interval,
                metrics=cfg.metrics,
                val_step=cfg.val_step,
            ),
            CallbackFactory.create("progress_bar", cfg.n_epoch),
        ]
        return callbacks

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    @staticmethod
    def _stop_requested(context: TrainContext) -> bool:
        stop_file = context.config.stop_file
        requested = bool(stop_file and Path(stop_file).exists())

        # A shared filesystem can become visible to ranks at slightly different
        # times. Make the decision collective so every rank enters checkpoint
        # collectives and exits the loop together.
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor(
                int(requested), device=get_current_device(), dtype=torch.int32
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            requested = bool(flag.item())
        return requested

    def _trainer_loop(self, param_path: Optional[str] = None, resume: bool = False):
        context = (
            TrainContextBuilder(self.train_config)
            .with_param_path(param_path, resume=resume)
            .build()
        )
        executor = context.executor
        self._call_callbacks("on_train_begin", context)

        try:
            context.model.train()
            stop_requested = False
            step_metrics = _StepMetricAccumulator()
            interval_started_at = time.perf_counter()
            interval_tokens = 0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(get_current_device())

            for epoch in range(context.epoch, context.config.n_epoch):
                context.epoch = epoch
                self._call_callbacks("on_epoch_begin", context)

                for batch in context.dataloader:
                    input_ids = batch.get("input_ids")
                    if isinstance(input_ids, torch.Tensor):
                        interval_tokens += input_ids.numel() * context.world_size
                    with executor.accumulate(context.model):
                        self._call_callbacks("on_batch_begin", context)
                        loss = context.strategy(batch)
                        step_metrics.add(loss, context.strategy.last_metrics)
                        stand_loss = loss / executor.grad_accum_steps
                        executor.backward(stand_loss)
                        context.consumed_samples += (
                            context.config.batch_per_device * context.world_size
                        )
                        self._call_callbacks("on_batch_end", context)

                        if executor.sync_gradients:
                            now = time.perf_counter()
                            context.step_time = now - interval_started_at
                            if interval_tokens > 0:
                                context.tokens_per_second = (
                                    interval_tokens / context.step_time
                                )
                            if torch.cuda.is_available():
                                context.peak_memory_gb = (
                                    torch.cuda.max_memory_allocated(
                                        get_current_device()
                                    )
                                    / 1e9
                                )
                                torch.cuda.reset_peak_memory_stats(get_current_device())
                            interval_started_at = now
                            interval_tokens = 0
                            step_metrics.materialize(context)
                            self._call_callbacks("on_optimizer_step", context)
                            context.optimizer.step()
                            context.strategy.on_optimizer_step()
                            context.optimizer.zero_grad()

                            if context.scheduler:
                                context.scheduler.step()
                            self._call_callbacks("on_optimizer_step_end", context)

                            if self._stop_requested(context):
                                stop_requested = True
                                break

                if stop_requested:
                    logger.info(
                        "Graceful stop requested at optimizer step %d; "
                        "saving final checkpoint.",
                        context.optimizer_step,
                    )
                    break

                self._call_callbacks("on_epoch_end", context)

        except Exception as e:
            logger.error("Training failed: %s", str(e), exc_info=True)
            self._call_callbacks("on_error", context)
            raise
        finally:
            self._call_callbacks("on_train_end", context)

    def train(self, param_path: Optional[str] = None, resume: bool = False):
        cfg = self.train_config
        spawn_parallel_fn(
            self._trainer_loop,
            backend=cfg.backend,
            world_size=cfg.nprocs,
            master_addr=cfg.master_addr,
            master_port=cfg.master_port,
            device_type=cfg.device_type,
            start_method=cfg.start_method,
            param_path=param_path,
            resume=resume,
        )
