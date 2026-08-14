import argparse
import logging
import os
from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.optim as optim
from torch import Tensor, nn
from torch.distributed.fsdp import ShardingStrategy

from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.dataset import DatasetFactory, dpo_collate_fn, grpo_collate_fn
from astrai.model import AutoRegressiveLM
from astrai.model.components.decoder_block import DecoderBlock
from astrai.trainer import SchedulerFactory, Trainer
from astrai.trainer.rollout import BaseRewardModel

logger = logging.getLogger(__name__)


class MatrixAwareOptimizer(optim.Optimizer):
    """Fused AdamW with optional Muon for visible unsharded 2D weights."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
        enable_muon: bool = False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, defaults)

        matrix_params: list[Tensor] = []
        other_params: list[Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if enable_muon and (
                param.dim() == 2
                and "norm" not in name
                and "bias" not in name
                and "embed" not in name
                and "lm_head" not in name
                and ".router." not in name
            ):
                matrix_params.append(param)
            else:
                other_params.append(param)

        self.muon = (
            optim.Muon(
                matrix_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                adjust_lr_fn=adjust_lr_fn,
            )
            if matrix_params
            else None
        )
        self.adamw = optim.AdamW(
            [{"params": other_params, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
            fused=True,
        )

        self.param_groups = [
            *(self.muon.param_groups if self.muon is not None else []),
            *self.adamw.param_groups,
        ]
        logger.info(
            "MatrixAwareOptimizer: Muon params=%d (%.3fB), "
            "fused AdamW params=%d (%.3fB)",
            len(matrix_params),
            sum(param.numel() for param in matrix_params) / 1e9,
            len(other_params),
            sum(param.numel() for param in other_params) / 1e9,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if self.muon is not None:
            self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        if self.muon is not None:
            self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        state = {"adamw": self.adamw.state_dict()}
        if self.muon is not None:
            state["muon"] = self.muon.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.muon is not None and "muon" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
        adamw_state = self._expand_zero2_adamw_state(state_dict["adamw"])
        self.adamw.load_state_dict(adamw_state)
        self.param_groups = [
            *(self.muon.param_groups if self.muon is not None else []),
            *self.adamw.param_groups,
        ]

    def _expand_zero2_adamw_state(self, state_dict: Dict[str, Any]):
        """Convert the old FSDP ZeRO-2 flat state to replicated AdamW state.

        SHARD_GRAD_OP stores all non-expert parameters as one flat optimizer
        shard per rank, while ignored EP expert tensors already have complete
        rank-local states.  NO_SHARD exposes the original parameters again.
        Reassemble the flat moments with NCCL, split them in original parameter
        order, and retain the local expert moments verbatim.
        """
        if self.muon is not None or not dist.is_available() or not dist.is_initialized():
            return state_dict
        saved_state = state_dict.get("state", {})
        params = self.adamw.param_groups[0]["params"]
        if not saved_state or not params:
            return state_dict
        first_entry = next(iter(saved_state.values()))
        if "exp_avg" not in first_entry:
            return state_dict

        # Rank-local grouped-expert parameters are 3D and remain complete in
        # each old rank checkpoint. Map them by traversal order rather than
        # numeric parameter ID so QKV/gate-up parameter fusion can change the
        # number of intervening non-expert parameters without losing moments.
        saved_expert_ids = [
            param_id
            for param_id in sorted(saved_state)
            if saved_state[param_id]["exp_avg"].ndim == 3
        ]
        current_expert_ids = [
            param_id for param_id, parameter in enumerate(params) if parameter.ndim == 3
        ]
        expert_state_by_current_id = {}
        saved_expert_cursor = 0
        expert_layout_changed = len(saved_expert_ids) != len(current_expert_ids)
        for current_id in current_expert_ids:
            current_shape = tuple(params[current_id].shape)
            if saved_expert_cursor >= len(saved_expert_ids):
                raise RuntimeError(
                    "cannot migrate optimizer state: missing rank-local expert state"
                )
            saved_id = saved_expert_ids[saved_expert_cursor]
            saved_entry = saved_state[saved_id]
            saved_shape = tuple(saved_entry["exp_avg"].shape)
            if saved_shape == current_shape:
                expert_state_by_current_id[current_id] = saved_entry
                saved_expert_cursor += 1
                continue
            # Legacy routed experts store up and gate separately as [E,F,D].
            # The fused layout stores [E,2F,D], preserving values by concat.
            if saved_expert_cursor + 1 < len(saved_expert_ids):
                next_entry = saved_state[saved_expert_ids[saved_expert_cursor + 1]]
                if (
                    len(current_shape) == 3
                    and saved_shape == tuple(next_entry["exp_avg"].shape)
                    and current_shape
                    == (saved_shape[0], 2 * saved_shape[1], saved_shape[2])
                ):
                    expert_state_by_current_id[current_id] = {
                        "step": saved_entry["step"],
                        "exp_avg": torch.cat(
                            [saved_entry["exp_avg"], next_entry["exp_avg"]], dim=1
                        ),
                        "exp_avg_sq": torch.cat(
                            [saved_entry["exp_avg_sq"], next_entry["exp_avg_sq"]],
                            dim=1,
                        ),
                    }
                    expert_layout_changed = True
                    saved_expert_cursor += 2
                    continue
            raise RuntimeError(
                "cannot migrate optimizer state: rank-local expert shape "
                f"{saved_shape} cannot map to {current_shape}"
            )
        if saved_expert_cursor != len(saved_expert_ids):
            raise RuntimeError(
                "cannot migrate optimizer state: unused rank-local expert states "
                f"({len(saved_expert_ids) - saved_expert_cursor})"
            )
        saved_expert_ids = set(saved_expert_ids)
        current_expert_ids = set(current_expert_ids)
        needs_migration = expert_layout_changed or any(
            param_id not in current_expert_ids
            and (
                param_id not in saved_state
                or tuple(saved_state[param_id]["exp_avg"].shape)
                != tuple(parameter.shape)
            )
            for param_id, parameter in enumerate(params)
        )
        if not needs_migration:
            return state_dict

        expected_numel = sum(
            parameter.numel()
            for param_id, parameter in enumerate(params)
            if param_id not in current_expert_ids
        )
        gathered_moments = {}
        device = params[0].device
        for key in ("exp_avg", "exp_avg_sq"):
            # FSDP maps a rank's contiguous flat shard back across multiple
            # original parameter IDs. Re-pack those fragments in ID order.
            local = torch.cat(
                [
                    saved_state[param_id][key].flatten()
                    for param_id in sorted(saved_state)
                    if param_id not in saved_expert_ids
                ]
            ).to(device=device, non_blocking=True)
            if local.numel() == expected_numel:
                # Replicated legacy AdamW checkpoint: only parameter fusion
                # changed IDs/shapes, so the local flat order is already full.
                gathered_moments[key] = local
                continue
            local_size = torch.tensor([local.numel()], device=device, dtype=torch.int64)
            size_list = [torch.empty_like(local_size) for _ in range(dist.get_world_size())]
            dist.all_gather(size_list, local_size)
            sizes = [int(value.item()) for value in size_list]
            max_size = max(sizes)
            if local.numel() < max_size:
                local = torch.nn.functional.pad(local, (0, max_size - local.numel()))
            gathered = torch.empty(
                dist.get_world_size() * max_size, dtype=local.dtype, device=device
            )
            dist.all_gather_into_tensor(gathered, local.contiguous())
            rank_chunks = gathered.view(dist.get_world_size(), max_size)
            full = torch.cat(
                [rank_chunks[rank, :size] for rank, size in enumerate(sizes)]
            )
            if full.numel() < expected_numel:
                raise RuntimeError(
                    "cannot migrate ZeRO-2 optimizer state: gathered flat size "
                    f"{full.numel()} < expected {expected_numel}"
                )
            # Any divisibility padding added by FSDP is at the end.
            gathered_moments[key] = full[:expected_numel]

        current_param_groups = self.adamw.state_dict()["param_groups"]
        for current_group, saved_group in zip(
            current_param_groups, state_dict["param_groups"]
        ):
            current_params = current_group["params"]
            current_group.update(
                {key: value for key, value in saved_group.items() if key != "params"}
            )
            current_group["params"] = current_params
        migrated = {"state": {}, "param_groups": current_param_groups}
        flat_step = next(
            saved_state[param_id]["step"]
            for param_id in sorted(saved_state)
            if param_id not in saved_expert_ids
        )
        offset = 0
        for param_id, parameter in enumerate(params):
            if param_id in current_expert_ids:
                migrated["state"][param_id] = expert_state_by_current_id[param_id]
                continue
            end = offset + parameter.numel()
            migrated["state"][param_id] = {
                "step": flat_step,
                "exp_avg": gathered_moments["exp_avg"][offset:end].view(
                    parameter.shape
                ),
                "exp_avg_sq": gathered_moments["exp_avg_sq"][offset:end].view(
                    parameter.shape
                ),
            }
            offset = end
        if offset != expected_numel:
            raise RuntimeError(
                f"optimizer migration consumed {offset} values, expected {expected_numel}"
            )
        logger.info(
            "Migrated ZeRO-2 AdamW state to replicated layout: %.3fB values, "
            "%d rank-local expert tensors preserved",
            expected_numel / 1e9,
            len(current_expert_ids),
        )
        return migrated


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train the AutoRegressiveLM model.")

    parser.add_argument(
        "--train_type",
        type=str,
        required=True,
        choices=["seq", "sft", "dpo", "grpo", "online_grpo", "online_dpo"],
        help="Train type.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="Path to the root directory of the dataset.",
    )
    parser.add_argument(
        "--param_path",
        type=str,
        required=True,
        help="Path to the model parameters or resume checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint at --param_path "
        "(restore epoch, consumed_samples, optimizer & scheduler state).",
    )

    parser.add_argument(
        "--n_epoch", type=int, default=1, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_per_device", type=int, default=1, help="Batch size per GPU."
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of iterations between each optimizer step.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for LR warmup.",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4, help="Max learning rate for training."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping. None disables clipping.",
    )
    parser.add_argument(
        "--optimizer_type",
        choices=["adamw", "muon"],
        default="adamw",
        help="Optimizer for visible 2D weights. Existing pretraining checkpoints use AdamW.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay (applied to Muon matrix params; non-matrix use 0).",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="Momentum factor for Muon optimizer.",
    )
    parser.add_argument(
        "--muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Nesterov momentum for Muon.",
    )
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz iteration steps for Muon.",
    )
    parser.add_argument(
        "--muon_adjust_lr",
        type=str,
        default="match_rms_adamw",
        choices=["original", "match_rms_adamw"],
        help="Muon learning rate adjustment strategy.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=3407, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pin memory",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Max length of the input sequence.",
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="Step size of the input sequence."
    )
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size.")
    parser.add_argument(
        "--grpo_clip_eps", type=float, default=0.2, help="GRPO clipping epsilon."
    )
    parser.add_argument(
        "--grpo_kl_coef", type=float, default=0.01, help="GRPO KL penalty coefficient."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="cross_entropy function label smoothing parameter",
    )

    # online rollout
    parser.add_argument(
        "--rollout_interval",
        type=int,
        default=512,
        help="Number of optimizer steps between online rollouts.",
    )
    parser.add_argument(
        "--rollout_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for online rollout.",
    )
    parser.add_argument(
        "--rollout_top_k",
        type=int,
        default=0,
        help="Top-k filtering for online rollout (0=disable).",
    )
    parser.add_argument(
        "--rollout_top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) filtering for online rollout.",
    )
    parser.add_argument(
        "--rollout_max_tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per response in rollout.",
    )

    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable activation checkpointing for DecoderBlock modules.",
    )
    parser.add_argument(
        "--loss_backend",
        type=str,
        default="torch",
        choices=["torch", "liger"],
        help=(
            "Language-model loss backend. Liger fuses the LM head and cross "
            "entropy without materializing full-vocabulary logits."
        ),
    )
    parser.add_argument(
        "--swiglu_backend",
        type=str,
        default=None,
        choices=["torch", "liger"],
        help=(
            "Override the model SwiGLU activation backend. Liger fuses SiLU "
            "and multiplication for shared and routed experts."
        ),
    )
    parser.add_argument(
        "--residual_norm_backend",
        type=str,
        default=None,
        choices=["torch", "liger"],
        help=(
            "Override the attention residual plus post-attention RMSNorm "
            "backend. Liger performs both operations in one kernel."
        ),
    )
    parser.add_argument(
        "--router_score_dtype",
        type=str,
        default=None,
        choices=["model", "fp32"],
        help=(
            "Override MoE routing score precision. fp32 avoids quantizing the "
            "full probability matrix before top-k selection."
        ),
    )
    parser.add_argument(
        "--deepep_cpu_sync",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override model DeepEP receive sizing. Disabling CPU sync uses "
            "fixed-capacity tensors and GPU-resident expert offsets."
        ),
    )
    parser.add_argument(
        "--attention_backend",
        choices=["auto", "flash_sdpa", "transformer_engine"],
        default=None,
        help="Attention kernel backend; Transformer Engine selects Hopper fused attention.",
    )
    parser.add_argument(
        "--fused_qkv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fuse GQA Q/K/V projections into one checkpoint-compatible GEMM.",
    )
    parser.add_argument(
        "--fused_mlp_gate_up",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fuse dense/shared and routed-expert up/gate projections.",
    )
    parser.add_argument(
        "--moe_route_scale_before_down",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override routing-weight placement for DeepEP experts. Enabling "
            "scales the narrower SwiGLU activation before the down projection."
        ),
    )

    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=5000,
        help="Number of iters between checkpoints.",
    )
    parser.add_argument(
        "--ckpt_keep_last",
        type=int,
        default=None,
        help="Keep only the newest N complete checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_after_first_step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a recovery checkpoint after optimizer step 1.",
    )
    parser.add_argument(
        "--stop_file",
        type=str,
        default=None,
        help=(
            "Shared file whose presence requests a graceful stop after the "
            "current optimizer step and writes a final checkpoint."
        ),
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoint",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=None,
        help="Ratio to split from training dataset for validation (e.g. 0.05).",
    )
    parser.add_argument(
        "--val_data_root_path",
        type=str,
        default=None,
        help=(
            "Path to a separate tokenized validation dataset. Preferred over "
            "--val_split for scaling comparisons and never mixed into training."
        ),
    )
    parser.add_argument(
        "--val_step",
        type=int,
        default=1000,
        help="Number of optimizer steps between validation runs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["loss", "lr", "grad_norm"],
        help="Metrics to log (e.g. --metrics loss lr val_loss). Default: loss lr grad_norm.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="checkpoint/logs",
        help="Directory for metric logs.",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="Start epoch for training."
    )
    parser.add_argument(
        "--start_samples",
        type=int,
        default=0,
        help="Start samples (per rank) for training.",
    )

    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master node address for distributed training.",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="Master node port for distributed training.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Distributed training backend.",
    )
    parser.add_argument("--nprocs", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp", "fsdp2"],
        help="Parallel training strategy (none, ddp, fsdp, fsdp2).",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
        help=(
            "FSDP sharding level: full_shard is ZeRO-3; shard_grad_op is "
            "ZeRO-2 and keeps full parameters resident through backward; "
            "no_shard replicates non-expert parameters and removes parameter "
            "all-gathers (rank-local EP experts remain excluded from FSDP)."
        ),
    )
    parser.add_argument(
        "--device_type", type=str, default="cuda", help="Device type to use."
    )
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method.",
    )
    parser.add_argument(
        "--neftune_alpha",
        type=float,
        default=0.0,
        help="NEFTune noise alpha (0=disabled, typical: 5.0).",
    )

    parser.add_argument(
        "--schedule_type",
        type=str,
        default="cosine",
        choices=["cosine", "sgdr", "wsd"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--min_rate",
        type=float,
        default=None,
        help="Minimum LR as fraction of base LR. Uses scheduler default if not set (cosine/sgdr: 0.05, wsd: 0.0).",
    )
    parser.add_argument(
        "--cycle_length",
        type=int,
        default=None,
        help="SGDR first cycle length in steps. Defaults to total_steps - warmup_steps.",
    )
    parser.add_argument(
        "--t_mult",
        type=int,
        default=2,
        help="SGDR cycle length multiplier per restart.",
    )
    parser.add_argument(
        "--stable_steps",
        type=int,
        default=None,
        help="WSD stable plateau steps. Required when --schedule_type wsd.",
    )
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=None,
        help="WSD decay steps. Defaults to total_steps - warmup_steps - stable_steps.",
    )

    args = parser.parse_args()

    return args


def create_model(config):
    return AutoRegressiveLM(config).to(dtype=torch.bfloat16)


def create_optimizer(model, **kwargs) -> MatrixAwareOptimizer:
    return MatrixAwareOptimizer(model, **kwargs)


def create_scheduler(
    optimizer: optim.Optimizer, **kwargs
) -> optim.lr_scheduler.LRScheduler:
    schedule_type = kwargs.pop("schedule_type")
    return SchedulerFactory.create(schedule_type, optimizer, **kwargs)


def compute_total_steps(
    dataset_len: int,
    n_epoch: int,
    batch_per_device: int,
    nprocs: int,
    grad_accum_steps: int,
) -> int:

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    samples_per_replica = ceil_div(dataset_len, nprocs)
    batches_per_replica = ceil_div(samples_per_replica, batch_per_device)
    total_steps = (batches_per_replica // grad_accum_steps) * n_epoch
    return total_steps


def train(
    train_type: str,
    param_path: str,
    data_root_path: str,
    resume: bool,
    n_epoch: int,
    batch_per_device: int,
    start_epoch: int,
    start_samples: int,
    grad_accum_steps: int,
    warmup_ratio: float,
    ckpt_interval: int,
    ckpt_keep_last: Optional[int],
    checkpoint_after_first_step: bool,
    stop_file: str,
    ckpt_dir: str,
    val_split: float,
    val_data_root_path: Optional[str],
    val_step: int,
    metrics: list[str],
    log_dir: str,
    max_grad_norm: float,
    optimizer_type: str,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
    gradient_checkpointing: bool,
    loss_backend: str,
    swiglu_backend: Optional[str],
    residual_norm_backend: Optional[str],
    router_score_dtype: Optional[str],
    deepep_cpu_sync: Optional[bool],
    attention_backend: Optional[str],
    fused_qkv: Optional[bool],
    fused_mlp_gate_up: Optional[bool],
    moe_route_scale_before_down: Optional[bool],
    window_size: int,
    stride: int,
    nprocs: int,
    parallel_mode: str,
    fsdp_sharding_strategy: str,
    device_type: str,
    backend: str,
    master_addr: str,
    master_port: str,
    start_method: str,
    neftune_alpha: float,
    schedule_type: str,
    min_rate: float,
    cycle_length: int,
    t_mult: int,
    stable_steps: int,
    decay_steps: int,
    **kwargs,
):
    assert train_type in [
        "seq",
        "sft",
        "dpo",
        "grpo",
        "online_grpo",
        "online_dpo",
    ]
    assert os.path.exists(param_path)
    if nprocs > 1 and parallel_mode == "none":
        raise ValueError(
            "--nprocs > 1 requires --parallel_mode to be 'ddp', 'fsdp', or 'fsdp2'"
        )
    if loss_backend != "torch" and train_type != "seq":
        raise ValueError("the Liger loss backend currently supports train_type=seq")

    # Load config
    config_path = os.path.join(param_path, "config.json")
    config = AutoRegressiveLMConfig.from_file(config_path)
    config.neftune_alpha = neftune_alpha
    if swiglu_backend is not None:
        config.swiglu_backend = swiglu_backend
    if residual_norm_backend is not None:
        config.residual_norm_backend = residual_norm_backend
    if router_score_dtype is not None:
        config.router_score_dtype = router_score_dtype
    if deepep_cpu_sync is not None:
        config.deepep_cpu_sync = deepep_cpu_sync
    if attention_backend is not None:
        config.attention_backend = attention_backend
    if fused_qkv is not None:
        config.fused_qkv = fused_qkv
    if fused_mlp_gate_up is not None:
        config.fused_mlp_gate_up = fused_mlp_gate_up
    if moe_route_scale_before_down is not None:
        config.moe_route_scale_before_down = moe_route_scale_before_down

    if window_size is None:
        window_size = config.max_position_embeddings

    strategy_kwargs = {
        "beta": kwargs.pop("dpo_beta"),
        "label_smoothing": kwargs.pop("label_smoothing"),
        "clip_eps": kwargs.pop("grpo_clip_eps"),
        "kl_coef": kwargs.pop("grpo_kl_coef"),
        "group_size": kwargs.pop("group_size"),
        "loss_backend": loss_backend,
    }

    rollout_interval = kwargs.pop("rollout_interval", 512)
    rollout_temperature = kwargs.pop("rollout_temperature", 0.7)
    rollout_top_k = kwargs.pop("rollout_top_k", 0)
    rollout_top_p = kwargs.pop("rollout_top_p", 0.9)
    rollout_max_tokens = kwargs.pop("rollout_max_tokens", 1024)
    reward_model_fn: Optional[Callable[[], BaseRewardModel]] = None

    executor_kwargs = {}
    if parallel_mode == "ddp":
        executor_kwargs.update(
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=True,
            bucket_cap_mb=100,
        )
    elif parallel_mode == "fsdp":
        executor_kwargs["sharding_strategy"] = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }[fsdp_sharding_strategy]

    model_fn = partial(create_model, config)
    dataset = DatasetFactory.load(
        train_type=train_type,
        load_path=data_root_path,
        window_size=window_size,
        stride=stride,
        tokenizer_path=param_path,
    )
    val_dataset = None
    if val_data_root_path is not None:
        if not os.path.exists(val_data_root_path):
            raise FileNotFoundError(
                f"validation dataset not found: {val_data_root_path}"
            )
        val_dataset = DatasetFactory.load(
            train_type=train_type,
            load_path=val_data_root_path,
            window_size=window_size,
            stride=stride,
            tokenizer_path=param_path,
        )

    optimizer_fn = partial(
        create_optimizer,
        lr=kwargs.pop("max_lr"),
        weight_decay=kwargs.pop("weight_decay"),
        momentum=kwargs.pop("muon_momentum"),
        nesterov=kwargs.pop("muon_nesterov"),
        ns_steps=kwargs.pop("muon_ns_steps"),
        adjust_lr_fn=kwargs.pop("muon_adjust_lr"),
        enable_muon=optimizer_type == "muon",
    )

    total_steps = compute_total_steps(
        len(dataset), n_epoch, batch_per_device, nprocs, grad_accum_steps
    )
    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, total_steps)

    unique_train_tokens = getattr(dataset, "token_count", None)
    if not unique_train_tokens:
        unique_train_tokens = len(dataset) * window_size
    if val_dataset is None and val_split is not None:
        unique_train_tokens = max(1, int(unique_train_tokens * (1.0 - val_split)))
    global_batch_tokens = batch_per_device * nprocs * grad_accum_steps * window_size
    planned_seen_tokens = total_steps * global_batch_tokens
    logger.info(
        "Scaling budget: D=%d unique tokens, K=%d planned seen tokens, "
        "K/D=%.4f, global_batch_tokens=%d, total_steps=%d",
        unique_train_tokens,
        planned_seen_tokens,
        planned_seen_tokens / unique_train_tokens,
        global_batch_tokens,
        total_steps,
    )

    scheduler_kwargs = {"warmup_steps": warmup_steps}

    if schedule_type == "cosine":
        scheduler_kwargs["lr_decay_steps"] = total_steps - warmup_steps
    elif schedule_type == "sgdr":
        scheduler_kwargs["cycle_length"] = cycle_length or (total_steps - warmup_steps)
        scheduler_kwargs["t_mult"] = t_mult
    elif schedule_type == "wsd":
        remaining = total_steps - warmup_steps
        stable_steps_ = stable_steps or max(1, int(remaining * 0.8))
        scheduler_kwargs["stable_steps"] = stable_steps_
        scheduler_kwargs["decay_steps"] = max(
            1, decay_steps or (remaining - stable_steps_)
        )

    if min_rate is not None:
        scheduler_kwargs["min_rate"] = min_rate

    scheduler_fn = partial(
        create_scheduler,
        schedule_type=schedule_type,
        **scheduler_kwargs,
    )

    grad_ckpt_modules = [DecoderBlock] if gradient_checkpointing else []

    collate_fn = None
    if train_type == "dpo":
        collate_fn = dpo_collate_fn
    elif train_type == "grpo":
        collate_fn = grpo_collate_fn
    elif train_type in ("online_grpo", "online_dpo"):
        collate_fn = None

    train_config = TrainConfig(
        model_fn=model_fn,
        strategy=train_type,
        dataset=dataset,
        val_dataset=val_dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch,
        batch_per_device=batch_per_device,
        start_epoch=start_epoch,
        start_samples=start_samples,
        ckpt_interval=ckpt_interval,
        ckpt_keep_last=ckpt_keep_last,
        checkpoint_after_first_step=checkpoint_after_first_step,
        stop_file=stop_file,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        nprocs=nprocs,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
        parallel_mode=parallel_mode,
        device_type=device_type,
        start_method=start_method,
        val_split=val_split,
        val_step=val_step,
        sequence_length=window_size,
        unique_train_tokens=unique_train_tokens,
        metrics=metrics,
        log_dir=log_dir,
        gradient_checkpointing_modules=grad_ckpt_modules,
        executor_kwargs=executor_kwargs,
        extra_kwargs=strategy_kwargs,
        neftune_alpha=neftune_alpha,
        collate_fn=collate_fn,
        rollout_interval=rollout_interval,
        rollout_temperature=rollout_temperature,
        rollout_top_k=rollout_top_k,
        rollout_top_p=rollout_top_p,
        rollout_max_tokens=rollout_max_tokens,
        reward_model_fn=reward_model_fn,
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=param_path, resume=resume)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    train(**vars(args))
