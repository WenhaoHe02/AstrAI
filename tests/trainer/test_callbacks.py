import os
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from astrai.config.train_config import TrainConfig
from astrai.model.components.decoder_block import DecoderBlock
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.train_callback import (
    CheckpointCallback,
    GradientCheckpointingCallback,
    TrainCallback,
    prune_checkpoint_dirs,
)
from astrai.trainer.trainer import Trainer


def test_gradient_checkpointing_enable_disable(test_model):
    """Enable wraps forward, _disable restores it."""
    model = test_model["model"]
    callback = GradientCheckpointingCallback(modules=[DecoderBlock])

    originals = [layer.forward for layer in model.layers]

    for layer in model.layers:
        callback._enable(layer)

    for layer in model.layers:
        assert hasattr(layer, "_original_forward")
        assert layer.forward is not originals[0]

    for layer in model.layers:
        callback._disable(layer)

    for layer in model.layers:
        assert not hasattr(layer, "_original_forward")


def test_checkpoint_after_first_step_is_one_shot():
    callback = CheckpointCallback(
        "unused", interval=5000, checkpoint_after_first_step=True
    )
    context = SimpleNamespace(optimizer_step=0)
    callback.on_train_begin(context)
    callback._save_checkpoint = Mock()

    context.optimizer_step = 1
    callback.on_optimizer_step_end(context)
    context.optimizer_step = 2
    callback.on_optimizer_step_end(context)

    callback._save_checkpoint.assert_called_once_with(context)


def test_checkpoint_after_first_step_is_relative_to_resume_step():
    callback = CheckpointCallback(
        "unused", interval=5000, checkpoint_after_first_step=True
    )
    context = SimpleNamespace(optimizer_step=48081)
    callback.on_train_begin(context)
    callback._save_checkpoint = Mock()

    context.optimizer_step = 48082
    callback.on_optimizer_step_end(context)
    context.optimizer_step = 48083
    callback.on_optimizer_step_end(context)

    callback._save_checkpoint.assert_called_once_with(context)


def test_checkpoint_on_error_does_not_save_partial_state():
    callback = CheckpointCallback("unused", interval=5000)
    callback._save_checkpoint = Mock()

    callback.on_error(SimpleNamespace(optimizer_step=7))

    callback._save_checkpoint.assert_not_called()


def test_prune_checkpoint_dirs_keeps_latest_complete(tmp_path):
    for step in (1, 251, 501, 751, 1001):
        checkpoint = tmp_path / f"epoch_0_step_{step}"
        checkpoint.mkdir()
        for name in ("model.safetensors", "config.json", "meta.json"):
            (checkpoint / name).touch()
        for rank in range(2):
            (checkpoint / f"optimizer.rank{rank}.pt").touch()
    incomplete = tmp_path / "epoch_0_step_1251"
    incomplete.mkdir()
    (incomplete / "model.safetensors").touch()

    removed = prune_checkpoint_dirs(tmp_path, keep_last=3, expected_optimizer_ranks=2)

    assert [path.name for path in removed] == ["epoch_0_step_1", "epoch_0_step_251"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "epoch_0_step_1001",
        "epoch_0_step_1251",
        "epoch_0_step_501",
        "epoch_0_step_751",
    ]


def test_stop_file_requests_graceful_stop(tmp_path):
    stop_file = tmp_path / "STOP"
    context = SimpleNamespace(
        config=SimpleNamespace(stop_file=str(stop_file)),
    )

    assert Trainer._stop_requested(context) is False
    stop_file.touch()
    assert Trainer._stop_requested(context) is True


def test_gradient_checkpointing_empty_modules_noop(test_model):
    """modules=None should leave forwards untouched."""
    model = test_model["model"]
    callback = GradientCheckpointingCallback()

    originals = [layer.forward for layer in model.layers]

    for layer in model.layers:
        callback._enable(layer)

    for layer, orig in zip(model.layers, originals):
        assert layer.forward is orig


def test_gradient_checkpointing_forward_unchanged(test_model):
    """Forward output unchanged after patching (no_grad)."""
    model = test_model["model"]
    device = test_model["device"]
    callback = GradientCheckpointingCallback(modules=[DecoderBlock])

    input_ids = torch.randint(0, 1000, (2, 32)).to(device)

    with torch.no_grad():
        ref = model(input_ids)["logits"].clone()

    for layer in model.layers:
        callback._enable(layer)

    with torch.no_grad():
        out = model(input_ids)["logits"]

    assert torch.equal(ref, out)


def test_gradient_checkpointing_backward(test_model):
    """backward passes gradients through checkpointed layers."""
    model = test_model["model"]
    device = test_model["device"]
    callback = GradientCheckpointingCallback(modules=[DecoderBlock])

    for layer in model.layers:
        callback._enable(layer)

    input_ids = torch.randint(0, 1000, (2, 32)).to(device)
    target_ids = torch.randint(0, 1000, (2, 32)).to(device)

    logits = model(input_ids)["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), target_ids.flatten()
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} gradient is None"

    for layer in model.layers:
        callback._disable(layer)

    model.zero_grad()
    for name, p in model.named_parameters():
        assert p.grad is None or p.grad.sum().item() == 0, f"{name} grad not zeroed"


def test_gradient_checkpointing_trainer_integration(base_test_env, random_dataset):
    """Gradient checkpointing runs end-to-end via Trainer."""

    def optimizer_fn(model):
        return torch.optim.AdamW(model.parameters())

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            "cosine", optim, warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    train_config = TrainConfig(
        model_fn=lambda: base_test_env["model"],
        strategy="seq",
        dataset=random_dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=base_test_env["test_dir"],
        log_dir=os.path.join(base_test_env["test_dir"], "logs"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=3,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        random_seed=42,
        device_type=base_test_env["device"],
        gradient_checkpointing_modules=[DecoderBlock],
    )

    trainer = Trainer(train_config)
    trainer.train()
    # no crash = callback correctly enabled/disabled


def test_callback_integration(base_test_env, random_dataset):
    """Test that all callbacks are properly integrated"""

    def optimizer_fn(model):
        return torch.optim.AdamW(model.parameters())

    def scheduler_fn(optim):
        return SchedulerFactory.create(
            "cosine", optim, warmup_steps=10, lr_decay_steps=10, min_rate=0.05
        )

    train_config = TrainConfig(
        model_fn=lambda: base_test_env["model"],
        strategy="seq",
        dataset=random_dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=base_test_env["test_dir"],
        log_dir=os.path.join(base_test_env["test_dir"], "logs"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=3,
        grad_accum_steps=1,
        max_grad_norm=1.0,
        random_seed=42,
        device_type=base_test_env["device"],
    )

    # Create custom callbacks to track calls
    callback_calls = []

    class TrackingCallback(TrainCallback):
        def on_train_begin(self, context):
            callback_calls.append("on_train_begin")

        def on_batch_end(self, context):
            callback_calls.append("on_batch_end")

        def on_epoch_end(self, context):
            callback_calls.append("on_epoch_end")

    trainer = Trainer(train_config, callbacks=[TrackingCallback()])

    trainer.train()

    # Verify callbacks were called
    assert "on_train_begin" in callback_calls
    assert "on_batch_end" in callback_calls
    assert "on_epoch_end" in callback_calls
