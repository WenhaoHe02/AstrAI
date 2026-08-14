from types import SimpleNamespace

import pytest

from astrai.config.train_config import TrainConfig
from astrai.trainer.train_context import TrainContext


def test_train_context_reports_scaling_coordinates():
    config = SimpleNamespace(
        batch_per_device=8,
        grad_accum_steps=4,
        sequence_length=2048,
        unique_train_tokens=46_160_000_000,
    )
    context = TrainContext(
        config=config,
        world_size=8,
        consumed_samples=8 * 8 * 4 * 100,
    )

    assert context.optimizer_step == 100
    assert context.global_batch_tokens == 524_288
    assert context.seen_tokens == 52_428_800
    assert context.effective_epochs == pytest.approx(
        52_428_800 / 46_160_000_000
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (("sequence_length", 0), ("unique_train_tokens", 0)),
)
def test_scaling_coordinates_must_be_positive(field, value):
    kwargs = {
        "model_fn": lambda: None,
        "strategy": "seq",
        "dataset": object(),
        "optimizer_fn": lambda model: None,
        "scheduler_fn": lambda optimizer: None,
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        TrainConfig(**kwargs)
