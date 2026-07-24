from types import SimpleNamespace

import pytest
import torch

from astrai.trainer.trainer import _StepMetricAccumulator


def test_step_metrics_are_averaged_at_optimizer_boundary():
    accumulator = _StepMetricAccumulator()
    accumulator.add(
        torch.tensor(2.0),
        {"language_model_loss": torch.tensor(1.5)},
    )
    accumulator.add(
        torch.tensor(4.0),
        {"language_model_loss": torch.tensor(3.5)},
    )
    context = SimpleNamespace(loss=0.0, language_model_loss=0.0)

    accumulator.materialize(context)

    assert context.loss == 3.0
    assert context.language_model_loss == 2.5


def test_step_metrics_reject_non_scalar_values():
    accumulator = _StepMetricAccumulator()

    with pytest.raises(TypeError, match="scalar tensor"):
        accumulator.add(
            torch.tensor(1.0),
            {"bad": torch.ones(2)},
        )


def test_step_metrics_require_stable_names_within_accumulation():
    accumulator = _StepMetricAccumulator()
    accumulator.add(torch.tensor(1.0), {"first": torch.tensor(2.0)})

    with pytest.raises(RuntimeError, match="metric names changed"):
        accumulator.add(torch.tensor(1.0), {"second": torch.tensor(2.0)})
