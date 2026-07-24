import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM

TINY_CONFIG = dict(
    vocab_size=128,
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=16,
    max_position_embeddings=64,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
)


CONFIGS = [
    pytest.param(
        {**TINY_CONFIG, "attn_type": "gqa", "ffn_type": "mlp"},
        id="gqa_mlp",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "mla",
            "ffn_type": "mlp",
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
        },
        id="mla_mlp",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            "router_aux_loss_coef": 0.01,
            "router_z_loss_coef": 0.001,
        },
        id="gqa_moe",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "mlp",
            "rope_theta": 100000.0,
        },
        id="gqa_rope_theta",
    ),
    pytest.param(
        {**TINY_CONFIG, "attn_type": "gqa", "ffn_type": "mlp", "use_qk_norm": True},
        id="gqa_qk_norm",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "mlp",
            "tie_word_embeddings": True,
        },
        id="gqa_tie_word_embeddings",
    ),
]


@pytest.mark.parametrize("config_kwargs", CONFIGS)
def test_model_forward(config_kwargs):
    config = AutoRegressiveLMConfig(**config_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoRegressiveLM(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert "logits" in output
    assert "hidden_states" in output
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert output["hidden_states"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    )
    assert not torch.isnan(output["logits"]).any()
    assert not torch.isnan(output["hidden_states"]).any()

    if config.ffn_type == "moe":
        assert output["router_loss"].ndim == 0
        assert output["router_aux_loss"].ndim == 0
        assert output["router_z_loss"].ndim == 0
        assert output["router_entropy"].ndim == 0
        assert output["router_expert_load"].shape == (config.n_routed_experts,)
        assert torch.allclose(
            output["router_expert_load"].sum(),
            torch.ones((), device=device),
        )


@pytest.mark.parametrize("config_kwargs", CONFIGS)
def test_model_forward_with_padding(config_kwargs):
    config = AutoRegressiveLMConfig(**config_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoRegressiveLM(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    input_mask[:, 4:] = False

    with torch.no_grad():
        output = model(input_ids, input_mask=input_mask)

    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(output["logits"]).any()


def test_moe_router_loss_backpropagates_to_router():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids)

    output["router_loss"].backward()

    for layer in model.layers:
        grad = layer.mlp.router.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_moe_router_statistics_match_reference_formulas():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    moe = AutoRegressiveLM(config).layers[0].mlp
    x = torch.randn(16, config.hidden_size)

    with torch.no_grad():
        _, _, _, expert_load, entropy = moe._routed_forward(x)
        logits = moe.router(x).float()
        probabilities = torch.softmax(logits, dim=-1)
        topk_indices = torch.topk(
            probabilities.to(x.dtype), config.n_activated_experts, dim=-1
        ).indices
        expected_load = (
            F.one_hot(topk_indices, num_classes=config.n_routed_experts)
            .float()
            .mean(dim=(0, 1))
        )
        expected_entropy = -torch.sum(
            probabilities * torch.log(probabilities), dim=-1
        ).mean()

    assert torch.equal(expert_load, expected_load)
    assert torch.allclose(entropy, expected_entropy, atol=1e-6, rtol=1e-6)


def test_single_shared_expert_returns_projection_without_extra_arithmetic():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    moe = AutoRegressiveLM(config).layers[0].mlp
    sentinel = torch.randn(16, config.hidden_size)

    class SentinelExpert(nn.Module):
        def forward(self, _):
            return sentinel

    moe.shared_experts[0] = SentinelExpert()

    assert moe._shared_forward(torch.randn_like(sentinel)) is sentinel


def test_mqa_uses_native_gqa_sdpa(monkeypatch):
    import astrai.model.components.attention as attention_module

    def fail_repeat(*args, **kwargs):
        raise AssertionError("MQA should not physically repeat KV heads")

    monkeypatch.setattr(attention_module, "repeat_kv", fail_repeat)
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids)
    assert output["logits"].shape == (2, 8, config.vocab_size)


def test_flash_sdpa_backend_keeps_cpu_reference_path():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        attention_backend="flash_sdpa",
        ffn_type="mlp",
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids)

    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert torch.isfinite(output["logits"]).all()


def test_invalid_attention_backend_fails_during_model_construction():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        attention_backend="unknown",
        ffn_type="mlp",
    )

    with pytest.raises(ValueError, match="attention_backend"):
        AutoRegressiveLM(config)


def test_liger_swiglu_backend_keeps_cpu_reference_path():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="moe",
        swiglu_backend="liger",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    model = AutoRegressiveLM(config).cpu()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids)
    output["logits"].float().mean().backward()

    for layer in model.layers:
        assert layer.mlp.shared_experts[0].swiglu_backend == "liger"
        assert layer.mlp.routed_experts.swiglu_backend == "liger"
    assert torch.isfinite(output["logits"]).all()


def test_invalid_swiglu_backend_fails_during_model_construction():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
        swiglu_backend="unknown",
    )

    with pytest.raises(ValueError, match="swiglu_backend"):
        AutoRegressiveLM(config)


def test_liger_residual_norm_backend_keeps_cpu_reference_path():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
        residual_norm_backend="liger",
    )
    reference_config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
        residual_norm_backend="torch",
    )
    torch.manual_seed(1234)
    reference = AutoRegressiveLM(reference_config)
    actual = AutoRegressiveLM(config)
    actual.load_state_dict(reference.state_dict())
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    expected = reference(input_ids)["logits"]
    output = actual(input_ids)["logits"]
    expected.float().sum().backward()
    output.float().sum().backward()

    assert torch.equal(output, expected)
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for actual_param, reference_param in zip(
        actual.parameters(), reference.parameters(), strict=True
    ):
        assert torch.equal(actual_param.grad, reference_param.grad)


def test_invalid_residual_norm_backend_fails_during_model_construction():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
        residual_norm_backend="unknown",
    )

    with pytest.raises(ValueError, match="residual_norm_backend"):
        AutoRegressiveLM(config)


def test_liger_backend_fuses_interblock_mlp_residuals(monkeypatch):
    from astrai.model.components.norm import RMSNorm

    calls = 0
    original = RMSNorm.forward_with_residual

    def counted(self, x, residual, backend="torch"):
        nonlocal calls
        calls += 1
        return original(self, x, residual, backend)

    monkeypatch.setattr(RMSNorm, "forward_with_residual", counted)
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
        residual_norm_backend="liger",
    )
    model = AutoRegressiveLM(config)
    model(torch.randint(0, config.vocab_size, (2, 8)))

    # One attention residual per block plus one MLP residual at every
    # inter-block boundary. The last MLP residual is added before final norm.
    assert calls == 2 * config.num_hidden_layers - 1


def test_liger_loss_backend_avoids_materializing_logits(monkeypatch):
    class FakeLigerFusedLinearCrossEntropyLoss:
        def __init__(self, label_smoothing=0.0):
            self.label_smoothing = label_smoothing

        def __call__(self, weight, hidden_states, target_ids):
            return F.cross_entropy(
                F.linear(hidden_states, weight).float(),
                target_ids,
                label_smoothing=self.label_smoothing,
            )

    liger_package = types.ModuleType("liger_kernel")
    liger_transformers = types.ModuleType("liger_kernel.transformers")
    liger_transformers.LigerFusedLinearCrossEntropyLoss = (
        FakeLigerFusedLinearCrossEntropyLoss
    )
    liger_package.transformers = liger_transformers
    monkeypatch.setitem(sys.modules, "liger_kernel", liger_package)
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", liger_transformers)

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    target_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(
        input_ids,
        target_ids=target_ids,
        loss_backend="liger",
    )
    output["language_model_loss"].backward()

    assert "logits" not in output
    assert output["language_model_loss"].ndim == 0
    assert torch.isfinite(output["language_model_loss"])
    assert model.lm_head.weight.grad is not None


def test_liger_loss_backend_requires_targets():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with pytest.raises(ValueError, match="target_ids"):
        model(input_ids, loss_backend="liger")
