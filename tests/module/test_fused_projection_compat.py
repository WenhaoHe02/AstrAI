import torch

from astrai.model.components.attention import GQA
from astrai.model.components.mlp import GroupedExperts, MLP


def _gqa(fused: bool):
    return GQA(
        dim=32,
        n_heads=4,
        n_kv_heads=2,
        use_qk_norm=False,
        norm_eps=1e-5,
        use_gated_attention=False,
        layer_id=0,
        fused_qkv=fused,
    )


def test_fused_qkv_loads_legacy_weights_and_round_trips():
    legacy = _gqa(False)
    fused = _gqa(True)
    fused.load_state_dict(legacy.state_dict(), strict=True)
    expected = torch.cat(
        [legacy.q_proj.weight, legacy.k_proj.weight, legacy.v_proj.weight], dim=0
    )
    assert torch.equal(fused.qkv_proj.weight, expected)

    restored = _gqa(False)
    restored.load_state_dict(fused.state_dict(), strict=True)
    assert torch.equal(restored.q_proj.weight, legacy.q_proj.weight)
    assert torch.equal(restored.k_proj.weight, legacy.k_proj.weight)
    assert torch.equal(restored.v_proj.weight, legacy.v_proj.weight)


def test_fused_shared_mlp_is_numerically_compatible_with_legacy_weights():
    legacy = MLP(32, 24, swiglu_backend="torch", fused_mlp_gate_up=False)
    fused = MLP(32, 24, swiglu_backend="torch", fused_mlp_gate_up=True)
    legacy.apply(
        lambda module: module.reset_parameters()
        if hasattr(module, "reset_parameters")
        else None
    )
    fused.load_state_dict(legacy.state_dict(), strict=True)
    inputs = torch.randn(3, 7, 32)
    assert torch.allclose(fused(inputs), legacy(inputs), rtol=1e-5, atol=1e-6)

    restored = MLP(32, 24, swiglu_backend="torch", fused_mlp_gate_up=False)
    restored.load_state_dict(fused.state_dict(), strict=True)
    assert torch.equal(restored.up.weight, legacy.up.weight)
    assert torch.equal(restored.gate.weight, legacy.gate.weight)


def test_fused_grouped_experts_load_legacy_weights_and_round_trip():
    kwargs = dict(
        dim=32,
        dim_ffn=24,
        n_experts=2,
        down_init_std=0.02,
        expert_parallel_size=1,
        swiglu_backend="torch",
    )
    legacy = GroupedExperts(**kwargs, fused_mlp_gate_up=False)
    fused = GroupedExperts(**kwargs, fused_mlp_gate_up=True)
    legacy.reset_parameters()
    fused.load_state_dict(legacy.state_dict(), strict=True)

    inputs = torch.randn(6, 32)
    counts = torch.tensor([2, 4], dtype=torch.int64)
    expected = torch.cat([legacy.up_weight, legacy.gate_weight], dim=1)
    assert torch.equal(fused.up_gate_weight, expected)
    assert torch.allclose(fused(inputs, counts), legacy(inputs, counts), rtol=1e-5, atol=1e-6)

    restored = GroupedExperts(**kwargs, fused_mlp_gate_up=False)
    restored.load_state_dict(fused.state_dict(), strict=True)
    assert torch.equal(restored.up_weight, legacy.up_weight)
    assert torch.equal(restored.gate_weight, legacy.gate_weight)
    assert torch.equal(restored.down_weight, legacy.down_weight)
