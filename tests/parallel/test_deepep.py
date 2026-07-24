from types import SimpleNamespace

import torch

from astrai.parallel import deepep


class _FakeElasticBuffer:
    """Single-rank reference with the subset of ElasticBuffer used by the bridge."""

    def dispatch(self, x, topk_idx=None, topk_weights=None, handle=None, **kwargs):
        if handle is None:
            num_tokens, num_topk = topk_idx.shape
            token_idx = torch.arange(num_tokens).repeat_interleave(num_topk)
            expert_idx = topk_idx.reshape(-1)
            order = torch.argsort(expert_idx)
            counts = torch.bincount(expert_idx, minlength=4).tolist()
            handle = SimpleNamespace(
                token_idx=token_idx,
                order=order,
                num_tokens=num_tokens,
                num_topk=num_topk,
                num_recv_tokens_per_expert_list=counts,
            )
            recv_weights = topk_weights.reshape(-1)[order]
        else:
            recv_weights = None

        recv_x = x[handle.token_idx][handle.order]
        return recv_x, None, recv_weights, handle, None

    def combine(self, x, handle, topk_weights=None, **kwargs):
        ungrouped = torch.empty_like(x)
        ungrouped[handle.order] = x
        output = x.new_zeros((handle.num_tokens, x.shape[-1]))
        output.index_add_(0, handle.token_idx, ungrouped)

        combined_weights = None
        if topk_weights is not None:
            ungrouped_weights = torch.empty_like(topk_weights)
            ungrouped_weights[handle.order] = topk_weights
            combined_weights = ungrouped_weights.view(
                handle.num_tokens, handle.num_topk
            )
        return output, combined_weights, None


def test_deepep_bridge_forward_and_backward(monkeypatch):
    monkeypatch.setattr(
        deepep,
        "_load_deep_ep",
        lambda: SimpleNamespace(topk_idx_t=torch.int64),
    )

    x = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    weights = torch.rand(5, 2, dtype=torch.double, requires_grad=True)
    topk_idx = torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1], [0, 3]])
    expert_scale = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.double)

    state = deepep.DeepEPDispatchState(
        buffer=_FakeElasticBuffer(),
        num_experts=4,
        num_max_tokens_per_rank=x.shape[0],
    )
    recv_x, recv_weights = deepep._Dispatch.apply(x, topk_idx, weights, state)
    repeated_scale = expert_scale.repeat_interleave(
        torch.tensor(state.counts, dtype=torch.int64)
    )
    expert_out = recv_x * repeated_scale.unsqueeze(-1)
    actual = deepep._Combine.apply(expert_out * recv_weights.unsqueeze(-1), state)

    reference = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for route in range(topk_idx.shape[1]):
            reference[token] += (
                x[token] * expert_scale[topk_idx[token, route]] * weights[token, route]
            )

    assert torch.allclose(actual, reference)

    actual.square().sum().backward()
    actual_x_grad = x.grad.clone()
    actual_weight_grad = weights.grad.clone()

    x.grad = None
    weights.grad = None
    reference.square().sum().backward()
    assert torch.allclose(actual_x_grad, x.grad)
    assert torch.allclose(actual_weight_grad, weights.grad)
