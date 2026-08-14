from unittest.mock import patch

import pytest

from astrai.model.components import mlp


@pytest.mark.parametrize(
    ("rank", "expected_ep_rank", "expected_dp_ranks", "expected_dp_rank"),
    [
        (0, 0, [0, 2], 0),
        (1, 1, [1, 3], 0),
        (2, 0, [0, 2], 1),
        (3, 1, [1, 3], 1),
    ],
)
def test_ep2_dp2_topology(
    rank, expected_ep_rank, expected_dp_ranks, expected_dp_rank
):
    mlp._EP_GROUP_CACHE.clear()
    mlp._EXPERT_DP_GROUP_CACHE.clear()
    created_groups = []

    def new_group(ranks):
        group = tuple(ranks)
        created_groups.append(group)
        return group

    with (
        patch.object(mlp.dist, "is_available", return_value=True),
        patch.object(mlp.dist, "is_initialized", return_value=True),
        patch.object(mlp.dist, "get_world_size", return_value=4),
        patch.object(mlp.dist, "get_rank", return_value=rank),
        patch.object(mlp.dist, "new_group", side_effect=new_group),
    ):
        ep_group, ep_rank = mlp._expert_parallel_group(2)
        dp_group, dp_rank, dp_size = mlp._expert_data_parallel_group(2)

    expected_ep_group = (0, 1) if rank < 2 else (2, 3)
    assert ep_group == expected_ep_group
    assert ep_rank == expected_ep_rank
    assert list(dp_group) == expected_dp_ranks
    assert dp_rank == expected_dp_rank
    assert dp_size == 2
    # All ranks must create all process groups in the same deterministic order.
    assert created_groups == [(0, 1), (2, 3), (0, 2), (1, 3)]


def test_no_expert_replica_when_ep_equals_world_size():
    mlp._EXPERT_DP_GROUP_CACHE.clear()
    with (
        patch.object(mlp.dist, "is_available", return_value=True),
        patch.object(mlp.dist, "is_initialized", return_value=True),
        patch.object(mlp.dist, "get_world_size", return_value=8),
    ):
        group, rank, size = mlp._expert_data_parallel_group(8)
    assert group is None
    assert rank == 0
    assert size == 1
