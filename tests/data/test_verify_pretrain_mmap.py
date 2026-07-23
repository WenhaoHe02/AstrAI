import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data.verify_pretrain_mmap import verify_dataset


def _write_shard(root: Path, part: int, shard: int, tokens: list[int]) -> None:
    shard_dir = (
        root / f"part_{part:02d}" / "__default__" / f"shard_{shard:04d}"
    )
    shard_dir.mkdir(parents=True)
    values = np.asarray(tokens, dtype=np.int32)
    values.tofile(shard_dir / "sequence.bin")
    (shard_dir / "meta.json").write_text(
        json.dumps({"sequence": {"shape": [len(tokens)], "dtype": "int32"}}),
        encoding="utf-8",
    )


def test_verify_dataset_counts_tokens_and_eos(tmp_path):
    _write_shard(tmp_path, 0, 0, [3, 10, 99])
    _write_shard(tmp_path, 1, 0, [99, 7])

    assert verify_dataset(tmp_path, eos_token_id=99, workers=2) == {
        "parts": 2,
        "shards": 2,
        "tokens": 5,
        "eos": 2,
        "min_token": 3,
        "max_token": 99,
    }


def test_verify_dataset_rejects_unexpected_shard_keys(tmp_path):
    _write_shard(tmp_path, 0, 0, [3, 99])
    meta_path = tmp_path / "part_00" / "__default__" / "shard_0000" / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "sequence": {"shape": [2], "dtype": "int32"},
                "position_ids": {"shape": [2], "dtype": "int32"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unexpected keys"):
        verify_dataset(tmp_path, eos_token_id=99, workers=1)
