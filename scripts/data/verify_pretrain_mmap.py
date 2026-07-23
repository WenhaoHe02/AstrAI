"""Verify a sharded pretraining mmap dataset and its EOS boundaries."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _inspect_shard(meta_path: Path, eos_token_id: int) -> dict[str, int]:
    shard_dir = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if set(meta) != {"sequence"}:
        raise ValueError(f"unexpected keys in {meta_path}: {sorted(meta)}")
    sequence = meta["sequence"]
    if sequence.get("dtype") != "int32":
        raise ValueError(f"unexpected dtype in {meta_path}: {sequence.get('dtype')}")
    shape = sequence.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or not isinstance(shape[0], int):
        raise ValueError(f"invalid shape in {meta_path}: {shape!r}")

    expected_entries = {"meta.json", "sequence.bin"}
    actual_entries = {path.name for path in shard_dir.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError(
            f"unexpected files in {shard_dir}: {sorted(actual_entries)}"
        )

    sequence_path = shard_dir / "sequence.bin"
    expected_bytes = shape[0] * np.dtype(np.int32).itemsize
    actual_bytes = sequence_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"size mismatch for {sequence_path}: {actual_bytes} != {expected_bytes}"
        )

    tokens = np.memmap(sequence_path, dtype=np.int32, mode="r")
    if tokens.size == 0:
        return {"tokens": 0, "eos": 0, "min_token": 0, "max_token": 0}
    return {
        "tokens": int(tokens.size),
        "eos": int(np.count_nonzero(tokens == eos_token_id)),
        "min_token": int(tokens.min()),
        "max_token": int(tokens.max()),
    }


def verify_dataset(root: Path, eos_token_id: int, workers: int) -> dict[str, int]:
    meta_paths = sorted(root.glob("part_*/__default__/shard_*/meta.json"))
    if not meta_paths:
        raise ValueError(f"no mmap shards found under {root}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        shard_stats = list(
            pool.map(lambda path: _inspect_shard(path, eos_token_id), meta_paths)
        )
    nonempty = [stats for stats in shard_stats if stats["tokens"]]
    return {
        "parts": len({path.parts[-4] for path in meta_paths}),
        "shards": len(meta_paths),
        "tokens": sum(stats["tokens"] for stats in shard_stats),
        "eos": sum(stats["eos"] for stats in shard_stats),
        "min_token": min(stats["min_token"] for stats in nonempty),
        "max_token": max(stats["max_token"] for stats in nonempty),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--expected-tokens", type=int)
    parser.add_argument("--expected-eos", type=int)
    parser.add_argument("--expected-parts", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.require_success and not (args.root / "_SUCCESS").is_file():
        raise SystemExit(f"missing success marker: {args.root / '_SUCCESS'}")

    stats = verify_dataset(args.root, args.eos_token_id, args.workers)
    expectations = {
        "tokens": args.expected_tokens,
        "eos": args.expected_eos,
        "parts": args.expected_parts,
    }
    mismatches = {
        key: {"actual": stats[key], "expected": expected}
        for key, expected in expectations.items()
        if expected is not None and stats[key] != expected
    }
    print(json.dumps(stats, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(f"verification failed: {json.dumps(mismatches, sort_keys=True)}")


if __name__ == "__main__":
    main()
