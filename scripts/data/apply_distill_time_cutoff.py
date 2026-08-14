#!/usr/bin/env python3
"""Split JSONL distillation outputs at an inclusive UTC timestamp cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def parse_time(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--at-or-after", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    cutoff = parse_time(args.cutoff)
    before: list[dict] = []
    rejected: list[dict] = []
    invalid_timestamp: list[str] = []
    input_hashes: dict[str, str] = {}

    for source in args.input:
        input_hashes[str(source)] = sha256(source)
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stamp = row.get("created_at") or row.get("completed_at")
                if not stamp:
                    invalid_timestamp.append(f"{source}:{line_number}:missing")
                    rejected.append(row)
                    continue
                try:
                    created_at = parse_time(str(stamp))
                except ValueError:
                    invalid_timestamp.append(f"{source}:{line_number}:{stamp}")
                    rejected.append(row)
                    continue
                (before if created_at < cutoff else rejected).append(row)

    before.sort(key=lambda row: row.get("created_at") or row.get("completed_at") or "")
    rejected.sort(key=lambda row: row.get("created_at") or row.get("completed_at") or "")
    atomic_jsonl(args.before, before)
    atomic_jsonl(args.at_or_after, rejected)

    def task_counts(rows: list[dict]) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get("task") or row.get("task_type") or "unknown") for row in rows).items()))

    manifest = {
        "policy": "reject every record whose timestamp is greater than or equal to cutoff",
        "cutoff_utc_inclusive": cutoff.isoformat(),
        "inputs": input_hashes,
        "strictly_before_cutoff": {"count": len(before), "tasks": task_counts(before)},
        "at_or_after_cutoff": {"count": len(rejected), "tasks": task_counts(rejected)},
        "invalid_timestamp_records_rejected": invalid_timestamp,
        "outputs": {
            "before": {"path": str(args.before), "sha256": sha256(args.before)},
            "at_or_after": {"path": str(args.at_or_after), "sha256": sha256(args.at_or_after)},
        },
        "training_eligible": False,
        "note": "The pre-cutoff subset remains quarantined pending manual quality review.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.manifest.with_name(f".{args.manifest.name}.{os.getpid()}.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
