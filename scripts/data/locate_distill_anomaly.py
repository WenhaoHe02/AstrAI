"""Locate the earliest hard anomaly in chronological distillation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL = re.compile(r"<final>(.*?)</final>", re.IGNORECASE | re.DOTALL)


def visible_content(row: dict[str, Any]) -> str:
    content = str((row.get("response") or {}).get("content") or "").strip()
    match = FINAL.search(content)
    return (match.group(1) if match else content).strip()


def prompt(row: dict[str, Any]) -> str:
    source = row.get("source") or {}
    for message in source.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def parse_time(value: Any) -> datetime:
    text = str(value or "")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    result = datetime.fromisoformat(text)
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def compact(value: str, limit: int = 500) -> str:
    return " ".join(value.split())[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--bin-seconds", type=int, default=5)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    rows.sort(key=lambda row: parse_time(row.get("created_at")))
    if not rows:
        raise SystemExit("no output rows")

    answer_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    request_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = visible_content(row)
        answer_groups[hashlib.sha256(answer.encode("utf-8")).hexdigest()].append(row)
        request_id = str(row.get("request_id") or "")
        if request_id:
            request_groups[request_id].append(row)

    hard: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = visible_content(row)
        if len(answer) < 8:
            hard["very_short"].append(row)
    for group in answer_groups.values():
        prompt_hashes = {
            hashlib.sha256(prompt(row).encode("utf-8")).hexdigest() for row in group
        }
        if len(group) > 1 and len(prompt_hashes) > 1:
            hard["cross_prompt_exact_answer"].extend(group)
    for group in request_groups.values():
        if len(group) > 1:
            hard["duplicate_request_id"].extend(group)

    hard_rows = {id(row): row for group in hard.values() for row in group}
    earliest = min(
        (parse_time(row.get("created_at")) for row in hard_rows.values()),
        default=None,
    )
    start = parse_time(rows[0].get("created_at"))
    bins: defaultdict[int, Counter[str]] = defaultdict(Counter)
    hard_ids = {
        name: {id(row) for row in group}
        for name, group in hard.items()
    }
    for row in rows:
        index = int((parse_time(row.get("created_at")) - start).total_seconds()) // args.bin_seconds
        bins[index]["success"] += 1
        for name, ids in hard_ids.items():
            if id(row) in ids:
                bins[index][name] += 1

    before = sum(
        parse_time(row.get("created_at")) < earliest for row in rows
    ) if earliest else len(rows)
    summary = {
        "total": len(rows),
        "start": start.isoformat(),
        "end": parse_time(rows[-1].get("created_at")).isoformat(),
        "hard_counts": {name: len({id(row) for row in group}) for name, group in hard.items()},
        "earliest_hard_anomaly": earliest.isoformat() if earliest else None,
        "strictly_before_cutoff": before,
        "at_or_after_cutoff": len(rows) - before,
        "timeline": {
            f"+{index * args.bin_seconds:03d}s": dict(counts)
            for index, counts in sorted(bins.items())
        },
    }
    print("ANOMALY_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    for name, group in hard.items():
        seen: set[str] = set()
        for row in sorted(group, key=lambda item: parse_time(item.get("created_at"))):
            task_id = str(row.get("task_id"))
            if task_id in seen:
                continue
            seen.add(task_id)
            print(
                "ANOMALY_ITEM",
                json.dumps(
                    {
                        "kind": name,
                        "created_at": row.get("created_at"),
                        "task_id": task_id,
                        "request_id": row.get("request_id"),
                        "prompt": compact(prompt(row)),
                        "answer": compact(visible_content(row)),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )


if __name__ == "__main__":
    main()
