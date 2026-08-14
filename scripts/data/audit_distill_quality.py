"""Audit exported distillation JSONL for visible-answer quality regressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REFUSAL = re.compile(
    r"\b(?:i (?:am|['’]m) (?:sorry|unable)|i cannot (?:help|assist)|as an ai|"
    r"i can(?:not|'t) comply)\b",
    re.IGNORECASE,
)
MOJIBAKE = re.compile(r"(?:ï¿½|�|Ã.|Â.|â(?:€|€™|€œ|€|€“|€”))")
THINK = re.compile(r"</?(?:think|analysis)>", re.IGNORECASE)
FINAL = re.compile(r"<final>(.*?)</final>", re.IGNORECASE | re.DOTALL)


def response_content(row: dict[str, Any]) -> str:
    response = row.get("response") or {}
    return response.get("content") or ""


def task_type(row: dict[str, Any]) -> str:
    source = row.get("source") or {}
    return str(source.get("task_type") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--sample-chars", type=int, default=900)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    rows.append(json.loads(line))

    flags: Counter[str] = Counter()
    flags_by_task: defaultdict[str, Counter[str]] = defaultdict(Counter)
    tasks: Counter[str] = Counter()
    finishes: Counter[str] = Counter()
    lengths: defaultdict[str, list[int]] = defaultdict(list)
    content_hashes: Counter[str] = Counter()
    flagged_ids: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        task = task_type(row)
        tasks[task] += 1
        content = response_content(row)
        lengths[task].append(len(content))
        finalizer = row.get("finalizer") or {}
        finish = str(
            finalizer.get("finish_reason")
            if finalizer
            else (row.get("response") or {}).get("finish_reason")
        )
        finishes[finish] += 1
        content_hashes[hashlib.sha256(content.strip().encode("utf-8")).hexdigest()] += 1
        checks = {
            "empty": not content.strip(),
            "very_short": 0 < len(content.strip()) < 12,
            "missing_final_tags": not bool(FINAL.search(content)),
            "think_tag_leak": bool(THINK.search(content)),
            "mojibake": bool(MOJIBAKE.search(content)),
            "refusal": bool(REFUSAL.search(content)),
            "finish_not_stop": finish != "stop",
        }
        for name, hit in checks.items():
            if hit:
                flags[name] += 1
                flags_by_task[task][name] += 1
                if len(flagged_ids[name]) < 10:
                    flagged_ids[name].append(str(row.get("task_id")))

    duplicates = sum(count - 1 for count in content_hashes.values() if count > 1)
    if duplicates:
        flags["duplicate_visible_answer"] = duplicates
    summary = {
        "total": len(rows),
        "tasks": dict(tasks),
        "finish_reasons": dict(finishes),
        "flags": dict(flags),
        "flags_by_task": {task: dict(values) for task, values in flags_by_task.items()},
        "flagged_task_ids": dict(flagged_ids),
        "content_chars": {
            task: {
                "min": min(values),
                "median": sorted(values)[len(values) // 2],
                "max": max(values),
            }
            for task, values in lengths.items()
        },
    }
    print("QUALITY_SUMMARY", json.dumps(summary, ensure_ascii=False, sort_keys=True))

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: str(value.get("created_at")), reverse=True):
        grouped[task_type(row)].append(row)
    for task in sorted(grouped):
        for row in grouped[task][: args.samples_per_task]:
            source = row.get("source") or {}
            messages = source.get("messages") or []
            prompt = next(
                (
                    str(message.get("content") or "")
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                "",
            )
            content = response_content(row)
            print(
                "QUALITY_SAMPLE",
                json.dumps(
                    {
                        "task": task,
                        "task_id": row.get("task_id"),
                        "prompt": prompt[: args.sample_chars],
                        "answer": content[: args.sample_chars],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )

    duplicate_hashes = {value for value, count in content_hashes.items() if count > 1}
    for row in rows:
        content = response_content(row)
        content_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        if (
            len(content.strip()) < 12
            or content_hash in duplicate_hashes
            or THINK.search(content)
            or REFUSAL.search(content)
            or MOJIBAKE.search(content)
        ):
            print(
                "QUALITY_FLAGGED",
                json.dumps(
                    {
                        "task": task_type(row),
                        "task_id": row.get("task_id"),
                        "content_chars": len(content),
                        "answer": content,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )


if __name__ == "__main__":
    main()
