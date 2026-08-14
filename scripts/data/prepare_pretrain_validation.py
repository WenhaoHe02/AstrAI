"""Build a small, fixed and auditable pretraining validation suite.

The suite is intentionally stored as plain JSONL so every checkpoint can be
evaluated against exactly the same bytes.  It is diagnostic data only and must
never be appended to the pretraining mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests


ROWS_URL = "https://datasets-server.huggingface.co/rows"


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    config: str
    split: str
    limit: int
    render: Callable[[dict], str]
    role: str
    start_offset: int = 0


def _plain_text(row: dict) -> str:
    # Different pretraining corpora use one of these conventional names.
    return str(
        row.get("text") or row.get("completion") or row.get("content") or ""
    ).strip()


def _gsm8k(row: dict) -> str:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    return f"{question}\n\nSolution:\n{answer}" if question and answer else ""


def _humaneval(row: dict) -> str:
    prompt = str(row.get("prompt") or "")
    solution = str(row.get("canonical_solution") or "")
    return (prompt + solution).strip() if prompt and solution else ""


SOURCES = (
    Source(
        name="zh_wikipedia",
        dataset="0xDing/wikipedia-cn-20230720-filtered",
        config="default",
        split="train",
        limit=4096,
        render=_plain_text,
        role="in_domain_language_ppl",
        # This external corpus is not part of the AstrAI training mixture.  A
        # tail range makes the selection stable even if viewer defaults change.
        start_offset=240000,
    ),
    Source(
        name="en_wikitext103",
        dataset="Salesforce/wikitext",
        config="wikitext-103-raw-v1",
        split="test",
        limit=4096,
        render=_plain_text,
        role="in_domain_language_ppl",
    ),
    Source(
        name="math_gsm8k",
        dataset="openai/gsm8k",
        config="main",
        split="test",
        limit=1319,
        render=_gsm8k,
        role="out_of_domain_diagnostic_ppl",
    ),
    Source(
        name="code_humaneval",
        dataset="openai/openai_humaneval",
        config="openai_humaneval",
        split="test",
        limit=164,
        render=_humaneval,
        role="out_of_domain_diagnostic_ppl",
    ),
)


def _get_rows(
    session: requests.Session,
    source: Source,
    offset: int,
    length: int,
    timeout: float,
    max_retries: int,
) -> list[dict]:
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                ROWS_URL,
                params={
                    "dataset": source.dataset,
                    "config": source.config,
                    "split": source.split,
                    "offset": offset,
                    "length": length,
                },
                timeout=timeout,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return response.json().get("rows") or []
        except requests.RequestException:
            if attempt >= max_retries:
                raise
        delay = min(30.0, 2**attempt) * random.uniform(0.8, 1.2)
        time.sleep(delay)
    raise RuntimeError("unreachable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_source(
    session: requests.Session,
    source: Source,
    output_dir: Path,
    batch_size: int,
    timeout: float,
    max_retries: int,
) -> dict:
    output = output_dir / f"{source.name}.jsonl"
    partial = output.with_suffix(".jsonl.partial")
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            complete_records = sum(1 for line in handle if line.strip())
        if complete_records:
            print(
                f"REUSE source={source.name} written={complete_records}", flush=True
            )
            return {
                "name": source.name,
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "start_offset": source.start_offset,
                "role": source.role,
                "records": complete_records,
                "file": output.name,
                "sha256": _sha256(output),
            }
    written = 0
    upstream_offset = source.start_offset
    if partial.exists():
        with partial.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        written = len(existing)
        if existing:
            upstream_offset = 1 + max(
                item["metadata"]["upstream_row"] for item in existing
            )
        print(f"RESUME source={source.name} written={written}", flush=True)

    mode = "a" if written else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        while written < source.limit:
            rows = _get_rows(
                session,
                source,
                upstream_offset,
                min(batch_size, source.limit - written),
                timeout,
                max_retries,
            )
            if not rows:
                break
            for wrapped in rows:
                row = wrapped.get("row") or {}
                text = source.render(row)
                upstream_row = wrapped.get("row_idx", upstream_offset)
                upstream_offset = max(upstream_offset, int(upstream_row) + 1)
                # Empty WikiText headings and tiny fragments make PPL unstable.
                if len(text) < 64:
                    continue
                item = {
                    "text": text,
                    "metadata": {
                        "category": source.name.split("_", 1)[0],
                        "role": source.role,
                        "upstream_dataset": source.dataset,
                        "upstream_config": source.config,
                        "upstream_split": source.split,
                        "upstream_row": upstream_row,
                    },
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
                if written >= source.limit:
                    break
            handle.flush()
            print(
                f"PROGRESS source={source.name} written={written}/{source.limit} "
                f"offset={upstream_offset}",
                flush=True,
            )
    os.replace(partial, output)
    return {
        "name": source.name,
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "start_offset": source.start_offset,
        "role": source.role,
        "records": written,
        "file": output.name,
        "sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be in [1, 100]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        entries = [
            _build_source(
                session,
                source,
                args.output_dir,
                args.batch_size,
                args.timeout,
                args.max_retries,
            )
            for source in SOURCES
        ]
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_eval_length": 2048,
        "training_exclusion": True,
        "notes": [
            "Never append these files to a training mixture.",
            "zh/en files track language-model PPL; math/code are diagnostics.",
            "Public benchmark contamination cannot be ruled out; compare trends, not absolute scores.",
        ],
        "sources": entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
