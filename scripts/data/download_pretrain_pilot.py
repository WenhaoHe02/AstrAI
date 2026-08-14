"""Download a bounded, auditable JSONL sample through HF's dataset server."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from pathlib import Path

import requests


ROWS_URL = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--source", required=True)
    parser.add_argument("--language", choices=("zh", "en"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--request-interval", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()
    if args.offset < 0 or args.records < 1:
        parser.error("--offset must be non-negative and --records must be positive")
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be in [1, 100]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    opener = gzip.open if args.output.name.endswith(".gz") else open
    written = 0
    if temporary.exists():
        with opener(temporary, "rt", encoding="utf-8") as handle:
            written = sum(1 for line in handle if line.strip())
        if written > args.records:
            raise SystemExit(
                f"partial output has {written:,} rows, more than requested "
                f"{args.records:,}; remove {temporary} and retry"
            )
        print(f"DOWNLOAD_RESUME written={written}", flush=True)
    session = requests.Session()
    try:
        mode = "at" if written else "wt"
        with opener(temporary, mode, encoding="utf-8") as handle:
            while written < args.records:
                length = min(args.batch_size, args.records - written)
                response = None
                for attempt in range(1, args.max_retries + 2):
                    try:
                        response = session.get(
                            ROWS_URL,
                            params={
                                "dataset": args.dataset,
                                "config": args.config,
                                "split": args.split,
                                "offset": args.offset + written,
                                "length": length,
                            },
                            timeout=args.timeout,
                        )
                    except requests.RequestException as exc:
                        if attempt > args.max_retries:
                            raise
                        delay = min(60.0, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                        print(
                            "DOWNLOAD_BACKOFF",
                            f"error={type(exc).__name__}",
                            f"attempt={attempt}",
                            f"seconds={delay:.1f}",
                            flush=True,
                        )
                        time.sleep(delay)
                        continue
                    if response.status_code not in {429, 500, 502, 503, 504}:
                        break
                    if attempt > args.max_retries:
                        response.raise_for_status()
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    if delay <= 0:
                        delay = min(60.0, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
                    print(
                        "DOWNLOAD_BACKOFF",
                        f"status={response.status_code}",
                        f"attempt={attempt}",
                        f"seconds={delay:.1f}",
                        flush=True,
                    )
                    time.sleep(delay)
                assert response is not None
                response.raise_for_status()
                rows = response.json().get("rows") or []
                if not rows:
                    raise RuntimeError(
                        f"dataset exhausted after {written:,} records"
                    )
                for wrapped in rows:
                    row = wrapped.get("row") or {}
                    text = row.get(args.text_key)
                    if not isinstance(text, str):
                        raise RuntimeError(
                            f"missing string field {args.text_key!r} at "
                            f"row {wrapped.get('row_idx')}"
                        )
                    metadata = {key: value for key, value in row.items() if key != args.text_key}
                    output = {
                        "text": text,
                        "metadata": {
                            "source": args.source,
                            "language": args.language,
                            "upstream_dataset": args.dataset,
                            "upstream_config": args.config,
                            "upstream_split": args.split,
                            "upstream_row": wrapped.get("row_idx"),
                            "upstream": metadata,
                        },
                    }
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.records:
                        break
                print(f"DOWNLOAD_PROGRESS written={written}", flush=True)
                handle.flush()
                time.sleep(args.request_interval)
        os.replace(temporary, args.output)
    finally:
        session.close()
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "offset": args.offset,
                "records": written,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
