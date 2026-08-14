"""Download a strict-license, high-score Stack-Edu code pilot."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


ROWS_URL = "https://datasets-server.huggingface.co/rows"
BLOB_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"
ALLOWED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-2-Clause-Views",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "CC0-1.0",
    "Zlib",
    "NCSA",
    "Artistic-2.0",
}


def get_with_retry(url: str, *, params=None, timeout: float = 60) -> requests.Response:
    for attempt in range(1, 9):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            response = None
        if response is not None and response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response
        if attempt == 8:
            if response is not None:
                response.raise_for_status()
            raise RuntimeError("upstream transport failed after 8 attempts")
        time.sleep(min(30.0, 2 ** (attempt - 1)) * random.uniform(0.8, 1.2))
    raise AssertionError("unreachable")


def strict_permissive(row: dict[str, Any]) -> bool:
    licenses = set(row.get("detected_licenses") or [])
    return (
        row.get("license_type") == "permissive"
        and bool(licenses)
        and licenses <= ALLOWED_LICENSES
    )


def candidates(language: str, count: int, min_score: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < count:
        response = get_with_retry(
            ROWS_URL,
            params={
                "dataset": "HuggingFaceTB/stack-edu",
                "config": language,
                "split": "train",
                "offset": offset,
                "length": 100,
            },
        )
        rows = response.json().get("rows") or []
        if not rows:
            raise RuntimeError(f"Stack-Edu/{language} exhausted at offset {offset}")
        for wrapped in rows:
            row = wrapped.get("row") or {}
            if int(row.get("int_score") or 0) < min_score:
                continue
            if not strict_permissive(row):
                continue
            if not 200 <= int(row.get("length_bytes") or 0) <= 2_000_000:
                continue
            selected.append({**row, "upstream_row": wrapped.get("row_idx")})
            if len(selected) >= count:
                break
        offset += len(rows)
        print(
            "STACK_EDU_SCAN",
            f"language={language}",
            f"offset={offset}",
            f"selected={len(selected)}",
            flush=True,
        )
    return selected


def fetch_content(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    response = get_with_retry(BLOB_URL.format(blob_id=row["blob_id"]))
    return row, gzip.decompress(response.content).decode("utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=("Python", "JavaScript", "TypeScript", "Java"),
    )
    parser.add_argument("--records-per-language", type=int, default=250)
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.records_per_language, args.min_score, args.download_workers) < 1:
        parser.error("record, score, and worker counts must be positive")

    selected: list[dict[str, Any]] = []
    for language in args.languages:
        selected.extend(candidates(language, args.records_per_language, args.min_score))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        downloaded = pool.map(fetch_content, selected)
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for index, (row, text) in enumerate(downloaded, start=1):
                metadata = {
                    "source": "stack-edu",
                    "language": "code",
                    "code_language": row.get("language"),
                    "upstream_dataset": "HuggingFaceTB/stack-edu",
                    **{key: value for key, value in row.items() if key != "language"},
                }
                handle.write(
                    json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False)
                    + "\n"
                )
                if index % 100 == 0:
                    print(f"STACK_EDU_DOWNLOAD written={index}", flush=True)
    os.replace(temporary, args.output)
    print(json.dumps({"records": len(selected), "output": str(args.output)}))


if __name__ == "__main__":
    main()
