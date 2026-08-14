"""Report status and output-length statistics for post-training distillation."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    connection = sqlite3.connect(path)
    try:
        counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            ).fetchall()
        )
        attempts = [
            list(row)
            for row in connection.execute(
                "SELECT attempts, status, COUNT(*) FROM tasks "
                "GROUP BY attempts, status ORDER BY attempts, status"
            )
        ]
        results = []
        for (raw,) in connection.execute(
            "SELECT result_json FROM tasks WHERE status='succeeded'"
        ):
            result = json.loads(raw)
            usage = result.get("usage") or {}
            results.append(
                (
                    usage.get("completion_tokens"),
                    result["response"].get("finish_reason"),
                    result.get("latency_seconds"),
                )
            )
        tokens = [row[0] for row in results if isinstance(row[0], int)]
        latencies = [row[2] for row in results if row[2] is not None]
        summary = {
            "counts": counts,
            "attempts": attempts,
            "finish_reasons": dict(Counter(row[1] for row in results)),
        }
        if tokens:
            summary["completion_tokens"] = {
                "min": min(tokens),
                "median": round(statistics.median(tokens)),
                "max": max(tokens),
            }
        if latencies:
            summary["median_latency_seconds"] = round(
                statistics.median(latencies), 1
            )
        summary["top_errors"] = [
            {"count": count, "error": error}
            for error, count in connection.execute(
                "SELECT SUBSTR(last_error, 1, 180), COUNT(*) FROM tasks "
                "WHERE last_error IS NOT NULL GROUP BY SUBSTR(last_error, 1, 180) "
                "ORDER BY COUNT(*) DESC LIMIT 5"
            )
        ]
        return summary
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("/mnt/nvme3/zbuser02/astrai-distill/state"),
    )
    parser.add_argument(
        "--run-id",
        default="posttrain-v4",
        help="SQLite filename prefix, for example posttrain-dsv4-public-v3",
    )
    args = parser.parse_args()
    report = {}
    for bucket in ("short", "medium", "long"):
        path = args.state_dir / f"{args.run_id}-{bucket}.sqlite3"
        report[bucket] = summarize(path)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
