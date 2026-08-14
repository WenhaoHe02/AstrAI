#!/usr/bin/env python3
"""Permanently remove generated responses at/after a cutoff from runner state."""

from __future__ import annotations

import argparse
import json
import sqlite3
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


def result_time(payload: str | None) -> datetime | None:
    if not payload:
        return None
    row = json.loads(payload)
    stamp = row.get("created_at") or row.get("completed_at")
    return parse_time(str(stamp)) if stamp else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", nargs="+", required=True, type=Path)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--purge-all-payloads",
        action="store_true",
        help="Also clear any remaining result/rejected payloads, including quality rejects.",
    )
    args = parser.parse_args()
    cutoff = parse_time(args.cutoff)

    grand_total = 0
    for path in args.database:
        db = sqlite3.connect(path)
        matches: list[str] = []
        for task_id, result_payload, rejected_payload in db.execute(
            "SELECT task_id, result_json, rejected_json FROM tasks "
            "WHERE result_json IS NOT NULL OR rejected_json IS NOT NULL"
        ):
            stamp = result_time(result_payload)
            if args.purge_all_payloads or (stamp is not None and stamp >= cutoff):
                matches.append(task_id)
        print(json.dumps({"database": str(path), "matched": len(matches), "execute": args.execute}))
        grand_total += len(matches)
        if args.execute and matches:
            db.execute("PRAGMA secure_delete=ON")
            marker = (
                "discarded_all_generated_payloads"
                if args.purge_all_payloads
                else f"discarded_by_time_cutoff:{cutoff.isoformat()}"
            )
            db.executemany(
                """UPDATE tasks
                   SET status='failed', result_json=NULL, rejected_json=NULL,
                       last_error=COALESCE(last_error, ?), next_attempt_at=0
                   WHERE task_id=?""",
                ((marker, task_id) for task_id in matches),
            )
            db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()
    print(json.dumps({"total_matched": grand_total, "execute": args.execute}))


if __name__ == "__main__":
    main()
