"""Move outage-related permanent failures back to the pending queue."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    connection = sqlite3.connect(args.state)
    try:
        has_tasks = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()
        if has_tasks is None:
            print("Recovered retryable failures: 0 (new state database)", flush=True)
            return
        cursor = connection.execute(
            """
            UPDATE tasks
            SET status='pending', next_attempt_at=0, updated_at=?
            WHERE status='failed' AND (
                last_error LIKE 'HTTP 429:%'
                OR last_error LIKE 'HTTP 5__:%'
                OR last_error LIKE '%transport failure%'
                OR last_error LIKE '%Duplicate request ID%'
            )
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        connection.commit()
        print(f"Recovered retryable failures: {cursor.rowcount}", flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
