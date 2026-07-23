"""Wait for a local process to exit, then replace this process with a command."""

import argparse
import os
import time
from pathlib import Path


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Path that must exist after the watched process exits",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("a command is required after --")
    while process_exists(args.pid):
        time.sleep(args.poll_seconds)
    missing = [path for path in args.require if not Path(path).exists()]
    if missing:
        raise SystemExit(
            "watched process exited without required outputs: " + ", ".join(missing)
        )
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
