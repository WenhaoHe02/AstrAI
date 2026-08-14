"""Prune old complete AstrAI checkpoints without touching partial writes."""

import argparse
import os
import time

from astrai.trainer.train_callback import prune_checkpoint_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--optimizer-ranks", type=int, default=0)
    parser.add_argument(
        "--watch-pid",
        type=int,
        help="Repeat while this process exists, then prune once more and exit.",
    )
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")

    while True:
        removed = prune_checkpoint_dirs(
            args.checkpoint_dir,
            keep_last=args.keep_last,
            expected_optimizer_ranks=args.optimizer_ranks,
        )
        for path in removed:
            print(f"REMOVED {path}", flush=True)
        print(f"PRUNED count={len(removed)} keep_last={args.keep_last}", flush=True)

        if args.watch_pid is None:
            break
        try:
            os.kill(args.watch_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
