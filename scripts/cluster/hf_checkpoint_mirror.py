"""Mirror complete recovery checkpoints to Hugging Face at a coarse interval."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from huggingface_hub import HfApi


LOGGER = logging.getLogger("astrai.hf_checkpoint_mirror")
STEP_RE = re.compile(r"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)$")


def complete_checkpoint(path: Path, world_size: int) -> tuple[int, int] | None:
    match = STEP_RE.match(path.name)
    if match is None or not path.is_dir():
        return None
    required = [
        "_SUCCESS",
        "config.json",
        "meta.json",
        "model.safetensors",
        "scheduler.pt",
    ]
    required.extend(f"optimizer.rank{rank}.pt" for rank in range(world_size))
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in required):
        return None
    return int(match.group("epoch")), int(match.group("step"))


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"uploaded": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_once(args: argparse.Namespace, api: HfApi, state: dict) -> bool:
    uploaded = set(state.get("uploaded", []))
    candidates: list[tuple[int, int, Path]] = []
    for path in args.checkpoint_root.glob("epoch_*_step_*"):
        identity = complete_checkpoint(path, args.world_size)
        if identity is None:
            continue
        epoch, step = identity
        if step < args.minimum_step or step % args.step_interval != 0:
            continue
        candidates.append((epoch, step, path))

    changed = False
    for epoch, step, path in sorted(candidates):
        key = path.name
        if key in uploaded:
            continue
        remote_path = f"{args.remote_prefix}/{key}"
        LOGGER.info("uploading complete checkpoint %s to %s/%s", path, args.repo_id, remote_path)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(path),
            path_in_repo=remote_path,
            commit_message=f"Mirror AstrAI 7B-A1B recovery checkpoint step {step}",
        )
        uploaded.add(key)
        state["uploaded"] = sorted(uploaded)
        state["last_uploaded_epoch"] = epoch
        state["last_uploaded_step"] = step
        state["last_uploaded_at"] = int(time.time())
        save_state(args.state_file, state)
        changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--remote-prefix", default="checkpoints/pretrain-7b-a1b")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--step-interval", type=int, default=10000)
    parser.add_argument("--minimum-step", type=int, default=10000)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"empty Hugging Face token file: {args.token_file}")
    api = HfApi(token=token)
    state = load_state(args.state_file)
    while True:
        try:
            run_once(args, api, state)
        except Exception:
            LOGGER.exception("checkpoint mirror pass failed; retaining state for retry")
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
