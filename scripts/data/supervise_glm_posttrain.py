"""Probe a stopped GLM teacher with persistent exponential backoff and resume it."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


BUCKETS = ("short", "medium", "long")
BACKOFF_SECONDS = (60, 120, 240, 480, 960, 1200)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def load_probe_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    initial = {"attempts": 0, "next_probe_at": time.time() + BACKOFF_SECONDS[0]}
    atomic_json(path, initial)
    return initial


def unfinished(state_dir: Path, run_id: str) -> int:
    count = 0
    for bucket in BUCKETS:
        path = state_dir / f"{run_id}-{bucket}.sqlite3"
        if not path.exists():
            continue
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            has_tasks = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchone()
            if has_tasks:
                count += connection.execute(
                    "SELECT COUNT(*) FROM tasks WHERE status IN ('pending','running')"
                ).fetchone()[0]
        finally:
            connection.close()
    return count


def services(run_version: str) -> list[str]:
    return [f"astral-glm-{run_version}@{bucket}.service" for bucket in BUCKETS]


def any_active(names: list[str]) -> bool:
    for name in names:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", name],
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def start_services(names: list[str]) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "start", *names],
        check=False,
    )
    return result.returncode == 0


def probe(base_url: str, model: str, api_key: str, timeout: float) -> tuple[bool, str]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": False,
    }
    try:
        response = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # The probe has no external side effects. A unique transport ID
                # prevents a gateway from rejecting a retry as a duplicate.
                "X-Request-ID": f"astrai-probe-{uuid.uuid4().hex}",
            },
            json=payload,
            timeout=httpx.Timeout(timeout, connect=min(20.0, timeout)),
        )
    except httpx.HTTPError as exc:
        return False, f"transport:{type(exc).__name__}"
    if not response.is_success:
        return False, f"http:{response.status_code}"
    try:
        body = response.json()
        if not body.get("choices"):
            return False, "invalid_response"
    except (ValueError, AttributeError):
        return False, "invalid_response"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="posttrain-v5")
    parser.add_argument("--run-version", default="v5")
    parser.add_argument("--state-dir", type=Path,
                        default=Path("/home/zbuser02/astrai-distill-state"))
    parser.add_argument("--probe-state", type=Path,
                        default=Path("/home/zbuser02/astrai-distill-state/probe-v5.json"))
    parser.add_argument("--credential-file", type=Path,
                        default=Path.home() / ".config/astrai/inferknock.env")
    parser.add_argument("--base-url", default="https://api.inferknock.ai/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    api_key = args.credential_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("empty API credential file")
    names = services(args.run_version)

    while True:
        remaining = unfinished(args.state_dir, args.run_id)
        if remaining == 0:
            print("DISTILL_SUPERVISOR_COMPLETE remaining=0", flush=True)
            return 0
        if any_active(names):
            # Active runners own outage handling. Do not emit competing probes.
            atomic_json(
                args.probe_state,
                {"attempts": 0, "next_probe_at": time.time() + BACKOFF_SECONDS[0]},
            )
            time.sleep(30)
            continue

        state = load_probe_state(args.probe_state)
        wait = max(0.0, float(state.get("next_probe_at", 0)) - time.time())
        if wait:
            time.sleep(min(wait, 30.0))
            continue

        ok, result = probe(args.base_url, args.model, api_key, args.timeout)
        if ok:
            atomic_json(
                args.probe_state,
                {"attempts": 0, "next_probe_at": time.time() + BACKOFF_SECONDS[0],
                 "last_result": "ok"},
            )
            print(f"DISTILL_PROBE_OK remaining={remaining}", flush=True)
            if not start_services(names):
                raise RuntimeError("failed to start distillation services")
            time.sleep(30)
            continue

        attempts = int(state.get("attempts", 0)) + 1
        delay = BACKOFF_SECONDS[min(attempts - 1, len(BACKOFF_SECONDS) - 1)]
        atomic_json(
            args.probe_state,
            {"attempts": attempts, "next_probe_at": time.time() + delay,
             "last_result": result},
        )
        print(
            f"DISTILL_PROBE_FAILED attempt={attempts} retry_in={delay}s result={result}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
