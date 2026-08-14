"""Resumable, auditable sequence distillation through an OpenAI-compatible API.

SQLite is the source of truth. Successful generations are stored transactionally
before they are materialized to JSONL, so a killed process can resume without
losing progress or silently duplicating samples.
"""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import hashlib
import json
import os
import random
import signal
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
SERVICE_UNAVAILABLE_STATUS = {408, 500, 502, 503, 504}
AUTH_STATUS = {401, 403}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(record: dict[str, Any], id_key: str) -> str:
    explicit = record.get(id_key)
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()[:24]


def task_id(source_id: str, sample_index: int, candidates: int) -> str:
    return source_id if candidates == 1 else f"{source_id}:{sample_index}"


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            target = email.utils.parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, target.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def retry_delay(attempt: int, base: float, maximum: float) -> float:
    capped = min(maximum, base * (2 ** max(0, attempt - 1)))
    return capped * random.uniform(0.75, 1.25)


def normalize_api_key(value: str | None, env_name: str) -> str:
    if value is None or not value.strip():
        raise SystemExit(f"missing API key environment variable: {env_name}")
    key = value.strip()
    if any(ord(char) < 33 or ord(char) == 127 for char in key):
        raise SystemExit(f"invalid control character in API key from {env_name}")
    return key


class StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                request_hash TEXT NOT NULL,
                request_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'running', 'succeeded', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL DEFAULT 0,
                result_json TEXT,
                rejected_json TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS tasks_status_retry
                ON tasks(status, next_attempt_at);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(tasks)")
        }
        if "rejected_json" not in columns:
            self.conn.execute("ALTER TABLE tasks ADD COLUMN rejected_json TEXT")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def bind_generation_config(self, config: dict[str, Any]):
        value = canonical_json(config)
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key='generation_config'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO metadata(key, value) VALUES ('generation_config', ?)",
                (value,),
            )
            self.conn.commit()
        elif row["value"] != value:
            raise ValueError(
                "generation settings differ from this state DB; resume with the "
                "original settings or use a new --state path"
            )

    def recover_interrupted(self) -> int:
        cursor = self.conn.execute(
            "UPDATE tasks SET status='pending', updated_at=? WHERE status='running'",
            (utc_now(),),
        )
        self.conn.commit()
        return cursor.rowcount

    def seed(
        self,
        input_path: Path,
        *,
        id_key: str,
        prompt_key: str,
        messages_key: str,
        candidates: int,
        system_prompt: str | None,
        limit: int | None,
    ) -> tuple[int, int]:
        inserted = 0
        seen = 0
        now = utc_now()
        with input_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                if limit is not None and seen >= limit:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {input_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"expected object at {input_path}:{line_number}")

                source_id = stable_id(record, id_key)
                raw_messages = record.get(messages_key)
                if raw_messages is not None:
                    if not isinstance(raw_messages, list) or not raw_messages:
                        raise ValueError(
                            f"invalid {messages_key!r} at line {line_number}"
                        )
                    messages = raw_messages
                else:
                    prompt = record.get(prompt_key)
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(
                            f"missing non-empty {prompt_key!r} or {messages_key!r} "
                            f"at line {line_number}"
                        )
                    messages = [{"role": "user", "content": prompt}]
                if system_prompt:
                    messages = [{"role": "system", "content": system_prompt}, *messages]

                for sample_index in range(candidates):
                    tid = task_id(source_id, sample_index, candidates)
                    request = {
                        "source": record,
                        "messages": messages,
                    }
                    request_json = canonical_json(request)
                    request_hash = hashlib.sha256(
                        request_json.encode("utf-8")
                    ).hexdigest()
                    existing = self.conn.execute(
                        "SELECT request_hash FROM tasks WHERE task_id=?", (tid,)
                    ).fetchone()
                    if existing is not None:
                        if existing["request_hash"] != request_hash:
                            raise ValueError(
                                f"task id {tid!r} already exists with different input; "
                                "use a new state DB or stable unique IDs"
                            )
                        continue
                    self.conn.execute(
                        """
                        INSERT INTO tasks(
                            task_id, source_id, sample_index, request_hash,
                            request_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            tid,
                            source_id,
                            sample_index,
                            request_hash,
                            request_json,
                            now,
                            now,
                        ),
                    )
                    inserted += 1
                seen += 1
                if inserted and inserted % 10_000 == 0:
                    self.conn.commit()
        self.conn.commit()
        return seen, inserted

    def retry_failed(self) -> int:
        cursor = self.conn.execute(
            """
            UPDATE tasks
            SET status='pending', attempts=0, next_attempt_at=0,
                last_error=NULL, updated_at=?
            WHERE status='failed'
            """,
            (utc_now(),),
        )
        self.conn.commit()
        return cursor.rowcount

    def claim_ready(self, limit: int) -> list[sqlite3.Row]:
        now_ts = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        rows = self.conn.execute(
            """
            SELECT task_id, source_id, sample_index, request_json, attempts
                 , last_error
            FROM tasks
            WHERE status='pending' AND next_attempt_at <= ?
            ORDER BY rowid
            LIMIT ?
            """,
            (now_ts, limit),
        ).fetchall()
        if rows:
            ids = [row["task_id"] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"""
                UPDATE tasks
                SET status='running', attempts=attempts+1, updated_at=?
                WHERE task_id IN ({placeholders})
                """,
                (utc_now(), *ids),
            )
        self.conn.commit()
        return rows

    def mark_success(self, tid: str, result: dict[str, Any]):
        self.conn.execute(
            """
            UPDATE tasks
            SET status='succeeded', result_json=?, last_error=NULL, updated_at=?
            WHERE task_id=?
            """,
            (canonical_json(result), utc_now(), tid),
        )
        self.conn.commit()

    def mark_retry(self, tid: str, error: str, delay: float):
        self.conn.execute(
            """
            UPDATE tasks
            SET status='pending', next_attempt_at=?, last_error=?, updated_at=?
            WHERE task_id=?
            """,
            (time.time() + delay, error[:4000], utc_now(), tid),
        )
        self.conn.commit()

    def mark_failed(self, tid: str, error: str):
        self.conn.execute(
            """
            UPDATE tasks
            SET status='failed', last_error=?, updated_at=?
            WHERE task_id=?
            """,
            (error[:4000], utc_now(), tid),
        )
        self.conn.commit()

    def save_rejected(self, tid: str, response: dict[str, Any]):
        """Persist a quality-rejected API response before retrying or failing."""
        self.conn.execute(
            """
            UPDATE tasks
            SET rejected_json=?, updated_at=?
            WHERE task_id=?
            """,
            (canonical_json(response), utc_now(), tid),
        )
        self.conn.commit()

    def counts(self) -> dict[str, int]:
        counts = {key: 0 for key in ("pending", "running", "succeeded", "failed")}
        for row in self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
        ):
            counts[row["status"]] = row["n"]
        return counts

    def next_retry_at(self) -> float | None:
        row = self.conn.execute(
            "SELECT MIN(next_attempt_at) AS value FROM tasks WHERE status='pending'"
        ).fetchone()
        return row["value"] if row and row["value"] is not None else None

    def export(self, output_path: Path, failed_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_export(
            output_path,
            """
            SELECT result_json AS line FROM tasks
            WHERE status='succeeded'
            ORDER BY rowid
            """,
        )
        self._atomic_export(
            failed_path,
            """
            SELECT json_object(
                'task_id', task_id,
                'source_id', source_id,
                'sample_index', sample_index,
                'attempts', attempts,
                'last_error', last_error,
                'rejected_response', CASE
                    WHEN rejected_json IS NULL THEN NULL ELSE json(rejected_json)
                END,
                'updated_at', updated_at
            ) AS line
            FROM tasks WHERE status='failed' ORDER BY rowid
            """,
        )

    def _atomic_export(self, path: Path, query: str):
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            for row in self.conn.execute(query):
                target.write(row["line"])
                target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)


@dataclass
class RunConfig:
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    max_attempts: int
    retry_base: float
    retry_max: float
    outage_threshold: int
    outage_cooldown: float
    extra_body: dict[str, Any]
    require_content: bool = False
    retry_finish_reasons: frozenset[str] = frozenset()
    stream: bool = False
    finalize_reasoning_only: bool = False
    finalizer_max_tokens: int = 4096
    finalizer_extra_body: dict[str, Any] | None = None


class OutageGate:
    def __init__(self, threshold: int, cooldown: float):
        self.threshold = threshold
        self.cooldown = cooldown
        self.consecutive_failures = 0
        self.consecutive_service_failures = 0
        self.blocked_until = 0.0
        self.fatal_error: str | None = None
        self.shutdown_error: str | None = None
        self.lock = asyncio.Lock()

    async def wait(self):
        while True:
            async with self.lock:
                delay = self.blocked_until - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(min(delay, 5.0))

    async def success(self):
        async with self.lock:
            self.consecutive_failures = 0
            self.consecutive_service_failures = 0
            self.blocked_until = 0.0

    async def retryable_failure(
        self, *, service_unavailable: bool = False, reason: str | None = None
    ):
        async with self.lock:
            if service_unavailable:
                self.consecutive_service_failures += 1
                if self.consecutive_service_failures >= self.threshold:
                    self.shutdown_error = reason or "upstream service unavailable"
                return
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.threshold:
                self.blocked_until = max(
                    self.blocked_until, time.monotonic() + self.cooldown
                )
                self.consecutive_failures = 0


def response_error(response: httpx.Response) -> str:
    body = response.text[:2000].replace("\n", " ")
    return f"HTTP {response.status_code}: {body}"


def response_fields(body: dict[str, Any]) -> tuple[str, str, str | None]:
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return (
        message.get("content") or "",
        message.get("reasoning_content") or message.get("reasoning") or "",
        choice.get("finish_reason"),
    )


async def finalize_reasoning(
    client: httpx.AsyncClient,
    *,
    request: dict[str, Any],
    reasoning: str,
    preliminary_content: str,
    truncated: bool,
    config: RunConfig,
    task_id: str,
) -> tuple[httpx.Response, dict[str, Any]]:
    """Turn a complete or truncated reasoning trace into a visible final response."""
    mode_instruction = (
        "The reasoning trace was cut off. Continue it only as much as needed to "
        "finish the solution, then return the final answer."
        if truncated
        else "The reasoning trace is complete. Do not redo it."
    )
    finalizer_messages = [
        {
            "role": "system",
            "content": (
                "You are a final-answer formatter. "
                + mode_instruction
                + " Return only the answer in <final>...</final>. For a "
                "programming task, the final answer must include the complete "
                "requested implementation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Original conversation:\n"
                + canonical_json(request["messages"])
                + "\n\nCompleted reasoning:\n"
                + reasoning
                + (
                    "\n\nPartial visible answer:\n" + preliminary_content
                    if preliminary_content.strip()
                    else ""
                )
                + "\n\nEmit only <final>...</final>."
            ),
        },
    ]
    payload = {
        "model": config.model,
        "messages": finalizer_messages,
        "max_tokens": config.finalizer_max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": config.stream,
        **(config.finalizer_extra_body or {}),
    }
    return await request_completion(
        client, payload, task_id=f"{task_id}-final", stream=config.stream
    )


async def process_one(
    row: sqlite3.Row,
    *,
    db: StateDB,
    client: httpx.AsyncClient,
    config: RunConfig,
    outage_gate: OutageGate,
) -> bool:
    await outage_gate.wait()
    request = json.loads(row["request_json"])
    attempt = int(row["attempts"]) + 1
    generation_max_tokens = config.max_tokens
    prior_error = row["last_error"] if "last_error" in row.keys() else None
    if prior_error and "finish_reason: length" in prior_error:
        # Repeating an identical truncated request wastes attempts. Increase
        # the completion budget deterministically while retaining the SQLite
        # attempt counter and the bucket's configured upper bound.
        generation_max_tokens = min(
            config.finalizer_max_tokens,
            config.max_tokens * (2 ** max(1, attempt - 1)),
        )
    payload = {
        "model": config.model,
        "messages": request["messages"],
        "max_tokens": generation_max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "stream": config.stream,
        **config.extra_body,
    }
    started = time.perf_counter()
    try:
        response, body = await request_completion(
            client,
            payload,
            # Preserve correlation with the logical task while giving each
            # retry a distinct upstream request ID. Some gateways reject an
            # exact ID reuse instead of treating it idempotently.
            task_id=f"{row['task_id']}-a{attempt}",
            stream=config.stream,
        )
        latency = time.perf_counter() - started
        if not response.is_success:
            error = response_error(response)
            if response.status_code in RETRYABLE_STATUS:
                await outage_gate.retryable_failure(
                    service_unavailable=(
                        response.status_code in SERVICE_UNAVAILABLE_STATUS
                    ),
                    reason=f"HTTP {response.status_code}",
                )
                delay = retry_after_seconds(response.headers.get("Retry-After"))
                delay = (
                    delay
                    if delay is not None
                    else retry_delay(attempt, config.retry_base, config.retry_max)
                )
                # Service outages and rate limits must survive indefinitely;
                # max_attempts is reserved for malformed/quality responses.
                db.mark_retry(row["task_id"], error, delay)
                return False
            db.mark_failed(row["task_id"], error)
            if response.status_code in AUTH_STATUS:
                outage_gate.fatal_error = error
            return False

        content, reasoning, finish_reason = response_fields(body)
        if not content.strip() and not reasoning.strip():
            raise ValueError("successful response contained no content")
        rejected_for_length = finish_reason in config.retry_finish_reasons
        rejected_for_missing_content = config.require_content and not content.strip()
        if rejected_for_length or rejected_for_missing_content:
            db.save_rejected(
                row["task_id"],
                {"stage": "generation", "body": body, "attempt": attempt},
            )
        finalizer = None
        if rejected_for_length or rejected_for_missing_content:
            if not config.finalize_reasoning_only or not reasoning.strip():
                if rejected_for_length:
                    raise ValueError(f"retryable finish_reason: {finish_reason}")
                raise ValueError(
                    "successful response contained reasoning but no final content"
                )
            final_response, final_body = await finalize_reasoning(
                client,
                request=request,
                reasoning=reasoning,
                preliminary_content=content,
                truncated=rejected_for_length,
                config=config,
                task_id=f"{row['task_id']}-a{attempt}",
            )
            if not final_response.is_success:
                if final_response.status_code in RETRYABLE_STATUS:
                    await outage_gate.retryable_failure(
                        service_unavailable=(
                            final_response.status_code in SERVICE_UNAVAILABLE_STATUS
                        ),
                        reason=f"finalizer HTTP {final_response.status_code}",
                    )
                    delay = retry_after_seconds(
                        final_response.headers.get("Retry-After")
                    )
                    db.mark_retry(
                        row["task_id"],
                        response_error(final_response),
                        delay
                        if delay is not None
                        else retry_delay(
                            attempt, config.retry_base, config.retry_max
                        ),
                    )
                    return False
                raise ValueError(
                    f"finalizer HTTP {final_response.status_code}: "
                    f"{final_response.text[:500]}"
                )
            final_content, final_reasoning, final_finish = response_fields(final_body)
            finalizer = {
                "content": final_content,
                "reasoning_content": final_reasoning,
                "finish_reason": final_finish,
                "request_id": final_body.get("id"),
            }
            db.save_rejected(
                row["task_id"],
                {
                    "stage": "finalizer",
                    "generation": body,
                    "finalizer": final_body,
                    "attempt": attempt,
                },
            )
            if final_finish in config.retry_finish_reasons:
                raise ValueError(f"finalizer finish_reason: {final_finish}")
            if not final_content.strip():
                raise ValueError("finalizer returned no visible final content")
            content = final_content
        result = {
            "task_id": row["task_id"],
            "source_id": row["source_id"],
            "sample_index": row["sample_index"],
            "source": request["source"],
            "messages": request["messages"],
            "teacher": config.model,
            "generation": {
                "max_tokens": generation_max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "extra_body": config.extra_body,
            },
            "response": {
                "content": content,
                "reasoning_content": reasoning,
                "finish_reason": finish_reason,
            },
            "finalizer": finalizer,
            "usage": body.get("usage"),
            "request_id": body.get("id"),
            "attempt": attempt,
            "latency_seconds": round(latency, 6),
            "created_at": utc_now(),
        }
        db.mark_success(row["task_id"], result)
        await outage_gate.success()
        return True
    except httpx.HTTPError as exc:
        await outage_gate.retryable_failure(
            service_unavailable=True,
            reason=f"{type(exc).__name__}: transport failure",
        )
        # Transport exceptions may include serialized request headers. Never
        # persist their message because it can contain credentials.
        error = f"{type(exc).__name__}: transport failure"
        db.mark_retry(
            row["task_id"],
            error,
            retry_delay(attempt, config.retry_base, config.retry_max),
        )
        return False
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        error = f"invalid_response: {type(exc).__name__}: {exc}"

    if config.max_attempts and attempt >= config.max_attempts:
        db.mark_failed(row["task_id"], error)
    else:
        db.mark_retry(
            row["task_id"],
            error,
            retry_delay(attempt, config.retry_base, config.retry_max),
        )
    return False


async def request_completion(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    *,
    task_id: str,
    stream: bool,
) -> tuple[httpx.Response, dict[str, Any]]:
    """Send one request and normalize full or SSE responses to one body."""
    headers = {"X-Request-ID": task_id}
    if not stream:
        response = await client.post("/chat/completions", json=payload, headers=headers)
        return response, response.json() if response.is_success else {}

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    response_id = None
    usage = None
    async with client.stream(
        "POST", "/chat/completions", json=payload, headers=headers
    ) as response:
        if not response.is_success:
            await response.aread()
            return response, {}
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            response_id = chunk.get("id") or response_id
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
            finish_reason = choice.get("finish_reason") or finish_reason

    body = {
        "id": response_id,
        "choices": [
            {
                "message": {
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
    return response, body


async def run_pending(
    db: StateDB,
    client: httpx.AsyncClient,
    config: RunConfig,
    concurrency: int,
    stop_event: asyncio.Event,
):
    gate = OutageGate(config.outage_threshold, config.outage_cooldown)
    started = time.monotonic()
    previous_success = db.counts()["succeeded"]
    processed_since_log = 0
    successful_since_log = 0

    def log_progress() -> None:
        nonlocal processed_since_log, successful_since_log
        counts = db.counts()
        elapsed = max(time.monotonic() - started, 1e-9)
        completed = counts["succeeded"] - previous_success
        print(
            "DISTILL_PROGRESS",
            f"succeeded={counts['succeeded']}",
            f"pending={counts['pending']}",
            f"failed={counts['failed']}",
            f"batch_ok={successful_since_log}",
            f"session_rate={completed / elapsed:.2f}/s",
            flush=True,
        )
        processed_since_log = 0
        successful_since_log = 0

    async def worker() -> None:
        nonlocal processed_since_log, successful_since_log
        while not stop_event.is_set():
            if gate.fatal_error:
                raise RuntimeError(
                    "authentication/authorization failure; stopping: "
                    f"{gate.fatal_error}"
                )
            if gate.shutdown_error:
                raise RuntimeError(
                    "service unavailable; automatic shutdown: "
                    f"{gate.shutdown_error}"
                )

            rows = db.claim_ready(1)
            if not rows:
                counts = db.counts()
                if counts["pending"] == 0 and counts["running"] == 0:
                    return
                next_at = db.next_retry_at()
                delay = 0.5 if next_at is None else max(
                    0.1, min(0.5, next_at - time.time())
                )
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                # Python 3.10 keeps asyncio.TimeoutError distinct from the
                # built-in TimeoutError (they became aliases in 3.11).
                # Catch the asyncio type explicitly so an idle retry queue
                # does not tear down every worker and restart the whole run.
                except asyncio.TimeoutError:
                    pass
                continue

            row = rows[0]
            try:
                outcome = await process_one(
                    row,
                    db=db,
                    client=client,
                    config=config,
                    outage_gate=gate,
                )
            except Exception as exc:
                # One malformed stream or unexpected library exception must
                # never tear down the process and disconnect every in-flight
                # request. Do not serialize exception messages: HTTP exceptions
                # may contain request headers/credentials.
                error = f"unhandled_task_exception: {type(exc).__name__}"
                print(
                    "DISTILL_TASK_EXCEPTION",
                    f"task_id={row['task_id']}",
                    f"type={type(exc).__name__}",
                    flush=True,
                )
                db.mark_retry(
                    row["task_id"],
                    error,
                    retry_delay(
                        int(row["attempts"]) + 1,
                        config.retry_base,
                        config.retry_max,
                    ),
                )
                outcome = False

            processed_since_log += 1
            successful_since_log += int(outcome)
            if processed_since_log >= concurrency:
                log_progress()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    done, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_EXCEPTION)
    done_results = await asyncio.gather(*done, return_exceptions=True)
    error = next(
        (result for result in done_results if isinstance(result, BaseException)),
        None,
    )
    if error is not None:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise error
    await asyncio.gather(*pending)
    if processed_since_log:
        log_progress()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--state", type=Path, required=True, help="SQLite state DB")
    parser.add_argument("--output", type=Path, required=True, help="Successful JSONL")
    parser.add_argument("--failed-output", type=Path, help="Permanent failures JSONL")
    parser.add_argument("--base-url", default="https://api.inferknock.ai/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--api-key-env", default="INFERKNOCK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--messages-key", default="messages")
    parser.add_argument("--system-prompt")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--connect-timeout", type=float, default=20.0)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 retries forever; attempts persist across restarts",
    )
    parser.add_argument("--retry-base", type=float, default=2.0)
    parser.add_argument("--retry-max", type=float, default=300.0)
    parser.add_argument("--outage-threshold", type=int, default=32)
    parser.add_argument("--outage-cooldown", type=float, default=120.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument(
        "--require-content",
        action="store_true",
        help="Retry responses that contain reasoning but no final content",
    )
    parser.add_argument(
        "--finalize-reasoning-only",
        action="store_true",
        help="Run a second short request when reasoning is complete but content is empty",
    )
    parser.add_argument("--finalizer-max-tokens", type=int, default=4096)
    parser.add_argument("--finalizer-extra-body-json", default="{}")
    parser.add_argument(
        "--retry-finish-reason",
        action="append",
        default=[],
        help="Retry this finish_reason (repeatable; e.g. --retry-finish-reason length)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use SSE streaming; recommended for long generations behind gateways",
    )
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    for name in (
        "concurrency",
        "candidates",
        "max_tokens",
        "finalizer_max_tokens",
        "outage_threshold",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.max_attempts < 0:
        raise SystemExit("--max-attempts cannot be negative")
    if args.system_prompt and args.system_prompt_file:
        raise SystemExit("use only one of --system-prompt and --system-prompt-file")


async def main_async(args: argparse.Namespace) -> int:
    validate_args(args)
    failed_path = args.failed_output or args.output.with_suffix(".failed.jsonl")
    db = StateDB(args.state)
    try:
        recovered = db.recover_interrupted()
        if recovered:
            print(f"Recovered {recovered} interrupted tasks", flush=True)
        if args.retry_failed:
            print(f"Reset {db.retry_failed()} failed tasks", flush=True)

        system_prompt = args.system_prompt
        if args.system_prompt_file:
            system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
        try:
            extra_body = json.loads(args.extra_body_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --extra-body-json: {exc}") from exc
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body-json must decode to an object")
        try:
            finalizer_extra_body = json.loads(args.finalizer_extra_body_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --finalizer-extra-body-json: {exc}") from exc
        if not isinstance(finalizer_extra_body, dict):
            raise SystemExit("--finalizer-extra-body-json must decode to an object")

        if not args.export_only:
            db.bind_generation_config(
                {
                    "base_url": args.base_url.rstrip("/"),
                    "model": args.model,
                    "candidates": args.candidates,
                    "id_key": args.id_key,
                    "prompt_key": args.prompt_key,
                    "messages_key": args.messages_key,
                    "system_prompt_sha256": hashlib.sha256(
                        (system_prompt or "").encode("utf-8")
                    ).hexdigest(),
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "extra_body": extra_body,
                    "require_content": args.require_content,
                    "retry_finish_reasons": sorted(set(args.retry_finish_reason)),
                    "stream": args.stream,
                }
            )
            seen, inserted = db.seed(
                args.input,
                id_key=args.id_key,
                prompt_key=args.prompt_key,
                messages_key=args.messages_key,
                candidates=args.candidates,
                system_prompt=system_prompt,
                limit=args.limit,
            )
            print(f"Seeded records={seen} new_tasks={inserted}", flush=True)

        if args.seed_only or args.export_only:
            db.export(args.output, failed_path)
            print("DISTILL_COUNTS", canonical_json(db.counts()), flush=True)
            return 0

        api_key = normalize_api_key(os.environ.get(args.api_key_env), args.api_key_env)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def request_stop(signame: str):
            print(f"DISTILL_SIGNAL signal={signame}", flush=True)
            stop_event.set()

        for signame in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, signame, None)
            if sig is not None:
                try:
                    loop.add_signal_handler(sig, request_stop, signame)
                except (NotImplementedError, RuntimeError):
                    pass

        limits = httpx.Limits(
            max_connections=args.concurrency,
            max_keepalive_connections=args.concurrency,
        )
        timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
        config = RunConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_attempts=args.max_attempts,
            retry_base=args.retry_base,
            retry_max=args.retry_max,
            outage_threshold=args.outage_threshold,
            outage_cooldown=args.outage_cooldown,
            extra_body=extra_body,
            require_content=args.require_content,
            retry_finish_reasons=frozenset(args.retry_finish_reason),
            stream=args.stream,
            finalize_reasoning_only=args.finalize_reasoning_only,
            finalizer_max_tokens=args.finalizer_max_tokens,
            finalizer_extra_body=finalizer_extra_body,
        )
        async with httpx.AsyncClient(
            base_url=args.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            limits=limits,
            http2=False,
        ) as client:
            while not stop_event.is_set():
                try:
                    await run_pending(
                        db, client, config, args.concurrency, stop_event
                    )
                    break
                except RuntimeError as exc:
                    if str(exc).startswith((
                        "authentication/authorization failure",
                        "service unavailable; automatic shutdown",
                    )):
                        print(
                            "DISTILL_AUTO_SHUTDOWN",
                            str(exc).split(":", 1)[0],
                            flush=True,
                        )
                        db.recover_interrupted()
                        return 75
                    frames = " > ".join(
                        f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                        for frame in traceback.extract_tb(exc.__traceback__)
                    )
                    print(
                        "DISTILL_RUNNER_EXCEPTION",
                        f"type={type(exc).__name__}",
                        f"frames={frames}",
                        flush=True,
                    )
                    db.recover_interrupted()
                    await asyncio.sleep(5)
                except Exception as exc:
                    # Keep the service alive on batch-level library/SQLite
                    # failures. Log stack locations but not exception text,
                    # which may contain serialized request credentials.
                    frames = " > ".join(
                        f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                        for frame in traceback.extract_tb(exc.__traceback__)
                    )
                    print(
                        "DISTILL_RUNNER_EXCEPTION",
                        f"type={type(exc).__name__}",
                        f"frames={frames}",
                        flush=True,
                    )
                    db.recover_interrupted()
                    await asyncio.sleep(5)

        db.export(args.output, failed_path)
        counts = db.counts()
        print("DISTILL_COUNTS", canonical_json(counts), flush=True)
        return 0 if counts["failed"] == 0 else 2
    finally:
        db.export(args.output, failed_path)
        db.close()


def main():
    try:
        raise SystemExit(asyncio.run(main_async(parse_args())))
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
