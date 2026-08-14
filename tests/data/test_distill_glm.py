import asyncio
import json

import httpx
import pytest

from scripts.data.distill_glm import (
    OutageGate,
    RunConfig,
    StateDB,
    normalize_api_key,
    process_one,
    run_pending,
    retry_after_seconds,
)


def config(**overrides):
    values = {
        "model": "glm-test",
        "max_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_attempts": 3,
        "retry_base": 0.01,
        "retry_max": 0.01,
        "outage_threshold": 2,
        "outage_cooldown": 0.01,
        "extra_body": {},
        "require_content": False,
        "retry_finish_reasons": frozenset(),
        "stream": False,
    }
    values.update(overrides)
    return RunConfig(**values)


def seed_one(tmp_path, candidates=1):
    source = tmp_path / "input.jsonl"
    source.write_text('{"id":"q1","prompt":"2+2="}\n', encoding="utf-8")
    db = StateDB(tmp_path / "state.sqlite3")
    assert db.seed(
        source,
        id_key="id",
        prompt_key="prompt",
        messages_key="messages",
        candidates=candidates,
        system_prompt="Solve carefully.",
        limit=None,
    ) == (1, candidates)
    return db


def test_seed_is_idempotent_and_export_is_atomic(tmp_path):
    db = seed_one(tmp_path, candidates=2)
    source = tmp_path / "input.jsonl"
    assert db.seed(
        source,
        id_key="id",
        prompt_key="prompt",
        messages_key="messages",
        candidates=2,
        system_prompt="Solve carefully.",
        limit=None,
    ) == (1, 0)
    rows = db.claim_ready(2)
    assert [row["task_id"] for row in rows] == ["q1:0", "q1:1"]
    db.mark_success("q1:0", {"task_id": "q1:0", "response": {"content": "4"}})
    db.mark_failed("q1:1", "bad response")
    output = tmp_path / "output.jsonl"
    failed = tmp_path / "failed.jsonl"
    db.export(output, failed)
    assert json.loads(output.read_text(encoding="utf-8"))["task_id"] == "q1:0"
    assert json.loads(failed.read_text(encoding="utf-8"))["attempts"] == 1
    db.close()


def test_interrupted_tasks_are_recovered(tmp_path):
    db = seed_one(tmp_path)
    assert len(db.claim_ready(1)) == 1
    assert db.counts()["running"] == 1
    assert db.recover_interrupted() == 1
    assert db.counts()["pending"] == 1
    db.close()


def test_retryable_http_error_then_success(tmp_path):
    db = seed_one(tmp_path)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, text="offline")
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "choices": [
                    {
                        "message": {"content": "4", "reasoning_content": "2+2=4"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 3},
            },
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            gate = OutageGate(2, 0.01)
            row = db.claim_ready(1)[0]
            assert not await process_one(
                row, db=db, client=client, config=config(), outage_gate=gate
            )
            row = db.claim_ready(1)[0]
            assert await process_one(
                row, db=db, client=client, config=config(), outage_gate=gate
            )

    asyncio.run(run())
    assert calls == 2
    assert db.counts()["succeeded"] == 1
    db.close()


def test_truncated_or_reasoning_only_response_is_retried(tmp_path):
    db = seed_one(tmp_path)
    responses = [
        {"content": "", "reasoning_content": "still thinking", "finish_reason": "length"},
        {"content": "<final>4</final>", "reasoning_content": "2+2=4", "finish_reason": "stop"},
    ]

    async def handler(request):
        message = responses.pop(0)
        finish_reason = message.pop("finish_reason")
        return httpx.Response(
            200,
            json={"choices": [{"message": message, "finish_reason": finish_reason}]},
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            gate = OutageGate(2, 0.01)
            strict = config(
                require_content=True,
                retry_finish_reasons=frozenset({"length"}),
            )
            assert not await process_one(
                db.claim_ready(1)[0],
                db=db,
                client=client,
                config=strict,
                outage_gate=gate,
            )
            await asyncio.sleep(0.02)
            assert await process_one(
                db.claim_ready(1)[0],
                db=db,
                client=client,
                config=strict,
                outage_gate=gate,
            )

    asyncio.run(run())
    assert db.counts()["succeeded"] == 1
    rejected = db.conn.execute("select rejected_json from tasks").fetchone()[0]
    assert json.loads(rejected)["body"]["choices"][0]["finish_reason"] == "length"
    db.close()


def test_reasoning_only_response_is_finalized_and_preserved(tmp_path):
    db = seed_one(tmp_path)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                json={
                    "id": "reasoning-1",
                    "choices": [{
                        "message": {"content": "", "reasoning_content": "2+2=4"},
                        "finish_reason": "stop",
                    }],
                },
            )
        payload = json.loads(request.content)
        assert payload["temperature"] == 0
        assert "Completed reasoning:\n2+2=4" in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "id": "final-1",
                "choices": [{
                    "message": {
                        "content": "<final>4</final>",
                        "reasoning_content": "",
                    },
                    "finish_reason": "stop",
                }],
            },
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            gate = OutageGate(2, 0.01)
            strict = config(require_content=True, finalize_reasoning_only=True)
            assert await process_one(
                db.claim_ready(1)[0],
                db=db,
                client=client,
                config=strict,
                outage_gate=gate,
            )

    asyncio.run(run())
    assert calls == 2
    row = db.conn.execute("select result_json,rejected_json from tasks").fetchone()
    result = json.loads(row[0])
    assert result["response"]["content"] == "<final>4</final>"
    assert result["response"]["reasoning_content"] == "2+2=4"
    assert json.loads(row[1])["stage"] == "finalizer"
    db.close()


def test_truncated_reasoning_is_continued_by_finalizer(tmp_path):
    db = seed_one(tmp_path)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{
                "message": {"content": "", "reasoning_content": "partial proof"},
                "finish_reason": "length",
            }]})
        payload = json.loads(request.content)
        assert "reasoning trace was cut off" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{
            "message": {"content": "<final>4</final>"},
            "finish_reason": "stop",
        }]})

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            gate = OutageGate(2, 0.01)
            strict = config(
                require_content=True,
                finalize_reasoning_only=True,
                retry_finish_reasons=frozenset({"length"}),
            )
            assert await process_one(
                db.claim_ready(1)[0], db=db, client=client,
                config=strict, outage_gate=gate,
            )

    asyncio.run(run())
    assert calls == 2
    assert db.counts()["succeeded"] == 1
    db.close()


def test_length_retry_doubles_generation_budget(tmp_path):
    db = seed_one(tmp_path)
    budgets = []

    async def handler(request):
        payload = json.loads(request.content)
        budgets.append(payload["max_tokens"])
        if len(budgets) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "partial"},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "complete"}, "finish_reason": "stop"}
                ]
            },
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            gate = OutageGate(2, 0.01)
            strict = config(
                max_tokens=32,
                finalizer_max_tokens=128,
                retry_finish_reasons=frozenset({"length"}),
            )
            assert not await process_one(
                db.claim_ready(1)[0], db=db, client=client,
                config=strict, outage_gate=gate,
            )
            await asyncio.sleep(0.02)
            assert await process_one(
                db.claim_ready(1)[0], db=db, client=client,
                config=strict, outage_gate=gate,
            )

    asyncio.run(run())
    assert budgets == [32, 64]
    assert db.counts()["succeeded"] == 1
    db.close()


def test_streaming_reasoning_and_content_are_assembled(tmp_path):
    db = seed_one(tmp_path)

    async def handler(request):
        chunks = [
            'data: {"id":"r1","choices":[{"delta":{"reasoning_content":"2+"}}]}\n\n',
            'data: {"id":"r1","choices":[{"delta":{"reasoning_content":"2=4"}}]}\n\n',
            'data: {"id":"r1","choices":[{"delta":{"content":"<final>4</final>"},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        return httpx.Response(200, content="".join(chunks))

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            gate = OutageGate(2, 0.01)
            strict = config(require_content=True, stream=True)
            assert await process_one(
                db.claim_ready(1)[0],
                db=db,
                client=client,
                config=strict,
                outage_gate=gate,
            )

    asyncio.run(run())
    row = db.conn.execute("select result_json from tasks").fetchone()
    result = json.loads(row[0])
    assert result["response"]["reasoning_content"] == "2+2=4"
    assert result["response"]["content"] == "<final>4</final>"
    db.close()


def test_retry_after_supports_seconds_and_http_dates():
    assert retry_after_seconds("2") == 2
    assert retry_after_seconds("not-a-date") is None


def test_api_key_is_trimmed_before_building_headers():
    assert normalize_api_key("  secret\r\n", "KEY") == "secret"


def test_unexpected_task_exception_does_not_kill_runner(tmp_path):
    db = seed_one(tmp_path)

    async def handler(request):
        raise RuntimeError("unexpected stream implementation failure")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            stop = asyncio.Event()
            task = asyncio.create_task(
                run_pending(db, client, config(), 1, stop)
            )
            await asyncio.sleep(0.05)
            stop.set()
            await task

    asyncio.run(run())
    row = db.conn.execute("select status,last_error from tasks").fetchone()
    assert row[0] == "pending"
    assert row[1] == "unhandled_task_exception: RuntimeError"
    db.close()


def test_runner_refills_slot_before_slowest_request_finishes(tmp_path):
    db = seed_one(tmp_path, candidates=3)
    calls = 0
    release_first = asyncio.Event()
    third_started = asyncio.Event()

    async def handler(request):
        nonlocal calls
        calls += 1
        call = calls
        if call == 1:
            await release_first.wait()
        if call == 3:
            third_started.set()
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "message": {"content": "ok", "reasoning_content": "done"},
                    "finish_reason": "stop",
                }]
            },
        )

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            runner = asyncio.create_task(
                run_pending(db, client, config(), 2, asyncio.Event())
            )
            await asyncio.wait_for(third_started.wait(), timeout=1.0)
            release_first.set()
            await runner

    asyncio.run(run())
    assert calls == 3
    assert db.counts()["succeeded"] == 3
    db.close()


def test_sustained_service_outage_stops_runner_and_preserves_pending(tmp_path):
    db = seed_one(tmp_path, candidates=2)

    async def handler(request):
        return httpx.Response(502, text="gateway unavailable")

    async def run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://example.test/v1"
        ) as client:
            with pytest.raises(RuntimeError, match="automatic shutdown"):
                await run_pending(
                    db,
                    client,
                    config(outage_threshold=2),
                    2,
                    asyncio.Event(),
                )

    asyncio.run(run())
    assert db.counts()["succeeded"] == 0
    assert db.counts()["failed"] == 0
    assert db.recover_interrupted() >= 0
    assert db.counts()["pending"] == 2
    db.close()
