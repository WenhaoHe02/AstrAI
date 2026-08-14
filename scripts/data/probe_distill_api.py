"""Probe an OpenAI-compatible distillation endpoint without logging secrets."""

import argparse
import asyncio
import os
import statistics
import time
from collections import Counter

import httpx


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://api.inferknock.ai/v1")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--api-key-env", default="INFERKNOCK_API_KEY")
    parser.add_argument("--show-first-error", action="store_true")
    return parser.parse_args()


async def main_async(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be positive")

    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 30.0))
    headers = {"Authorization": f"Bearer {api_key}"}
    statuses = Counter()
    latencies = []
    finish_reasons = Counter()
    first_error = []

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        headers=headers,
        limits=limits,
        timeout=timeout,
        http2=False,
    ) as client:
        gate = asyncio.Semaphore(args.concurrency)

        async def request_one(index):
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"只回复 OK。探针编号 {index}",
                    }
                ],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            }
            async with gate:
                started = time.perf_counter()
                try:
                    response = await client.post("/chat/completions", json=payload)
                    elapsed = time.perf_counter() - started
                    statuses[f"http_{response.status_code}"] += 1
                    if response.is_success:
                        body = response.json()
                        choice = body.get("choices", [{}])[0]
                        message = choice.get("message") or {}
                        content = message.get("content") or ""
                        reasoning = message.get("reasoning_content") or ""
                        if content.strip() or reasoning.strip():
                            statuses["nonempty"] += 1
                        else:
                            statuses["empty"] += 1
                        finish_reasons[str(choice.get("finish_reason"))] += 1
                        latencies.append(elapsed)
                    elif args.show_first_error and not first_error:
                        first_error.append(response.text[:800].replace("\n", " "))
                except httpx.TimeoutException:
                    statuses["timeout"] += 1
                except httpx.HTTPError as exc:
                    statuses[f"transport_{type(exc).__name__}"] += 1
                except (KeyError, ValueError, TypeError):
                    statuses["invalid_json"] += 1

        wall_started = time.perf_counter()
        await asyncio.gather(*(request_one(i) for i in range(args.requests)))
        wall_seconds = time.perf_counter() - wall_started

    print(
        "API_PROBE",
        f"requests={args.requests}",
        f"concurrency={args.concurrency}",
        f"wall_seconds={wall_seconds:.3f}",
        f"request_rate={args.requests / wall_seconds:.2f}",
        "statuses="
        + ",".join(f"{key}:{value}" for key, value in sorted(statuses.items())),
        "finish_reasons="
        + ",".join(f"{key}:{value}" for key, value in sorted(finish_reasons.items())),
    )
    if latencies:
        ordered = sorted(latencies)
        print(
            "API_LATENCY",
            f"p50={statistics.median(ordered):.3f}",
            f"p90={ordered[int(0.90 * (len(ordered) - 1))]:.3f}",
            f"p99={ordered[int(0.99 * (len(ordered) - 1))]:.3f}",
            f"max={ordered[-1]:.3f}",
        )
    if first_error:
        print("API_FIRST_ERROR", first_error[0])


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
