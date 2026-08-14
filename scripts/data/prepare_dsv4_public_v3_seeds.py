"""Build a deduplicated 16K DSV4 mix from NVIDIA's public post-train set.

Only user requests (and tool schemas for tool-use examples) are sent to the
teacher.  Upstream assistant answers are represented by hashes and lengths for
audit, never copied into the student target.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


RUN_ID = "posttrain-dsv4-public-v3"
DATASET = "nvidia/Nemotron-Post-Training-Dataset-v1"
REVISION = "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8"
CHAT_DATASET = "HuggingFaceH4/ultrafeedback_binarized:train_prefs"
CHAT_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
MATH_DATASET = "open-r1/OpenR1-Math-220k:default:shard0"
MATH_REVISION = "e4e141ec9dea9f8326f4d347be56105859b2bd68"
CODE_DATASET = "nvidia/OpenCodeReasoning:split_0:shard0"
CODE_REVISION = "20a1ca19c0d050fe9057fc08339d6b370ec1c67a"
TARGETS = {"chat": 4096, "math": 4096, "code": 4096, "stem": 2048, "tool_calling": 2048}
BUCKETS = ("short", "medium", "long")
SYSTEMS = {
    "chat": (
        "You are a natural, knowledgeable conversational assistant. Follow the user's intent "
        "and constraints, answer directly, admit uncertainty, and never fabricate facts or "
        "citations. Put the complete response in <final>...</final>."
    ),
    "math": (
        "You are a rigorous mathematics teacher. Solve independently with a concise, "
        "checkable derivation, verify the result, and put the complete solution in "
        "<final>...</final>."
    ),
    "code": (
        "You are an expert software and competitive-programming teacher. Derive a correct "
        "solution, explain its invariant, complexity and edge cases, then provide complete "
        "code in the requested language inside <final>...</final>."
    ),
    "stem": (
        "You are a careful STEM tutor. Work from first principles, distinguish facts from "
        "assumptions, check units and conclusions, and put the complete answer in "
        "<final>...</final>."
    ),
    "tool_calling": (
        "You are a reliable tool-using assistant. Use only the supplied tools, never claim a "
        "tool ran before observing its result, and ask for missing required arguments. Return "
        "the best next assistant turn as either a user-facing response or one valid JSON tool "
        "call inside <final>...</final>."
    ),
}


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    source = pq.ParquetFile(path)
    columns = [
        name
        for name in ("uuid", "license", "generator", "version", "category", "reasoning", "messages", "metadata")
        if name in source.schema_arrow.names
    ]
    for batch in source.iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


def first_user(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            value = text(message.get("content"))
            if value:
                return value
    return ""


def first_assistant(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            value = text(message.get("content"))
            if value:
                return value
    return ""


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def user_prompt_hashes(seed_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in sorted(seed_dir.glob("*.jsonl")):
        # Outputs and rejects live elsewhere; this directory contains seeds.
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = first_user(row.get("messages")) if isinstance(row, dict) else ""
                if prompt:
                    hashes.add(digest(normalized(prompt)))
    return hashes


class StableSample:
    def __init__(self, limit: int, namespace: str) -> None:
        self.limit = limit
        self.namespace = namespace
        self.heap: list[tuple[int, str, dict[str, Any]]] = []
        self.keys: set[str] = set()

    def add(self, key: str, value: dict[str, Any]) -> None:
        if key in self.keys:
            return
        score = int(digest(self.namespace, key)[:16], 16)
        item = (-score, key, value)
        if len(self.heap) < self.limit:
            heapq.heappush(self.heap, item)
            self.keys.add(key)
        elif item > self.heap[0]:
            removed = heapq.heapreplace(self.heap, item)
            self.keys.remove(removed[1])
            self.keys.add(key)

    def values(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heap, key=lambda item: (-item[0], item[1]))]


def tool_prompt(user: str, metadata: dict[str, Any]) -> str:
    tools = metadata.get("tools")
    if not isinstance(tools, list) or not tools:
        return ""
    return (
        "Available tools (JSON):\n"
        + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        + "\n\nUser request:\n"
        + user
        + "\n\nGenerate only the best next assistant turn."
    )


def acceptable_prompt(task: str, prompt: str) -> bool:
    lower = normalized(prompt)
    minimum = 40 if task == "chat" else 80
    maximum = 20_000 if task == "tool_calling" else 14_000
    if not minimum <= len(prompt) <= maximum or prompt == "-":
        return False
    if len(set(lower.split())) < 5:
        return False
    # Remove empty/meta conversations and obvious requests for hidden reasoning.
    blocked = ("please ask the first question", "show your hidden chain of thought")
    return not any(fragment in lower for fragment in blocked)


def build_task(path: Path, task: str, excluded: set[str]) -> list[dict[str, Any]]:
    sample = StableSample(TARGETS[task], f"{RUN_ID}:{task}")
    seen: set[str] = set()
    audit: Counter[str] = Counter()
    message_example = ""
    for row in parquet_rows(path):
        audit["rows"] += 1
        if not message_example:
            message_example = repr(row.get("messages"))[:1000]
        if text(row.get("license")).casefold().replace("-", " ") != "cc by 4.0":
            audit["license_rejected"] += 1
            continue
        audit["license_accepted"] += 1
        user = first_user(row.get("messages"))
        metadata = parse_metadata(row.get("metadata"))
        prompt = tool_prompt(user, metadata) if task == "tool_calling" else user
        if not acceptable_prompt(task, prompt):
            audit["prompt_rejected"] += 1
            continue
        audit["prompt_accepted"] += 1
        prompt_hash = digest(normalized(prompt))
        if prompt_hash in excluded or prompt_hash in seen:
            audit["duplicate_rejected"] += 1
            continue
        seen.add(prompt_hash)
        reference = first_assistant(row.get("messages"))
        source_id = text(row.get("uuid")) or prompt_hash[:24]
        rid = digest(RUN_ID, DATASET, task, source_id, prompt)[:24]
        sample.add(
            source_id,
            {
                "id": f"dsv4-v3-{task}-{rid}",
                "task_type": task,
                "language": "en",
                "source_dataset": f"{DATASET}:{task}",
                "source_revision": REVISION,
                "source_id": source_id,
                "prompt_version": "dsv4-public-v3",
                "teacher_target": "deepseek-v4-flash",
                "distill_partition": "public_v3",
                "source_meta": {
                    "license": text(row.get("license")),
                    "generator": text(row.get("generator")),
                    "upstream_reasoning": text(row.get("reasoning")),
                    "reference_sha256": digest(reference) if reference else None,
                    "reference_chars": len(reference),
                },
                "messages": [
                    {"role": "system", "content": SYSTEMS[task]},
                    {"role": "user", "content": prompt},
                ],
            },
        )
    values = sample.values()
    if len(values) != TARGETS[task]:
        raise RuntimeError(
            f"{task}: selected {len(values)} of {TARGETS[task]}; "
            f"audit={dict(audit)}; messages={message_example}"
        )
    return values


def build_chat(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    sample = StableSample(TARGETS["chat"], f"{RUN_ID}:ultrafeedback-chat")
    seen: set[str] = set()
    source = pq.ParquetFile(path)
    columns = ["prompt", "prompt_id", "chosen", "score_chosen", "score_rejected"]
    for batch in source.iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            prompt = text(row.get("prompt"))
            chosen = first_assistant(row.get("chosen"))
            chosen_score = float(row.get("score_chosen") or 0.0)
            rejected_score = float(row.get("score_rejected") or 0.0)
            if not acceptable_prompt("chat", prompt) or not chosen:
                continue
            if chosen_score - rejected_score < 1.0:
                continue
            prompt_hash = digest(normalized(prompt))
            if prompt_hash in excluded or prompt_hash in seen:
                continue
            seen.add(prompt_hash)
            source_id = text(row.get("prompt_id")) or prompt_hash[:24]
            rid = digest(RUN_ID, CHAT_DATASET, source_id, prompt)[:24]
            sample.add(
                source_id,
                {
                    "id": f"dsv4-v3-chat-{rid}",
                    "task_type": "chat",
                    "language": "en",
                    "source_dataset": CHAT_DATASET,
                    "source_revision": CHAT_REVISION,
                    "source_id": source_id,
                    "prompt_version": "dsv4-public-v3",
                    "teacher_target": "deepseek-v4-flash",
                    "distill_partition": "public_v3",
                    "source_meta": {
                        "license": "MIT",
                        "reference_sha256": digest(chosen),
                        "reference_chars": len(chosen),
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                    },
                    "messages": [
                        {"role": "system", "content": SYSTEMS["chat"]},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
    values = sample.values()
    if len(values) != TARGETS["chat"]:
        raise RuntimeError(f"chat: selected {len(values)} of {TARGETS['chat']}")
    return values


def build_math(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    sample = StableSample(TARGETS["math"], f"{RUN_ID}:openr1-math")
    seen: set[str] = set()
    source = pq.ParquetFile(path)
    columns = ["problem", "answer", "uuid", "source", "problem_type", "correctness_count"]
    for batch in source.iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            prompt = text(row.get("problem"))
            reference = text(row.get("answer"))
            if not acceptable_prompt("math", prompt):
                continue
            if int(row.get("correctness_count") or 0) < 1:
                continue
            prompt_hash = digest(normalized(prompt))
            if prompt_hash in excluded or prompt_hash in seen:
                continue
            seen.add(prompt_hash)
            source_id = text(row.get("uuid")) or prompt_hash[:24]
            rid = digest(RUN_ID, MATH_DATASET, source_id, prompt)[:24]
            sample.add(
                source_id,
                make_simple_record(
                    rid=rid,
                    task="math",
                    prompt=prompt,
                    source_dataset=MATH_DATASET,
                    source_revision=MATH_REVISION,
                    source_id=source_id,
                    license_name="Apache-2.0",
                    reference=reference,
                    extra_meta={"source": row.get("source"), "problem_type": row.get("problem_type")},
                ),
            )
    values = sample.values()
    if len(values) != TARGETS["math"]:
        raise RuntimeError(f"math: selected {len(values)} of {TARGETS['math']}")
    return values


def build_code(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    sample = StableSample(TARGETS["code"], f"{RUN_ID}:open-code-reasoning")
    seen: set[str] = set()
    source = pq.ParquetFile(path)
    columns = ["id", "input", "output", "solution", "source", "dataset", "difficulty", "license"]
    for batch in source.iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            prompt = text(row.get("input"))
            reference = text(row.get("solution")) or text(row.get("output"))
            if not acceptable_prompt("code", prompt):
                continue
            prompt_hash = digest(normalized(prompt))
            if prompt_hash in excluded or prompt_hash in seen:
                continue
            seen.add(prompt_hash)
            source_id = text(row.get("id")) or prompt_hash[:24]
            rid = digest(RUN_ID, CODE_DATASET, source_id, prompt)[:24]
            sample.add(
                source_id,
                make_simple_record(
                    rid=rid,
                    task="code",
                    prompt=prompt,
                    source_dataset=CODE_DATASET,
                    source_revision=CODE_REVISION,
                    source_id=source_id,
                    license_name=text(row.get("license")) or "CC-BY-4.0",
                    reference=reference,
                    extra_meta={
                        "source": row.get("source"),
                        "dataset": row.get("dataset"),
                        "difficulty": row.get("difficulty"),
                    },
                ),
            )
    values = sample.values()
    if len(values) != TARGETS["code"]:
        raise RuntimeError(f"code: selected {len(values)} of {TARGETS['code']}")
    return values


def make_simple_record(
    *,
    rid: str,
    task: str,
    prompt: str,
    source_dataset: str,
    source_revision: str,
    source_id: str,
    license_name: str,
    reference: str,
    extra_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"dsv4-v3-{task}-{rid}",
        "task_type": task,
        "language": "en",
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_id": source_id,
        "prompt_version": "dsv4-public-v3",
        "teacher_target": "deepseek-v4-flash",
        "distill_partition": "public_v3",
        "source_meta": {
            "license": license_name,
            "reference_sha256": digest(reference) if reference else None,
            "reference_chars": len(reference),
            **extra_meta,
        },
        "messages": [
            {"role": "system", "content": SYSTEMS[task]},
            {"role": "user", "content": prompt},
        ],
    }


def response_bucket(item: dict[str, Any]) -> str:
    chars = int(item["source_meta"].get("reference_chars") or 0)
    if chars <= 5_000:
        return "short"
    if chars <= 18_000:
        return "medium"
    return "long"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--chat-source", type=Path, required=True)
    parser.add_argument("--math-source", type=Path, required=True)
    parser.add_argument("--code-source", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()

    excluded = user_prompt_hashes(args.seed_dir)
    records: list[dict[str, Any]] = [
        *build_chat(args.chat_source, excluded),
        *build_math(args.math_source, excluded),
        *build_code(args.code_source, excluded),
    ]
    for task in ("stem", "tool_calling"):
        records.extend(build_task(args.source_root / f"{task}-0000.parquet", task, excluded))

    if len(records) != sum(TARGETS.values()):
        raise RuntimeError(f"expected {sum(TARGETS.values())} records, got {len(records)}")
    ids = {item["id"] for item in records}
    prompts = {digest(normalized(first_user(item["messages"]))) for item in records}
    if len(ids) != len(records) or len(prompts) != len(records):
        raise RuntimeError("duplicate ids or prompts survived v3 filtering")

    buckets = {bucket: [] for bucket in BUCKETS}
    for item in records:
        buckets[response_bucket(item)].append(item)
    args.seed_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "total": len(records),
        "public_source_only": True,
        "sources": [
            {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0"},
            {"dataset": CHAT_DATASET, "revision": CHAT_REVISION, "license": "MIT"},
            {"dataset": MATH_DATASET, "revision": MATH_REVISION, "license": "Apache-2.0"},
            {"dataset": CODE_DATASET, "revision": CODE_REVISION, "license": "CC-BY-4.0; upstream-dependent"},
        ],
        "tasks": dict(Counter(item["task_type"] for item in records)),
        "buckets": {},
        "prior_prompt_hashes_excluded": len(excluded),
    }
    for bucket, items in buckets.items():
        items.sort(key=lambda item: digest(args.run_id, bucket, item["id"]))
        path = args.seed_dir / f"{args.run_id}-{bucket}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for item in items:
                target.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["buckets"][bucket] = {
            "count": len(items),
            "tasks": dict(Counter(item["task_type"] for item in items)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path = args.seed_dir / f"{args.run_id}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
