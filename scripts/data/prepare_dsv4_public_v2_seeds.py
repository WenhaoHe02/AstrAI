"""Build an 8,192-record, public-source-only DSV4 distillation mix.

The teacher sees prompts and public trajectory prefixes, never the upstream
assistant answer.  Upstream answers are represented only by hashes for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BUCKETS = ("short", "medium", "long")
TARGETS = {"general_zh": 4096, "minecraft_qa": 2048, "agent_next_action": 2048}
SYSTEMS = {
    "general_zh": (
        "你是一名严谨、自然的中文助手。准确回答问题，需要推理时给出简洁、可核验的推导；"
        "不虚构事实或引用，并把最终回答放在 <final>...</final> 中。"
    ),
    "minecraft_qa": (
        "You are a careful Minecraft knowledge assistant. Answer from stable game mechanics, "
        "state edition or version uncertainty when relevant, do not invent commands or recipes, "
        "and put the complete answer in <final>...</final>."
    ),
    "agent_next_action": (
        "You are reviewing a public tool-agent trajectory. Follow its policy and available tool "
        "schema exactly. Produce the best next assistant turn without assuming an unobserved tool "
        "succeeded. Return either a user-facing message or one tool call as valid JSON inside "
        "<final>...</final>."
    ),
}


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


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
        return [item[2] for item in sorted(self.heap, key=lambda x: (-x[0], x[1]))]


def make_record(
    source_dataset: str,
    source_id: str,
    task_type: str,
    language: str,
    prompt: str,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    rid = digest("posttrain-dsv4-public-v2", source_dataset, source_id, prompt)[:24]
    return {
        "id": f"dsv4-public-{task_type}-{rid}",
        "task_type": task_type,
        "language": language,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "prompt_version": "dsv4-public-v2",
        "teacher_target": "deepseek-v4-flash",
        "distill_partition": "public_v2",
        "source_meta": source_meta,
        "messages": [
            {"role": "system", "content": SYSTEMS[task_type]},
            {"role": "user", "content": prompt},
        ],
    }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def existing_prompt_hashes(seed_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in seed_dir.glob("posttrain-v[45]-*.jsonl"):
        for item in read_jsonl(path):
            messages = item.get("messages")
            if not isinstance(messages, list):
                continue
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    text = clean_text(message.get("content"))
                    if text:
                        hashes.add(digest(normalized(text)))
                    break
    return hashes


def chinese_records(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    sample = StableSample(TARGETS["general_zh"], "public-v2:general-zh")
    seen: set[str] = set()
    for row in read_jsonl(path):
        instruction = clean_text(row.get("instruction")) or clean_text(row.get("prompt"))
        extra = clean_text(row.get("input"))
        prompt = instruction if not extra else f"{instruction}\n\n补充输入：\n{extra}"
        reference = clean_text(row.get("output")) or clean_text(row.get("response"))
        prompt_hash = digest(normalized(prompt))
        # Chinese questions are often semantically complete in far fewer
        # characters than English prompts.  Keep only human-verified rows,
        # but use a Chinese-appropriate length floor.
        if not (20 <= len(prompt) <= 6000 and len(reference) >= 20):
            continue
        if row.get("human_verified") is not True:
            continue
        if prompt_hash in excluded or prompt_hash in seen:
            continue
        seen.add(prompt_hash)
        source_id = clean_text(row.get("id")) or prompt_hash[:24]
        sample.add(
            source_id,
            make_record(
                "m-a-p/COIG-CQIA:COIG-CQIA-full",
                source_id,
                "general_zh",
                "zh",
                prompt,
                {"reference_sha256": digest(reference)},
            ),
        )
    result = sample.values()
    if len(result) != TARGETS["general_zh"]:
        raise RuntimeError(f"Chinese public sample short: {len(result)}")
    return result


VERSION_QUESTION = re.compile(
    r"(?:\b(?:snapshot|pre-release|alpha|beta)\b|\b(?:java|bedrock) edition\s+\d|\b\d+\.\d+(?:\.\d+)?\b)",
    re.IGNORECASE,
)


def minecraft_records(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Minecraft dataset must be a JSON array")
    sample = StableSample(TARGETS["minecraft_qa"], "public-v2:minecraft")
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        instruction = clean_text(row.get("instruction"))
        extra = clean_text(row.get("input"))
        reference = clean_text(row.get("output"))
        prompt = instruction if not extra else f"{instruction}\n\nContext:\n{extra}"
        prompt_hash = digest(normalized(prompt))
        if not (30 <= len(prompt) <= 2500 and 20 <= len(reference) <= 6000):
            continue
        if VERSION_QUESTION.search(prompt) or prompt_hash in excluded or prompt_hash in seen:
            continue
        seen.add(prompt_hash)
        source_id = f"row-{index}"
        sample.add(
            source_id,
            make_record(
                "Aiwensile2/Minecraft_QA-pairs_Instruction_Dataset",
                source_id,
                "minecraft_qa",
                "en",
                prompt,
                {"reference_sha256": digest(reference), "license": "CC-BY-NC-SA-3.0"},
            ),
        )
    result = sample.values()
    if len(result) != TARGETS["minecraft_qa"]:
        raise RuntimeError(f"Minecraft public sample short: {len(result)}")
    return result


def agent_prompt(messages: list[Any], tools: Any, assistant_index: int) -> str:
    prefix = messages[:assistant_index]
    return (
        "Available tools (JSON):\n"
        + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        + "\n\nConversation prefix (JSON):\n"
        + json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
        + "\n\nGenerate only the best next assistant turn."
    )


def agent_records(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    sample = StableSample(TARGETS["agent_next_action"], "public-v2:nemotron-agent")
    seen: set[str] = set()
    for row in read_jsonl(path):
        messages = row.get("messages")
        tools = row.get("tools")
        source_id = clean_text(row.get("uuid"))
        if not source_id or not isinstance(messages, list) or not isinstance(tools, list) or not tools:
            continue
        candidates = [
            i for i, message in enumerate(messages)
            if i > 0 and isinstance(message, dict) and message.get("role") == "assistant"
        ]
        if not candidates:
            continue
        assistant_index = candidates[int(digest(source_id)[:8], 16) % len(candidates)]
        target = messages[assistant_index]
        prompt = agent_prompt(messages, tools, assistant_index)
        prompt_hash = digest(normalized(prompt))
        if not (300 <= len(prompt) <= 16000):
            continue
        if prompt_hash in excluded or prompt_hash in seen:
            continue
        seen.add(prompt_hash)
        sample.add(
            source_id,
            make_record(
                "nvidia/Nemotron-Agentic-v1:interactive_agent",
                source_id,
                "agent_next_action",
                "en",
                prompt,
                {
                    "target_turn_sha256": digest(json.dumps(target, ensure_ascii=False, sort_keys=True)),
                    "source_license": clean_text(row.get("license")) or "CC-BY-4.0",
                    "assistant_turn_index": assistant_index,
                },
            ),
        )
    result = sample.values()
    if len(result) != TARGETS["agent_next_action"]:
        raise RuntimeError(f"Nemotron public sample short: {len(result)}")
    return result


def bucketize(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets = {bucket: [] for bucket in BUCKETS}
    for task_type in TARGETS:
        items = [item for item in records if item["task_type"] == task_type]
        items.sort(key=lambda item: digest("public-v2:bucket", str(item["id"])))
        short_end = round(len(items) * 0.55)
        medium_end = short_end + round(len(items) * 0.28)
        buckets["short"].extend(items[:short_end])
        buckets["medium"].extend(items[short_end:medium_end])
        buckets["long"].extend(items[medium_end:])
    for bucket in BUCKETS:
        buckets[bucket].sort(key=lambda item: digest("public-v2:order", str(item["id"])))
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="posttrain-dsv4-public-v2")
    args = parser.parse_args()

    excluded = existing_prompt_hashes(args.seed_dir)
    records = [
        *chinese_records(
            args.source_root / "public-v2/COIG-CQIA-full.jsonl", excluded
        ),
        *minecraft_records(
            args.source_root / "public-v2/minecraft_instruction_dataset.json", excluded
        ),
        *agent_records(
            args.source_root / "public-v2/nemotron_interactive_agent.jsonl", excluded
        ),
    ]
    if len(records) != 8192 or len({item["id"] for item in records}) != 8192:
        raise RuntimeError("expected 8,192 unique public records")

    buckets = bucketize(records)
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "total": len(records),
        "public_source_only": True,
        "tasks": dict(Counter(item["task_type"] for item in records)),
        "languages": dict(Counter(item["language"] for item in records)),
        "buckets": {},
    }
    for bucket, items in buckets.items():
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
