"""Build the balanced AstrAI post-training v5 teacher seed set.

The v5 mix deliberately compensates for v4's English-heavy distribution so
the cumulative v4+v5 request mix is approximately 1:1 Chinese/English. Source
responses are retained only as audit metadata; the teacher sees the prompt (or
the prompt plus a rejected draft for critique tasks).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SYSTEMS = {
    "general_zh": (
        "你是一名严谨的中文助教。准确回答问题；需要推理时给出简洁、可核验的推导，"
        "不要虚构事实或引用，最终回答放在 <final>...</final> 中。"
    ),
    "math_reasoning": (
        "You are a rigorous mathematics teacher. Solve the problem independently, "
        "show a concise checkable derivation, verify the result, and put the final "
        "answer in <final>...</final>."
    ),
    "code_reasoning": (
        "You are an expert programming teacher. Derive a correct algorithm, explain "
        "its invariant, complexity, and edge cases, then provide a complete solution "
        "in the language requested by the user inside <final>...</final>."
    ),
    "general_en": (
        "You are a careful general assistant. Follow every constraint, reason "
        "concisely when needed, do not fabricate facts or citations, and put the "
        "final response in <final>...</final>."
    ),
    "critique_en": (
        "You are reviewing a proposed answer. Identify substantive errors, omissions, "
        "or instruction-following failures, then write a corrected self-contained "
        "answer in <final>...</final>."
    ),
}


def text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def record(dataset: str, source_id: str, task: str, language: str, prompt: str,
           meta: dict[str, Any]) -> dict[str, Any]:
    rid = digest("posttrain-v5", dataset, source_id, task, prompt)[:24]
    return {
        "id": f"{task}-{rid}",
        "task_type": task,
        "language": language,
        "source_dataset": dataset,
        "source_id": source_id,
        "prompt_version": "glm52-posttrain-v2",
        "source_meta": meta,
        "messages": [
            {"role": "system", "content": SYSTEMS[task]},
            {"role": "user", "content": prompt},
        ],
    }


class StableSample:
    """Keep the N records with the smallest deterministic hash scores."""

    def __init__(self, limit: int, namespace: str):
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
        return [x[2] for x in sorted(self.heap, key=lambda x: (-x[0], x[1]))]


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def chinese_records(stem: Path, coig: Path) -> list[dict[str, Any]]:
    # stem_zh contains 1,327 prompts above the quality/length floor. Keep a
    # little margin instead of relaxing that floor, and fill the remainder
    # from the broader COIG-CQIA Chinese instruction split.
    targets = ((stem, 1200, "lcccher/Chinese-Instruct:stem_zh", "stem"),
               (coig, 1552, "lcccher/Chinese-Instruct:coig-cqia", "coig"))
    out = []
    for path, count, dataset, namespace in targets:
        sample = StableSample(count, namespace)
        for row in jsonl_rows(path):
            prompt = text(row.get("prompt"))
            response = text(row.get("response"))
            if not 80 <= len(prompt) <= 10_000:
                continue
            sid = text(row.get("id")) or digest(prompt)
            sample.add(sid, record(dataset, sid, "general_zh", "zh", prompt,
                                   {"reference_response": response}))
        values = sample.values()
        if len(values) != count:
            raise RuntimeError(f"{dataset}: selected {len(values)} of {count}")
        out.extend(values)
    return out


def first_user(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return text(message.get("content"))
    return ""


def mot_records(root: Path) -> list[dict[str, Any]]:
    specs = {
        "open-r1/OpenR1-Math-220k": ("math_reasoning", 320),
        "open-r1/codeforces-cots": ("code_reasoning", 320),
        "nvidia/Llama-Nemotron-Post-Training-Dataset": ("general_en", 256),
    }
    samples = {source: StableSample(count, f"mot:{source}")
               for source, (_, count) in specs.items()}
    for path in sorted(root.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024,
                                           columns=["messages", "num_tokens", "source"]):
            for row in batch.to_pylist():
                source = row.get("source")
                if source not in specs:
                    continue
                prompt = first_user(row.get("messages"))
                if not 100 <= len(prompt) <= 16_000:
                    continue
                task, _ = specs[source]
                sid = digest(source, prompt)
                samples[source].add(
                    sid,
                    record(
                        f"open-r1/Mixture-of-Thoughts:{source}", sid, task, "en", prompt,
                        {"upstream_source": source, "upstream_num_tokens": row.get("num_tokens")},
                    ),
                )
    out = []
    for source, (_, count) in specs.items():
        values = samples[source].values()
        if len(values) != count:
            raise RuntimeError(f"{source}: selected {len(values)} of {count}")
        out.extend(values)
    return out


def assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return text(message.get("content"))
    return ""


def critique_records(path: Path) -> list[dict[str, Any]]:
    sample = StableSample(448, "ultrafeedback")
    parquet = pq.ParquetFile(path)
    columns = ["prompt", "prompt_id", "chosen", "rejected", "score_chosen", "score_rejected"]
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            prompt = text(row.get("prompt"))
            rejected = assistant_text(row.get("rejected"))
            chosen = assistant_text(row.get("chosen"))
            chosen_score = float(row.get("score_chosen") or 0.0)
            rejected_score = float(row.get("score_rejected") or 0.0)
            if not (80 <= len(prompt) <= 8_000 and 80 <= len(rejected) <= 12_000):
                continue
            if not chosen or chosen_score - rejected_score < 1.0:
                continue
            sid = text(row.get("prompt_id")) or digest(prompt)
            critique = f"User request:\n{prompt}\n\nProposed answer:\n{rejected}"
            sample.add(
                sid,
                record(
                    "HuggingFaceH4/ultrafeedback_binarized:train_prefs", sid,
                    "critique_en", "en", critique,
                    {"chosen_reference": chosen, "chosen_score": chosen_score,
                     "rejected_score": rejected_score},
                ),
            )
    values = sample.values()
    if len(values) != 448:
        raise RuntimeError(f"ultrafeedback: selected {len(values)} of 448")
    return values


def bucketize(records: list[dict[str, Any]], output_dir: Path) -> None:
    buckets = {"short": [], "medium": [], "long": []}
    # Assign within every task type so each bucket preserves the global mix.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        grouped.setdefault(item["task_type"], []).append(item)
    for task, items in grouped.items():
        items.sort(key=lambda x: digest("bucket", task, x["id"]))
        n = len(items)
        short_end = round(n * 0.55)
        medium_end = short_end + round(n * 0.28)
        buckets["short"].extend(items[:short_end])
        buckets["medium"].extend(items[short_end:medium_end])
        buckets["long"].extend(items[medium_end:])
    output_dir.mkdir(parents=True, exist_ok=True)
    for bucket, items in buckets.items():
        items.sort(key=lambda x: digest("order", bucket, x["id"]))
        path = output_dir / f"posttrain-v5-{bucket}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for item in items:
                target.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(json.dumps({"bucket": bucket, "count": len(items)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = [
        *chinese_records(
            args.source_root / "additional/chinese-instruct/stem_zh.jsonl",
            args.source_root / "coig-cqia.jsonl",
        ),
        *mot_records(args.source_root / "additional/mixture-of-thoughts"),
        *critique_records(args.source_root / "additional/ultrafeedback/train_prefs.parquet"),
    ]
    ids = {item["id"] for item in records}
    if len(records) != 4096 or len(ids) != len(records):
        raise RuntimeError(f"expected 4096 unique records, got {len(records)} / {len(ids)}")
    counts: dict[str, int] = {}
    languages: dict[str, int] = {}
    for item in records:
        counts[item["task_type"]] = counts.get(item["task_type"], 0) + 1
        languages[item["language"]] = languages.get(item["language"], 0) + 1
    print(json.dumps({"total": len(records), "tasks": counts, "languages": languages}, sort_keys=True))
    bucketize(records, args.output_dir)


if __name__ == "__main__":
    main()
