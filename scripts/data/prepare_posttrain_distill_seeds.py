"""Build a balanced, auditable post-training distillation seed set.

The output only contains source prompts and metadata. Teacher responses are
generated separately by ``distill_glm.py`` so generation can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SYSTEM_PROMPTS = {
    "math_reasoning": (
        "You are a rigorous mathematics teacher. Solve the problem independently. "
        "Use a concise, checkable derivation, verify the result, and place the final "
        "answer in <final>...</final>. Do not mention the source answer."
    ),
    "code_reasoning": (
        "You are an expert competitive-programming teacher. Derive a correct "
        "algorithm, explain the key invariant and complexity, consider edge cases, "
        "then provide a complete Python 3 solution in <final>...</final>."
    ),
    "general_zh": (
        "你是一名严谨的中文助教。准确遵循用户要求；需要推理时给出简洁、可核验的推导，"
        "不要虚构事实或引用，最终回答放在 <final>...</final> 中。"
    ),
    "general_en": (
        "You are a careful general assistant. Follow every constraint, use concise "
        "and checkable reasoning when needed, do not fabricate facts or citations, "
        "and put the final response in <final>...</final>."
    ),
    "critique_math": (
        "You are checking a proposed mathematical solution. Identify the first "
        "substantive error, if any, repair the reasoning, verify the answer, and put "
        "the corrected solution in <final>...</final>."
    ),
    "critique_code": (
        "You are reviewing a proposed programming solution. Check correctness, "
        "complexity, input/output handling and edge cases. Repair it when necessary "
        "and put the corrected explanation and Python 3 code in <final>...</final>."
    ),
}


def rows(path: Path, columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def stable_shuffle(items: list[dict[str, Any]], seed: int, namespace: str) -> None:
    rng = random.Random(f"{seed}:{namespace}")
    rng.shuffle(items)


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def make_record(
    *,
    source_id: str,
    source_dataset: str,
    task_type: str,
    language: str,
    prompt: str,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    # Some upstream corpora reuse a problem ID for multiple variants. Include
    # the exact prompt so every semantically distinct request remains unique,
    # while a rerun with unchanged inputs is still idempotent.
    digest = hashlib.sha256(
        f"{source_dataset}\0{source_id}\0{task_type}\0{prompt}".encode("utf-8")
    ).hexdigest()[:24]
    return {
        "id": f"{task_type}-{digest}",
        "task_type": task_type,
        "language": language,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "prompt_version": "glm52-posttrain-v1",
        "source_meta": source_meta,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[task_type]},
            {"role": "user", "content": prompt},
        ],
    }


def select_math(path: Path, count: int, critique_count: int, seed: int):
    candidates = []
    for row in rows(
        path,
        [
            "problem",
            "answer",
            "problem_type",
            "question_type",
            "source",
            "uuid",
            "generations",
            "correctness_math_verify",
            "correctness_count",
        ],
    ):
        problem = clean_text(row["problem"])
        if not 80 <= len(problem) <= 12_000 or int(row.get("correctness_count") or 0) < 1:
            continue
        candidates.append(row)
    stable_shuffle(candidates, seed, "math")
    result = []
    for row in candidates[:count]:
        result.append(
            make_record(
                source_id=row["uuid"],
                source_dataset="open-r1/OpenR1-Math-220k:default:shard0",
                task_type="math_reasoning",
                language="zh" if any("\u4e00" <= c <= "\u9fff" for c in row["problem"]) else "en",
                prompt=row["problem"],
                source_meta={
                    "source": row["source"],
                    "problem_type": row["problem_type"],
                    "question_type": row["question_type"],
                    "reference_answer": row["answer"],
                },
            )
        )
    critique_pool = candidates[count : count + critique_count * 3]
    for row in critique_pool:
        generations = row.get("generations") or []
        verdicts = row.get("correctness_math_verify") or []
        draft = next(
            (text for text, ok in zip(generations, verdicts) if text and not ok),
            None,
        )
        if not draft:
            continue
        prompt = f"Problem:\n{row['problem']}\n\nProposed solution:\n{draft[:14_000]}"
        result.append(
            make_record(
                source_id=row["uuid"],
                source_dataset="open-r1/OpenR1-Math-220k:default:shard0",
                task_type="critique_math",
                language="en",
                prompt=prompt,
                source_meta={"reference_answer": row["answer"], "source": row["source"]},
            )
        )
        if sum(x["task_type"] == "critique_math" for x in result) >= critique_count:
            break
    return result


def select_code(path: Path, count: int, critique_count: int, seed: int):
    candidates = []
    for row in rows(
        path,
        ["id", "input", "output", "source", "license", "dataset", "split", "difficulty", "solution"],
    ):
        prompt = clean_text(row["input"])
        if 200 <= len(prompt) <= 12_000 and prompt != "-":
            candidates.append(row)
    stable_shuffle(candidates, seed, "code")
    result = []
    for row in candidates[:count]:
        result.append(
            make_record(
                source_id=row["id"],
                source_dataset="nvidia/OpenCodeReasoning:split_0:shard0",
                task_type="code_reasoning",
                language="en",
                prompt=row["input"],
                source_meta={
                    "source": row["source"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "difficulty": row["difficulty"],
                    "license": row["license"],
                    "reference_solution": row["solution"],
                },
            )
        )
    for row in candidates[count : count + critique_count]:
        draft = clean_text(row["output"])
        if not draft:
            continue
        prompt = f"Problem:\n{row['input']}\n\nProposed solution:\n{draft[:14_000]}"
        result.append(
            make_record(
                source_id=row["id"],
                source_dataset="nvidia/OpenCodeReasoning:split_0:shard0",
                task_type="critique_code",
                language="en",
                prompt=prompt,
                source_meta={
                    "source": row["source"],
                    "dataset": row["dataset"],
                    "reference_solution": row["solution"],
                },
            )
        )
    return result


def conversation_user_text(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ""
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("from") or message.get("role")
        if role in {"human", "user"}:
            return clean_text(message.get("value") or message.get("content"))
    return ""


def select_general(path: Path, zh_count: int, en_count: int, seed: int):
    selected: dict[str, list[dict[str, Any]]] = {"zh": [], "en": []}
    for row in rows(path):
        language = clean_text(row.get("langdetect")).lower()
        language = "zh" if language.startswith("zh") else "en" if language.startswith("en") else ""
        if not language:
            continue
        prompt = conversation_user_text(row.get("conversations"))
        if 80 <= len(prompt) <= 10_000:
            selected[language].append({**row, "_prompt": prompt})
    result = []
    for language, count in (("zh", zh_count), ("en", en_count)):
        stable_shuffle(selected[language], seed, f"general-{language}")
        for row in selected[language][:count]:
            task_type = f"general_{language}"
            source_id = clean_text(row.get("id")) or hashlib.sha256(
                row["_prompt"].encode("utf-8")
            ).hexdigest()
            result.append(
                make_record(
                    source_id=source_id,
                    source_dataset="manifoldlabs/Infinity-Instruct:0625:shard0",
                    task_type=task_type,
                    language=language,
                    prompt=row["_prompt"],
                    source_meta={"source": row.get("source"), "label": row.get("label")},
                )
            )
    return result


def select_chinese_jsonl(path: Path, count: int, seed: int):
    candidates = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = clean_text(row.get("prompt"))
            if 80 <= len(prompt) <= 10_000:
                candidates.append({**row, "_prompt": prompt})
    stable_shuffle(candidates, seed, "chinese-supplement")
    result = []
    for row in candidates[:count]:
        source_id = clean_text(row.get("id")) or hashlib.sha256(
            row["_prompt"].encode("utf-8")
        ).hexdigest()
        result.append(
            make_record(
                source_id=source_id,
                source_dataset="lcccher/Chinese-Instruct:coig-cqia",
                task_type="general_zh",
                language="zh",
                prompt=row["_prompt"],
                source_meta={"reference_response": row.get("response")},
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-parquet", type=Path, required=True)
    parser.add_argument("--code-parquet", type=Path, required=True)
    parser.add_argument("--general-parquet", type=Path, required=True)
    parser.add_argument("--chinese-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--math", type=int, default=768)
    parser.add_argument("--code", type=int, default=512)
    parser.add_argument("--general-zh", type=int, default=384)
    parser.add_argument("--general-en", type=int, default=384)
    parser.add_argument("--chinese-supplement", type=int, default=261)
    parser.add_argument("--critique-math", type=int, default=128)
    parser.add_argument("--critique-code", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        *select_math(args.math_parquet, args.math, args.critique_math, args.seed),
        *select_code(args.code_parquet, args.code, args.critique_code, args.seed),
        *select_general(args.general_parquet, args.general_zh, args.general_en, args.seed),
    ]
    if args.chinese_jsonl:
        records.extend(
            select_chinese_jsonl(
                args.chinese_jsonl, args.chinese_supplement, args.seed
            )
        )
    # Upstream shards can contain byte-identical duplicate rows. Calling the
    # teacher twice would waste quota and would violate the state DB's unique ID
    # invariant, so collapse exact requests before shuffling and exporting.
    records = list({record["id"]: record for record in records}.values())
    stable_shuffle(records, args.seed, "combined")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")
            counts[record["task_type"]] = counts.get(record["task_type"], 0) + 1
    print(json.dumps({"total": len(records), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
