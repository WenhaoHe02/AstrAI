"""Evaluate GSM8K with generated-answer exact match."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from tqdm import tqdm

from astrai.inference import InferenceEngine
from astrai.model import AutoModel
from astrai.tokenize import AutoTokenizer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    return format(number.normalize(), "f")


def extract_gold(text: str) -> str | None:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return normalize_number(match.group(1) if match else None)


def extract_prediction(text: str) -> str | None:
    marked = re.findall(
        r"(?:####|final answer(?: is)?\s*[:=]?)\s*\$?"
        r"([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    candidates = marked or NUMBER_RE.findall(text)
    return normalize_number(candidates[-1] if candidates else None)


def load_problems(path: Path) -> list[dict]:
    problems = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = item["text"]
            question, separator, solution = text.partition("\n\nSolution:\n")
            if not separator:
                raise ValueError("GSM8K validation record has no Solution separator")
            gold = extract_gold(solution)
            if gold is None:
                raise ValueError("GSM8K validation record has no #### answer")
            problems.append({"question": question, "gold": gold})
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--param-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    problems = load_problems(args.data_path)
    if args.max_samples is not None:
        problems = problems[: args.max_samples]
    model = AutoModel.from_pretrained(
        args.param_path,
        config_overrides={
            "expert_parallel_size": 1,
            "expert_dispatch_backend": "torch",
            "deepep_overlap_with_compute": False,
            "moe_shared_expert_overlap": False,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output.with_suffix(".jsonl")
    correct = 0
    total = 0
    with InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=args.batch_size,
        max_seq_len=2048,
        max_prompt_len=1536,
    ) as engine, detail_path.open("w", encoding="utf-8") as detail:
        for start in tqdm(
            range(0, len(problems), args.batch_size), desc="GSM8K", unit="batch"
        ):
            batch = problems[start : start + args.batch_size]
            prompts = [
                "Solve the problem. Show your reasoning, then write the final "
                "numeric answer as #### number.\n\nQuestion: "
                + item["question"]
                + "\n\nSolution:\n"
                for item in batch
            ]
            outputs = engine.generate(
                prompt=prompts,
                stream=False,
                max_tokens=args.max_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            if isinstance(outputs, str):
                outputs = [outputs]
            for item, output in zip(batch, outputs):
                prediction = extract_prediction(output)
                passed = prediction == item["gold"]
                correct += int(passed)
                total += 1
                detail.write(
                    json.dumps(
                        {
                            **item,
                            "prediction": prediction,
                            "correct": passed,
                            "response": output,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            detail.flush()

    result = {
        "benchmark": "GSM8K test",
        "metric": "exact_match",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
