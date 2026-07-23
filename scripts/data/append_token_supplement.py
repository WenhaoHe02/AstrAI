"""Append an exact-size language supplement as a pretokenized mmap part."""

from __future__ import annotations

import argparse
import array
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

from astrai.tokenize import AutoTokenizer
from scripts.data.balance_pretrain import expand, iter_records


def count_nonempty_documents(path: Path) -> int:
    return sum(
        1
        for item in iter_records([path])
        if isinstance(item.get("text"), str) and item["text"].strip()
    )


def locate_document_offset(
    paths: list[Path], counts: list[int], skip_documents: int
) -> tuple[list[Path], int]:
    if len(paths) != len(counts):
        raise ValueError("paths and counts must have the same length")
    remaining = skip_documents
    for ordinal, count in enumerate(counts):
        if remaining < count:
            return paths[ordinal:], remaining
        remaining -= count
    if remaining == 0:
        return [], 0
    raise RuntimeError(
        f"cannot skip {skip_documents:,} documents; source has "
        f"{sum(counts):,}"
    )


def build_supplement(
    records: Iterable[dict],
    encode: Callable[[str], list[int]],
    eos_token_id: int,
    target_tokens: int,
) -> tuple[array.array, dict[str, int | bool]]:
    tokens = array.array("i")
    documents = 0
    truncated_final_document = False
    for item in records:
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        ids = encode(text)
        if not ids:
            continue
        remaining = target_tokens - len(tokens)
        if remaining <= 0:
            break
        if len(ids) + 1 <= remaining:
            tokens.extend(ids)
            tokens.append(eos_token_id)
        else:
            if remaining > 1:
                tokens.extend(ids[: remaining - 1])
            tokens.append(eos_token_id)
            truncated_final_document = True
        documents += 1
        if len(tokens) == target_tokens:
            break

    if len(tokens) != target_tokens:
        raise RuntimeError(
            f"source exhausted at {len(tokens):,} tokens, expected {target_tokens:,}"
        )
    return tokens, {
        "tokens": len(tokens),
        "documents": documents,
        "eos": documents,
        "truncated_final_document": truncated_final_document,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--skip-documents", type=int, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-part", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--count-workers", type=int, default=32)
    args = parser.parse_args()
    if args.skip_documents < 0:
        parser.error("--skip-documents must be non-negative")
    if args.target_tokens < 1:
        parser.error("--target-tokens must be positive")
    if args.count_workers < 1:
        parser.error("--count-workers must be positive")
    if args.output_part.exists():
        raise SystemExit(f"output part already exists: {args.output_part}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise SystemExit("tokenizer does not define eos_token_id")

    paths = expand(args.source)
    with ProcessPoolExecutor(max_workers=min(args.count_workers, len(paths))) as pool:
        counts = list(pool.map(count_nonempty_documents, paths))
    selected_paths, intra_file_skip = locate_document_offset(
        paths, counts, args.skip_documents
    )
    nonempty_records = (
        item
        for item in iter_records(selected_paths)
        if isinstance(item.get("text"), str) and item["text"].strip()
    )
    remaining_records = islice(nonempty_records, intra_file_skip, None)
    tokens, stats = build_supplement(
        remaining_records,
        lambda text: tokenizer.encode(text, add_special_tokens=True),
        eos_token_id,
        args.target_tokens,
    )

    temporary = args.output_part.with_name(args.output_part.name + ".tmp")
    shard_dir = temporary / "__default__" / "shard_0000"
    try:
        shard_dir.mkdir(parents=True)
        sequence_path = shard_dir / "sequence.bin"
        with sequence_path.open("wb") as handle:
            tokens.tofile(handle)
        (shard_dir / "meta.json").write_text(
            json.dumps(
                {"sequence": {"shape": [len(tokens)], "dtype": "int32"}}
            ),
            encoding="utf-8",
        )
        provenance = {
            **stats,
            "language": args.language,
            "source": [str(path) for path in paths],
            "skip_documents": args.skip_documents,
            "start_source": str(selected_paths[0]),
            "intra_file_skip": intra_file_skip,
            "eos_token_id": eos_token_id,
        }
        (temporary / "_SUCCESS").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        os.replace(temporary, args.output_part)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
