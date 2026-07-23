"""Create a deterministic, tokenizer-exact language-balanced JSONL stream."""

import argparse
import gzip
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, TextIO

from tokenizers import Tokenizer


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_records(paths: list[Path]) -> Iterator[dict]:
    for path in sorted(paths):
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
                yield from batch.to_pylist()
            continue
        with open_text(path, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(candidate.rglob("*.jsonl*"))
        else:
            paths.extend(Path().glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched: {patterns}")
    return paths


def iter_tokenized(
    paths: list[Path], tokenizer: Tokenizer, batch_size: int
) -> Iterator[tuple[dict, list[int]]]:
    batch: list[dict] = []
    for item in iter_records(paths):
        if item.get("text", "").strip():
            batch.append(item)
        if len(batch) < batch_size:
            continue
        encoded = tokenizer.encode_batch(
            [item["text"] for item in batch], add_special_tokens=False
        )
        for record, encoding in zip(batch, encoded):
            if encoding.ids:
                yield record, encoding.ids
        batch = []
    if batch:
        encoded = tokenizer.encode_batch(
            [item["text"] for item in batch], add_special_tokens=False
        )
        for record, encoding in zip(batch, encoded):
            if encoding.ids:
                yield record, encoding.ids


def count_tokens(paths: list[Path], tokenizer: Tokenizer, batch_size: int) -> int:
    return sum(
        len(token_ids)
        for _, token_ids in iter_tokenized(paths, tokenizer, batch_size)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh", nargs="+", required=True)
    parser.add_argument("--en", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokens-per-language",
        required=True,
        help="Exact target per language, or 'auto' to use the smaller corpus",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    paths = {"zh": expand(args.zh), "en": expand(args.en)}
    if args.tokens_per_language == "auto":
        available = {
            language: count_tokens(source_paths, tokenizer, args.batch_size)
            for language, source_paths in paths.items()
        }
        target = min(available.values())
        print(json.dumps({"available_tokens": available, "target": target}))
    else:
        target = int(args.tokens_per_language)
    if target < 1:
        raise SystemExit("token target must be positive")

    streams = {
        language: iter_tokenized(source_paths, tokenizer, args.batch_size)
        for language, source_paths in paths.items()
    }
    counts = {"zh": 0, "en": 0}
    docs = {"zh": 0, "en": 0}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        handle = stack.enter_context(open_text(output, "wt"))
        while min(counts.values()) < target:
            language = min(counts, key=counts.get)
            try:
                item, token_ids = next(streams[language])
            except StopIteration as exc:
                raise RuntimeError(
                    f"{language} exhausted at {counts[language]:,} tokens"
                ) from exc
            length = len(token_ids)
            remaining = target - counts[language]
            if length > remaining:
                token_ids = token_ids[:remaining]
                item["text"] = tokenizer.decode(token_ids)
                length = remaining
            item["language"] = language
            item["token_count"] = length
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[language] += length
            docs[language] += 1

    stats_path = output.with_suffix(output.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps({"tokens": counts, "documents": docs}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"tokens": counts, "documents": docs}))


if __name__ == "__main__":
    main()
