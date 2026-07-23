"""Create a deterministic, tokenizer-exact language-balanced JSONL stream."""

import argparse
import array
import gzip
import json
import os
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path
from typing import Iterator, TextIO

from tokenizers import Tokenizer


TOKEN_COUNT = struct.Struct("<Q")
_INDEX_TOKENIZER: Tokenizer | None = None


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
    # Every producer and consumer of the compact token index must use the
    # identical file order.  ``Path.rglob`` ordering is filesystem-dependent.
    return sorted(paths)


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


def write_token_index(
    paths: list[Path], tokenizer: Tokenizer, batch_size: int, output: Path
) -> int:
    """Tokenize once and persist one exact uint64 token count per document."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    total = 0
    try:
        with temporary.open("wb") as handle:
            for _, token_ids in iter_tokenized(paths, tokenizer, batch_size):
                length = len(token_ids)
                handle.write(TOKEN_COUNT.pack(length))
                total += length
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return total


def _init_index_worker(tokenizer_path: str) -> None:
    global _INDEX_TOKENIZER
    _INDEX_TOKENIZER = Tokenizer.from_file(tokenizer_path)


def _write_index_part(
    task: tuple[int, Path, int, Path],
) -> tuple[int, int]:
    ordinal, source, batch_size, part = task
    if _INDEX_TOKENIZER is None:
        raise RuntimeError("token index worker was not initialized")
    total = write_token_index([source], _INDEX_TOKENIZER, batch_size, part)
    return ordinal, total


def write_token_index_parallel(
    paths: list[Path],
    tokenizer_path: str,
    batch_size: int,
    output: Path,
    workers: int,
) -> int:
    """Build per-file indexes concurrently, then concatenate in source order."""

    paths = sorted(paths)

    if workers <= 1 or len(paths) <= 1:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return write_token_index(paths, tokenizer, batch_size, output)

    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.with_name(output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [parts_dir / f"{ordinal:05d}.u64" for ordinal in range(len(paths))]
    tasks = [
        (ordinal, source, batch_size, part_paths[ordinal])
        for ordinal, source in enumerate(paths)
    ]
    totals = [0] * len(tasks)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=get_context("spawn"),
            initializer=_init_index_worker,
            initargs=(tokenizer_path,),
        ) as pool:
            futures = [pool.submit(_write_index_part, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                ordinal, total = future.result()
                totals[ordinal] = total
                print(
                    f"indexed {completed}/{len(tasks)} files: "
                    f"{paths[ordinal].name} ({total:,} tokens)",
                    flush=True,
                )

        with temporary.open("wb") as destination:
            for part in part_paths:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        for part in part_paths:
            part.unlink(missing_ok=True)
        try:
            parts_dir.rmdir()
        except OSError:
            pass
    return sum(totals)


def token_index_stats(index: Path) -> tuple[int, int]:
    """Return ``(documents, text_tokens)`` from a compact uint64 index."""

    size = index.stat().st_size
    if size % TOKEN_COUNT.size:
        raise RuntimeError(f"invalid token index byte length: {index}")
    total = 0
    with index.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            values = array.array("Q")
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            total += sum(values)
    return size // TOKEN_COUNT.size, total


def validate_reusable_index(index: Path, paths: list[Path]) -> tuple[int, int]:
    if not index.is_file():
        raise RuntimeError(f"reusable token index does not exist: {index}")
    newest_source = max(path.stat().st_mtime_ns for path in paths)
    if index.stat().st_mtime_ns < newest_source:
        raise RuntimeError(f"token index is older than its source data: {index}")
    return token_index_stats(index)


def iter_indexed(
    paths: list[Path], index: Path, document_token_overhead: int = 0
) -> Iterator[tuple[dict, int, None]]:
    """Pair source records with counts produced by :func:`write_token_index`."""

    with index.open("rb") as counts:
        for item in iter_records(paths):
            if not item.get("text", "").strip():
                continue
            packed = counts.read(TOKEN_COUNT.size)
            if len(packed) != TOKEN_COUNT.size:
                raise RuntimeError(f"token index ended before source records: {index}")
            yield (
                item,
                TOKEN_COUNT.unpack(packed)[0] + document_token_overhead,
                None,
            )
        if counts.read(1):
            raise RuntimeError(f"token index contains extra records: {index}")


def iter_with_tokens(
    paths: list[Path],
    tokenizer: Tokenizer,
    batch_size: int,
    document_token_overhead: int = 0,
) -> Iterator[tuple[dict, int, list[int]]]:
    for item, token_ids in iter_tokenized(paths, tokenizer, batch_size):
        yield item, len(token_ids) + document_token_overhead, token_ids


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
    parser.add_argument(
        "--index-workers",
        type=int,
        default=1,
        help="Processes used to build token-count indexes (default: 1)",
    )
    parser.add_argument(
        "--reuse-token-indexes",
        action="store_true",
        help="Reuse existing .tokens.u64 files after freshness validation",
    )
    parser.add_argument(
        "--document-token-overhead",
        type=int,
        default=0,
        help="Tokens added per document downstream, such as one EOS token",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.index_workers < 1:
        parser.error("--index-workers must be positive")
    if args.document_token_overhead < 0:
        parser.error("--document-token-overhead must be non-negative")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    paths = {"zh": expand(args.zh), "en": expand(args.en)}
    if args.tokens_per_language == "auto":
        output = Path(args.output)
        indexes = {
            language: output.with_name(output.name + f".{language}.tokens.u64")
            for language in paths
        }
        if args.reuse_token_indexes:
            index_stats = {
                language: validate_reusable_index(
                    indexes[language], source_paths
                )
                for language, source_paths in paths.items()
            }
        else:
            index_stats = {}
            for language, source_paths in paths.items():
                text_tokens = write_token_index_parallel(
                    source_paths,
                    args.tokenizer,
                    args.batch_size,
                    indexes[language],
                    args.index_workers,
                )
                documents, indexed_tokens = token_index_stats(indexes[language])
                if indexed_tokens != text_tokens:
                    raise RuntimeError(
                        f"token index sum mismatch for {language}: "
                        f"{indexed_tokens} != {text_tokens}"
                    )
                index_stats[language] = (documents, text_tokens)
        available = {
            language: text_tokens + documents * args.document_token_overhead
            for language, (documents, text_tokens) in index_stats.items()
        }
        target = min(available.values())
        print(
            json.dumps(
                {
                    "available_tokens": available,
                    "indexed_documents": {
                        language: stats[0] for language, stats in index_stats.items()
                    },
                    "document_token_overhead": args.document_token_overhead,
                    "target": target,
                }
            )
        )
        streams = {
            language: iter_indexed(
                source_paths,
                indexes[language],
                args.document_token_overhead,
            )
            for language, source_paths in paths.items()
        }
    else:
        target = int(args.tokens_per_language)
        streams = {
            language: iter_with_tokens(
                source_paths,
                tokenizer,
                args.batch_size,
                args.document_token_overhead,
            )
            for language, source_paths in paths.items()
        }
    if target < 1:
        raise SystemExit("token target must be positive")
    counts = {"zh": 0, "en": 0}
    docs = {"zh": 0, "en": 0}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        handle = stack.enter_context(open_text(output, "wt"))
        while min(counts.values()) < target:
            language = min(counts, key=counts.get)
            try:
                item, length, token_ids = next(streams[language])
            except StopIteration as exc:
                raise RuntimeError(
                    f"{language} exhausted at {counts[language]:,} tokens"
                ) from exc
            remaining = target - counts[language]
            if length > remaining:
                text_tokens_to_keep = remaining - args.document_token_overhead
                if text_tokens_to_keep < 1:
                    raise RuntimeError(
                        "exact target would require an empty final document; "
                        "choose a nearby explicit token target"
                    )
                if token_ids is None:
                    token_ids = tokenizer.encode(
                        item["text"], add_special_tokens=False
                    ).ids
                token_ids = token_ids[:text_tokens_to_keep]
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
