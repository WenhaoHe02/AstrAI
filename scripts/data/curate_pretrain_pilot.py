"""Deterministically clean a bounded pretraining pilot with MinHash/LSH.

This local pilot mirrors the production policy without requiring DataTrove:
structural rules -> exact content hash -> 5-gram MinHash -> 9x10 LSH bands.
For the full corpus, use ``curate_pretrain.py minhash`` in the curation env.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import numpy as np

try:
    import xxhash
except ImportError:  # pragma: no cover - production/dev environments have xxhash
    xxhash = None


WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\W\d_]", re.UNICODE)
CODE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|==|!=|<=|>=|->|=>|::|[^\s]"
)


def open_text(path: Path, mode: str) -> TextIO:
    compressed = path.name.endswith(".gz") or path.name.endswith(".gz.tmp")
    return gzip.open(path, mode, encoding="utf-8") if compressed else path.open(mode, encoding="utf-8")


def iter_records(paths: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with open_text(path, "rt") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {path}:{line_number}")
                yield path, line_number, value


def basic_quality_reason(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_useful_ratio: float,
    max_same_char_run: int,
    max_duplicate_line_ratio: float | None = 0.30,
) -> str | None:
    text = text.strip()
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    useful = 0
    last = ""
    run = 0
    for char in text:
        category = unicodedata.category(char)
        if char.isspace() or category[0] in {"L", "N", "P", "S"}:
            useful += 1
        if char == last:
            run += 1
            if run > max_same_char_run:
                return "same_char_run"
        else:
            last = char
            run = 1
    if useful / len(text) < min_useful_ratio:
        return "low_useful_char_ratio"

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if max_duplicate_line_ratio is not None and len(nonempty_lines) >= 10:
        duplicate_lines = len(nonempty_lines) - len(set(nonempty_lines))
        if duplicate_lines / len(nonempty_lines) > max_duplicate_line_ratio:
            return "duplicate_lines"
    return None


def tokens_for_minhash(text: str, language: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    if language == "zh":
        # Keep individual CJK characters while preserving Latin/code words.
        return WORD_RE.findall(normalized)
    if language == "code":
        return CODE_RE.findall(normalized)
    return re.findall(r"[a-z0-9_]+", normalized)


def shingle_hashes(tokens: list[str], ngram: int) -> np.ndarray:
    if len(tokens) < ngram:
        shingles = ["\x1f".join(tokens)] if tokens else []
    else:
        shingles = ["\x1f".join(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    unique = set(shingles)
    if xxhash is not None:
        values = [xxhash.xxh64_intdigest(value) for value in unique]
    else:
        values = [
            int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")
            for value in unique
        ]
    return np.asarray(values, dtype=np.uint64)


def permutation_coefficients(count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0xA57A1)
    a = rng.integers(1, np.iinfo(np.uint64).max, size=count, dtype=np.uint64)
    a |= np.uint64(1)
    b = rng.integers(0, np.iinfo(np.uint64).max, size=count, dtype=np.uint64)
    return a, b


def minhash_signature(
    hashes: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    if hashes.size == 0:
        return np.full(a.shape, np.iinfo(np.uint64).max, dtype=np.uint64)
    result = np.full(a.shape, np.iinfo(np.uint64).max, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for start in range(0, hashes.size, chunk_size):
            chunk = hashes[start : start + chunk_size, None]
            result = np.minimum(result, (chunk * a[None, :] + b[None, :]).min(axis=0))
    return result


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Input order is the deterministic survivor policy.
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass
class KeptRecord:
    item: dict[str, Any]
    signature: np.ndarray


def curate(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [Path(value).resolve() for value in args.input]
    counts: Counter[str] = Counter()
    exact_seen: set[str] = set()
    records: list[KeptRecord] = []
    permutations = args.bands * args.hashes_per_band
    a, b = permutation_coefficients(permutations)

    for _, _, item in iter_records(inputs):
        counts["input"] += 1
        text = item.get(args.text_key)
        if not isinstance(text, str):
            counts["missing_text"] += 1
            continue
        reason = basic_quality_reason(
            text,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_useful_ratio=args.min_useful_ratio,
            max_same_char_run=args.max_same_char_run,
            max_duplicate_line_ratio=None if args.language == "code" else 0.30,
        )
        if reason:
            counts[reason] += 1
            continue
        digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if digest in exact_seen:
            counts["exact_duplicate"] += 1
            continue
        exact_seen.add(digest)
        tokens = tokens_for_minhash(text, args.language)
        signature = minhash_signature(shingle_hashes(tokens, args.ngram), a, b)
        records.append(KeptRecord(item=item, signature=signature))

    union_find = UnionFind(len(records))
    buckets: dict[bytes, int] = {}
    for index, record in enumerate(records):
        for band in range(args.bands):
            start = band * args.hashes_per_band
            stop = start + args.hashes_per_band
            key = band.to_bytes(2, "little") + record.signature[start:stop].tobytes()
            previous = buckets.get(key)
            if previous is None:
                buckets[key] = index
            else:
                union_find.union(previous, index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    source_counts: Counter[str] = Counter()
    with open_text(temporary, "wt") as handle:
        for index, record in enumerate(records):
            if union_find.find(index) != index:
                counts["minhash_lsh_duplicate"] += 1
                continue
            handle.write(json.dumps(record.item, ensure_ascii=False) + "\n")
            counts["kept"] += 1
            metadata = record.item.get("metadata")
            source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
            source_counts[str(source)] += 1
    os.replace(temporary, args.output)
    report = {
        "algorithm": {
            "exact_hash": "sha256-stripped-text",
            "minhash_ngram": args.ngram,
            "lsh_bands": args.bands,
            "hashes_per_band": args.hashes_per_band,
            "approximate_threshold": round((1 / args.bands) ** (1 / args.hashes_per_band), 6),
        },
        "language": args.language,
        "counts": dict(sorted(counts.items())),
        "kept_by_source": dict(sorted(source_counts.items())),
        "inputs": [str(path) for path in inputs],
        "output": str(args.output.resolve()),
    }
    report_path = args.report or args.output.with_name(args.output.name + ".stats.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--language", choices=("zh", "en", "code"), required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=2_000_000)
    parser.add_argument("--min-useful-ratio", type=float, default=0.70)
    parser.add_argument("--max-same-char-run", type=int, default=256)
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument("--bands", type=int, default=9)
    parser.add_argument("--hashes-per-band", type=int, default=10)
    args = parser.parse_args()
    if min(args.min_chars, args.max_chars, args.ngram, args.bands, args.hashes_per_band) < 1:
        parser.error("size and MinHash parameters must be positive")
    if args.min_chars > args.max_chars:
        parser.error("--min-chars cannot exceed --max-chars")
    if not 0 <= args.min_useful_ratio <= 1:
        parser.error("--min-useful-ratio must be in [0, 1]")
    print(json.dumps(curate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
