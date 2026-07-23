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


def iter_jsonl(paths: list[Path]) -> Iterator[dict]:
    for path in sorted(paths):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zh", nargs="+", required=True)
    parser.add_argument("--en", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens-per-language", type=int, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    streams = {"zh": iter_jsonl(expand(args.zh)), "en": iter_jsonl(expand(args.en))}
    counts = {"zh": 0, "en": 0}
    docs = {"zh": 0, "en": 0}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        handle = stack.enter_context(open_text(output, "wt"))
        while min(counts.values()) < args.tokens_per_language:
            language = min(counts, key=counts.get)
            try:
                item = next(streams[language])
            except StopIteration as exc:
                raise RuntimeError(
                    f"{language} exhausted at {counts[language]:,} tokens"
                ) from exc
            text = item.get("text", "")
            if not text:
                continue
            length = len(tokenizer.encode(text, add_special_tokens=False).ids)
            if not length:
                continue
            remaining = args.tokens_per_language - counts[language]
            if length > remaining:
                token_ids = tokenizer.encode(
                    text, add_special_tokens=False
                ).ids[:remaining]
                item["text"] = tokenizer.decode(token_ids)
                length = len(token_ids)
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
