"""Verify a fixed pretraining validation suite and report effective tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrai.tokenize import AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("training_exclusion"):
        raise RuntimeError("manifest must mark the suite as excluded from training")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    seen_text = set()
    result = []
    for source in manifest["sources"]:
        path = args.data_dir / source["file"]
        actual_sha = sha256(path)
        if actual_sha != source["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")
        records = 0
        raw_tokens = 0
        effective_tokens = 0
        duplicate_records = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"invalid text record in {path}")
                text_hash = hashlib.sha256(text.encode("utf-8")).digest()
                if text_hash in seen_text:
                    duplicate_records += 1
                seen_text.add(text_hash)
                token_count = len(tokenizer.encode(text))
                records += 1
                raw_tokens += token_count
                effective_tokens += min(token_count, args.max_length)
        if records != source["records"]:
            raise RuntimeError(
                f"record count mismatch for {path}: {records} != {source['records']}"
            )
        result.append(
            {
                "name": source["name"],
                "records": records,
                "raw_tokens": raw_tokens,
                "effective_tokens": effective_tokens,
                "duplicate_records_across_suite": duplicate_records,
                "sha256": actual_sha,
            }
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
