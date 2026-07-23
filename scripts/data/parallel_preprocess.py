"""Split one balanced JSONL and preprocess its parts concurrently."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def split_jsonl(source: Path, parts_dir: Path, workers: int) -> list[Path]:
    """Use GNU split to create deterministic, line-aligned byte-balanced parts."""

    manifest_path = parts_dir / "manifest.json"
    expected = {
        "source": str(source.resolve()),
        "size": source.stat().st_size,
        "mtime_ns": source.stat().st_mtime_ns,
        "workers": workers,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parts = sorted(parts_dir.glob("part_*.jsonl"))
        if manifest == expected and len(parts) == workers:
            return parts

    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)
    width = max(2, len(str(workers - 1)))
    prefix = parts_dir / "part_"
    subprocess.run(
        [
            "split",
            "-n",
            f"l/{workers}",
            "-d",
            "-a",
            str(width),
            "--additional-suffix=.jsonl",
            str(source),
            str(prefix),
        ],
        check=True,
    )
    parts = sorted(parts_dir.glob("part_*.jsonl"))
    if len(parts) != workers:
        raise RuntimeError(f"split produced {len(parts)} parts, expected {workers}")
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    return parts


def preprocess_part(
    ordinal: int,
    source: Path,
    output_root: Path,
    config: str,
    tokenizer_path: str,
    batch_size: int | None,
) -> tuple[int, Path]:
    target = output_root / f"part_{ordinal:02d}"
    marker = target / "_SUCCESS"
    if marker.exists():
        return ordinal, target
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/tools/preprocess.py",
        str(source),
        "-o",
        str(target),
        "-c",
        config,
        "--tokenizer_path",
        tokenizer_path,
    ]
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    log_path = output_root / f"part_{ordinal:02d}.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    marker.write_text(
        json.dumps({"input": str(source), "bytes": source.stat().st_size}),
        encoding="utf-8",
    )
    return ordinal, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    parts_dir = source.with_name(source.name + ".parts")
    parts = split_jsonl(source, parts_dir, args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                preprocess_part,
                ordinal,
                part,
                output_root,
                args.config,
                args.tokenizer_path,
                args.batch_size,
            )
            for ordinal, part in enumerate(parts)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            ordinal, target = future.result()
            print(
                f"preprocessed {completed}/{len(parts)} parts: "
                f"part {ordinal} -> {target}",
                flush=True,
            )

    (output_root / "_SUCCESS").write_text(
        json.dumps({"input": str(source), "workers": args.workers}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
