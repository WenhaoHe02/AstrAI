"""Large-scale pretraining data curation with DataTrove.

The pipeline is deliberately language-aware and conservative: source datasets
have already received substantial upstream filtering, so this stage removes
clearly broken/repetitive documents and near duplicates without applying an
English-only quality classifier to Chinese text.
"""

import argparse
import unicodedata
from collections.abc import Iterator

from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.filters import GopherRepetitionFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig


class RowGroupParquetReader(ParquetReader):
    """Shard large Parquet files by contiguous row-group chunks.

    DataTrove's standard disk reader assigns whole files to ranks. Datasets
    such as Dolma can contain thousands of row groups in only a few files, so
    file-level sharding leaves most CPU cores idle. This reader constructs
    deterministic virtual chunks and assigns each chunk to exactly one rank.
    """

    name = "📒 Parquet row-group chunks"

    def __init__(self, *args, row_groups_per_chunk: int = 16, **kwargs):
        if row_groups_per_chunk < 1:
            raise ValueError("row_groups_per_chunk must be positive")
        super().__init__(*args, **kwargs)
        self.row_groups_per_chunk = row_groups_per_chunk

    def _work_units(self) -> list[tuple[str, int, int]]:
        import pyarrow.parquet as pq

        files = self.data_folder.list_files(
            recursive=self.recursive, glob_pattern=self.glob_pattern
        )
        units: list[tuple[str, int, int]] = []
        for filepath in files:
            with self.data_folder.open(filepath, "rb") as handle:
                groups = pq.ParquetFile(handle).metadata.num_row_groups
            for start in range(0, groups, self.row_groups_per_chunk):
                units.append(
                    (filepath, start, min(start + self.row_groups_per_chunk, groups))
                )
        return units

    def run(
        self, data=None, rank: int = 0, world_size: int = 1
    ) -> Iterator[Document]:
        import pyarrow.parquet as pq

        if data:
            yield from data
        units = self._work_units()[rank::world_size]
        emitted = 0
        skipped = 0
        columns = [self.text_key, self.id_key] if not self.read_metadata else None
        for filepath, start, stop in units:
            self.stat_update("input_row_group_chunks")
            self.stat_update("input_row_groups", value=stop - start)
            documents = 0
            with self.data_folder.open(filepath, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                row_in_chunk = 0
                for batch in parquet.iter_batches(
                    batch_size=self.batch_size,
                    row_groups=range(start, stop),
                    columns=columns,
                ):
                    with self.track_time("batch"):
                        rows = batch.to_pylist()
                    for row in rows:
                        document = self.get_document_from_dict(
                            row,
                            filepath,
                            (start << 32) | row_in_chunk,
                        )
                        row_in_chunk += 1
                        if document is None:
                            continue
                        if skipped < self.skip:
                            skipped += 1
                            continue
                        if self.limit != -1 and emitted >= self.limit:
                            break
                        self.update_doc_stats(document)
                        emitted += 1
                        documents += 1
                        yield document
                    if self.limit != -1 and emitted >= self.limit:
                        break
            self.stat_update("documents", value=documents, unit="row_group_chunk")
            if self.limit != -1 and emitted >= self.limit:
                break


class BasicPretrainQualityFilter(BaseFilter):
    """Reject structurally broken text without imposing a language style."""

    name = "AstrAI basic pretrain quality"

    def __init__(
        self,
        min_chars: int = 200,
        max_chars: int = 2_000_000,
        min_useful_ratio: float = 0.70,
        max_same_char_run: int = 256,
        exclusion_writer=None,
    ):
        super().__init__(exclusion_writer=exclusion_writer)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_useful_ratio = min_useful_ratio
        self.max_same_char_run = max_same_char_run

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        text = doc.text.strip()
        if len(text) < self.min_chars:
            return False, "too_short"
        if len(text) > self.max_chars:
            return False, "too_long"

        useful = 0
        last = ""
        run = 0
        for char in text:
            category = unicodedata.category(char)
            if char.isspace() or category[0] in {"L", "N", "P", "S"}:
                useful += 1
            if char == last:
                run += 1
                if run > self.max_same_char_run:
                    return False, "same_char_run"
            else:
                last = char
                run = 1

        if useful / len(text) < self.min_useful_ratio:
            return False, "low_useful_char_ratio"
        return True


def quality(args: argparse.Namespace) -> None:
    removed = JsonlWriter(f"{args.output}/removed") if args.keep_removed else None
    reader_type = RowGroupParquetReader if args.row_groups_per_chunk else ParquetReader
    reader_options = {}
    if args.row_groups_per_chunk:
        reader_options["row_groups_per_chunk"] = args.row_groups_per_chunk
    reader = reader_type(
        args.input,
        glob_pattern=args.glob,
        text_key=args.text_key,
        default_metadata={"source": args.source, "language": args.language},
        **reader_options,
    )
    executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            BasicPretrainQualityFilter(exclusion_writer=removed),
            GopherRepetitionFilter(language=args.language, exclusion_writer=removed),
            JsonlWriter(
                f"{args.output}/kept",
                output_filename="${rank}.jsonl.gz",
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=f"{args.logs}/quality-{args.source}",
    )
    executor.run()


def minhash(args: argparse.Namespace) -> None:
    # 9 bands x 10 hashes gives an approximate similarity threshold of
    # (1 / 9) ** (1 / 10) = 0.803. 64-bit hashes keep collision risk low.
    config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=9,
        hashes_per_bucket=10,
        n_grams=5,
    )
    reader = JsonlReader(args.input, glob_pattern="*.jsonl.gz")
    base = args.work

    signatures = LocalPipelineExecutor(
        pipeline=[
            reader,
            MinhashDedupSignature(
                output_folder=f"{base}/signatures",
                config=config,
                language=args.language,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=f"{args.logs}/minhash-{args.source}/signatures",
    )
    buckets = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{base}/signatures",
                output_folder=f"{base}/buckets",
                config=config,
            )
        ],
        tasks=config.num_buckets,
        workers=min(args.workers, config.num_buckets),
        logging_dir=f"{args.logs}/minhash-{args.source}/buckets",
        depends=signatures,
    )
    clusters = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{base}/buckets",
                output_folder=f"{base}/remove_ids",
                config=config,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=f"{args.logs}/minhash-{args.source}/clusters",
        depends=buckets,
    )
    filtered = LocalPipelineExecutor(
        pipeline=[
            reader,
            MinhashDedupFilter(input_folder=f"{base}/remove_ids"),
            JsonlWriter(args.output, output_filename="${rank}.jsonl.gz"),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=f"{args.logs}/minhash-{args.source}/filter",
        depends=clusters,
    )
    filtered.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quality_parser = subparsers.add_parser("quality")
    quality_parser.add_argument("--input", required=True)
    quality_parser.add_argument("--output", required=True)
    quality_parser.add_argument("--logs", required=True)
    quality_parser.add_argument("--source", required=True)
    quality_parser.add_argument("--language", choices=("en", "zh"), required=True)
    quality_parser.add_argument("--text-key", default="text")
    quality_parser.add_argument("--glob", default="**/*.parquet")
    quality_parser.add_argument("--tasks", type=int, default=64)
    quality_parser.add_argument("--workers", type=int, default=64)
    quality_parser.add_argument(
        "--row-groups-per-chunk",
        type=int,
        default=0,
        help="Shard parquet inputs by N row groups instead of by whole file",
    )
    quality_parser.add_argument("--keep-removed", action="store_true")
    quality_parser.set_defaults(func=quality)

    minhash_parser = subparsers.add_parser("minhash")
    minhash_parser.add_argument("--input", required=True)
    minhash_parser.add_argument("--output", required=True)
    minhash_parser.add_argument("--work", required=True)
    minhash_parser.add_argument("--logs", required=True)
    minhash_parser.add_argument("--source", required=True)
    minhash_parser.add_argument("--language", choices=("en", "zh"), required=True)
    minhash_parser.add_argument("--tasks", type=int, default=64)
    minhash_parser.add_argument("--workers", type=int, default=64)
    minhash_parser.set_defaults(func=minhash)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tasks < 1 or args.workers < 1:
        raise SystemExit("--tasks and --workers must be positive")
    if getattr(args, "row_groups_per_chunk", 0) < 0:
        raise SystemExit("--row-groups-per-chunk cannot be negative")
    args.func(args)


if __name__ == "__main__":
    main()
