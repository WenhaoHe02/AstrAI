from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


pytest.importorskip("datatrove")


SCRIPT = Path(__file__).parents[2] / "scripts" / "data" / "curate_pretrain.py"
SPEC = spec_from_file_location("curate_pretrain", SCRIPT)
assert SPEC and SPEC.loader
curate_pretrain = module_from_spec(SPEC)
SPEC.loader.exec_module(curate_pretrain)


def test_row_group_reader_shards_every_row_exactly_once(tmp_path):
    table = pa.table(
        {
            "id": [f"id-{i}" for i in range(10)],
            "text": [f"document number {i}" for i in range(10)],
        }
    )
    pq.write_table(table, tmp_path / "input.parquet", row_group_size=2)

    documents = []
    for rank in range(3):
        reader = curate_pretrain.RowGroupParquetReader(
            str(tmp_path),
            glob_pattern="*.parquet",
            row_groups_per_chunk=2,
            default_metadata={"source": "fixture", "language": "en"},
        )
        documents.extend(reader.run(rank=rank, world_size=3))

    assert len(documents) == 10
    assert {document.id for document in documents} == {f"id-{i}" for i in range(10)}
    assert {document.text for document in documents} == {
        f"document number {i}" for i in range(10)
    }
    assert all(document.metadata["source"] == "fixture" for document in documents)
    assert all(document.metadata["language"] == "en" for document in documents)


def test_row_group_reader_rejects_nonpositive_chunks(tmp_path):
    try:
        curate_pretrain.RowGroupParquetReader(
            str(tmp_path), row_groups_per_chunk=0
        )
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("zero-sized row-group chunks must fail")
