import gzip
import json
import struct
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "data" / "balance_pretrain.py"
SPEC = spec_from_file_location("balance_pretrain", SCRIPT)
assert SPEC and SPEC.loader
balance_pretrain = module_from_spec(SPEC)
SPEC.loader.exec_module(balance_pretrain)


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def encode_batch(self, texts, add_special_tokens=False):
        assert not add_special_tokens
        return [_Encoding(list(range(len(text.split())))) for text in texts]


def test_token_index_round_trip_skips_empty_records(tmp_path):
    source = tmp_path / "records.jsonl.gz"
    records = [
        {"text": "one two", "id": 1},
        {"text": "   ", "id": 2},
        {"text": "three four five", "id": 3},
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    index = tmp_path / "counts.u64"
    total = balance_pretrain.write_token_index(
        [source], _Tokenizer(), batch_size=2, output=index
    )

    assert total == 5
    assert index.stat().st_size == 2 * balance_pretrain.TOKEN_COUNT.size
    assert list(balance_pretrain.iter_indexed([source], index)) == [
        (records[0], 2, None),
        (records[2], 3, None),
    ]


def test_token_index_detects_truncation(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"text":"one two"}\n', encoding="utf-8")
    index = tmp_path / "counts.u64"
    index.write_bytes(b"")

    try:
        list(balance_pretrain.iter_indexed([source], index))
    except RuntimeError as exc:
        assert "ended before source records" in str(exc)
    else:
        raise AssertionError("a truncated token index must fail")


def test_token_index_stats_and_document_overhead(tmp_path):
    source = tmp_path / "records.jsonl"
    records = [{"text": "one two"}, {"text": "three four five"}]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    index = tmp_path / "counts.u64"
    index.write_bytes(struct.pack("<QQ", 2, 3))

    assert balance_pretrain.token_index_stats(index) == (2, 5)
    assert list(balance_pretrain.iter_indexed([source], index, 1)) == [
        (records[0], 3, None),
        (records[1], 4, None),
    ]
