import argparse
import gzip
import json

from scripts.data.curate_pretrain_pilot import (
    basic_quality_reason,
    curate,
    tokens_for_minhash,
)


def test_language_aware_tokens():
    assert tokens_for_minhash("Hello, WORLD 42", "en") == ["hello", "world", "42"]
    assert tokens_for_minhash("你好 world", "zh") == ["你", "好", "world"]
    assert tokens_for_minhash("def f(x): return x >= 2", "code") == [
        "def",
        "f",
        "(",
        "x",
        ")",
        ":",
        "return",
        "x",
        ">=",
        "2",
    ]


def test_basic_quality_rejects_short_and_repeated_lines():
    assert basic_quality_reason(
        "short",
        min_chars=20,
        max_chars=1000,
        min_useful_ratio=0.7,
        max_same_char_run=20,
    ) == "too_short"
    repeated = "\n".join(["same useful line"] * 20)
    assert basic_quality_reason(
        repeated,
        min_chars=20,
        max_chars=1000,
        min_useful_ratio=0.7,
        max_same_char_run=20,
    ) == "duplicate_lines"


def test_curate_removes_exact_and_near_duplicates(tmp_path):
    base = "This is a useful educational document about stars and planets. " * 12
    near = base.replace("stars", "bright stars", 1)
    other = "A separate technical explanation of Python functions and testing. " * 12
    source = tmp_path / "input.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for text in (base, base, near, other):
            handle.write(json.dumps({"text": text, "metadata": {"source": "test"}}) + "\n")
    output = tmp_path / "kept.jsonl.gz"
    args = argparse.Namespace(
        input=[str(source)],
        output=output,
        report=None,
        language="en",
        text_key="text",
        min_chars=20,
        max_chars=10_000,
        min_useful_ratio=0.7,
        max_same_char_run=256,
        ngram=5,
        bands=9,
        hashes_per_band=10,
    )
    report = curate(args)
    assert report["counts"]["exact_duplicate"] == 1
    assert report["counts"]["minhash_lsh_duplicate"] == 1
    assert report["counts"]["kept"] == 2
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        assert len(handle.readlines()) == 2
