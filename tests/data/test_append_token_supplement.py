from pathlib import Path

import pytest

from scripts.data.append_token_supplement import (
    build_supplement,
    locate_document_offset,
)


def test_build_supplement_has_exact_target_and_eos_boundaries():
    records = [{"text": "abcd"}, {"text": "ef"}, {"text": "ghijkl"}]
    tokens, stats = build_supplement(
        records,
        encode=lambda text: [ord(char) for char in text],
        eos_token_id=999,
        target_tokens=9,
    )

    assert list(tokens) == [97, 98, 99, 100, 999, 101, 102, 999, 999]
    assert stats == {
        "tokens": 9,
        "documents": 3,
        "eos": 3,
        "truncated_final_document": True,
    }


def test_locate_document_offset_skips_whole_files():
    paths = [Path("a"), Path("b"), Path("c")]
    assert locate_document_offset(paths, [3, 5, 7], 6) == (paths[1:], 3)
    assert locate_document_offset(paths, [3, 5, 7], 8) == (paths[2:], 0)
    with pytest.raises(RuntimeError, match="source has 15"):
        locate_document_offset(paths, [3, 5, 7], 16)
