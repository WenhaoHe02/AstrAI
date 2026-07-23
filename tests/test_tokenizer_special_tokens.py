from tokenizers import Tokenizer, models

from astrai.tokenize import AutoTokenizer


def test_short_special_token_names_support_conventional_accessors():
    backend = Tokenizer(
        models.WordLevel(vocab={"<eos>": 0, "<unk>": 1}, unk_token="<unk>")
    )
    tokenizer = AutoTokenizer()
    tokenizer._tokenizer = backend
    tokenizer._special_token_map = {"eos": "<eos>", "unk": "<unk>"}

    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.eos_token_id == 0
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.unk_token_id == 1
