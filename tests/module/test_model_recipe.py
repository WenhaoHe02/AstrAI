import json
from pathlib import Path


def test_12b_gqa_moe_recipe_parameter_count():
    recipe_path = (
        Path(__file__).parents[2]
        / "recipes"
        / "astrai-12b-gqa-moe"
        / "config.json"
    )
    config = json.loads(recipe_path.read_text(encoding="utf-8"))

    dim = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    head_dim = dim // config["num_attention_heads"]
    embedding = 2 * config["vocab_size"] * dim
    attention = (
        2 * dim * dim + 2 * dim * config["num_key_value_heads"] * head_dim
    )
    expert = 3 * dim * config["intermediate_size"]
    router = dim * config["n_routed_experts"]
    total_ffn = expert * (
        config["n_routed_experts"] + config["n_shared_experts"]
    ) + router
    active_ffn = expert * (
        config["n_activated_experts"] + config["n_shared_experts"]
    ) + router
    total = embedding + n_layers * (attention + 2 * dim + total_ffn) + dim
    active = embedding + n_layers * (attention + 2 * dim + active_ffn) + dim

    assert config["num_attention_heads"] // config["num_key_value_heads"] == 6
    assert config["num_key_value_heads"] == 4
    assert total == 12_230_200_320
    assert active == 3_246_001_152


def test_7b_a1b_gqa_moe_recipe_parameter_count():
    recipe_path = (
        Path(__file__).parents[2]
        / "recipes"
        / "astrai-7b-a1b-gqa-moe"
        / "config.json"
    )
    config = json.loads(recipe_path.read_text(encoding="utf-8"))

    dim = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    head_dim = dim // config["num_attention_heads"]
    embedding = config["vocab_size"] * dim
    attention = (
        2 * dim * dim + 2 * dim * config["num_key_value_heads"] * head_dim
    )
    expert = 3 * dim * config["intermediate_size"]
    router = dim * config["n_routed_experts"]
    total_ffn = expert * (
        config["n_routed_experts"] + config["n_shared_experts"]
    ) + router
    active_ffn = expert * (
        config["n_activated_experts"] + config["n_shared_experts"]
    ) + router
    total = embedding + n_layers * (attention + 2 * dim + total_ffn) + dim
    active = embedding + n_layers * (attention + 2 * dim + active_ffn) + dim

    assert config["tie_word_embeddings"] is True
    assert config["expert_parallel_size"] == 1
    assert config["expert_dispatch_backend"] == "torch"
    assert config["num_attention_heads"] // config["num_key_value_heads"] == 4
    assert total == 6_998_099_968
    assert active == 1_052_674_048
