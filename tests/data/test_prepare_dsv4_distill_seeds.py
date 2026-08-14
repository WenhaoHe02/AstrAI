from scripts.data.prepare_dsv4_distill_seeds import (
    minecraft_record,
    stable_sample,
)


def test_stable_sample_is_order_independent():
    records = [{"id": f"r{i}"} for i in range(20)]
    expected = stable_sample(records, 7, "test")
    assert stable_sample(reversed(records), 7, "test") == expected
    assert len({item["id"] for item in expected}) == 7


def test_minecraft_records_are_unique_and_grounded():
    records = [minecraft_record("short", index) for index in range(64)]
    assert len({item["id"] for item in records}) == len(records)
    for item in records:
        assert item["task_type"].startswith("minecraft_")
        assert item["distill_partition"] == "minecraft_complement"
        assert item["teacher_target"] == "deepseek-v4-flash"
        system, user = item["messages"]
        assert system["role"] == "system"
        assert "不得使用 give、tp、fill" in system["content"]
        assert user["role"] == "user"
        assert "库存=" in user["content"]
