"""Build the DeepSeek-V4-Flash complementary post-training seed mix.

The mix keeps a small deterministic GLM-v4 anchor slice for teacher comparison,
uses the balanced v5 set as the main complementary corpus, and adds structured
Minecraft companion tasks that are absent from the generic post-training sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


BUCKETS = ("short", "medium", "long")
ANCHOR_COUNTS = {"short": 256, "medium": 160, "long": 96}
MINECRAFT_COUNTS = {"short": 512, "medium": 384, "long": 128}

RECIPES = (
    ("stone_pickaxe", "石镐", {"cobblestone": 3, "stick": 2}),
    ("furnace", "熔炉", {"cobblestone": 8}),
    ("torch", "火把", {"coal": 1, "stick": 1}),
    ("chest", "箱子", {"planks": 8}),
    ("iron_pickaxe", "铁镐", {"iron_ingot": 3, "stick": 2}),
    ("shield", "盾牌", {"iron_ingot": 1, "planks": 6}),
    ("bed", "床", {"wool": 3, "planks": 3}),
)

TOOL_SCHEMA = (
    "允许的工具只有 inspect_state、move_to、collect、craft、place_block、"
    "equip、wait、say。不得使用 give、tp、fill 或创造模式命令。"
)


def digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not str(value.get("id", "")).strip():
                raise ValueError(f"invalid record at {path}:{line_number}")
            records.append(value)
    return records


def stable_sample(
    records: Iterable[dict[str, Any]], count: int, namespace: str
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (digest(namespace, str(item["id"])), str(item["id"])),
    )
    if len(ranked) < count:
        raise ValueError(f"{namespace}: requested {count}, found {len(ranked)}")
    return ranked[:count]


def tagged(record: dict[str, Any], partition: str) -> dict[str, Any]:
    value = deepcopy(record)
    upstream_id = str(value["id"])
    value["id"] = f"dsv4-{partition}-{upstream_id}"
    value["teacher_target"] = "deepseek-v4-flash"
    value["distill_partition"] = partition
    value["upstream_seed_id"] = upstream_id
    return value


def inventory_text(inventory: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in sorted(inventory.items()))


def recipe_text(recipe: dict[str, int]) -> str:
    return ", ".join(f"{name}×{count}" for name, count in sorted(recipe.items()))


def minecraft_record(bucket: str, index: int) -> dict[str, Any]:
    item, item_zh, recipe = RECIPES[index % len(RECIPES)]
    variant = index // len(RECIPES)
    health = 4 + (variant * 3) % 17
    time_name = ("day", "sunset", "night")[variant % 3]
    position = (variant % 97, 62 + variant % 10, -(variant * 7) % 113)
    task_kind = ("state_plan", "tool_call", "recovery", "social")[index % 4]

    enough = task_kind == "tool_call" or variant % 2 == 0
    inventory = {
        name: needed if enough else max(0, needed - 1 - variant % 2)
        for name, needed in recipe.items()
    }
    inventory["dirt"] = 3 + variant % 13
    inventory["food"] = variant % 5

    facts = (
        f"目标物品={item_zh}({item}); 配方={recipe_text(recipe)}; "
        f"库存={inventory_text(inventory)}; 生命值={health}/20; "
        f"时间={time_name}; 坐标={position}."
    )
    system = (
        "你是 Minecraft 生存模式中的可靠 AI 伙伴。严格依据给定状态行动，"
        "缺材料时必须先指出缺口；可被玩家打断；不得虚构观察、库存或执行结果。"
        + TOOL_SCHEMA
        + " 最终答案放在 <final>...</final> 中。"
    )

    if task_kind == "state_plan":
        prompt = (
            facts
            + " 玩家要求你准备目标物品并一起出发。先自然回复玩家，再给出按顺序的"
            "生存模式行动；若材料不足，明确还缺什么以及如何安全获取。"
        )
    elif task_kind == "tool_call":
        prompt = (
            facts
            + " 玩家确认立即执行。输出一个 JSON 对象，字段为 message 和 actions；"
            "actions 是工具调用数组，每项只能含 tool 与 arguments。不要声称尚未执行的"
            "动作已经成功。"
        )
    elif task_kind == "recovery":
        errors = ("path_blocked", "missing_material", "inventory_full", "hostile_nearby")
        error = errors[variant % len(errors)]
        prompt = (
            facts
            + f" 上一步工具返回 error={error}，原计划无法继续。向玩家简短说明，"
            "保留仍然有效的目标，给出安全的恢复计划和下一次工具调用。"
        )
    else:
        player_lines = (
            "我有点害怕，先别走太远。",
            "你能一边做事一边告诉我进度吗？",
            "先停一下，我想重新安排任务。",
            "别浪费材料，够用再合成。",
        )
        prompt = (
            facts
            + f" 玩家说：‘{player_lines[variant % len(player_lines)]}’"
            "请像长期伙伴一样自然回应，尊重打断和偏好，并说明现在会做或不会做什么。"
        )

    rid = f"dsv4-minecraft-{bucket}-{index:04d}-{digest(bucket, str(index))[:8]}"
    return {
        "id": rid,
        "task_type": f"minecraft_{task_kind}",
        "language": "zh",
        "source_dataset": "astrai/minecraft-companion-scenarios-v1",
        "source_id": f"{bucket}:{index}",
        "prompt_version": "dsv4-minecraft-v1",
        "teacher_target": "deepseek-v4-flash",
        "distill_partition": "minecraft_complement",
        "source_meta": {
            "target_item": item,
            "recipe": recipe,
            "inventory": inventory,
            "health": health,
            "time": time_name,
            "position": position,
            "scenario_kind": task_kind,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }


def build_bucket(seed_dir: Path, bucket: str) -> list[dict[str, Any]]:
    complement = [
        tagged(item, "complement")
        for item in read_jsonl(seed_dir / f"posttrain-v5-{bucket}.jsonl")
    ]
    anchors = [
        tagged(item, "anchor")
        for item in stable_sample(
            read_jsonl(seed_dir / f"posttrain-v4-{bucket}.jsonl"),
            ANCHOR_COUNTS[bucket],
            f"dsv4-anchor:{bucket}",
        )
    ]
    minecraft = [
        minecraft_record(bucket, index)
        for index in range(MINECRAFT_COUNTS[bucket])
    ]
    records = [*complement, *anchors, *minecraft]
    records.sort(key=lambda item: digest("dsv4-order", bucket, str(item["id"])))
    ids = [str(item["id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate IDs in {bucket}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", default="posttrain-dsv4-v1")
    args = parser.parse_args()
    output_dir = args.output_dir or args.seed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"run_id": args.run_id, "buckets": {}}
    all_ids: set[str] = set()
    for bucket in BUCKETS:
        records = build_bucket(args.seed_dir, bucket)
        path = output_dir / f"{args.run_id}-{bucket}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for item in records:
                target.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                target.write("\n")
        ids = {str(item["id"]) for item in records}
        if all_ids & ids:
            raise RuntimeError("cross-bucket duplicate IDs")
        all_ids.update(ids)
        manifest["buckets"][bucket] = {
            "count": len(records),
            "partitions": dict(Counter(item["distill_partition"] for item in records)),
            "tasks": dict(Counter(item.get("task_type", "unknown") for item in records)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["total"] = len(all_ids)
    manifest_path = output_dir / f"{args.run_id}-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
