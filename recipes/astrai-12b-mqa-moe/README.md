# AstrAI 12B MQA-MoE pretraining recipe

This recipe defines a 12.155B-total, 3.171B-active decoder model:

- 32 layers, hidden size 3072, 24 query heads, one KV head (MQA)
- 16 routed experts, one shared expert, top-2 routing
- 8-way expert parallelism (two routed experts per H200) with grouped GEMM
- expert intermediate size 2176
- 100K vocabulary with untied input/output embeddings
- 2048-token bulk pretraining windows; the model retains a 32K RoPE limit for a later context-extension stage

The initial data mix should be sampled by token count, not by file count. Start with a 1:1 Chinese/English token ratio and deduplicate both sources against the validation set. `opencsg/chinese-cosmopedia` and `emozilla/dolma-v1_7-30B` are initial candidate sources, not a complete production mixture.

## Prepare a fresh parameter directory

Copy the existing tokenizer files into a new parameter directory, then replace its model config with this recipe's `config.json`. Do not copy the 1B model weights: the changed width, depth, MQA layout, and MoE experts are not checkpoint-compatible.

## Preprocess

```bash
python scripts/tools/preprocess.py data/*.jsonl \
  -o data-bin/pretrain-2048 \
  -c recipes/astrai-12b-mqa-moe/pretrain-2048.json \
  --tokenizer_path params/astrai-12b-mqa-moe
```

The binary storage is memory-mapped by the dataset reader, so the tokenized corpus does not need to fit in RAM.

For the production corpus, run `scripts/data/curate_pretrain.py quality` once
per language, followed by its `minhash` subcommand. The configured 9 bands x
10 hashes use 64-bit hashes and give an approximate near-duplicate threshold
of 0.803. Keep Chinese (`--language zh`) and English (`--language en`) in
separate MinHash runs so each uses the correct word splitter.

Build the final mixture from the deduplicated outputs with the model's actual
tokenizer, rather than balancing files, bytes, or document counts:

```bash
python scripts/data/balance_pretrain.py \
  --zh data/dedup/zh --en data/dedup/en \
  --tokenizer params/astrai-12b-mqa-moe/tokenizer.json \
  --output data/pretrain-balanced.jsonl \
  --tokens-per-language auto --batch-size 512
```

`auto` counts both cleaned corpora and selects exactly the smaller token count
from each language. It persists compact `.tokens.u64` indexes while counting,
so the output pass does not tokenize the complete corpora a second time. The
`bfd_split` preprocessing recipe then preserves long documents by splitting
them into 2048-token chunks instead of truncating them.

## Smoke training

Start with a tiny processed shard before using the complete corpus:

```bash
python scripts/tools/train.py \
  --nprocs=8 \
  --parallel_mode=fsdp \
  --train_type=seq \
  --data_root_path=data-bin/smoke-2048 \
  --param_path=params/astrai-12b-mqa-moe \
  --batch_per_device=1 \
  --grad_accum_steps=32 \
  --gradient_checkpointing \
  --window_size=2048 \
  --warmup_ratio=0.02 \
  --max_lr=2e-4 \
  --weight_decay=0.1 \
  --max_grad_norm=1.0 \
  --schedule_type=wsd \
  --ckpt_interval=1000 \
  --metrics loss language_model_loss router_loss router_aux_loss router_z_loss router_entropy expert_load_min expert_load_max expert_load_cv lr grad_norm
```

This is a correctness recipe, not the final throughput configuration. Increase `batch_per_device` only after measuring peak memory, tokens/s, and router balance on the target node.
