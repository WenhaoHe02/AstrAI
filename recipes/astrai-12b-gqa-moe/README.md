# AstrAI 12B GQA-MoE pretraining recipe

This recipe defines a 12.230B-total, 3.246B-active decoder model:

- 32 layers, hidden size 3072, 24 query heads, four KV heads (6:1 GQA)
- 16 routed experts, one shared expert, top-2 routing
- 8-way expert parallelism (two routed experts per H200) with grouped GEMM
- DeepEP V2 expand dispatch/combine for the EP communication path
- expert intermediate size 2176
- 100K vocabulary with untied input/output embeddings
- 2048-token bulk pretraining windows; the model retains a 32K RoPE limit for a later context-extension stage

The initial data mix should be sampled by token count, not by file count. Start with a 1:1 Chinese/English token ratio and deduplicate both sources against the validation set. `opencsg/chinese-cosmopedia` and `emozilla/dolma-v1_7-30B` are initial candidate sources, not a complete production mixture.

## Prepare a fresh parameter directory

Copy the existing tokenizer files into a new parameter directory, then replace its model config with this recipe's `config.json`. Do not copy the 1B model weights: the changed width, depth, GQA layout, and MoE experts are not checkpoint-compatible. MQA checkpoints from the earlier one-KV-head recipe are also incompatible with the four-KV-head K/V projection shapes and must not be resumed.

## Preprocess

```bash
python scripts/tools/preprocess.py data/*.jsonl \
  -o data-bin/pretrain-2048 \
  -c recipes/astrai-12b-gqa-moe/pretrain-2048.json \
  --tokenizer_path params/astrai-12b-gqa-moe
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
  --tokenizer params/astrai-12b-gqa-moe/tokenizer.json \
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
  --fsdp_sharding_strategy=shard_grad_op \
  --loss_backend=liger \
  --train_type=seq \
  --data_root_path=data-bin/smoke-2048 \
  --param_path=params/astrai-12b-gqa-moe \
  --batch_per_device=1 \
  --grad_accum_steps=32 \
  --gradient_checkpointing \
  --window_size=2048 \
  --warmup_ratio=0.02 \
  --max_lr=2e-4 \
  --weight_decay=0.1 \
  --max_grad_norm=1.0 \
  --schedule_type=wsd \
  --ckpt_interval=250 \
  --metrics loss language_model_loss router_loss router_aux_loss router_z_loss router_entropy expert_load_min expert_load_max expert_load_cv step_time tokens_per_second peak_memory_gb lr grad_norm
```

This is a correctness recipe, not the final throughput configuration. Increase `batch_per_device` only after measuring peak memory, tokens/s, and router balance on the target node.

For this 8xH200 recipe, `shard_grad_op` is the selected ZeRO-2 path. Routed
expert weights are already rank-local under expert parallelism, while the
roughly 1.9B non-routed parameters fit comfortably unsharded during compute.
Unlike ZeRO-3 (`full_shard`), ZeRO-2 does not reshard those parameters after
forward and therefore avoids the backward all-gather, including the extra
pressure from activation checkpoint recomputation. Set
`ASTRAI_FSDP_SHARDING=full_shard` in the nightly launcher for the lower-memory
rollback path.

The nightly fast path uses `batch_per_device=4`, `grad_accum_steps=8`, and no
activation checkpointing. This preserves the previous 524,288 tokens per
optimizer step (`8 x 4 x 8 x 2048`) while reducing the number of Python,
DeepEP, and loss launches from 32 microbatches to 8. ZeRO-2 plus rank-local
experts only keeps about 6.34GB of BF16 parameters resident per H200, so the
141GB cards have ample room to retain block activations. If the first exact
memory probe disproves that budget, set `ASTRAI_BATCH_PER_DEVICE=1`,
`ASTRAI_GRAD_ACCUM_STEPS=32`, and `ASTRAI_GRADIENT_CHECKPOINTING=1` to restore
the conservative path without changing the effective batch.

Use the logged `step_time`, global `tokens_per_second`, and `peak_memory_gb`
after discarding the first two warmup optimizer steps. The acceptance gate is
finite loss/gradients, healthy routing, no OOM, and a sustained step time below
the existing 10.3-second baseline at the same 524,288 tokens per step.

The selected `liger` loss backend uses Liger's fused linear cross-entropy, so
the 2048x100K full-vocabulary logits and their roughly 0.82GB FP32 cast are not
materialized. Install `liger-kernel==0.8.1` in the training environment before
the smoke run. Set `ASTRAI_LOSS_BACKEND=torch` to retain the original PyTorch
LM-head plus FP32 cross-entropy path without changing checkpoint parameters.
Validate its loss/gradient numerics and exact-shape throughput on H200 with:

```bash
python scripts/tools/benchmark_training_loss.py --backend torch-fp32 --check
python scripts/tools/benchmark_training_loss.py --backend torch-native --check
python scripts/tools/benchmark_training_loss.py --backend liger --check
```

## DeepEP validation

The 12B recipe sets `expert_dispatch_backend` to `deepep`. AstrAI uses the
DeepEP V2 `ElasticBuffer` API in expand mode, so received tokens arrive grouped
by local expert and feed the existing grouped GEMM directly. The original
PyTorch all-to-all implementation remains available by setting the backend to
`torch`.

DeepEP V2 must be installed separately in the training environment. Before a
formal run, execute both the correctness smoke test and the isolated routed
expert benchmark on all eight GPUs:

When PyTorch was compiled against an older NCCL than DeepEP's Gin runtime,
leave `ASTRAI_DEEPEP_REUSE_NCCL_COMM=0` (the nightly default). This makes
DeepEP create a communicator linked against its own NCCL runtime instead of
reusing PyTorch's private communicator pointer. Only opt back into reuse after
the two NCCL versions match and the eight-rank smoke test passes.

```bash
python scripts/tools/check_deepep_env.py

torchrun --standalone --nproc-per-node=8 \
  scripts/tools/smoke_expert_parallel.py --backend deepep
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/smoke_expert_parallel.py --compare-backends
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/smoke_expert_parallel.py --compare-backends --shared-expert-overlap
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/smoke_expert_parallel.py --compare-backends \
  --shared-expert-overlap --no-cpu-sync

torchrun --standalone --nproc-per-node=8 \
  scripts/tools/benchmark_expert_dispatch.py --backend torch
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/benchmark_expert_dispatch.py --backend deepep
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/benchmark_expert_dispatch.py --backend deepep --expert-alignment 128
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/benchmark_expert_dispatch.py --backend deepep \
  --shared-experts 1 --shared-expert-overlap
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/benchmark_expert_dispatch.py --backend deepep \
  --shared-experts 1 --shared-expert-overlap --overlap-with-compute
torchrun --standalone --nproc-per-node=8 \
  scripts/tools/benchmark_expert_dispatch.py --backend deepep \
  --expert-alignment 128 --shared-experts 1 --shared-expert-overlap \
  --overlap-with-compute --no-cpu-sync
```

Do not start formal training unless the DeepEP smoke test has finite forward
outputs and gradients. The checked-in recipe enables
`deepep_overlap_with_compute` because shared-expert/communication overlap is
wired end to end; disable both overlap flags together when isolating failures
or comparing against the conservative path.

## Selected Hopper fast path

The checked-in recipe selects the mature Hopper-oriented path directly:

- PyTorch fused Flash-SDPA for full-sequence causal GQA training;
- DeepEP V2 expert dispatch with 128-token expert alignment;
- DeepEP's compute-overlap mode and shared-expert side-stream overlap;
- Liger fused SwiGLU for both shared and routed experts;
- Liger fused residual-add plus RMSNorm after attention and across block
  boundaries after the MLP.

Flash-SDPA is forced on CUDA so an unsupported shape fails loudly instead of
silently falling back to the slow math kernel. CPU development retains the
automatic reference path. The SwiGLU backend is selected at launch so resumed
checkpoints created before this option also use it; set
`ASTRAI_SWIGLU_BACKEND=torch` for an immediate rollback. To roll back the
residual-norm fusion independently, set `ASTRAI_RESIDUAL_NORM_BACKEND=torch`.
To roll back the full model recipe conservatively, set
`attention_backend=auto`, `expert_dispatch_backend=torch`,
`deepep_expert_alignment=1`, `deepep_overlap_with_compute=false`,
`deepep_cpu_sync=true`, and `moe_shared_expert_overlap=false`.

Routing softmax, top-k selection, and normalization stay in FP32, matching the
stable router path used by mature MoE implementations. Only the selected
weights are cast to BF16 on the non-DeepEP fallback; DeepEP consumes FP32
weights directly. Set `ASTRAI_ROUTER_SCORE_DTYPE=model` to reproduce the older
BF16 top-k behavior.

On the DeepEP path, routing weights are applied to the 2176-wide SwiGLU
activation before the bias-free down projection instead of to its 3072-wide
output. The operations are algebraically equivalent, while the row-scaling
kernel moves about 29% fewer elements. Compare
`--scale-before-down` against `--no-scale-before-down` with the expert dispatch
benchmark before formal training. Set `ASTRAI_ROUTE_SCALE_BEFORE_DOWN=0` to
restore the original post-projection placement without changing checkpoints.

Compare the activation alone at the shared-expert shape (8192 rows) and the
balanced routed-expert receive shape (2048 rows) before the first formal run:

```bash
python scripts/tools/benchmark_training_swiglu.py --backend torch --check
python scripts/tools/benchmark_training_swiglu.py --backend liger --check
python scripts/tools/benchmark_training_swiglu.py --backend torch --rows 2048
python scripts/tools/benchmark_training_swiglu.py --backend liger --rows 2048
```

The residual-norm benchmark exercises both returned tensors and both backward
paths, matching the decoder block rather than timing forward normalization
alone:

```bash
python scripts/tools/benchmark_training_residual_norm.py --backend torch --check
python scripts/tools/benchmark_training_residual_norm.py --backend liger --check
```

An optional `--no-deepep_cpu_sync` training override uses fixed-capacity DeepEP
receive tensors and GPU-resident expert offsets to remove per-layer CPU shape
synchronization. It is not the default for the B=4/no-checkpoint path: the
worst-case receive capacity is 8x the balanced token count, and retaining those
grouped-GEMM activations across 32 layers can erase H200's memory headroom. Gate
it with the `--no-cpu-sync` expert smoke/benchmark above and the logged peak
memory before enabling it for a formal run.

The 2048-token pretraining path is a full-sequence causal GQA workload, not a
decode split-KV workload. The benchmark helper remains available for the first
H200 validation window at the model's exact head shape:

```bash
python scripts/tools/benchmark_training_attention.py --backend torch-auto --check
python scripts/tools/benchmark_training_attention.py --backend torch-flash --check
python scripts/tools/benchmark_training_attention.py --backend flash-attn --check
```

## Router balancing policy

Keep `router_aux_loss_coef=0.01` during the initial training window. The current
greedy top-2 router uses Switch-style auxiliary balancing and does not yet have
an aux-loss-free dynamic expert-bias controller, so setting the coefficient to
zero can allow early expert collapse. Keep `router_z_loss_coef=0.001` to bound
router logits.

Watch an EMA of `expert_load_cv`, `expert_load_min`, and `expert_load_max`, not a
single microbatch. After warmup, a persistent load CV above 0.3 or experts with
near-zero load is a reason to keep or strengthen balancing. Only reduce the aux
coefficient toward 0.001--0.005 after routing remains healthy for a sustained
window.
