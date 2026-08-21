# Astral12B-A3B to Minecraft-agent SOTA: Argus handoff

This document is the operational and research handoff for Argus. The desired end state is simple to state and hard to achieve:

> Given the instruction “train a model that reaches state-of-the-art Minecraft agent performance on this server,” Argus should autonomously manage the research loop and produce a reproducible, usable model with strong Minecraft agent ability and retained general chat quality.

“Minecraft agent” means acting in a real harness from observations through valid actions/tools over long horizons. It does not mean scoring well on Minecraft trivia alone.

## 1. Current system

### Model

`Astral12B-A3B` is a 12.15B-total, approximately 3.17B-active decoder model:

- vocabulary 100,000; hidden size 3,072; 32 layers;
- GQA with 24 query heads and 4 KV heads;
- maximum configured position length 32,768;
- MoE with 16 routed experts, one shared expert, and top-2 routed experts per token;
- router scores in FP32, auxiliary load-balancing loss 0.01, router z-loss 0.001;
- fused QKV and fused gate/up projection;
- DeepEP expert dispatch, alignment 128, compute/communication overlap;
- Flash SDPA attention path.

The repository includes a compatibility fix for variable-length attention: current Transformer Engine expects `is_causal=True` for THD input, while older interfaces represented causality through the window. Keep this compatibility when upgrading libraries.

### Pretraining data and processing

The active corpus contains approximately 46.158B unique tokens. The two main public sources are:

- `opencsg/chinese-cosmopedia` for Chinese;
- `emozilla/dolma-v1_7-30B` for English.

The selected mixture is 1:1 Chinese:English. Inputs were structurally filtered, exact-deduplicated, and near-deduplicated with 5-gram MinHash/LSH. Token allocation is quota-based rather than document-count-based. Data is pretokenized into mmap shards.

The active 2,048-token packing format carries document-aware attention and position metadata plus EOS boundaries. This matters: EOS alone does not prevent cross-document attention. Preserve document isolation in any repack or longer-context stage.

Relevant code includes:

- `scripts/data/curate_pretrain.py`
- `scripts/data/curate_pretrain_pilot.py`
- `scripts/data/launch_pretrain_pipeline.sh`
- `scripts/data/balance_pretrain.py`

Before changing the corpus, save a manifest with dataset revision, license, selection rule, language/domain proportions, document/token counts, filter statistics, dedup parameters, tokenizer revision, and hashes.

### Current B200 training baseline

At handoff, the working topology is 8x B200 with EP2 x DP4. The recent batch-8 configuration uses:

- per-rank micro-batch 8;
- gradient accumulation 4;
- sequence length 2,048;
- 524,288 tokens per optimizer update;
- BF16, DeepEP, fused QKV/MLP, Flash SDPA;
- observed steady throughput roughly 234k-237k tokens/s after compilation;
- peak device allocation roughly 145 GB by the trainer metric, within a nominal 180 GB device.

Treat these figures as a baseline, not a promise. Re-read live telemetry. EP1 x DP8 may be faster if every rank can fit all experts and optimizer state, because it removes expert dispatch communication, but it increases replicated model/optimizer memory. Test it independently with a lower micro-batch and ZeRO/FSDP if needed. Do not replace the healthy EP2 run until an end-to-end identical-token test wins on throughput, memory safety, loss parity, and checkpoint resume.

The data mmap is large. Multiprocessing `spawn` attempted to pickle it into `/dev/shm` and failed; `fork` shares the mapping and is currently required. Keep allocator expandable segments enabled. Compile/JIT startup must be excluded from steady-state measurements.

### Checkpoint discipline

Only directories containing `_SUCCESS` are complete. A safe resume test must verify model, optimizer, scheduler, scaler/state metadata, RNG/data position, and step number. Checkpoint creation is asynchronous, so a newly created directory may be incomplete for a period. Never select it merely because its numeric suffix is newest.

The previous retention preference was to keep few local checkpoints due to capacity. Argus should make remote uploads reliable before enforcing aggressive local retention.

## 2. Existing data curation and distillation work

### Quality pipeline

The pretraining quality flow uses structural/content filters, exact hashing, MinHash near-duplicate removal, language-aware sampling, and token-level quota balancing. Extend it with measurable rejection reasons rather than opaque scalar “quality” alone. Protect code, math, tables, multilingual text, and short high-value documents from filters tuned only for prose.

For downstream data, always separate:

1. source prompt/task;
2. teacher reasoning or private scratch work, when legally and operationally usable;
3. visible final answer;
4. tool/action trace and observations;
5. automated validators and rejection reasons;
6. provenance, prompt version, model, timestamp, sampling parameters, and response metadata.

### Resumable teacher generation

`scripts/data/distill_glm.py` uses SQLite as the source of truth for resumable, idempotent generation and records requests/responses. Related preparation, audit, recovery, and quarantine scripts include:

- `scripts/data/prepare_posttrain_distill_seeds.py`
- `scripts/data/prepare_posttrain_v5_seeds.py`
- `scripts/data/prepare_dsv4_distill_seeds.py`
- `scripts/data/prepare_dsv4_public_v2_seeds.py`
- `scripts/data/prepare_dsv4_public_v3_seeds.py`
- `scripts/data/audit_distill_quality.py`
- `scripts/data/recover_retryable_distill.py`
- `scripts/data/report_distill_status.py`
- `scripts/data/locate_distill_anomaly.py`
- `scripts/data/apply_distill_time_cutoff.py`
- `scripts/data/purge_distill_after_cutoff.py`

GLM-5.2 generation used short/medium/long output buckets. Earlier runs exposed two important failure modes: responses that contained reasoning but no usable final answer, and gateway `client_gone`/SSE timeout behavior. Systemd user services fixed local runner disappearance but not upstream idle timeouts. Preserve successful rows, make retries idempotent, and retry only transport errors, 429, and 5xx with bounded exponential backoff. Three consecutive outage probes are sufficient before cooling down.

DeepSeek-V4-Flash generation was fast, but a suspected KV-cache/gateway anomaly appeared after an identified time boundary. The policy was deliberately conservative: everything after that cutoff was invalidated, including superficially good rows. Do not train on those rows. Re-audit surviving outputs and maintain a quarantine manifest.

Public downstream seeds include `open-r1/Mixture-of-Thoughts` and NVIDIA Nemotron post-training material where selected by the scripts. Argus must inspect the actual SQLite/JSONL stores and report current counts; this handoff intentionally does not freeze stale counts.

## 3. Minecraft capability program

### Reference stack

Use the primary repositories, pin revisions, and document deviations:

- Numen: https://github.com/Dwinovo/minecraft-numen
- archived Numen MCP reference: https://github.com/Dwinovo/numen-mcp
- Voyager: https://github.com/MineDojo/Voyager
- MineStudio: https://github.com/CraftJarvis/MineStudio
- OpenHA: https://github.com/CraftJarvis/OpenHA
- MineExplorer: https://github.com/Jometeorie/MineExplorer
- SmartPlay: https://github.com/microsoft/SmartPlay

Numen is the first integration target because it supplies the MCP/tool-agent interface and feedback loop closest to the intended deployment. MineStudio/OpenHA/MineExplorer broaden embodied task coverage. Voyager is useful for skill-library and lifelong-curriculum ideas. SmartPlay contributes comparable agent-evaluation patterns.

Keep separate leaderboards for tool/MCP agents and pixels-to-actions/VLA agents. Mixing them into a single “SOTA” number is misleading because observations, action spaces, compute budgets, and external tools differ.

### Demonstration schema

Each trajectory should contain immutable environment identity and seed; objective; observation; compact grounded plan; chosen tool/action with arguments; raw result; updated verified state; progress/reward; error classification; recovery decision; terminal result. Capture both successes and instructive recoveries, but do not imitate long unproductive loops.

Teacher agents may propose plans and actions, but environment state and task completion must be verified by the harness. Synthetic examples are allowed when validators accept them and provenance remains explicit. Prefer diverse public tasks and procedural held-out environments over self-generated prompt templates alone.

### Curriculum and optimization

Start with short valid-action tasks before long-horizon RL:

1. observation parsing, inventory queries, movement and legal tool calls;
2. navigation, locating entities/blocks, basic gathering and crafting;
3. prerequisite planning and tech-tree milestones;
4. recovery from missing resources, failed actions, death, and route obstruction;
5. long-horizon survival, building, exploration, and combat;
6. collaborative/multi-agent and unseen-world variants.

Use behavior cloning/SFT first, then preference optimization over paired trajectories, then online RL. RL rewards must be environment-derived where possible. Audit for loops, inventory spoofing, invalid actions, save/reload abuse, evaluator leakage, and reward hacking.

### Preserve chat capability

Minecraft traces should not dominate every update. Maintain a replay mixture containing high-quality Chinese/English chat, instruction following, code, math/reasoning, safety calibration, and concise response behavior. Track both aggregate and slice-level regression; a strong MC agent that can no longer hold a normal conversation is not a successful release.

## 4. Infra research loop

Maintain a stable reference command and compare candidate changes with identical token counts, warm-up exclusion, and repeated steady-state windows. Record:

- tokens/s and optimizer updates/hour;
- estimated model FLOPs utilization with the exact FLOP convention stated;
- median/p90/p99 step time and first-step compile time;
- allocated/reserved/driver memory;
- attention, GEMM, routing, dispatch, collective, dataloader, and optimizer timing;
- per-expert tokens, load CV, capacity drops/overflow, and auxiliary losses;
- checkpoint duration, pause time, bytes, and verified restore time.

Priorities are: eliminate redundant collectives/copies; overlap DeepEP dispatch with compute; use proven cuBLAS/cuBLASLt or Transformer Engine paths when faster than hand kernels; fuse only when end-to-end profiling supports it; improve input locality and pinned/asynchronous transfer; make checkpoint save asynchronous and atomic; benchmark EP1/2 and DP/ZeRO trade-offs. FP8 training is out of scope unless the owner explicitly reauthorizes it.

Write structured results to `artifacts/argus/experiments/` and human summaries to `docs/ARGUS_STATUS.md`. Failed experiments are valuable and must be recorded with the reason they lost.

## 5. Evaluation contract

The Minecraft suite needs deterministic reset, fixed budgets, multiple seeds, held-out scenario generation, and comparable action/tool permissions. Minimum metrics:

- task success with confidence intervals;
- normalized reward and milestone completion;
- median environment steps and wall time;
- survival/death and unique-item/tech-tree progression;
- valid/invalid tool-call rate;
- recovery success after injected or natural failures;
- repeated-action/loop rate;
- generalization to held-out seeds, maps, and task wording.

Every promoted model also runs the protected chat suite: bilingual sampled conversations, instruction following, knowledge, math, code, safety/refusal calibration, perplexity where useful, and latency/memory. Store prompts, evaluator versions, outputs, and scoring code. Use blind or rule-based environment verification rather than relying solely on another model’s subjective score.

## 6. Hugging Face and reporting

The model repository is `kkmdadf/Astral12B-A3B`. A protected credential is already provisioned on the server and the owner has authorized Argus to use it. Read it at runtime; do not copy its value into source, configuration committed to Git, reports, shell history, or process arguments.

Upload policy:

- only complete `_SUCCESS` checkpoints;
- every 500 optimizer steps or 6 hours when a newer complete checkpoint exists;
- immediately for a promoted best-MC or major recovery milestone;
- upload asynchronously with retry/backoff so training never waits on the network;
- attach commit, config, tokenizer, step/token count, optimizer/scheduler state when resumable, data manifest hashes, training/evaluation metrics, lineage, and a checksum manifest;
- verify remote file inventory/checksums before marking an upload complete.

Update `docs/ARGUS_STATUS.md` with current training state, newest safe checkpoint, latest upload, data audit counts, MC benchmark status, infra baseline/candidates, blockers, and next experiment. Argus should leave a concise status at least every six hours and after any failure/recovery or promoted improvement.

## 7. Definition of done

The program is complete only when all of the following are true:

1. A versioned Astral checkpoint reproducibly leads the selected Minecraft agent benchmark track under matched tools and budgets, with confidence intervals and public/replayable evaluation artifacts.
2. It generalizes to protected seeds/tasks and survives adversarial checks for leakage and reward hacking.
3. General Chinese/English chat and instruction-following performance stays within the declared non-regression budget.
4. The full pretraining, curation, distillation, SFT/preference/RL, infra, checkpoint, and evaluation lineage is documented well enough to reproduce a technical report.
5. Model artifacts and required metadata are safely present in Hugging Face, and code/documentation changes are committed to the authorized repository without credentials.

