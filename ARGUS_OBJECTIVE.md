# Argus objective: train an MC-agent SOTA Astral model

## Mission

Autonomously turn Astral12B-A3B into a state-of-the-art Minecraft agent model while preserving strong Chinese/English chat capability. The primary target is embodied/tool-using Minecraft agent performance, not Minecraft trivia or text-only QA.

Operate with approximately zero human intervention: observe, form hypotheses, run controlled experiments, measure, keep only demonstrated improvements, document results, and continue. Ask for human input only for irreversible actions, new credentials/permissions, public-release decisions, or genuinely ambiguous benchmark definitions.

## Non-negotiable safety and continuity rules

1. Do not interrupt or reconfigure a healthy live training job merely to try an idea. First prepare and benchmark the replacement independently; switch only from a complete checkpoint after it wins and can resume safely.
2. Never delete or overwrite source datasets, SQLite distillation state, accepted examples, complete checkpoints, logs, or provenance records. Quarantine questionable data instead.
3. A checkpoint is usable/uploadable only when its save is complete and `_SUCCESS` exists. Never upload a partially written checkpoint.
4. Never expose credentials in Git, logs, reports, command-line arguments, or model artifacts. Read Hugging Face credentials from the pre-provisioned protected file/environment only.
5. Keep experiments reproducible: record commit, command, environment, data manifest/checksums, seed, topology, hardware, metrics, and checkpoint ancestry.
6. Change one major variable at a time and compare against an identical-token baseline after warm-up. Do not report JIT/compile steps as steady-state throughput.
7. Preserve general chat ability. Every Minecraft capability gain must pass the chat/non-regression suite before promotion.

## Current model and training baseline

- Model: `Astral12B-A3B`, decoder-only GQA MoE, 12.15B total / 3.17B active.
- Architecture: hidden size 3072, 32 layers, 24 query heads, 4 KV heads, 16 routed experts + 1 shared expert, top-2 routing, context capacity 32,768.
- Router regularization: auxiliary load-balancing coefficient 0.01 and z-loss coefficient 0.001.
- Pretraining corpus: approximately 46.158B unique tokens at 1:1 Chinese:English, derived from `opencsg/chinese-cosmopedia` and `emozilla/dolma-v1_7-30B`.
- Current training sequences: 2,048 tokens with document-aware attention/position IDs and EOS boundaries, pretokenized mmap storage.
- Current B200 baseline: 8 GPUs, EP2 x DP4, per-rank micro-batch 8, gradient accumulation 4, 524,288 tokens/global update, DeepEP dispatch, BF16, fused QKV/MLP and Flash SDPA. Recent steady throughput is approximately 234k-237k tokens/s. Treat live values and logs as authoritative.
- EP1 x DP8 is an experiment, not the production default: it removes expert communication but replicates all experts and optimizer state. Benchmark it only if memory is safe, preferably with lower micro-batch and/or ZeRO/FSDP sharding.

Read `docs/ARGUS_MC_SOTA_HANDOFF.md` before acting. Audit live state rather than assuming paths, counts, process IDs, or service availability remain unchanged.

## Required autonomous workstreams

### 1. Protect and improve pretraining

- Monitor loss, throughput, MFU, GPU memory, load balance, invalid values, dataloader stalls, checkpoint latency, and recovery correctness.
- Preserve the 1:1 Chinese:English allocation unless a measured ablation justifies a change.
- Maintain document boundaries and causal isolation. Do not regress to EOS-only packing without document masking.
- Improve infra continuously using controlled benchmarks: communication overlap, DeepEP, attention/GEMM/library kernels, checkpointing, dataloading, topology, and compiler/JIT behavior. Do not enable FP8 training unless explicitly authorized.

### 2. Curate and distill post-training data

- Use public, attributable task seeds and retain source/license/provenance.
- Audit all existing GLM-5.2 and DeepSeek-V4-Flash outputs before use. Require a complete visible final answer; keep reasoning separately when present.
- Exclude malformed, truncated, refusal-heavy, state-hallucinating, duplicate, contaminated, or post-cutoff suspect DSV4 records. Never silently revive quarantined data.
- Preserve SQLite resume/idempotency. Retry only transport outages, HTTP 429, and 5xx with bounded exponential backoff; quality failures require review or regeneration with a corrected prompt.
- Build a balanced mixture for chat, instruction following, code, math/reasoning, tool use, Minecraft knowledge grounded in state, planning, recovery, and long-horizon trajectories.

### 3. Build the Minecraft agent stack

Use these projects as primary references and integration targets:

- Numen harness/API: https://github.com/Dwinovo/minecraft-numen
- Numen MCP history: https://github.com/Dwinovo/numen-mcp
- Voyager: https://github.com/MineDojo/Voyager
- MineStudio: https://github.com/CraftJarvis/MineStudio
- OpenHA: https://github.com/CraftJarvis/OpenHA
- MineExplorer: https://github.com/Jometeorie/MineExplorer
- SmartPlay: https://github.com/microsoft/SmartPlay

Prioritize the Numen MCP/tool-agent track first. Pin Minecraft/server/mod/harness versions, make reset and seeds deterministic, version the observation and action schema, and log full trajectories. The model must ground decisions in observations rather than inventing inventory, coordinates, entities, recipes, or action outcomes.

Build curriculum stages in this order: primitives and tool validity; navigation and perception; gathering and crafting; resource/tech-tree milestones; recovery from failures; long-horizon building and survival; combat; collaborative/multi-agent tasks; held-out seeds and environment variants.

### 4. Post-training and RL

1. Collect successful demonstrations from capable teacher agents and scripted/rule planners, including failed attempts with corrected recovery traces.
2. Supervised fine-tune on observation-thought/action-result trajectories plus a protected general-chat mixture.
3. Add preference training for grounded, efficient, recoverable plans rather than verbose but ineffective reasoning.
4. Run online RL only after deterministic evaluation and replayable environments exist. Start with dense curriculum rewards and advance to sparse long-horizon success.
5. Reward task success, milestone completion, inventory correctness, survival, valid tool calls, efficiency, exploration, and recovery. Penalize illegal calls, repeated loops, hallucinated state, avoidable death, timeouts, and reward hacking.

Do not optimize hidden chain-of-thought text as a user-visible product. Distill useful plans, verifiable intermediate state, actions, tool outputs, and concise final explanations.

## Evaluation and promotion gates

Maintain separate train/dev/test seeds and protect test scenarios from teacher generation. Report at least:

- Minecraft: success rate, normalized reward, median steps/time, milestone/tech-tree completion, unique-item acquisition, survival, tool-call validity, recovery rate, loop rate, held-out-seed generalization, and confidence intervals.
- Agent baselines: compare under the same harness/version/budget against relevant Numen/Voyager/MineStudio/OpenHA/MineExplorer or reproducible published baselines.
- Chat non-regression: Chinese and English instruction/chat sets, IFEval-style instruction following, general knowledge, math, code, safety/refusal calibration, and human-readable sampled conversations.
- Infra: steady tokens/s, model FLOPs utilization, step time percentiles, peak memory, communication fraction, expert-load CV/dropped tokens, dataloader wait, compile time, and checkpoint save/restore time.

A candidate becomes “best” only when Minecraft held-out performance improves materially and the protected chat suite remains within the recorded regression budget. Never claim SOTA without a pinned, reproducible comparison and uncertainty reporting.

## Reporting and artifact policy

- Update `docs/ARGUS_STATUS.md` at startup, after every meaningful experiment, and at least every 6 hours while active.
- Append machine-readable experiment records under `artifacts/argus/experiments/`; never rewrite history.
- Publish an infra evaluation at least every 500 optimizer steps or 6 hours, whichever occurs first, when a live run supplies enough data.
- Upload the latest complete checkpoint to `kkmdadf/Astral12B-A3B` after each 500-step milestone or 6-hour interval, plus every promoted/best-MC milestone. Deduplicate when possible and never block training on upload failure.
- Each upload must include model/config/tokenizer, optimizer/scheduler state when intended for resume, training metadata, source commit, data manifest/checksums, metrics, and provenance. Retry asynchronously and record success/failure.

## First actions

1. Inspect the live trainer, latest complete checkpoint, logs, GPU topology, filesystem persistence, free capacity, repo status, and Argus backend/model health. Do not stop training.
2. Create/update `docs/ARGUS_STATUS.md` with verified facts and an explicit risk register.
3. Establish the immutable baseline evaluation/telemetry schema.
4. Integrate a minimal deterministic Numen smoke task without consuming the training GPUs unnecessarily.
5. Audit existing post-training/distillation databases and produce accepted/quarantined manifests with counts and reasons.
6. Propose and run the smallest safe experiment that advances Minecraft capability or infra efficiency, then continue the loop.

