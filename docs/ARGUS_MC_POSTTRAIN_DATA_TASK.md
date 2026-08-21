# Argus task: Minecraft interaction data and post-training interface

## Why this task exists

The unresolved post-training problem is not merely finding more instruction
data. Minecraft is an interactive, partially observable, long-horizon
environment. We need to decide exactly what the model receives, what it emits,
how an executor applies the output, and how the resulting transition is stored
for SFT, preference learning and RL. A dataset is unusable until this contract
is explicit and validated end to end.

## Objective

Research existing public Minecraft-agent systems and datasets, then propose a
concrete, implementation-ready data and environment design for a text-first
AstrAI 7B-A1B agent. The final model must retain normal conversational ability
while pursuing state-of-the-art Minecraft agent performance. Do not assume a
vision encoder in the base model; compare text/state, tool-use and optional
external-vision-adapter designs explicitly.

## Questions that must be answered

1. **Observation contract:** Which environment state is observable at each
   step: chat, task, inventory, equipment, health/hunger, position/orientation,
   biome/time/weather, nearby blocks/entities, recent events, recipes,
   achievements, map/memory and executor errors? Separate information genuinely
   available to the player from privileged simulator state.
2. **Action contract:** Compare raw keyboard/mouse actions, MineRL-style
   discrete controls, structured JSON tool calls and hierarchical skills.
   Specify the exact action grammar, validation, retries, timeouts, cancellation
   and error-return format.
3. **Temporal representation:** Define episode, turn and transition boundaries;
   timestamps/ticks; observation deltas; action duration; rewards; termination;
   failure and recovery. Explain how long trajectories are summarized or
   retrieved within 32K context, and what later 128K extension would improve.
4. **Training targets:** State precisely which tokens receive loss for behavior
   cloning/SFT. Keep observable plans or concise rationales distinct from
   hidden chain-of-thought. Define pair/group records for DPO or reward-model
   training and transition/batch records for offline and online RL.
5. **Data sources:** Inventory relevant public environments, agents,
   demonstrations and task suites. For each, record license, version, schema,
   modality, approximate scale, task coverage, quality risks and conversion
   cost. Distinguish real interaction traces from synthetic instructions and
   model-generated rollouts.
6. **Quality and safety:** Specify schema validation, impossible-action checks,
   reward hacking checks, duplicate/near-duplicate handling, abnormal episode
   rejection, train/eval contamination control and provenance. Never expose
   sealed evaluation trajectories to training.
7. **Capability balance:** Propose replay/mixing that prevents Minecraft
   post-training from destroying general chat, reasoning, code and tool-use
   ability. Include measurable regression gates.
8. **Evaluation:** Define task families, success metrics, sample efficiency,
   recovery, long-horizon memory, collaboration and chat quality. Separate
   development, validation and sealed test sets and report confidence intervals.

## Required deliverables

Produce artifacts, not only a prose survey:

1. `docs/mc/landscape.md`: evidence-backed comparison of candidate projects,
   datasets and environment APIs with primary-source links.
2. `docs/mc/data_contract.md`: exact observation/action/episode contracts and
   loss masks, including at least one fully worked trajectory.
3. `schemas/mc_episode.schema.json`: versioned JSON Schema for canonical
   episodes, with provenance and split fields.
4. `scripts/mc/validate_episode.py`: streaming validator with actionable error
   categories and aggregate statistics.
5. `scripts/mc/convert_<source>.py`: at least one resumable public-source
   converter into the canonical schema; preserve source IDs and licenses.
6. `recipes/mc/posttrain-mixture.json`: proposed token/example mixture for
   general chat, MC SFT, recovery, preference and RL stages, with justification.
7. `docs/mc/experiment_plan.md`: staged pilot-to-scale plan, compute/storage
   estimates, ablations, acceptance gates and rollback criteria.
8. A small, legally redistributable pilot shard plus manifest, checksums,
   validation report and a round-trip environment smoke log.

## Decision process

First produce a short decision memo comparing no more than three viable
interface designs. Select one default and one fallback using implementation
cost, trainability, inference latency, information leakage, compatibility with
the text-only 7B-A1B model and expected MC performance. Then implement the
schema, validator and one converter and demonstrate an end-to-end pilot.

Record all research queries, source URLs, assumptions, commands, diffs,
artifact hashes, validation results, failures and decisions in the Argus trace.
Planning text is not evidence of completion. Ask the operator only when a real
permission, licensing, irreversible-data or product-choice blocker remains.
