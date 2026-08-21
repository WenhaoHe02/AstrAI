# Operator handoff: AstrAI 7B-A1B pivot

Trace event type: `operator_handoff`

Provenance: this handoff summarizes work performed by the operator and Codex.
Argus must import it into its trace without rewriting the provenance or
claiming the implementation as an autonomous Argus action.

## Completed before import

- Selected and exactly counted a 6.998B-total / 1.053B-active architecture:
  24x2048, GQA 16Q/4KV, 32 routed + one shared expert, top-2, FFN 1344,
  tied 100K embeddings.
- Removed expert parallelism from the new recipe. The target topology is EP1
  x DP8 and the routed experts use the local grouped-GEMM path.
- Pinned `openbmb/Ultra-FineWeb` at revision
  `02c85641e3d19a854be2e09139c25adaa9518063` and specified a 300B-token
  initial budget of 120B Chinese + 180B English tokens.
- Preserved EOS document boundaries and document-reset position IDs for the
  2048-token bulk-pretraining stage.
- Added parameter-count regression coverage and an independent 7B launcher.

## Required Argus work

1. Maintain an append-only trace of decisions, commands, diffs, benchmarks,
   failures, retries, data manifests, checksums, training metrics and uploads.
2. Audit the local EP1 grouped-GEMM route on B200 and propose changes only when
   an exact-shape forward/backward benchmark proves a sustained improvement.
3. Build a resumable Ultra-FineWeb shard pipeline with pinned source revision,
   rotating local cache, token quotas, EOS/document metadata, validation
   decontamination, checksums and an explicit consumed-shard cursor.
4. Safely stop the old 12B run only after a complete checkpoint exists. Never
   reinterpret a 12B checkpoint as 7B; the 7B model starts from random weights
   and reuses only the tokenizer.
5. Run DP8 micro-batch 8/12/16 smoke probes, discard compile warmup, and select
   the highest sustained global tokens/s configuration that remains stable.
6. Monitor loss, grad norm, router entropy/load CV, tokens/s, peak memory,
   checkpoint completeness and dataset cursor. Upload complete checkpoints and
   trace artifacts to the configured Hugging Face repository at the agreed
   interval.
7. After base pretraining is stable, continue the MC-agent plan: Minecraft
   trajectory collection, SFT, preference/reward data, RL, recovery evaluation
   and reproducible public benchmark reporting. Do not branch into unrelated
   paper exercises.
