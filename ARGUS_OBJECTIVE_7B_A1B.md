# Argus objective: autonomously train AstrAI 7B-A1B for MC-agent SOTA

Treat the 7B-A1B pivot as the only active objective. Import
`docs/OPERATOR_HANDOFF_7B_A1B.md` into the append-only trace as an
`operator_handoff`, preserving its provenance. Then execute and continuously
improve the complete pipeline: Ultra-FineWeb acquisition and deterministic
shard manifests, token-based 120B-ZH/180B-EN selection, contamination control,
EOS/document-aware preprocessing, rotating-cache recovery, DP8 training,
checkpoint verification and Hugging Face publication.

The architecture is `recipes/astrai-7b-a1b-gqa-moe/config.json`: 6.998B total,
1.053B active, GQA, EP1, DP8. Maximize measured B200 throughput without FP8.
Use exact-shape forward/backward benchmarks and preserve correctness; do not
enable EP or DeepEP unless the operator explicitly changes the objective.

Every material action must produce a trace event containing timestamp, actor,
command or diff, input/output artifact paths, hashes when applicable, measured
result, decision and rollback. Never report planning as completed work. Do not
pause for nonessential approval; ask only when permissions, irreversible data
loss, or a materially ambiguous product decision blocks progress.

## GPU throughput ownership

Argus owns sustained GPU efficiency for every training run. After initialization,
JIT warmup and checkpoint I/O are excluded, record per-GPU SM utilization,
memory use, power, step time, global tokens/s, MFU, input wait and rank skew as
rolling statistics. Treat a five-minute rolling median SM utilization below 80%
or a persistent slow-rank spread above 10 percentage points as an incident unless
an identified checkpoint, evaluation or data-stage transition explains it.

For each incident, distinguish input starvation, CPU tokenization, storage I/O,
host synchronization, DDP/NCCL, attention, expert routing/GEMM, optimizer and
checkpoint overhead. Apply changes only after exact-shape measurement. The
primary optimization target is stable global tokens/s and MFU at finite loss,
healthy routing and complete checkpoints; never inflate utilization by adding
useless work. Probe micro-batches 16, 20, 24, 28 and 32 in increasing order,
including first-step optimizer-state allocation and allocator fragmentation,
and retain at least 10% physical-memory safety margin for the selected setting.

Once stable base pretraining is underway, prepare the Minecraft-agent stages:
environment integration using public MC-agent projects as references,
trajectory schema and collection, supervised post-training, preference/reward
data, RL, recovery/collaboration tasks and leakage-controlled evaluation. The
end goal is a strong conversational model with SOTA MC agent capability, not a
benchmark-only policy.

The immediate post-training research assignment is specified in
`docs/ARGUS_MC_POSTTRAIN_DATA_TASK.md`. Execute it concurrently with base-model
training where it uses only CPU, storage and network resources. Its first gate
is an explicit observation/action/episode contract; do not download or generate
large trajectory corpora before that contract and a streaming validator exist.
