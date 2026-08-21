# AstrAI 7B-A1B GQA-MoE pretraining recipe

This is the throughput-oriented, single-node B200 pivot recipe:

- 6,998,099,968 total parameters and 1,052,674,048 active parameters;
- 24 layers, hidden size 2048, 16 query heads and four KV heads;
- 32 routed experts, one shared expert and top-2 routing;
- expert parallel size one and eight data-parallel replicas;
- local BF16 grouped GEMMs for routed experts, with no DeepEP or all-to-all;
- tied 100K-token input/output embeddings;
- 2048-token bulk pretraining with document-reset position IDs, while retaining
  a 32K architectural limit for a later context-extension stage.

The initial 300B-token budget uses the complete advertised 120B Chinese-token
portion of `openbmb/Ultra-FineWeb` and samples 180B English tokens. This 40:60
ratio avoids repeating Chinese data merely to manufacture a 1:1 ratio. The
dataset revision is pinned in `data-mixture.json`. Ultra-FineWeb has already
undergone upstream quality filtering, but AstrAI still performs exact/MinHash
deduplication across any additional sources and validation contamination
filtering. Sampling and accounting are always by tokenizer tokens.

The architecture is intentionally incompatible with Astral12B-A3B weights.
Reuse the tokenizer only; initialize model and optimizer state from scratch.
The old run must first stop at an optimizer boundary and finish a complete
checkpoint.

For the first B200 smoke, run DP8 with DDP/no sharding. Probe micro-batches 8,
12, and 16 after compilation and select by sustained tokens/s, not allocated
memory alone. Keep the global optimizer batch at 524,288 tokens by adjusting
gradient accumulation. Only enable ZeRO/FSDP if replicated optimizer state
does not fit: sharding adds communication that this EP-free design is meant to
avoid.
