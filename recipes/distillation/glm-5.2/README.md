# GLM-5.2 sequence distillation

`scripts/data/distill_glm.py` runs OpenAI-compatible teacher generation with a
SQLite resume ledger. The database is the source of truth; JSONL files are
atomically materialized from successful and permanently failed tasks.

Input records may contain either `messages` or a `prompt` string. Give every
record a stable `id` when possible:

```json
{"id":"math-000001","prompt":"Solve 2+2 and explain the steps."}
```

Keep the API key in the environment, never in a config file or shell history:

```bash
export INFERKNOCK_API_KEY='...'
python scripts/data/distill_glm.py \
  --input /mnt/nvme6/astrai/distill/problems.jsonl \
  --state /mnt/nvme6/astrai/distill/glm-5.2.sqlite3 \
  --output /mnt/nvme6/astrai/distill/glm-5.2.jsonl \
  --model glm-5.2 \
  --concurrency 256 \
  --candidates 8 \
  --max-tokens 4096
```

Run the exact same command after a crash or API outage. Completed task IDs are
skipped, interrupted requests return to `pending`, retry counts persist, and
retryable HTTP/network failures use exponential backoff. A shared circuit
breaker pauses new waves after repeated failures. By default retries are
unlimited; set `--max-attempts N` to create a permanent failure queue.

Useful maintenance commands:

```bash
# Import tasks without calling the API.
python scripts/data/distill_glm.py ... --seed-only

# Rebuild JSONL files from SQLite without calling the API.
python scripts/data/distill_glm.py ... --export-only

# Reset permanent failures for another API window.
python scripts/data/distill_glm.py ... --retry-failed
```

Each successful line preserves the source record, exact messages, teacher,
candidate index, response/reasoning fields, usage, request ID, attempt count,
latency, and timestamp. This makes later filtering and technical reporting
auditable. Correctness verification and rejection sampling should run after
generation; unverified teacher traces must not be used directly for SFT.
