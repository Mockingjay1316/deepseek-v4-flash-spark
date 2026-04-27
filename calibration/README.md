# Calibration-based REAP pruning for V4-Flash on DGX Spark

Replaces `--keep-strategy first_n` (keeps experts `0..N-1` by index) with a calibration-driven selection that scores every expert by its actual contribution to the layer output. Per-expert REAP score:

```
S_j = mean over tokens routed to expert j of:  g_j(x) · ‖f_j(x)‖₂
```

where `g_j(x)` is the gate's routing weight and `f_j(x)` is expert j's output before routing-weight scaling. Reference: [arXiv 2510.13999](https://arxiv.org/html/2510.13999v1).

Hash-routed layers (0..2) are not pruned — `tid2eid` indexes the full pool.

## Architecture: pipeline-parallel + pure REAP + length-sorted batching

For each layer in order (embed → 0..n_layers-1 → mtp.0):

1. Build a fresh standalone `Block` at full **256 routed experts** (~4 GB resident).
2. Load layer L's weights from upstream HF (the unpruned 256 versions).
3. Forward calibration through this layer — length-sorted batches reading per-sequence input activations from the previous layer's on-disk cache and writing per-sequence output activations to the next layer's cache.
4. For score layers: hook `MoE.forward` to record REAP scores from valid (non-padded) positions only.
5. Score layer → pick top-N → write `keep_indices[L]`. Free the block.

Three properties this gets us that older harness-based approaches don't:

- **Pure REAP**: each layer scored against UNPRUNED upstream activations. Eliminates the compounding bias of sequential-pruning approaches (where layer L was scored against an upstream that had been pruned 0..L-1 times).
- **Tiny memory footprint**: ~10–15 GB peak (one block + small batch worth of activations + Python/CUDA), regardless of which layer we're on or how many we've processed.
- **Batched calibration**: length-sorted bins → padding waste minimized → 2–4× throughput vs per-sequence forward.

## What's in this folder

| Path | Purpose |
|---|---|
| `reap_score.py` | Pipelined REAP algorithm: HFShardIndex, ScoreBuffer, length-sorted batching, on-disk activation cache, per-layer build/load/score/free, end-to-end orchestrator. |
| `build_keep_indices.py` | Top-level CLI. |
| `calibration_loader.py` | Fetch + tokenize a calibration recipe from HF datasets, per-source disk cache. |
| `recipes/reap_full.json` | Cerebras's published agentic-reasoner mix (24,576 × 16K tokens). Default. Tractable on a single Spark via the pipelined sweep. |
| `recipes/spark_default.json` | 2,560 × 4K-token iteration mix. ~30–60 min on Spark. |
| `analysis/plot_scores.py` | Read `expert_scores.json`, emit per-layer markdown report + ASCII histograms. |

## How to run

### Prerequisites

- HF datasets package (one-time): `pip install datasets`
- Upstream HF V4-Flash checkpoint at `$HF_CKPT` (~149 GB)
- Tokenizer files (any pruned ckpt dir or the HF dir itself)

### One-shot end-to-end

```bash
python calibration/build_keep_indices.py \
    --hf-ckpt-path $HF_CKPT \
    --tokenizer-path runs/pruned-128-firstn \
    --out-dir runs/keep_indices/reap \
    --n-keep 128 \
    --batch-size 4
```

Defaults to `recipes/reap_full.json`. Switch to `--recipe calibration/recipes/spark_default.json` for fast iteration.

Memory peak: ~10–15 GB. Wall time: depends on calibration size; spark_default ~30–60 min, reap_full a few hours.

Output (under `--out-dir`):

```
runs/keep_indices/reap/
├── keep_indices.json    # consumed by inference/convert.py --keep-strategy indices
├── expert_scores.json   # full per-expert score vectors (preserved for analysis)
├── score_summary.csv    # per-layer aggregates: min/max/mean/median/std/top1-vs-topN ratio
├── chunks/              # per-layer score JSONs (resume granularity)
└── activations/         # per-seq h cache (rolled forward; deleted after sweep)
```

### Materialize the pruned safetensors

```bash
python inference/convert.py \
    --hf-ckpt-path $HF_CKPT \
    --save-path runs/pruned-128-reap \
    --n-experts 256 --n-experts-score 128 \
    --n-hash-layers 3 --n-layers 43 \
    --keep-strategy indices \
    --keep-indices-json runs/keep_indices/reap/keep_indices.json \
    --model-parallel 1 --expert-dtype fp4
```

Then run inference normally (`scripts/try_generate_chat.py`, `scripts/chat_interactive.py`, etc.) against `runs/pruned-128-reap`.

### Inspect the score distributions

```bash
python calibration/analysis/plot_scores.py \
    runs/keep_indices/reap/expert_scores.json \
    --out runs/keep_indices/reap/score_report.md
```

Cross-layer summary table (top-1/top-N ratios, boundary gaps, never-selected counts) and per-layer ASCII histograms. Use this to decide whether per-layer non-uniform expert counts or intra-expert SVD compression is worth pursuing in v3.

## Memory profile

At any moment the pipelined sweep holds:

- One `Block` (or `ParallelEmbedding` during bootstrap), full 256 experts: ~4 GB
- Current batch's input activations on CUDA: ~250 MB (batch_size=4 × max-seqlen 4K × HC=4 × dim=4096 × 2 bytes)
- Current batch's output activations on CUDA: ~250 MB
- ScoreBuffer + Python + CUDA overhead: ~5 GB

**Peak ~10–15 GB on a 119 GB Spark.** Per-layer disk I/O is the throughput floor (~600 MB read + 600 MB write per layer for spark_default; correspondingly larger for reap_full).

## Resume

The runner is fully resumable. Per-layer chunks are written to `chunks/layers.{L}.json` as each layer scores; activation caches are rolled forward (each layer's input dir is deleted once the next layer's output dir is complete). Restart with the same `--out-dir` to pick up where it left off.

## Recipe customization

Edit any recipe JSON or write a new one. The runner accepts any path via `--recipe`. Fields:

```json
{
  "name": "...",
  "max_seq_len": 4096,
  "sources": [
    {
      "name": "human-readable-id",
      "hf_id": "owner/dataset",
      "split": "train",
      "config": "optional_config",
      "n_samples": 1024,
      "text_fields": ["field1", "field2"],
      "join_with": "\n\n",                // for non-message data
      "messages_format": "concat_role_content"  // for [{role,content}] data
    }
  ]
}
```

Per-source caches are keyed on the source spec hash; changing one source doesn't invalidate the others.

## Tradeoffs vs `first_n`

| Property | `first_n` | This pipelined REAP |
|---|---|---|
| Calibration cost | none | minutes to hours |
| Calibration data needed | none | a few M tokens minimum |
| Quality vs unpruned | poor (commit-failure on reasoning) | targets paper-grade |
| Per-expert audit trail | none | full `expert_scores.json` |
| Compounding bias | n/a | none (pure REAP) |
| Memory peak | n/a | ~10–15 GB |
