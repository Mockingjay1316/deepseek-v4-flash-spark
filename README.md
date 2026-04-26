# DeepSeek-V4-Flash on a single DGX Spark (GB10)

Run [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — a 284 B-parameter / 13 B-active MoE with a 1 M-token context window — on a single NVIDIA DGX Spark (GB10 Superchip, 128 GB unified LPDDR5X). The released checkpoint is ~149 GB and does not fit; this fork ships a half-prune of the learned-router experts (256 → 128) plus a memory-friendly loader so the resulting ~85 GB model loads and serves at the full 1 M context with ~17 GB of headroom.

This is a port, not a re-train. All weights are derived from the official DeepSeek release — distribute them through the HuggingFace channel, not this repo.

## Status

| What | State |
| --- | --- |
| 8-/32-/64-/128-expert pruned conversion | Working |
| Single-GPU inference at `max_seq_len=1M` | Working, ~5 tok/s decode, ~98 GB peak |
| Reasoning-quality coherence on chat prompts | Verified on small math word problems |
| Calibration-based expert selection (better than `first_n`) | Not yet — needs a bigger machine to log routing on the unpruned model |
| Multi-GPU / multi-node | Out of scope for this fork |

## What changed vs. upstream

The upstream `inference/` tree assumes you can fit the full 256-expert model. Three categories of change make a single-GB10 deployment work:

1. **Per-layer expert-count plumbing** (`inference/model.py`, `inference/config.json`)
   - New `ModelArgs.n_routed_experts_score` field (default 128) governs the routed-expert pool size in *learned-router* layers (3–42).
   - Hash-routed layers 0–2 keep the full 256 experts because their `tid2eid` lookup table indexes the entire pool.
   - `n_activated_experts` (top-k per token) stays at **6** in every layer — pruning shrinks the candidate pool, not the activation count.
   - `Gate`, `MoE`, and `Block` branch on `layer_id < n_hash_layers` to pick the right pool size.

2. **Pruning-aware checkpoint converter** (`inference/convert.py`)
   - New CLI: `--n-experts-score N`, `--keep-strategy {first_n,indices}`, `--keep-indices-json PATH`.
   - Drops experts beyond the kept set in learned layers, slices `gate.weight` / `gate.bias` rows to match, and renumbers expert IDs into a contiguous `[0, N)` range. Hash layers pass through unchanged.
   - FP4 (`torch.float4_e2m1fn_x2`) and FP8 paths both supported; scale tensors slice along the same expert axis as their weights.

3. **Memory-friendly loader** (`inference/load_streaming.py`, new file)
   - The default `safetensors.torch.load_model` calls `load_file` first, which materializes the entire shard in memory before copying into the already-allocated CUDA tensors. On GB10 the mmap'd page cache and the CUDA tensors share the same physical RAM, so peak load memory is ~2× the on-disk size — enough to OOM at 85 GB.
   - `load_direct` reads each tensor with a single `pread`, copies it into the model, then calls `posix_fadvise(POSIX_FADV_DONTNEED)` so the kernel drops the just-read pages from page cache. Peak load drops from ~170 GB to ~95 GB and load time drops from "OOM" to 46 s for the 128-expert checkpoint.

The TileLang attention kernels (`inference/kernel.py`) are unchanged — they're already expert-count agnostic.

## Hardware & software

| Component | Tested version |
| --- | --- |
| Machine | DGX Spark (GB10 Superchip, sm_121, 128 GB LPDDR5X unified, aarch64) |
| OS | Ubuntu/Linux 6.17 |
| Python | 3.12 |
| CUDA | 13.0.88 |
| PyTorch | 2.11.0+cu130 |
| safetensors | 0.7.0 |
| TileLang | 0.1.8 (with `apache-tvm-ffi==0.1.9` pinned — see `scripts/build_env.sh`) |
| `fast_hadamard_transform` | latest from upstream git, built `--no-build-isolation` |

GB10-specific gotchas (rediscovering these costs ~an hour each):
- TileLang's `sparse_attn` defaults exceed GB10's 99 KB shared-memory ceiling. The kernel in this repo is sized to fit (`block=16, num_stages=1`).
- `nvidia-smi --query-gpu=memory.used` returns `[N/A]` on GB10; use `free -h` or `/proc/meminfo` for memory pressure.
- PyTorch wheels typically top out at sm_120; sm_121 PTX-JIT works for vendor kernels but custom extensions need `TORCH_CUDA_ARCH_LIST=12.1` at build time.

## Quickstart

### 1. Clone and build the environment

```bash
git clone https://github.com/Mockingjay1316/deepseek-v4-flash-spark.git
cd deepseek-v4-flash-spark
# Activate a Python 3.12 environment of your choice (conda, venv, etc.)
pip install -r inference/requirements.txt
bash scripts/build_env.sh   # installs fast_hadamard_transform from git, pins apache-tvm-ffi
```

### 2. Download the upstream weights

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir /path/to/hf-checkpoint
```

The HF FP4 checkpoint is ~149 GB.

### 3. Convert to a pruned reference checkpoint

The converter reads the HF safetensors, drops experts beyond the kept set in learned layers, and writes a single sharded reference file. Recipe for the production 128-expert prune:

```bash
python inference/convert.py \
    --hf-ckpt-path /path/to/hf-checkpoint \
    --save-path /path/to/v4flash-pruned-128 \
    --n-experts 256 \
    --n-experts-score 128 \
    --n-hash-layers 3 \
    --n-layers 43 \
    --keep-strategy first_n \
    --model-parallel 1 \
    --expert-dtype fp4
```

Convert time on GB10: ~2 min. Output checkpoint size for various prune ratios:

| `--n-experts-score` | On-disk size | Notes |
| --- | --- | --- |
| 8 | 24 GB | Sanity-test only; output quality is poor |
| 32 | 36 GB | Loads comfortably; quality marginal |
| 64 | 52 GB | Borderline on GB10's RAM ceiling |
| **128** | **85 GB** | **Recommended for GB10. Coherent generation.** |
| 256 (no prune) | ~149 GB | Won't fit on GB10 |

### 4. Generate text

Plain prompt completion (raw text in, raw text out):

```bash
python scripts/try_generate.py \
    --ckpt-path /path/to/v4flash-pruned-128 \
    --n-experts-score 128 \
    --prompt "Hello, my name is" \
    --max-new-tokens 64 \
    --temperature 0.7
```

Chat-formatted reasoning (uses `encoding/encoding_dsv4.py` to apply the official prompt template):

```bash
python scripts/try_generate_chat.py \
    --ckpt-path /path/to/v4flash-pruned-128 \
    --n-experts-score 128 \
    --prompt "A train leaves Chicago at 9 AM going east at 60 mph. Another leaves New York at 10 AM going west at 80 mph. The cities are 800 miles apart. At what time will the trains meet? Reason step by step." \
    --max-new-tokens 200 \
    --temperature 0.7 \
    --max-seq-len 8192
```

`scripts/mem_probe.py` wraps any of the above to record `/proc/meminfo` and `/proc/<pid>/status` per second:

```bash
python scripts/mem_probe.py --label gen-128 --log /tmp/mem.csv -- \
    python scripts/try_generate_chat.py --ckpt-path /path/to/v4flash-pruned-128 ...
```

### 5. (Optional) The upstream `generate.py` runner

The original `inference/generate.py` is preserved unchanged for compatibility with the upstream chat loop:

```bash
torchrun --nproc-per-node 1 inference/generate.py \
    --ckpt-path /path/to/v4flash-pruned-128 \
    --config inference/config.json \
    --interactive
```

Note: `generate.py` uses `safetensors.torch.load_model` and so will hit the OOM problem on the 128-expert checkpoint. Either patch it to call `load_direct` from `inference/load_streaming.py`, or use `scripts/try_generate*.py` which already does.

## Measured numbers on GB10 (128-expert prune)

| `max_seq_len` | Load | Decode | Peak Δavail RAM |
| --- | --- | --- | --- |
| 512 | 42 s | 4.07 tok/s | -89.9 GB |
| 8 K | 51 s | 5.12 tok/s | -90.2 GB |
| 64 K | 43 s | 4.73 tok/s | -91.0 GB |
| 256 K | 43 s | 4.82 tok/s | -92.0 GB |
| 1 M | 44 s | 4.78 tok/s | -97.9 GB |

KV cache scales sublinearly with context length because most of the 43 layers use `compress_ratio=128` (8× compressed); only 21 layers use `compress_ratio=4`. The 1 M test pre-allocates KV cache + freqs_cis for full 1 M and still leaves ~17 GB headroom.

Single-batch decode is bandwidth- and launch-overhead-bound on this hardware, not compute-bound: at ~5 tok/s we're using ~15% of GB10's 273 GB/s memory bandwidth and well under 0.1% of its FP4 compute peak. The realistic ceiling on this chip with optimized MoE kernels (grouped-experts GEMM) is ~10–20 tok/s. See conversation logs for the full analysis.

## Repository layout

```
deepseek-v4-flash-spark/
├── config.json                         # upstream HF config (read-only)
├── model.safetensors.index.json        # upstream weight index (read-only)
├── tokenizer.json, tokenizer_config.json, generation_config.json
├── LICENSE                             # MIT, from upstream
├── inference/
│   ├── model.py                        # PATCHED: per-layer expert-pool plumbing
│   ├── convert.py                      # PATCHED: --n-experts-score, --keep-strategy
│   ├── config.json                     # PATCHED: adds n_routed_experts_score=128
│   ├── load_streaming.py               # NEW: load_direct (pread + fadvise DONTNEED)
│   ├── kernel.py                       # PATCHED: GB10 shared-mem ceiling fixes
│   ├── generate.py                     # upstream chat loop (uses old loader)
│   ├── README.md                       # upstream README
│   └── requirements.txt
├── encoding/                           # vendored chat-prompt encoder (DeepSeek-V4 format)
│   ├── encoding_dsv4.py
│   ├── README.md
│   └── tests/
└── scripts/                            # NEW (this fork)
    ├── build_env.sh                    # GB10-specific install recipe
    ├── try_generate.py                 # plain text generation
    ├── try_generate_chat.py            # chat-formatted generation
    └── mem_probe.py                    # per-second RAM/swap profiler wrapper
```

## Known caveats

- **`first_n` expert selection drifts on long generations.** Greedy decoding loops on simple prompts ("The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. ..."). Sampling at `temperature=0.7` produces coherent output. A calibration-based selection (running the unpruned 256-expert model on a corpus, ranking experts by routing frequency, and picking top-128 per layer) would close most of the quality gap, but requires hardware large enough to run the unpruned model.
- **Decode is launch-overhead-bound.** The Python expert-dispatch loop in `MoE.forward` issues ~770 GEMM kernel launches per token. Batching active experts into a single grouped GEMM (DeepGEMM MegaMoE-style) is the largest single perf win available.
- **The `inference/generate.py` interactive loop is untouched** and uses the old loader. If you want to run it on the 128-expert checkpoint, patch the `load_model` calls to `load_direct`.
- **No multi-GPU / TP support tested.** Convert.py has the `--model-parallel` plumbing from upstream, but this fork has only been validated at `mp=1` on a single GB10.
- **Pruned checkpoints are NOT distributed.** Build them locally from the upstream HF release by running `inference/convert.py`.

## Provenance

- Architecture, weights, tokenizer, and reference inference code: [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (MIT).
- Pruning patches and the `load_direct` loader: this fork.
- The "safetensors doubles memory on coherent-memory GPUs" diagnosis is the same root cause as [ComfyUI issue #10896](https://github.com/comfyanonymous/ComfyUI/issues/10896), discovered independently while debugging the OOM on this checkpoint.

## License

MIT, inherited from upstream. See `LICENSE`.
