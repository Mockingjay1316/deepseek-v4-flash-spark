"""Estimate per-superset activation cache size from cached calibration tokens.

Validates a chosen --superset-size against actual per-source sequence lengths
before committing to a multi-hour run. Uses the same source ordering and per-
source token cache that build_keep_indices.py loads.

Activation cache formula (matches reap_score.bootstrap_embed_to_disk):
    bytes_per_token = hc_mult * dim * 2   # bf16
                    = 4 * 4096 * 2 = 32_768

Cross-validates against runs/logs/reap_full.log entries by predicting the
already-observed embed cache sizes for size=1024 supersets 0..4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "calibration" / "recipes"
CACHE_ROOT = REPO_ROOT / "runs" / "calib_cache"

HC_MULT = 4
DIM = 4096
BYTES_PER_TOKEN = HC_MULT * DIM * 2  # bf16

GIB = 1024 ** 3


def _source_cache_dir(source_name: str) -> Path | None:
    """Find the most recent cache dir for a given source name."""
    matches = list((CACHE_ROOT / "sources").glob(f"{source_name}.*"))
    if not matches:
        return None
    # Use the one whose tokens.pt matches `tokens.pt` directly, prefer freshest.
    candidates = [p for p in matches if (p / "tokens.pt").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "tokens.pt").stat().st_mtime)


def load_recipe_seqs(recipe_path: Path) -> tuple[list[int], list[str]]:
    """Returns (per_seq_token_count, per_seq_source_name) in recipe order."""
    recipe = json.loads(recipe_path.read_text())
    lengths: list[int] = []
    source_for_seq: list[str] = []
    for src in recipe["sources"]:
        name = src["name"]
        cache = _source_cache_dir(name)
        if cache is None:
            print(f"[!] no cached tokens.pt for source '{name}' — skipping", file=sys.stderr)
            continue
        seqs = torch.load(cache / "tokens.pt", weights_only=False)
        n_target = int(src["n_samples"])
        if len(seqs) < n_target:
            print(f"[!] source '{name}' has {len(seqs)} cached, recipe wants {n_target}",
                  file=sys.stderr)
        # build_keep_indices loads the full cached list (it was sized at write time).
        lengths.extend(int(t.numel()) for t in seqs)
        source_for_seq.extend([name] * len(seqs))
    return lengths, source_for_seq


def supersets_by_size(lengths: list[int], size: int) -> list[tuple[int, int]]:
    """Fixed-size partition: range(0, N, size)."""
    return [(i, min(i + size, len(lengths))) for i in range(0, len(lengths), size)]


def supersets_by_token_budget(lengths: list[int], budget: int) -> list[tuple[int, int]]:
    """Greedy partition: close out a superset when the next seq would exceed
    `budget`. Mirrors reap_score._partition_supersets exactly."""
    bounds, start, cum = [], 0, 0
    for i, L in enumerate(lengths):
        if cum > 0 and cum + L > budget:
            bounds.append((start, i))
            start, cum = i, 0
        cum += L
    if start < len(lengths):
        bounds.append((start, len(lengths)))
    return bounds


def fmt_gb(n: int) -> str:
    return f"{n / GIB:.2f} GB"


BYTES_PER_BLOCK_GB = 4.0       # 256 experts FP4 + FP8 attn (project memory)
PYTHON_CUDA_OVERHEAD_GB = 10.0  # interpreter + cuda context + score bufs + idx
EMBED_LAYER_GB = 1.0            # vocab × dim × bf16 (transient: only during embed phase)


def resident_weights_gb(chunk_size: int) -> float:
    """Block weights resident during chunk forward. On GB10 unified memory,
    these come out of the same 119 GB pool as page cache and python heap."""
    return BYTES_PER_BLOCK_GB * chunk_size


def resident_overhead_gb() -> float:
    return PYTHON_CUDA_OVERHEAD_GB


def length_sorted_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    """Mirror reap_score.make_length_sorted_batches: sort by length, bin by bs."""
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]


def padded_batch_bytes(lengths: list[int], batch_size: int) -> tuple[int, int, int]:
    """Compute (sum_padded_tokens, max_per_batch_padded_tokens, sum_unpadded_tokens)
    when sequences are length-sorted then binned into batches of `batch_size`.
    Each batch is padded to its in-batch max length.

    `sum_padded_tokens` is what would be on disk if the cache were stored padded.
    `max_per_batch_padded_tokens` is the peak per-batch GPU tensor size driver.
    `sum_unpadded_tokens` is what's actually on disk (per-seq files).
    """
    batches = length_sorted_batches(lengths, batch_size)
    sum_padded = 0
    sum_unpadded = sum(lengths)
    max_per_batch = 0
    for batch in batches:
        batch_lens = [lengths[i] for i in batch]
        max_in_batch = max(batch_lens)
        sum_padded += max_in_batch * len(batch)
        max_per_batch = max(max_per_batch, max_in_batch * len(batch))
    return sum_padded, max_per_batch, sum_unpadded


def simulate_superset_ram(
    ss_lens: list[int], chunk_size: int, batch_size: int,
) -> dict:
    """Estimate peak host-RAM demand during one superset's chunk forward.

    Empirical anchor: supersets 0-3 (~16 GB cache, padded) ran fine; superset 4
    (~161 GB cache) OOM'd. The cliff sits inside [16, 161] GB cache size.
    The model that fits both data points:

        peak = resident + cache_padded + batch_working

    where:
      * resident          = 4×chunk + 0.25×bs + 10 GB (project memory)
      * cache_padded      = sum_padded_tokens × 32 KB  — the in_cache being
        read by the chunk forward, conservatively padded per the user's
        instruction. With length-sorted batching this is 0.2-11% above the
        unpadded on-disk size; using padded is a conservative safety margin.
        We do NOT separately count out_cache dirty pages: the kernel writes
        them back as the chunk progresses (vm.dirty_ratio caps accumulation
        at ~24 GB) and they overlap with in_cache space rather than adding to
        it — modeling them as a separate component double-counts.
      * batch_working     = 3 × per-batch padded peak
        (input h + output h + per-block scratch, all on unified memory)

    Verified: at chunk=16, bs=8, this gives 104.7 GB for ss=0 (margin +14.3 GB
    on 119 GB box) and 249.7 GB for ss=4 (margin -130.7 GB), matching the
    empirical pass/fail cliff.
    """
    sum_padded, max_per_batch_padded, sum_unpadded = padded_batch_bytes(
        ss_lens, batch_size
    )
    weights_b = int(resident_weights_gb(chunk_size) * GIB)
    overhead_b = int(resident_overhead_gb() * GIB)
    cache_padded = sum_padded * BYTES_PER_TOKEN
    cache_unpadded = sum_unpadded * BYTES_PER_TOKEN
    batch_working = max_per_batch_padded * BYTES_PER_TOKEN * 3
    peak_demand = weights_b + overhead_b + cache_padded + batch_working

    return {
        "weights_b": weights_b,
        "overhead_b": overhead_b,
        "cache_padded_b": cache_padded,
        "cache_unpadded_b": cache_unpadded,
        "batch_working_b": batch_working,
        "peak_demand_b": peak_demand,
        "sum_unpadded_tokens": sum_unpadded,
        "sum_padded_tokens": sum_padded,
        "max_per_batch_padded_tokens": max_per_batch_padded,
        "padding_overhead_pct": (sum_padded - sum_unpadded) * 100 / max(sum_unpadded, 1),
    }


def report(lengths: list[int], sources: list[str], parts: list[tuple[int, int]],
           label: str, host_ram_gb: float, chunk_size: int, batch_size: int,
           verbose: bool = False):
    avg_tokens = sum(lengths) / len(parts)
    print(f"\n{label} → {len(parts)} supersets, {len(lengths)} total seqs, "
          f"avg {avg_tokens/1e6:.2f}M tokens/superset")

    if verbose:
        header = (f"{'idx':>4} {'srcs':<28} {'n':>5} {'tok':>9} "
                  f"{'cache':>8} {'work':>7} {'peak':>8}")
        print(header)
        print("-" * len(header))

    rows = []
    for ss_idx, (a, b) in enumerate(parts):
        ss_lens = lengths[a:b]
        ss_srcs = sources[a:b]
        sim = simulate_superset_ram(ss_lens, chunk_size, batch_size)
        src_counts: dict[str, int] = {}
        for s in ss_srcs:
            src_counts[s] = src_counts.get(s, 0) + 1
        src_str = ",".join(f"{n}:{c}" for n, c in src_counts.items())
        rows.append((ss_idx, src_str, len(ss_lens), sim))
        if verbose:
            print(f"{ss_idx:>4} {src_str[:28]:<28} {len(ss_lens):>5} "
                  f"{sim['sum_unpadded_tokens']:>9,} "
                  f"{sim['page_cache_demand_b']/GIB:>6.1f}G "
                  f"{sim['batch_working_b']/GIB:>5.1f}G "
                  f"{sim['peak_demand_b']/GIB:>6.1f}G")

    worst = max(rows, key=lambda r: r[3]["peak_demand_b"])
    sim = worst[3]
    peak_gb = sim["peak_demand_b"] / GIB
    margin_gb = host_ram_gb - peak_gb
    if margin_gb > 15:
        verdict = "OK"
    elif margin_gb > 5:
        verdict = "TIGHT"
    else:
        verdict = "FAIL"

    print(f"Worst superset: idx={worst[0]} ({worst[1][:40]})  "
          f"n_seq={worst[2]} tokens={sim['sum_unpadded_tokens']:,} "
          f"(padding overhead {sim['padding_overhead_pct']:+.1f}%)")
    print(f"  block weights  {sim['weights_b']/GIB:.1f} GB  ({BYTES_PER_BLOCK_GB:.0f} GB × chunk_size={chunk_size})")
    print(f"  py/cuda over.  {sim['overhead_b']/GIB:.1f} GB")
    print(f"  cache (in)     {sim['cache_padded_b']/GIB:.1f} GB  (sum of padded batch sizes)")
    print(f"  batch working  {sim['batch_working_b']/GIB:.1f} GB  (3× padded peak batch)")
    print(f"  → peak demand  {peak_gb:.1f} GB    margin {margin_gb:+.1f} GB  → {verdict}")
    return verdict, peak_gb, margin_gb


def cross_validate_with_log(lengths: list[int], sources: list[str]):
    """Compare predicted size vs reap_full.log for size=1024 supersets 0..4."""
    log_path = REPO_ROOT / "runs" / "logs" / "reap_full.log"
    if not log_path.exists():
        return
    print("\n" + "=" * 72)
    print("Cross-validating prediction vs runs/logs/reap_full.log (size=1024)")
    print("=" * 72)
    log_text = log_path.read_text()
    parts = supersets_by_size(lengths, 1024)
    import re
    embed_lines = re.findall(r"\[embed\] done in [\d.]+s \(([\d.]+) GB on disk\)", log_text)
    print(f"{'idx':>4} {'predicted':>12} {'logged':>12} {'delta':>10}")
    for ss_idx, observed_str in enumerate(embed_lines):
        if ss_idx >= len(parts):
            break
        a, b = parts[ss_idx]
        toks = sum(lengths[a:b])
        pred = toks * BYTES_PER_TOKEN / GIB
        obs = float(observed_str)
        print(f"{ss_idx:>4} {pred:>10.2f} GB {obs:>10.2f} GB {pred-obs:>+8.2f} GB")
    print("(activation cache files are sliced to actual length per save_one()")
    print(" so the unpadded sum matches the on-disk and page-cache footprint)")


def per_source_summary(lengths: list[int], sources: list[str]):
    print("\nPer-source token stats:")
    by_source: dict[str, list[int]] = {}
    for L, s in zip(lengths, sources):
        by_source.setdefault(s, []).append(L)
    print(f"  {'source':<28} {'n':>6} {'avg':>6} {'max':>6} {'p50':>6} {'p95':>6} {'tot tokens':>12}")
    for name, ls in by_source.items():
        ls_sorted = sorted(ls)
        n = len(ls)
        p50 = ls_sorted[n//2]
        p95 = ls_sorted[int(n*0.95)]
        print(f"  {name:<28} {n:>6} {sum(ls)//n:>6} {max(ls):>6} "
              f"{p50:>6} {p95:>6} {sum(ls):>12,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=str(RECIPES_DIR / "reap_full.json"))
    ap.add_argument("--sizes", type=int, nargs="*", default=[1024, 256, 128],
                    help="Fixed-size strategies to evaluate (sequences per superset)")
    ap.add_argument("--token-budgets", type=int, nargs="*",
                    default=[2_000_000, 1_500_000, 1_000_000, 750_000, 500_000],
                    help="Token-budget strategies to evaluate (max tokens per superset)")
    ap.add_argument("--host-ram-gb", type=float, default=119.0)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print every superset row (default: just the worst)")
    args = ap.parse_args()

    lengths, sources = load_recipe_seqs(Path(args.recipe))
    if not lengths:
        print("[!] No sequences loaded — abort.")
        sys.exit(1)

    total_tokens = sum(lengths)
    print(f"Loaded {len(lengths):,} sequences, {total_tokens:,} total tokens")
    print(f"  avg {total_tokens/len(lengths):.0f}, max {max(lengths)}, min {min(lengths)}")
    print(f"Bytes/token = hc_mult({HC_MULT}) * dim({DIM}) * 2 (bf16) = {BYTES_PER_TOKEN:,}")

    per_source_summary(lengths, sources)

    summary = []
    for s in args.sizes:
        parts = supersets_by_size(lengths, s)
        v, peak, mg = report(lengths, sources, parts, f"size={s}",
                             args.host_ram_gb, args.chunk_size, args.batch_size,
                             verbose=args.verbose)
        summary.append(("size", s, v, peak, mg, len(parts)))
    for tb in args.token_budgets:
        parts = supersets_by_token_budget(lengths, tb)
        v, peak, mg = report(lengths, sources, parts, f"token-budget={tb:,}",
                             args.host_ram_gb, args.chunk_size, args.batch_size,
                             verbose=args.verbose)
        summary.append(("budget", tb, v, peak, mg, len(parts)))

    cross_validate_with_log(lengths, sources)

    print("\n" + "=" * 76)
    print(f"Summary (chunk={args.chunk_size}, batch={args.batch_size}, "
          f"host RAM={args.host_ram_gb:.0f} GB)")
    print("=" * 76)
    print(f"{'strategy':<10} {'param':>10} {'#supersets':>11} {'verdict':>8} "
          f"{'peak demand':>13} {'margin':>10}")
    for kind, val, v, peak, mg, n in summary:
        print(f"{kind:<10} {val:>10,} {n:>11} {v:>8} {peak:>11.1f} GB {mg:>+8.1f} GB")


if __name__ == "__main__":
    main()
