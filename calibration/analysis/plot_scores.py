"""Read an `expert_scores.json` and emit a markdown report + ASCII histograms.

Reports per layer:
  - Score distribution shape (min/max/mean/median/std).
  - top-1 vs top-N ratio (how peaked is the distribution).
  - Score gap at the keep boundary (S[N-1] vs S[N]) — large gap = clean separation,
    small gap = pruning is throwing away barely-different experts.
  - ASCII histogram of scores.

These statistics are what you'd use to decide whether to do per-layer
non-uniform expert counts in a v2 pruning pass.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


def hist_bars(values: list[float], n_bins: int = 24, width: int = 40) -> list[str]:
    """Return ASCII bars for a histogram of `values`."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [f"  {lo:.3e}  | {'#'*width}  ({len(values)})"]
    bins = [0] * n_bins
    edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]
    for v in values:
        b = min(int((v - lo) / (hi - lo) * n_bins), n_bins - 1)
        bins[b] += 1
    cmax = max(bins)
    out = []
    for i, c in enumerate(bins):
        bar = "#" * int(c / cmax * width) if cmax > 0 else ""
        out.append(f"  {edges[i]:.3e}..{edges[i+1]:.3e} | {bar:<{width}}  ({c})")
    return out


def per_layer_report(layer_id, entry: dict) -> list[str]:
    scores = entry["scores"]
    kept = set(entry["kept"])
    n_keep = entry["n_keep"]
    diags = entry.get("diagnostics", {})
    n = len(scores)
    s_sorted = sorted(scores, reverse=True)

    s_min = min(scores); s_max = max(scores)
    s_mean = sum(scores) / n
    s_med = s_sorted[n // 2]
    s_std = math.sqrt(sum((x - s_mean) ** 2 for x in scores) / n)

    top1 = s_sorted[0]
    last_kept = s_sorted[n_keep - 1] if n_keep <= n else float("nan")
    first_dropped = s_sorted[n_keep] if n_keep < n else float("nan")
    boundary_gap = (last_kept - first_dropped) if not math.isnan(first_dropped) else float("nan")
    boundary_gap_rel = (boundary_gap / last_kept) if last_kept > 0 else float("nan")
    top1_to_topN = (top1 / last_kept) if last_kept > 0 else float("inf")

    out = [
        f"## {layer_id}",
        "",
        f"- experts: **{n}**, keep: **{n_keep}**",
        f"- score range: [{s_min:.3e}, {s_max:.3e}]",
        f"- mean ± std: {s_mean:.3e} ± {s_std:.3e}, median: {s_med:.3e}",
        f"- top-1 / top-N ratio: **{top1_to_topN:.2f}×** "
        f"(closer to 1 = flatter distribution)",
        f"- boundary gap (kept[N-1] vs dropped[0]): {boundary_gap:.3e} "
        f"({boundary_gap_rel:.1%} of kept[N-1])",
    ]
    if diags:
        out.append(
            f"- routing diagnostics: {diags.get('n_never_selected', 0)} of {n} experts "
            f"never selected; selections per expert "
            f"min/mean/max = {diags.get('min_selections','?')}/"
            f"{diags.get('mean_selections','?'):.1f}/"
            f"{diags.get('max_selections','?')}"
        )
    out.append("")
    out.append("score histogram (all experts):")
    out.append("```")
    out.extend(hist_bars(scores, n_bins=24, width=36))
    out.append("```")
    out.append("")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scores_json", help="path to expert_scores.json")
    p.add_argument("--out", default=None, help="markdown output path (default: stdout)")
    args = p.parse_args()

    data = json.loads(Path(args.scores_json).read_text())
    lines: list[str] = []
    lines.append(f"# REAP score report — `{args.scores_json}`")
    lines.append("")

    # Cross-layer summary first (the table you'd use to decide non-uniform ratios).
    lines.append("## cross-layer summary")
    lines.append("")
    lines.append("| layer | n_experts | n_keep | top1/topN | boundary_gap_rel | never_selected |")
    lines.append("|---|---|---|---|---|---|")
    def _sort_key(kv):
        # Keys are like "layers.3" or "mtp.0". Sort layers.* numerically, MTP last.
        k = kv[0]
        if k.startswith("layers."):
            return (0, int(k.split(".")[1]))
        return (1, k)
    for L_str, entry in sorted(data.items(), key=_sort_key):
        scores = entry["scores"]
        n = len(scores); n_keep = entry["n_keep"]
        s_sorted = sorted(scores, reverse=True)
        top1 = s_sorted[0]; last_kept = s_sorted[n_keep - 1]
        first_dropped = s_sorted[n_keep] if n_keep < n else float("nan")
        gap_rel = (last_kept - first_dropped) / last_kept if last_kept > 0 and not math.isnan(first_dropped) else float("nan")
        ratio = (top1 / last_kept) if last_kept > 0 else float("inf")
        ns = entry.get("diagnostics", {}).get("n_never_selected", "?")
        lines.append(f"| {L_str} | {n} | {n_keep} | {ratio:.2f}× | {gap_rel:.1%} | {ns} |")
    lines.append("")
    lines.append("Reading guide:")
    lines.append("- **top1/topN close to 1** = flat distribution; pruning more is risky here, consider keeping more experts.")
    lines.append("- **boundary_gap_rel large** = clean separation between kept and dropped; safe to prune at this threshold.")
    lines.append("- **boundary_gap_rel small (<5%)** = the pruner is throwing away experts that look almost identical to kept ones; strong signal that this layer would benefit from either more keepers or intra-expert decomposition (Tier-2 SVD).")
    lines.append("- **never_selected > 0** = calibration is too small or routing collapses on this layer; verify before trusting the prune.")
    lines.append("")

    # Per-layer detail.
    lines.append("## per-layer detail")
    lines.append("")
    def _sort_key(kv):
        # Keys are like "layers.3" or "mtp.0". Sort layers.* numerically, MTP last.
        k = kv[0]
        if k.startswith("layers."):
            return (0, int(k.split(".")[1]))
        return (1, k)
    for L_str, entry in sorted(data.items(), key=_sort_key):
        lines.extend(per_layer_report(L_str, entry))

    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
