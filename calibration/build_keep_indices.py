"""Top-level CLI: pipeline-parallel + pure REAP + length-sorted batched calibration.

End-to-end:
  1. Load tokenizer.
  2. Load + tokenize the calibration recipe (per-source disk cache).
  3. Run the pipelined REAP sweep (calibration/reap_score.py::reap_full_sweep):
     for each layer L, build a fresh standalone Block at full 256 experts, load
     it from upstream HF, forward calibration through it (length-sorted batches
     reading per-seq h from disk, writing per-seq h_out), score, free.
  4. Write keep_indices.json + per-layer score JSONs + score_summary.csv.
  5. Print the convert.py command needed to materialize the pruned safetensors.

Memory peak: ~10-15 GB (one Block + small batch worth of activations) regardless
of layer index. No long-lived harness.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ModelArgs  # noqa: E402
from calibration_loader import load_calibration  # noqa: E402
from reap_score import reap_full_sweep  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-ckpt-path", required=True,
                   help="Path to the upstream HF DeepSeek-V4-Flash checkpoint dir.")
    p.add_argument("--config", default=str(REPO / "inference" / "config.json"))
    p.add_argument("--tokenizer-path", required=True,
                   help="Dir with tokenizer.json (the HF dir, or any pruned ckpt).")
    p.add_argument("--recipe", default=str(Path(__file__).resolve().parent / "recipes" / "reap_full.json"),
                   help="Calibration recipe JSON. Default: reap_full.json (paper-grade, "
                        "feasible on Spark via pipelined sweep). spark_default.json for "
                        "fast iteration.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory (under runs/, please). Will hold chunks/, "
                        "activations/, keep_indices.json, score_summary.csv.")
    p.add_argument("--n-keep", type=int, default=128,
                   help="Experts to keep per learned-router layer (default 128).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Length-sorted batch size for the calibration forward (default 4).")
    p.add_argument("--chunk-size", type=int, default=8,
                   help="Number of consecutive layers held resident per pipeline stage. "
                        "Larger = fewer disk-cache rolls (less I/O). Per-Block resident "
                        "cost is ~4 GB; budget ~4*chunk_size + 0.25*batch_size + 10 GB total.")
    p.add_argument("--superset-size", type=int, default=0,
                   help="Split calibration into supersets of this many sequences each. "
                        "Caps inter-chunk activation disk cache to one superset's worth "
                        "(necessary for reap_full's ~5TB single-batch footprint). "
                        "0 = no split (one superset = full data). For reap_full, try "
                        "superset_size=4096 → ~6 supersets, peak disk ~870GB.")
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="Max prompt length the standalone Blocks are built for. "
                        "Default 0 = auto-set from recipe's max_seq_len.")
    p.add_argument("--cache-root", default="runs/calib_cache")
    p.add_argument("--no-mtp", dest="process_mtp", action="store_false",
                   help="Skip scoring the MTP block (the MoE inside it).")
    p.add_argument("--drop-intermediate-caches", dest="keep_intermediate_caches",
                   action="store_false",
                   help="Delete each chunk's input cache as soon as the chunk completes "
                        "(saves ~116GB per chunk on disk, but means a restart has to re-roll "
                        "activations forward from the deepest existing cache). "
                        "Default: keep all caches (resume from any layer cheaply).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    with open(args.config) as f:
        cfg = json.load(f)
    cfg["max_batch_size"] = args.batch_size
    with open(args.recipe) as rf:
        recipe = json.load(rf)
    recipe_max = int(recipe.get("max_seq_len", 4096))
    cfg["max_seq_len"] = args.max_seq_len if args.max_seq_len > 0 else recipe_max
    # We don't use n_routed_experts_score in the pipelined path (each Block we
    # build is at full 256), but ModelArgs requires it; pin to n_keep so the
    # field is consistent with what convert.py will materialize later.
    cfg["n_routed_experts_score"] = args.n_keep
    base_args = ModelArgs(**cfg)
    print(f"[main] base config: n_layers={base_args.n_layers} "
          f"n_hash_layers={base_args.n_hash_layers} "
          f"n_routed_experts={base_args.n_routed_experts} "
          f"n_mtp_layers={base_args.n_mtp_layers}", flush=True)
    print(f"[main] harness-free pipelined sweep: max_seq_len={cfg['max_seq_len']} "
          f"max_batch_size={cfg['max_batch_size']}", flush=True)

    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    torch.set_default_device("cuda")

    from transformers import AutoTokenizer
    print(f"[main] loading tokenizer from {args.tokenizer_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"[main] loading calibration ({args.recipe})", flush=True)
    calib_seqs = load_calibration(args.recipe, tok, cache_root=args.cache_root)
    print(f"[main] {len(calib_seqs)} sequences, "
          f"{sum(int(s.numel()) for s in calib_seqs):,} tokens", flush=True)

    print(f"\n[main] starting pipelined REAP sweep: batch_size={args.batch_size}, "
          f"n_keep={args.n_keep}", flush=True)
    t_sweep = time.time()
    per_layer = reap_full_sweep(
        hf_ckpt_path=args.hf_ckpt_path,
        base_args=base_args,
        calib_seqs=calib_seqs,
        out_dir=str(out_dir),
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        n_keep=args.n_keep,
        process_mtp=args.process_mtp,
        keep_intermediate_caches=args.keep_intermediate_caches,
        superset_size=args.superset_size if args.superset_size > 0 else None,
    )
    print(f"\n[main] REAP sweep finished in {(time.time()-t_sweep)/60:.1f} min", flush=True)

    # Write expert_scores.json (full per-expert score vectors, kept for offline analysis).
    expert_scores = {
        k: {"scores": v["scores"], "kept": v["kept"], "n_keep": v["n_keep"],
            "diagnostics": v["diagnostics"]}
        for k, v in per_layer.items()
    }
    (out_dir / "expert_scores.json").write_text(json.dumps(expert_scores, indent=2))

    # CSV summary per layer.
    with open(out_dir / "score_summary.csv", "w") as f:
        f.write("layer,n_routed_experts,n_keep,score_min,score_max,score_mean,"
                "score_median,score_std,top1_to_topN_ratio,n_never_selected\n")
        for layer_key, v in per_layer.items():
            s = torch.tensor(v["scores"], dtype=torch.float64)
            n = v["n_keep"]
            sorted_s, _ = s.sort(descending=True)
            top1 = float(sorted_s[0].item())
            topN = float(sorted_s[n - 1].item()) if n <= len(sorted_s) else float("nan")
            ratio = top1 / topN if topN > 0 else float("inf")
            f.write(
                f"{layer_key},{int(s.numel())},{n},"
                f"{float(s.min()):.6e},{float(s.max()):.6e},{float(s.mean()):.6e},"
                f"{float(s.median()):.6e},{float(s.std()):.6e},{ratio:.3f},"
                f"{v['diagnostics']['n_never_selected']}\n"
            )

    print(f"\n[main] artifacts written to {out_dir}/")
    print(f"  keep_indices.json     ({len(per_layer)} layers scored)")
    print(f"  expert_scores.json    (full {sum(len(v['scores']) for v in per_layer.values())} expert scores)")
    print(f"  score_summary.csv")
    print(f"  chunks/               (per-layer score JSONs for resume)")
    print()
    print("Next step: materialize the pruned safetensors with")
    print(f"  python inference/convert.py \\")
    print(f"      --hf-ckpt-path {args.hf_ckpt_path} \\")
    print(f"      --save-path runs/pruned-{args.n_keep}-reap-pipelined \\")
    print(f"      --n-experts {base_args.n_routed_experts} \\")
    print(f"      --n-experts-score {args.n_keep} \\")
    print(f"      --n-hash-layers {base_args.n_hash_layers} \\")
    print(f"      --n-layers {base_args.n_layers} \\")
    print(f"      --keep-strategy indices \\")
    print(f"      --keep-indices-json {out_dir / 'keep_indices.json'} \\")
    print(f"      --model-parallel 1 --expert-dtype fp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
