"""Fetch + tokenize the calibration mix described by a recipe JSON.

Each source in the recipe is loaded from HuggingFace datasets, projected to a
single text string, tokenized to at most `max_seq_len` tokens, and cached to
disk **per source** (so changing one source doesn't invalidate the others).

If a source fails to download (e.g. gated dataset, network error), the loader
prints a clear warning, skips that source, and continues with the rest. This is
load-bearing for multi-day runs where mid-flight HF outages would otherwise
waste hours of cached work.

Output format: a list of `torch.LongTensor` with shape `[seq_len_i]`, one per
calibration sample.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


def _normalize_text_from_messages(messages: list) -> str:
    """Render a [{"role":..., "content":...}, ...] list to a flat string."""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _project_sample_to_text(sample: dict, source_spec: dict) -> str:
    text_fields = source_spec.get("text_fields", [])
    if source_spec.get("messages_format") == "concat_role_content":
        assert len(text_fields) == 1, source_spec
        msgs = sample.get(text_fields[0])
        if msgs is None:
            return ""
        # Some HF datasets store the messages list as a JSON-encoded string (e.g.
        # SWE-bench/SWE-smith-trajectories). Parse if needed; if parsing fails,
        # the silent-skip in _normalize_text_from_messages will still drop it.
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except (json.JSONDecodeError, TypeError):
                return ""
        return _normalize_text_from_messages(msgs)
    join_with = source_spec.get("join_with", "\n\n")
    pieces = []
    for f in text_fields:
        v = sample.get(f)
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        pieces.append(str(v))
    return join_with.join(pieces)


def _source_cache_key(source: dict, max_seq_len: int, tokenizer_name_or_path: str) -> str:
    payload = json.dumps({
        "source": source,
        "max_seq_len": max_seq_len,
        "tokenizer": tokenizer_name_or_path,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _tokenize_one_source(
    source: dict, max_seq_len: int, tokenizer, cache_root: Path,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Load + tokenize a single source. Returns (token_seqs, summary).

    Uses per-source disk cache keyed on (source spec + max_seq_len + tokenizer).
    Failure raises — caller decides whether to skip or abort.
    """
    cache_key = _source_cache_key(source, max_seq_len, getattr(tokenizer, "name_or_path", "?"))
    cache_dir = cache_root / "sources" / f"{source['name']}.{cache_key}"
    cache_file = cache_dir / "tokens.pt"
    if cache_file.exists():
        seqs = torch.load(cache_file, weights_only=False)
        print(f"[calib] {source['name']}: loaded {len(seqs)} cached sequences "
              f"from {cache_file}", flush=True)
        return seqs, {"source": source["name"], "n_sequences": len(seqs), "cache_hit": True}

    from datasets import load_dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_target = int(source["n_samples"])
    print(f"[calib] {source['name']}: streaming {n_target} samples from "
          f"{source['hf_id']} (max_len={max_seq_len})", flush=True)
    load_kwargs = {"split": source["split"], "streaming": True}
    if "config" in source:
        ds = load_dataset(source["hf_id"], source["config"], **load_kwargs)
    else:
        ds = load_dataset(source["hf_id"], **load_kwargs)

    seqs: list[torch.Tensor] = []
    for sample in ds:
        text = _project_sample_to_text(sample, source)
        if not text or len(text) < 32:
            continue
        tokens = tokenizer.encode(
            text, add_special_tokens=False,
            truncation=True, max_length=max_seq_len,
        )
        if len(tokens) < 16:
            continue
        seqs.append(torch.tensor(tokens, dtype=torch.long))
        if len(seqs) >= n_target:
            break
    avg_len = (sum(int(t.numel()) for t in seqs) / max(len(seqs), 1))
    print(f"[calib] {source['name']}: kept {len(seqs)}/{n_target} (avg len {avg_len:.0f} tokens)",
          flush=True)

    torch.save(seqs, cache_file)
    summary = {
        "source": source["name"],
        "hf_id": source["hf_id"],
        "n_sequences": len(seqs),
        "total_tokens": sum(int(t.numel()) for t in seqs),
        "avg_len": float(avg_len),
        "cache_file": str(cache_file),
        "cache_hit": False,
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return seqs, summary


def load_calibration(
    recipe_path: str,
    tokenizer,
    cache_root: str = "runs/calib_cache",
    skip_on_fail: bool = True,
) -> list[torch.Tensor]:
    """Load (or rebuild) the calibration token sequences.

    `skip_on_fail=True` (default) prints a warning and continues if a source fails
    to download (e.g. gated dataset, network error). The returned list is the
    union of all successfully loaded sources.
    """
    with open(recipe_path) as f:
        recipe = json.load(f)
    cache_root_p = Path(cache_root)
    cache_root_p.mkdir(parents=True, exist_ok=True)
    max_len = int(recipe.get("max_seq_len", 4096))

    all_seqs: list[torch.Tensor] = []
    all_summaries: list[dict] = []
    for src in recipe["sources"]:
        try:
            seqs, summary = _tokenize_one_source(src, max_len, tokenizer, cache_root_p)
            all_seqs.extend(seqs)
            all_summaries.append(summary)
        except Exception as e:
            err_summary = {
                "source": src.get("name"),
                "hf_id": src.get("hf_id"),
                "skipped": True,
                "error": f"{type(e).__name__}: {e}",
            }
            all_summaries.append(err_summary)
            if skip_on_fail:
                print(f"\n[calib] WARNING: source '{src.get('name')}' "
                      f"({src.get('hf_id')}) failed; SKIPPING and continuing.",
                      flush=True)
                print(f"[calib]   reason: {type(e).__name__}: {e}", flush=True)
                if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                    print(f"[calib]   hint: visit {src.get('hf_id')} on HF Hub to request "
                          f"access, or replace this source in the recipe with an open alternative.",
                          flush=True)
                continue
            raise

    # Top-level summary file (per-recipe roll-up of all source attempts).
    summary_path = cache_root_p / "last_recipe_summary.json"
    summary_path.write_text(json.dumps({
        "recipe_name": recipe.get("name"),
        "total_sequences": len(all_seqs),
        "total_tokens": sum(int(t.numel()) for t in all_seqs),
        "sources": all_summaries,
    }, indent=2))

    if not all_seqs:
        raise RuntimeError(
            "calibration loader: every source failed; cannot proceed. "
            f"See {summary_path} for per-source errors."
        )
    print(f"[calib] total: {len(all_seqs)} sequences, "
          f"{sum(int(t.numel()) for t in all_seqs):,} tokens "
          f"(across {sum(1 for s in all_summaries if not s.get('skipped'))} sources)",
          flush=True)
    return all_seqs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--cache-root", default="/tmp/v4f_calib_cache")
    p.add_argument("--no-skip-on-fail", dest="skip_on_fail", action="store_false")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    seqs = load_calibration(args.recipe, tok, args.cache_root, args.skip_on_fail)
    print(f"OK: {len(seqs)} sequences, total {sum(int(s.numel()) for s in seqs):,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
