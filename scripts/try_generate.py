"""Quick text-generation probe on a pruned V4-Flash checkpoint.

Mirrors generate.py's setup (default device flip, etc.) but bypasses
encoding_dsv4 so you can feed plain text, not a chat message list.

Usage:
  python try_generate.py --ckpt-path /path/to/pruned \
      --n-experts-score 8 --prompt "Hello, my name is" --max-new-tokens 32
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_model
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "inference"))
from model import Transformer, ModelArgs  # noqa: E402
from load_streaming import load_direct  # noqa: E402


def sample(logits, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate_tokens(
    model: Transformer,
    prompt_tokens: List[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
) -> List[int]:
    total_len = min(model.max_seq_len, max_new_tokens + len(prompt_tokens))
    tokens = torch.full((1, total_len), -1, dtype=torch.long)
    tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long)
    prev_pos = 0
    prompt_mask = tokens != -1
    n_prompt = len(prompt_tokens)
    for cur_pos in range(n_prompt, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        nxt = sample(logits, temperature) if temperature > 0 else logits.argmax(dim=-1)
        # during the prompt overlap no-op path (we start at n_prompt so it's all generated)
        nxt = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], nxt)
        tokens[0, cur_pos] = nxt[0]
        if nxt[0].item() == eos_id:
            break
        prev_pos = cur_pos
    return tokens[0, n_prompt : cur_pos + 1].tolist()


@torch.inference_mode()
def generate_tokens_mtp(
    model: Transformer,
    prompt_tokens: List[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 0.0,
) -> tuple[List[int], dict]:
    """MTP speculative decoding (k=1, greedy).

    Plumbing for the trained MTPBlock as a draft head, then verifies the draft
    via a 1-token main forward at the next position. Acceptance rate is recorded
    but the per-step verify currently uses a SINGLE-TOKEN main forward (rather
    than a batched 2-token verify), so this path does NOT yield a wall-clock
    speedup yet.

    The 2-token batched verify -- which is what produces the actual speedup --
    is blocked on `inference/kernel.py:sparse_attn_kernel` JIT-binding its
    symbolic q-seqlen on first call (seqlen=1 from regular decode), so a later
    seqlen=2 verify call fails with a shape assertion. Resolving that needs a
    real fix to either the TileLang kernel template or the sparse_attn cache
    key (we tried both; both have second-order issues with prefill calls
    sharing the cache).

    Greedy only. Sampling MTP requires rejection sampling for valid spec-decode
    semantics and is out of scope.
    """
    if temperature > 0:
        raise NotImplementedError("MTP speculative decode currently supports greedy only.")
    if not getattr(model, "mtp", None) or len(model.mtp) == 0:
        raise RuntimeError("model has no MTP block (n_mtp_layers must be >= 1).")
    mtp = model.mtp[0]

    total_len = min(model.max_seq_len, max_new_tokens + len(prompt_tokens))
    tokens = torch.full((1, total_len), -1, dtype=torch.long)
    n_prompt = len(prompt_tokens)
    tokens[0, :n_prompt] = torch.tensor(prompt_tokens, dtype=torch.long)

    stats = {"accepted": 0, "rejected": 0, "main_calls": 0}

    # Init: prefill, predict t_{n_prompt}.
    logits_init, h_init = model.forward(
        tokens[:, :n_prompt], start_pos=0, return_h=True,
    )
    stats["main_calls"] += 1
    next_tok = logits_init.argmax(dim=-1)  # [1] = t_{n_prompt}
    tokens[0, n_prompt] = next_tok[0]
    if int(next_tok[0].item()) == eos_id:
        return tokens[0, n_prompt:n_prompt + 1].tolist(), stats

    # Draft t_{n_prompt+1}.
    draft = int(mtp(
        h_init[:, -1:], start_pos=n_prompt,
        input_ids=next_tok.view(1, 1).to(torch.int64),
    ).argmax(dim=-1)[0].item())

    cur_pos = n_prompt + 1

    while cur_pos < total_len:
        # 1-token main forward at position cur_pos-1 -> predicts t_{cur_pos} (true)
        # AND advances main KV by one. Output h is at position cur_pos-1.
        m_logits, m_h = model.forward(
            tokens[:, cur_pos - 1:cur_pos], start_pos=cur_pos - 1,
            return_h=True,
        )
        stats["main_calls"] += 1
        true_at_cur = int(m_logits.argmax(dim=-1)[0].item())

        if true_at_cur == draft:
            stats["accepted"] += 1
        else:
            stats["rejected"] += 1
        # Either way commit the true value.
        tokens[0, cur_pos] = true_at_cur
        if true_at_cur == eos_id:
            cur_pos += 1
            break
        cur_pos += 1
        if cur_pos >= total_len:
            break
        # Next draft from MTP using h at the just-committed position.
        draft = int(mtp(
            m_h[:, -1:], start_pos=cur_pos - 1,
            input_ids=tokens[:, cur_pos - 1:cur_pos].to(torch.int64),
        ).argmax(dim=-1)[0].item())

    return tokens[0, n_prompt:cur_pos].tolist(), stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--config", default=str(REPO / "inference" / "config.json"))
    p.add_argument("--n-experts-score", type=int, required=True)
    p.add_argument("--prompt", default="Hello, my name is")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy; >0 = Gumbel-max sampling.")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--direct-load", action="store_true", default=True,
                   help="Use mmap-free pread+fadvise loader. Default True to avoid 2x RAM peak.")
    p.add_argument("--no-direct-load", dest="direct_load", action="store_false")
    p.add_argument("--mtp", action="store_true",
                   help="Use MTP speculative decoding (greedy only). "
                        "Drafts one extra token per step from the trained MTP block.")
    args = p.parse_args()

    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    with open(args.config) as f:
        cfg = json.load(f)
    cfg["n_routed_experts_score"] = args.n_experts_score
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = args.max_seq_len
    model_args = ModelArgs(**cfg)

    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)

    t0 = time.time()
    with torch.device("cuda"):
        model = Transformer(model_args)
    print(f"[{time.time()-t0:.1f}s] built model")

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    eos_id = tokenizer.eos_token_id
    print(f"eos_id: {eos_id}")

    t0 = time.time()
    shard_files = sorted(Path(args.ckpt_path).glob("model*-mp*.safetensors"))
    if args.direct_load:
        load_direct(model, [str(s) for s in shard_files])
    else:
        for shard in shard_files:
            load_model(model, str(shard), strict=False)
    print(f"[{time.time()-t0:.1f}s] loaded weights")
    torch.set_default_device("cuda")
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    print(f"prompt ({len(prompt_ids)} tokens): {args.prompt!r}")
    print(f"prompt ids: {prompt_ids}")

    t0 = time.time()
    if args.mtp:
        if args.temperature > 0:
            print("[warn] --mtp requires greedy; forcing temperature=0")
        out_ids, mtp_stats = generate_tokens_mtp(
            model, prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_id=eos_id,
            temperature=0.0,
        )
    else:
        out_ids = generate_tokens(
            model, prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_id=eos_id,
            temperature=args.temperature,
        )
        mtp_stats = None
    dt = time.time() - t0
    completion = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(f"\n[{dt:.1f}s to generate {len(out_ids)} new tokens, "
          f"{len(out_ids) / dt:.2f} tok/s]")
    if mtp_stats is not None:
        total = mtp_stats["accepted"] + mtp_stats["rejected"]
        accept_rate = mtp_stats["accepted"] / max(total, 1)
        print(f"[mtp] accepted={mtp_stats['accepted']} rejected={mtp_stats['rejected']} "
              f"accept_rate={accept_rate*100:.1f}%  main_calls={mtp_stats['main_calls']}")
        print(f"[mtp] NOTE: current path uses 1-token verify (no wall-clock speedup); "
              f"batched 2-token verify is blocked on sparse_attn shape binding (see "
              f"scripts/try_generate.py docstring).")
    print()
    print("=== PROMPT ===")
    print(args.prompt)
    print("=== COMPLETION ===")
    print(completion)
    print("=== IDS ===")
    print(out_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
