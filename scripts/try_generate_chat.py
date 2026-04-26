"""Chat-formatted generation probe.

Wraps `try_generate` with the official encoding_dsv4 chat template so a single
{"role":"user","content":...} message gets proper BOS/role tokens before
generation. Use this for reasoning-quality tests; use try_generate.py for raw
prompt-completion / context-length stress tests.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "inference"))
sys.path.insert(0, str(REPO / "encoding"))
from model import Transformer, ModelArgs  # noqa: E402
from load_streaming import load_direct  # noqa: E402
from encoding_dsv4 import encode_messages  # noqa: E402


def sample(logits, temperature: float = 1.0):
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate_tokens(model, prompt_tokens, max_new_tokens, eos_id, temperature):
    total_len = min(model.max_seq_len, max_new_tokens + len(prompt_tokens))
    tokens = torch.full((1, total_len), -1, dtype=torch.long)
    tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long)
    n_prompt = len(prompt_tokens)
    prev_pos = 0
    cur_pos = n_prompt
    for cur_pos in range(n_prompt, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        nxt = sample(logits, temperature) if temperature > 0 else logits.argmax(dim=-1)
        tokens[0, cur_pos] = nxt[0]
        if nxt[0].item() == eos_id:
            break
        prev_pos = cur_pos
    return tokens[0, n_prompt:cur_pos + 1].tolist()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--config", default=str(REPO / "inference" / "config.json"))
    p.add_argument("--n-experts-score", type=int, required=True)
    p.add_argument("--prompt", required=True, help="User-message content")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    p.add_argument("--reasoning-effort", default=None)
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
    print(f"[{time.time()-t0:.1f}s] built model", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    eos_id = tokenizer.eos_token_id

    t0 = time.time()
    shard_files = sorted(Path(args.ckpt_path).glob("model*-mp*.safetensors"))
    load_direct(model, [str(s) for s in shard_files])
    print(f"[{time.time()-t0:.1f}s] loaded weights", flush=True)
    torch.set_default_device("cuda")
    model.eval()

    chat_text = encode_messages(
        [{"role": "user", "content": args.prompt}],
        thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort,
    )
    prompt_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    print(f"chat prompt ({len(prompt_ids)} tokens, max_seq_len={args.max_seq_len})", flush=True)

    t0 = time.time()
    out_ids = generate_tokens(
        model, prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=eos_id,
        temperature=args.temperature,
    )
    dt = time.time() - t0
    completion = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(f"\n[{dt:.1f}s to generate {len(out_ids)} tokens, {len(out_ids)/dt:.2f} tok/s]")
    print()
    print("=== USER ===")
    print(args.prompt)
    print("=== ASSISTANT ===")
    print(completion)
    return 0


if __name__ == "__main__":
    sys.exit(main())
