import json
import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file

LAYER_ID_RE = re.compile(r"^(layers|mtp)\.(\d+)\.")
EXPERT_ID_RE = re.compile(r"(^|\.)experts\.(\d+)\.")


def parse_layer_key(name: str, n_layers: int) -> tuple[str, int | None, int]:
    """Return (prefix, layer_id, effective_layer_id) for a tensor name.

    - prefix is 'layers' or 'mtp' or '' for global tensors (embed, head, etc.)
    - layer_id is the raw layer index within that prefix (or None for global)
    - effective_layer_id is the id used to decide hash-vs-learned:
        layers.L  -> L
        mtp.M     -> n_layers + M     (always >= n_hash_layers, always learned)
        global    -> -1
    """
    m = LAYER_ID_RE.search(name)
    if m is None:
        return "", None, -1
    prefix = m.group(1)
    lid = int(m.group(2))
    eff = lid if prefix == "layers" else n_layers + lid
    return prefix, lid, eff


def load_keep_indices(path: str | None) -> dict[str, list[int]]:
    """Load per-layer keep-index JSON, with keys like 'layers.3' or 'mtp.0'
    mapping to sorted lists of original expert indices to keep."""
    if path is None:
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        idxs = sorted(int(x) for x in v)
        # Require sorted deduped list; this keeps the renumbering deterministic
        # and makes the 'new_index = keep.index(old)' O(log n) via bisect below.
        assert len(set(idxs)) == len(idxs), f"duplicate indices for {k}"
        out[k] = idxs
    return out


FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Casts a tensor from e2m1fn to e4m3fn losslessly.
    """
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low  = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)
    # 6.0 * 2^6 = 384 < 448; 6.0 * 2^7 = 768 > 448; so MAX_OFFSET_BITS = 6
    MAX_OFFSET_BITS = 6

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    # bOut, bIn, 128, 128
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    # bOut, bIn, 128*4
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    ## bOut, bIn, 1
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    # bOut, bIn, 128*4
    offset = scale / scale_max_offset_bits
    # bOut, bIn, 128, 128
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),

    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def main(
    hf_ckpt_path,
    save_path,
    n_experts,
    mp,
    expert_dtype,
    n_experts_score=None,
    n_hash_layers=3,
    n_layers=43,
    keep_strategy="first_n",
    keep_indices_path=None,
):
    """
    Converts and saves model checkpoint files into the reference format.

    For DeepSeek-V4-Flash half-expert pruning: hash-routed layers
    (layer_id < n_hash_layers) keep all `n_experts` routed experts because
    `tid2eid` indexes the full table. Learned-router layers keep only
    `n_experts_score` experts, chosen by `keep_strategy`:
      - "first_n": keep expert ids 0..n_experts_score-1
      - "indices": keep ids listed in `keep_indices_path` (JSON keyed by
                   "layers.L" / "mtp.M"); non-contiguous sets are renumbered
                   to a contiguous [0, n_experts_score) range.
    MTP blocks are treated as learned-router layers (they have no tid2eid).
    """
    if n_experts_score is None:
        n_experts_score = n_experts  # no pruning
    assert n_experts_score <= n_experts
    assert n_experts_score % mp == 0 and n_experts % mp == 0
    torch.set_num_threads(8)
    n_local_experts_hash = n_experts // mp
    n_local_experts_score = n_experts_score // mp
    keep_indices = load_keep_indices(keep_indices_path)
    if keep_strategy == "indices":
        assert keep_indices, "--keep-strategy=indices requires --keep-indices-json"

    state_dicts = [{} for _ in range(mp)]
    # stats for post-run assertions
    kept_experts_per_layer_key: dict[str, set[int]] = {}

    for file_path in tqdm(sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
                    continue
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):    # without .weight
                    key = name.split(".")[-1]
                else:
                    key = name.split(".")[-2]
                if key in mapping:
                    new_key, dim = mapping[key]
                else:
                    new_key, dim = key, None
                name = name.replace(key, new_key)

                # Per-layer hash vs learned classification (after rename).
                prefix, lid, eff_lid = parse_layer_key(name, n_layers)
                is_hash_layer = (prefix == "layers") and (lid is not None) and (lid < n_hash_layers)
                layer_key = f"{prefix}.{lid}" if prefix else None

                is_expert_tensor = "experts" in name and "shared_experts" not in name
                is_gate_table = name.endswith(".gate.weight") or name.endswith(".gate.bias")

                # ---- expert drop / renumber (learned layers only) ----
                new_idx_for_mp: int | None = None
                if is_expert_tensor:
                    m = EXPERT_ID_RE.search(name)
                    assert m is not None, f"expert tensor with no id: {name}"
                    old_idx = int(m.group(2))
                    if is_hash_layer:
                        new_idx_for_mp = old_idx
                    else:
                        if keep_strategy == "first_n":
                            if old_idx >= n_experts_score:
                                continue  # drop
                            new_idx = old_idx
                        else:  # "indices"
                            keep = keep_indices.get(layer_key)
                            assert keep is not None, f"keep_indices missing entry for {layer_key}"
                            assert len(keep) == n_experts_score, (
                                f"{layer_key}: expected {n_experts_score} kept indices, got {len(keep)}"
                            )
                            try:
                                new_idx = keep.index(old_idx)
                            except ValueError:
                                continue  # drop expert not in keep set
                        name = EXPERT_ID_RE.sub(rf"\1experts.{new_idx}.", name, count=1)
                        new_idx_for_mp = new_idx
                    kept_experts_per_layer_key.setdefault(layer_key, set()).add(old_idx)

                # ---- gate.weight / gate.bias row slicing (learned layers only) ----
                if is_gate_table and not is_hash_layer and prefix:
                    if keep_strategy == "first_n":
                        param = param[:n_experts_score].contiguous()
                    else:
                        keep = keep_indices.get(layer_key)
                        assert keep is not None, f"keep_indices missing entry for {layer_key}"
                        assert len(keep) == n_experts_score, (
                            f"{layer_key}: expected {n_experts_score} kept indices, got {len(keep)}"
                        )
                        param = param[keep].contiguous()

                for i in range(mp):
                    new_param = param
                    if is_expert_tensor:
                        layer_n_local = n_local_experts_hash if is_hash_layer else n_local_experts_score
                        if new_idx_for_mp < i * layer_n_local or new_idx_for_mp >= (i + 1) * layer_n_local:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    # ---- post-run invariant checks ----
    for layer_key, kept in kept_experts_per_layer_key.items():
        prefix_k, _, lid_s = layer_key.partition(".")
        lid_i = int(lid_s)
        if prefix_k == "layers" and lid_i < n_hash_layers:
            assert kept == set(range(n_experts)), (
                f"hash layer {layer_key} missing experts: kept={sorted(kept)[:5]}.. "
                f"({len(kept)}/{n_experts})"
            )
        else:
            assert len(kept) == n_experts_score, (
                f"learned layer {layer_key}: kept {len(kept)} experts, expected {n_experts_score}"
            )

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        names = list(state_dicts[i].keys())
        for name in names:
            if name.endswith("wo_a.weight"):
                weight = state_dicts[i][name]
                scale = state_dicts[i].pop(name.replace("weight", "scale"))
                weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()
                state_dicts[i][name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
            elif "experts" in name and state_dicts[i][name].dtype == torch.int8:
                if expert_dtype == "fp8":
                    scale_name = name.replace("weight", "scale")
                    weight = state_dicts[i].pop(name)
                    scale = state_dicts[i].pop(scale_name)
                    state_dicts[i][name], state_dicts[i][scale_name] = cast_e2m1fn_to_e4m3fn(weight, scale)
                else:
                    state_dicts[i][name] = state_dicts[i][name].view(torch.float4_e2m1fn_x2)
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file in ["tokenizer.json", "tokenizer_config.json"]:
        old_file_path = os.path.join(hf_ckpt_path, file)
        new_file_path = os.path.join(save_path, file)
        if os.path.exists(old_file_path):
            shutil.copyfile(old_file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True,
                        help="Total routed experts in the source checkpoint (usually 256).")
    parser.add_argument("--n-experts-score", type=int, default=None,
                        help="Pruned expert count for learned-router layers. "
                             "Defaults to --n-experts (no pruning).")
    parser.add_argument("--n-hash-layers", type=int, default=3,
                        help="Number of hash-routed layers from the start (unaffected by pruning).")
    parser.add_argument("--n-layers", type=int, default=43,
                        help="Number of transformer body layers (used to map MTP ids).")
    parser.add_argument("--keep-strategy", choices=["first_n", "indices"], default="first_n",
                        help="How to choose which experts to keep in learned layers.")
    parser.add_argument("--keep-indices-json", type=str, default=None,
                        help="JSON file mapping 'layers.L' / 'mtp.M' -> list of original "
                             "expert ids to keep; required when --keep-strategy=indices.")
    parser.add_argument("--model-parallel", type=int, required=True)
    parser.add_argument("--expert-dtype", type=str, choices=["fp8", "fp4"], required=False, default=None)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    if args.n_experts_score is not None:
        assert args.n_experts_score % args.model_parallel == 0, \
            "n_experts_score must be divisible by model parallelism"
        assert args.n_experts_score <= args.n_experts
    main(
        args.hf_ckpt_path,
        args.save_path,
        args.n_experts,
        args.model_parallel,
        args.expert_dtype,
        n_experts_score=args.n_experts_score,
        n_hash_layers=args.n_hash_layers,
        n_layers=args.n_layers,
        keep_strategy=args.keep_strategy,
        keep_indices_path=args.keep_indices_json,
    )
