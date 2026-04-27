"""REAP (Router-weighted Expert Activation Pruning) scoring on V4-Flash.

Reference: https://arxiv.org/html/2510.13999v1

Per-expert metric in a layer:
    S_j = mean over tokens routed to expert j of:  g_j(x) · ||f_j(x)||_2
where g_j is the gate's routing weight to expert j and f_j(x) is expert j's
output before routing-weight scaling.

Architecture: pipeline-parallel + pure REAP + length-sorted batched calibration.

For each layer L in order (embed → 0..n_layers-1 → mtp.0):
  - Build a fresh standalone Block (or ParallelEmbedding) at full 256 routed
    experts (~4 GB resident per Block).
  - Load layer L's weights from upstream HF (the unpruned 256 versions).
  - Forward calibration (length-sorted batches) through this layer, reading
    per-sequence input activations from the previous layer's on-disk cache and
    writing per-sequence output activations to the next layer's cache.
  - For score layers: hook MoE.forward to record REAP scores from valid
    (non-padded) positions only.
  - Score layer L → pick top-N → write keep_indices[L]. Free the block.

Pure REAP (each layer scored against UNPRUNED upstream) eliminates the
compounding bias of sequential-pruning approaches. Memory peak is ~10-15 GB
regardless of which layer we're on.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "inference"))

import model as inference_model  # noqa: E402
from model import (  # noqa: E402
    Block, MoE, ModelArgs, MTPBlock, ParallelEmbedding,
)
from convert import mapping as CONVERT_NAME_MAPPING  # noqa: E402


def init_inference_module_globals(args: ModelArgs) -> None:
    """Set the module-level globals in inference/model.py that Transformer.__init__
    normally sets. We need this because the pipelined sweep builds standalone
    Blocks/Embeds without ever instantiating a full Transformer — so default_dtype,
    scale_fmt, and scale_dtype would otherwise stay at their stale module defaults
    (which would mismatch the loaded weight scales and trip the fp4_gemm kernel).

    Mirrors Transformer.__init__:782-789.
    """
    inference_model.world_size = 1
    inference_model.rank = 0
    inference_model.default_dtype = (
        torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    )
    inference_model.scale_fmt = (
        "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
    )
    inference_model.scale_dtype = (
        torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
    )


# ---------------------------------------------------------------------------
# Score buffer
# ---------------------------------------------------------------------------

@dataclass
class ScoreBuffer:
    """Per-expert REAP accumulator for one MoE layer (CPU-resident, float64)."""
    n_routed_experts: int
    reap_sum: torch.Tensor = field(init=False)
    reap_count: torch.Tensor = field(init=False)
    gate_weight_sum: torch.Tensor = field(init=False)
    output_norm_sum: torch.Tensor = field(init=False)

    def __post_init__(self):
        n = self.n_routed_experts
        self.reap_sum = torch.zeros(n, dtype=torch.float64)
        self.reap_count = torch.zeros(n, dtype=torch.float64)
        self.gate_weight_sum = torch.zeros(n, dtype=torch.float64)
        self.output_norm_sum = torch.zeros(n, dtype=torch.float64)

    def add(self, expert_id: int, gate_weights: torch.Tensor, output_norms: torch.Tensor) -> None:
        n = int(gate_weights.numel())
        if n == 0:
            return
        gw = gate_weights.detach().to("cpu", dtype=torch.float64)
        on = output_norms.detach().to("cpu", dtype=torch.float64)
        self.reap_sum[expert_id] += float((gw * on).sum().item())
        self.reap_count[expert_id] += n
        self.gate_weight_sum[expert_id] += float(gw.sum().item())
        self.output_norm_sum[expert_id] += float(on.sum().item())

    def compute_reap(self) -> torch.Tensor:
        s = torch.zeros_like(self.reap_sum)
        nz = self.reap_count > 0
        s[nz] = self.reap_sum[nz] / self.reap_count[nz]
        return s

    def diagnostics(self) -> dict:
        sel = self.reap_count
        return {
            "min_selections": int(sel.min().item()),
            "max_selections": int(sel.max().item()),
            "mean_selections": float(sel.mean().item()),
            "n_never_selected": int((sel == 0).sum().item()),
            "n_routed_experts": int(self.n_routed_experts),
        }

    def to_state(self) -> dict:
        """Serialize raw aggregates (sum/count) for resume persistence."""
        return {
            "n_routed_experts": int(self.n_routed_experts),
            "reap_sum": self.reap_sum.tolist(),
            "reap_count": self.reap_count.tolist(),
            "gate_weight_sum": self.gate_weight_sum.tolist(),
            "output_norm_sum": self.output_norm_sum.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "ScoreBuffer":
        buf = cls(n_routed_experts=int(state["n_routed_experts"]))
        buf.reap_sum = torch.tensor(state["reap_sum"], dtype=torch.float64)
        buf.reap_count = torch.tensor(state["reap_count"], dtype=torch.float64)
        buf.gate_weight_sum = torch.tensor(state["gate_weight_sum"], dtype=torch.float64)
        buf.output_norm_sum = torch.tensor(state["output_norm_sum"], dtype=torch.float64)
        return buf

    def add_state(self, other: "ScoreBuffer") -> None:
        """Accumulate another buffer's aggregates in place (for cross-superset summing)."""
        assert self.n_routed_experts == other.n_routed_experts
        self.reap_sum += other.reap_sum
        self.reap_count += other.reap_count
        self.gate_weight_sum += other.gate_weight_sum
        self.output_norm_sum += other.output_norm_sum


# ---------------------------------------------------------------------------
# HF safetensors index + name remapping (mirrors convert.py)
# ---------------------------------------------------------------------------

class HFShardIndex:
    """Maps every tensor name in the upstream HF shards to (path, byte_range, dtype, shape)."""

    _DTYPE_MAP = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2, "F8_E8M0": torch.float8_e8m0fnu,
        "F4": torch.float4_e2m1fn_x2,
        "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
        "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }

    def __init__(self, hf_ckpt_path: str):
        self.hf_path = Path(hf_ckpt_path)
        self.entries: dict[str, tuple[Path, int, int, str, tuple]] = {}
        for shard in sorted(self.hf_path.glob("*.safetensors")):
            fd = os.open(str(shard), os.O_RDONLY)
            try:
                (hdr_len,) = struct.unpack("<Q", os.pread(fd, 8, 0))
                hdr = json.loads(os.pread(fd, hdr_len, 8).decode("utf-8"))
                hdr.pop("__metadata__", None)
                data_start = 8 + hdr_len
                for name, spec in hdr.items():
                    start, end = spec["data_offsets"]
                    self.entries[name] = (
                        shard, data_start + start, end - start, spec["dtype"], tuple(spec["shape"])
                    )
            finally:
                os.close(fd)

    def read_tensor(self, name: str) -> torch.Tensor:
        shard, off, nbytes, dtype_str, shape = self.entries[name]
        dtype = self._DTYPE_MAP[dtype_str]
        if dtype_str == "F4":
            shape = list(shape)
            shape[-1] = shape[-1] // 2
            shape = tuple(shape)
        fd = os.open(str(shard), os.O_RDONLY)
        try:
            buf = os.pread(fd, nbytes, off)
            t = torch.frombuffer(buf, dtype=dtype).reshape(shape).clone()
            try:
                os.posix_fadvise(fd, off, nbytes, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
            return t
        finally:
            os.close(fd)


def _hf_to_ref_name(hf_name: str) -> str:
    """Apply convert.py's HF -> reference name remapping (mirrors convert.py:167-183)."""
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]
    if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
        return ""
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    if key in CONVERT_NAME_MAPPING:
        new_key, _ = CONVERT_NAME_MAPPING[key]
        name = name.replace(key, new_key)
    return name


def _build_ref_to_hf_map(idx: HFShardIndex) -> dict[str, str]:
    out = {}
    for hf_name in idx.entries:
        ref = _hf_to_ref_name(hf_name)
        if ref:
            out[ref] = hf_name
    return out


# ---------------------------------------------------------------------------
# Length-sorted batching
# ---------------------------------------------------------------------------

def make_length_sorted_batches(seqs: List[torch.Tensor], batch_size: int) -> List[List[int]]:
    """Sort sequence indices by length, bin into batches. Within a bin, padding waste = (max_len - min_len) per slot."""
    sorted_idx = sorted(range(len(seqs)), key=lambda i: int(seqs[i].numel()))
    return [sorted_idx[i:i + batch_size] for i in range(0, len(sorted_idx), batch_size)]


def pad_input_ids(seqs_subset: List[torch.Tensor], pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pad token-id sequences to the longest in the batch. Returns (ids[b,L], valid[b,L] bool, lengths)."""
    max_len = max(int(s.numel()) for s in seqs_subset)
    bsz = len(seqs_subset)
    padded = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    valid = torch.zeros(bsz, max_len, dtype=torch.bool)
    lengths = []
    for j, s in enumerate(seqs_subset):
        n = int(s.numel())
        padded[j, :n] = s
        valid[j, :n] = True
        lengths.append(n)
    return padded, valid, lengths


def pad_h_batch(h_list: List[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    """Pad a list of [seqlen_i, hc, dim] tensors to a [bsz, max_len, hc, dim] batch."""
    assert all(t.dim() == 3 for t in h_list)
    max_len = max(int(t.shape[0]) for t in h_list)
    hc = h_list[0].shape[1]
    dim = h_list[0].shape[2]
    out = torch.zeros(len(h_list), max_len, hc, dim, dtype=h_list[0].dtype, device=h_list[0].device)
    lengths = []
    for j, t in enumerate(h_list):
        n = int(t.shape[0])
        out[j, :n] = t
        lengths.append(n)
    return out, lengths


# ---------------------------------------------------------------------------
# On-disk activation cache (per-sequence files; rolled forward layer by layer)
# ---------------------------------------------------------------------------

def cache_dir_for(out_dir: str | Path, layer_idx: int, superset_idx: int = 0) -> Path:
    """Per-superset, per-layer input activation cache.
    layer_idx=0 means input to layer 0 (= embed output).
    superset_idx isolates each outer-superset's caches so disk peak stays bounded."""
    return Path(out_dir) / "activations" / f"superset_{superset_idx:02d}" / f"input_to_layer_{layer_idx:02d}"


@torch.no_grad()
def save_one(t: torch.Tensor, cache_dir: str | Path, seq_idx: int) -> int:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    cpu_t = t.detach().cpu().contiguous()
    torch.save(cpu_t, p / f"seq_{seq_idx:06d}.pt")
    return int(cpu_t.numel()) * int(cpu_t.element_size())


@torch.no_grad()
def load_one(cache_dir: str | Path, seq_idx: int) -> torch.Tensor:
    return torch.load(str(Path(cache_dir) / f"seq_{seq_idx:06d}.pt"), weights_only=False)


def cache_size_bytes(cache_dir: str | Path) -> int:
    p = Path(cache_dir)
    return sum(f.stat().st_size for f in p.glob("seq_*.pt")) if p.exists() else 0


def cache_count(cache_dir: str | Path) -> int:
    p = Path(cache_dir)
    return sum(1 for _ in p.glob("seq_*.pt")) if p.exists() else 0


def delete_cache_dir(cache_dir: str | Path) -> None:
    p = Path(cache_dir)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def fmt_gb(nbytes: int) -> str:
    return f"{nbytes / 1024 / 1024 / 1024:.2f} GB"


# ---------------------------------------------------------------------------
# Standalone block / embed builders
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_unpruned_block(layer_id: int, base_args: ModelArgs, device: str = "cuda") -> Block:
    """Construct a Block at full 256 routed experts (overrides any pruning in base_args)."""
    args = ModelArgs(**{**base_args.__dict__, "n_routed_experts_score": 256})
    with torch.device(device):
        block = Block(layer_id, args)
    return block


@torch.no_grad()
def load_block_from_hf(block: Block, layer_id: int, idx: HFShardIndex, layer_prefix: str = "layers"
) -> tuple[int, int]:
    """Load all weights for one standalone Block from upstream HF.

    layer_prefix: "layers" for the body, "mtp" for the MTP block.
    Returns (n_loaded, n_skipped).
    """
    ref_to_hf = _build_ref_to_hf_map(idx)
    sd = block.state_dict()
    n_loaded, n_skipped = 0, 0
    for ref_name, target in sd.items():
        full_ref = f"{layer_prefix}.{layer_id}.{ref_name}"
        hf_name = ref_to_hf.get(full_ref)
        if hf_name is None:
            n_skipped += 1
            continue
        t = idx.read_tensor(hf_name)
        if t.dtype == torch.int8 and target.dtype == torch.float4_e2m1fn_x2:
            t = t.view(torch.float4_e2m1fn_x2)
        if t.shape != target.shape:
            n_skipped += 1
            continue
        target.copy_(t.to(target.device, dtype=target.dtype))
        n_loaded += 1
    return n_loaded, n_skipped


@torch.no_grad()
def build_unpruned_mtp_block(layer_id: int, base_args: ModelArgs, device: str = "cuda") -> MTPBlock:
    """Construct an MTPBlock at full 256 routed experts. layer_id should be n_layers+0=43
    so Block.__init__ takes the score-routed branch and picks compress_ratios[43]=0."""
    args = ModelArgs(**{**base_args.__dict__, "n_routed_experts_score": 256})
    with torch.device(device):
        block = MTPBlock(layer_id, args)
    return block


@torch.no_grad()
def build_and_load_embed(base_args: ModelArgs, idx: HFShardIndex, device: str = "cuda") -> ParallelEmbedding:
    with torch.device(device):
        emb = ParallelEmbedding(base_args.vocab_size, base_args.dim)
    ref_to_hf = _build_ref_to_hf_map(idx)
    hf_name = ref_to_hf.get("embed.weight")
    if hf_name is None:
        raise RuntimeError("HF source missing embed.weight (after rename)")
    t = idx.read_tensor(hf_name)
    emb.weight.data.copy_(t.to(emb.weight.device, dtype=emb.weight.dtype))
    return emb


# ---------------------------------------------------------------------------
# Hooked MoE.forward — masks padded positions from REAP score accumulation
# ---------------------------------------------------------------------------

def _make_hooked_moe_forward(score_buf: ScoreBuffer, valid_mask_flat: torch.Tensor) -> Callable:
    """Replace MoE.forward to record REAP scores ONLY at valid (non-padded) positions.

    Output for ALL positions (including padded) is still computed so the batched
    shape stays correct; we just don't accumulate scores for padded slots, and
    we drop padded positions when saving outputs to disk.
    """
    def hooked_forward(self: MoE, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            w_per_token = weights[idx, top]
            raw_out = expert(x[idx])
            y[idx] += w_per_token.unsqueeze(-1) * raw_out
            v = valid_mask_flat[idx]
            if v.any():
                score_buf.add(i, w_per_token[v], raw_out.float().norm(dim=-1)[v])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)
    return hooked_forward


# ---------------------------------------------------------------------------
# Per-layer step
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_chunk(
    layer_ids: List[int],
    layer_prefix: str,
    base_args: ModelArgs,
    idx: HFShardIndex,
    calib_seqs: List[torch.Tensor],
    batches: List[List[int]],
    in_cache: Path,
    out_cache: Path,
    score_each: List[bool],
    n_keep: int = 128,
    log_every_n_batches: int = 32,
    arch_layer_ids: Optional[List[int]] = None,
    external_score_bufs: Optional[dict] = None,
) -> dict[int, dict]:
    """Forward all calibration batches through a CHUNK of N consecutive Blocks.

    Holds N standalone unpruned Blocks resident at once. Disk I/O happens once
    per chunk (read in_cache at start, write out_cache at end), so total disk
    traffic is ~N× lower than processing one layer at a time. Each individual
    layer is still scored separately if its score_each entry is True.

    Memory cost: ~N × 4 GB blocks + small per-batch activations.

    `layer_ids` are HF-source ids (used both to look up tensors at
    `{layer_prefix}.{id}.*` and to key the returned dict). `arch_layer_ids` are
    the architecture ids passed to `Block.__init__` — usually the same as
    `layer_ids`, but DIFFERENT for the MTP block: HF id=0 (under "mtp.0.*")
    while the architecture id must be `n_layers + 0` so Block's own per-layer
    branching (compress_ratios index, hash-vs-score MoE) lines up with how the
    full Transformer would build it.

    Returns {layer_id: score_artifact_dict} keyed by the HF-side `layer_ids`.
    """
    assert len(layer_ids) == len(score_each)
    assert len(layer_ids) > 0
    if arch_layer_ids is None:
        arch_layer_ids = layer_ids
    assert len(arch_layer_ids) == len(layer_ids)

    # 1. Build all blocks in chunk.
    print(f"  [chunk {layer_prefix}.{layer_ids[0]}..{layer_ids[-1]}] "
          f"building {len(layer_ids)} unpruned blocks from HF ...", flush=True)
    t0 = time.time()
    blocks: list[Block] = []
    # External score_bufs (if any) persist across calls — caller manages lifecycle.
    score_bufs: dict[int, ScoreBuffer] = external_score_bufs if external_score_bufs is not None else {}
    for hf_L, arch_L, sc in zip(layer_ids, arch_layer_ids, score_each):
        block = build_unpruned_block(arch_L, base_args)
        n_loaded, _ = load_block_from_hf(block, hf_L, idx, layer_prefix=layer_prefix)
        blocks.append(block)
        if sc and hf_L not in score_bufs:
            score_bufs[hf_L] = ScoreBuffer(n_routed_experts=block.ffn.n_routed_experts)
    print(f"  [chunk] all {len(layer_ids)} blocks loaded in {time.time()-t0:.1f}s "
          f"(scoring: {sorted(L for L, sc in zip(layer_ids, score_each) if sc)})", flush=True)

    # 2. Forward all batches through entire chunk in RAM, save final output.
    out_cache.mkdir(parents=True, exist_ok=True)
    t_fwd = time.time()
    for bi, batch_indices in enumerate(batches):
        seqs = [calib_seqs[i] for i in batch_indices]
        ids_padded, valid_mask, lengths = pad_input_ids(seqs)
        ids_padded = ids_padded.to("cuda")
        valid_mask = valid_mask.to("cuda")
        valid_mask_flat = valid_mask.flatten()

        h_list = [load_one(in_cache, i).to("cuda", non_blocking=False) for i in batch_indices]
        h, _ = pad_h_batch(h_list)
        del h_list

        for L, block in zip(layer_ids, blocks):
            if L in score_bufs:
                block.ffn.forward = _make_hooked_moe_forward(
                    score_bufs[L], valid_mask_flat
                ).__get__(block.ffn, MoE)
            h = block(h, 0, ids_padded)

        for j, seq_idx in enumerate(batch_indices):
            save_one(h[j, :lengths[j]], out_cache, seq_idx)
        del h

        if (bi + 1) % log_every_n_batches == 0:
            elapsed = time.time() - t_fwd
            rate = (bi + 1) / elapsed
            print(f"  [chunk {layer_ids[0]}..{layer_ids[-1]}] batches {bi+1}/{len(batches)} "
                  f"({rate:.2f} batch/s, {elapsed:.0f}s)", flush=True)

    print(f"  [chunk] forward sweep done in {time.time()-t_fwd:.1f}s "
          f"(out cache {fmt_gb(cache_size_bytes(out_cache))})", flush=True)

    # 3. Compute scores per layer, free blocks.
    out: dict[int, dict] = {}
    for L in layer_ids:
        if L not in score_bufs:
            continue
        scores = score_bufs[L].compute_reap()
        top = torch.topk(scores, n_keep)
        out[L] = {
            "scores": scores.tolist(),
            "kept": sorted(top.indices.tolist()),
            "n_keep": n_keep,
            "diagnostics": score_bufs[L].diagnostics(),
        }

    for block in blocks:
        del block
    blocks.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return out


@torch.no_grad()
def process_mtp_chunk(
    base_args: ModelArgs,
    idx: HFShardIndex,
    calib_seqs: List[torch.Tensor],
    batches: List[List[int]],
    in_cache: Path,
    out_cache: Path,
    n_keep: int = 128,
    log_every_n_batches: int = 32,
    external_score_buf: Optional[ScoreBuffer] = None,
) -> dict:
    """Score the MTP block using the REAL MTPBlock.forward path.

    Differences vs body process_chunk:
      - Builds MTPBlock (which adds e_proj, h_proj, enorm, hnorm, MTP-specific HC-head params)
      - Wires `block.embed` from a freshly-loaded ParallelEmbedding so MTPBlock.forward's
        `e = self.embed(input_ids)` works
      - Skips the final `head()` call (we don't need logits for REAP scoring;
        the MoE we hook fires inside the inherited Block.forward)
      - This means the MoE sees the proper `e_proj(e) + h_proj(h)` mixed input,
        not the body-style raw h
    """
    print(f"  [mtp.0] build + load 256-expert MTPBlock from HF ...", flush=True)
    t0 = time.time()
    block = build_unpruned_mtp_block(layer_id=base_args.n_layers, base_args=base_args)
    n_loaded, _ = load_block_from_hf(block, hf_layer_id=0, idx=idx, layer_prefix="mtp")
    block.embed = build_and_load_embed(base_args, idx)
    # `block.head` is required by MTPBlock.forward's assert and final call, but we
    # never reach the head call in our custom forward — leave as-is.
    print(f"  [mtp.0] loaded {n_loaded} MTPBlock tensors + embed in "
          f"{time.time()-t0:.1f}s", flush=True)

    # Caller-managed score buf if provided (for accumulating across supersets).
    score_buf = external_score_buf if external_score_buf is not None else ScoreBuffer(
        n_routed_experts=block.ffn.n_routed_experts
    )

    # Custom forward that runs MTP-specific input mixing + the inherited Block path,
    # but stops before the head call (we only need MoE scoring, not logits).
    def mtp_forward_for_scoring(self, x, start_pos, input_ids):
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        # super().forward = Block.forward (attention + MoE + HC residual)
        return Block.forward(self, x, start_pos, input_ids)

    out_cache.mkdir(parents=True, exist_ok=True)
    t_fwd = time.time()
    for bi, batch_indices in enumerate(batches):
        seqs = [calib_seqs[i] for i in batch_indices]
        ids_padded, valid_mask, lengths = pad_input_ids(seqs)
        ids_padded = ids_padded.to("cuda")
        valid_mask = valid_mask.to("cuda")

        h_list = [load_one(in_cache, i).to("cuda") for i in batch_indices]
        h, _ = pad_h_batch(h_list)
        del h_list

        # Install hooked MoE.forward bound to this batch's valid_mask.
        block.ffn.forward = _make_hooked_moe_forward(
            score_buf, valid_mask.flatten()
        ).__get__(block.ffn, MoE)

        h_out = mtp_forward_for_scoring(block, h, 0, ids_padded)

        for j, seq_idx in enumerate(batch_indices):
            save_one(h_out[j, :lengths[j]], out_cache, seq_idx)
        del h, h_out

        if (bi + 1) % log_every_n_batches == 0:
            elapsed = time.time() - t_fwd
            print(f"  [mtp.0] batches {bi+1}/{len(batches)} "
                  f"({(bi+1)/elapsed:.2f} batch/s, {elapsed:.0f}s)", flush=True)

    print(f"  [mtp.0] forward sweep done in {time.time()-t_fwd:.1f}s", flush=True)

    scores = score_buf.compute_reap()
    top = torch.topk(scores, n_keep)
    out = {
        "scores": scores.tolist(),
        "kept": sorted(top.indices.tolist()),
        "n_keep": n_keep,
        "diagnostics": score_buf.diagnostics(),
    }
    del block
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def bootstrap_embed_to_disk(
    base_args: ModelArgs,
    idx: HFShardIndex,
    calib_seqs: List[torch.Tensor],
    batches: List[List[int]],
    out_cache: Path,
) -> None:
    print(f"[embed] forwarding {len(calib_seqs)} sequences in {len(batches)} batches",
          flush=True)
    t0 = time.time()
    emb = build_and_load_embed(base_args, idx)
    out_cache.mkdir(parents=True, exist_ok=True)
    for bi, batch_indices in enumerate(batches):
        seqs = [calib_seqs[i] for i in batch_indices]
        ids_padded, _, lengths = pad_input_ids(seqs)
        ids_padded = ids_padded.to("cuda")
        h = emb(ids_padded)
        h = h.unsqueeze(2).repeat(1, 1, base_args.hc_mult, 1)
        for j, seq_idx in enumerate(batch_indices):
            save_one(h[j, :lengths[j]], out_cache, seq_idx)
        del h
        if (bi + 1) % 32 == 0:
            print(f"  [embed] {bi+1}/{len(batches)} batches "
                  f"({(bi+1)/(time.time()-t0):.1f} batch/s)", flush=True)
    del emb
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[embed] done in {time.time()-t0:.1f}s "
          f"({fmt_gb(cache_size_bytes(out_cache))} on disk)", flush=True)


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------

def _aggregate_to_artifact(buf: ScoreBuffer, n_keep: int) -> dict:
    """Compute final REAP score artifact from an accumulated ScoreBuffer."""
    scores = buf.compute_reap()
    top = torch.topk(scores, n_keep)
    return {
        "scores": scores.tolist(),
        "kept": sorted(top.indices.tolist()),
        "n_keep": n_keep,
        "diagnostics": buf.diagnostics(),
        "raw_state": buf.to_state(),  # for resume across supersets
    }


def _load_aggregate_buffer(path: Path) -> Optional[ScoreBuffer]:
    """Load a persisted aggregate ScoreBuffer from a chunk JSON, or None if absent / no raw_state."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    state = data.get("raw_state")
    if state is None:
        return None
    return ScoreBuffer.from_state(state)


def reap_full_sweep(
    hf_ckpt_path: str,
    base_args: ModelArgs,
    calib_seqs: List[torch.Tensor],
    out_dir: str,
    batch_size: int = 4,
    chunk_size: int = 8,
    n_keep: int = 128,
    process_mtp: bool = True,
    keep_intermediate_caches: bool = True,
    superset_size: Optional[int] = None,
) -> dict[str, dict]:
    """Pipeline-parallel + pure REAP + length-sorted batched calibration.

    With `superset_size`: split calibration into supersets of that many sequences
    each, process each superset through the full pipeline, accumulate REAP scores
    globally across supersets. Caps the inter-chunk activation disk cache to one
    superset's worth of activations instead of the full data — necessary for
    large recipes (reap_full ≈ 5 TB at 1 superset, fits at ≥3 supersets).

    `superset_size=None` (default) runs as a single superset = old behavior.

    Layout under out_dir:
        chunks/layers.{L}.json           per-layer aggregate state (cumulative across supersets)
        chunks/mtp.0.json                same for MTP
        chunks/supersets_done.json       list of completed superset indices (resume marker)
        activations/superset_NN/input_to_layer_NN/   per-superset, per-layer per-seq h
        keep_indices.json                final union written at end
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    chunks_dir = out_dir_p / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    init_inference_module_globals(base_args)

    print(f"[reap] indexing HF safetensors ...", flush=True)
    t0 = time.time()
    idx = HFShardIndex(hf_ckpt_path)
    print(f"[reap] indexed {len(idx.entries):,} tensors in {time.time()-t0:.1f}s", flush=True)

    # Build supersets.
    n_total = len(calib_seqs)
    if superset_size is None or superset_size >= n_total:
        supersets = [list(range(n_total))]
    else:
        supersets = [list(range(i, min(i + superset_size, n_total)))
                     for i in range(0, n_total, superset_size)]
    print(f"[reap] {len(supersets)} superset(s), avg {n_total/len(supersets):.0f} seqs each",
          flush=True)

    # Persistent score buffers across supersets. Initialize from any saved aggregate
    # state so resume continues from where we left off.
    # Pre-create empty buffers for ALL scoring layers so process_chunk / process_mtp_chunk
    # always have somewhere to accumulate (no None first-superset edge case).
    body_score_bufs: dict[int, ScoreBuffer] = {}
    for L in range(base_args.n_hash_layers, base_args.n_layers):
        loaded = _load_aggregate_buffer(chunks_dir / f"layers.{L}.json")
        body_score_bufs[L] = loaded if loaded is not None else ScoreBuffer(n_routed_experts=256)
    if process_mtp and base_args.n_mtp_layers > 0:
        loaded = _load_aggregate_buffer(chunks_dir / "mtp.0.json")
        mtp_score_buf: ScoreBuffer = loaded if loaded is not None else ScoreBuffer(n_routed_experts=256)
    else:
        mtp_score_buf = None

    # Track which supersets are complete (for resume).
    done_marker = chunks_dir / "supersets_done.json"
    done_supersets = set(json.loads(done_marker.read_text())) if done_marker.exists() else set()
    if done_supersets:
        print(f"[reap] resuming: {len(done_supersets)} of {len(supersets)} "
              f"supersets already complete", flush=True)

    # Process each superset.
    for ss_idx, ss_seq_indices in enumerate(supersets):
        if ss_idx in done_supersets:
            print(f"\n[reap] superset {ss_idx}/{len(supersets)-1}: SKIPPING (done)",
                  flush=True)
            continue

        ss_seqs = [calib_seqs[i] for i in ss_seq_indices]
        print(f"\n{'='*72}\n[reap] superset {ss_idx}/{len(supersets)-1}: "
              f"{len(ss_seqs)} sequences\n{'='*72}", flush=True)
        t_ss = time.time()

        _process_one_superset(
            ss_idx=ss_idx, ss_seqs=ss_seqs,
            base_args=base_args, idx=idx,
            out_dir=out_dir, chunks_dir=chunks_dir,
            batch_size=batch_size, chunk_size=chunk_size, n_keep=n_keep,
            process_mtp=process_mtp,
            body_score_bufs=body_score_bufs,
            mtp_score_buf=mtp_score_buf,   # mutated in place by process_mtp_chunk
        )

        # Mark superset done + persist aggregates (per-layer JSONs already written by callee).
        done_supersets.add(ss_idx)
        done_marker.write_text(json.dumps(sorted(done_supersets)))

        # Reclaim this superset's disk cache (its activations are no longer needed).
        if not keep_intermediate_caches:
            ss_act_dir = Path(out_dir) / "activations" / f"superset_{ss_idx:02d}"
            if ss_act_dir.exists():
                import shutil
                shutil.rmtree(ss_act_dir, ignore_errors=True)
                print(f"[reap] reclaimed superset {ss_idx} activation cache", flush=True)

        print(f"\n[reap] superset {ss_idx} done in {(time.time()-t_ss)/60:.1f} min", flush=True)

    # Final aggregate → keep_indices + per-layer artifacts.
    out: dict[str, dict] = {}
    for L, buf in sorted(body_score_bufs.items()):
        out[f"layers.{L}"] = _aggregate_to_artifact(buf, n_keep)
    if mtp_score_buf is not None:
        out["mtp.0"] = _aggregate_to_artifact(mtp_score_buf, n_keep)
    keep_indices = {k: v["kept"] for k, v in out.items()}
    (out_dir_p / "keep_indices.json").write_text(json.dumps(keep_indices, indent=2))
    return out


def _process_one_superset(
    ss_idx: int,
    ss_seqs: List[torch.Tensor],
    base_args: ModelArgs,
    idx: HFShardIndex,
    out_dir: str,
    chunks_dir: Path,
    batch_size: int,
    chunk_size: int,
    n_keep: int,
    process_mtp: bool,
    body_score_bufs: dict,
    mtp_score_buf: Optional[ScoreBuffer],
) -> None:
    """Forward one superset through embed → body chunks → MTP, accumulating into shared
    score buffers. Per-superset disk caches are isolated so peak disk = one superset only."""
    n_seqs = len(ss_seqs)
    print(f"[reap] making length-sorted batches: bs={batch_size}", flush=True)
    batches = make_length_sorted_batches(ss_seqs, batch_size)
    print(f"[reap] {len(batches)} batches (avg {n_seqs/len(batches):.1f} seqs/batch)",
          flush=True)

    # Bootstrap embed for THIS superset (cheap, ~1 min for 5K sequences).
    cache_0 = cache_dir_for(out_dir, 0, superset_idx=ss_idx)
    if cache_count(cache_0) == n_seqs:
        print(f"[reap] superset {ss_idx} embed cache present, skipping", flush=True)
    else:
        delete_cache_dir(cache_0)
        bootstrap_embed_to_disk(base_args, idx, ss_seqs, batches, cache_0)

    print(f"[reap] chunk_size={chunk_size} → "
          f"{(base_args.n_layers + chunk_size - 1) // chunk_size} body chunks", flush=True)

    # Detect latest valid input cache *for this superset* for resume within a superset.
    latest_cache_layer: Optional[int] = None
    for L in range(base_args.n_layers + 1, -1, -1):
        if cache_count(cache_dir_for(out_dir, L, superset_idx=ss_idx)) == n_seqs:
            latest_cache_layer = L
            break

    # Body chunks.
    for chunk_start in range(0, base_args.n_layers, chunk_size):
        layer_ids = list(range(chunk_start, min(chunk_start + chunk_size, base_args.n_layers)))
        score_each = [L >= base_args.n_hash_layers for L in layer_ids]
        in_cache = cache_dir_for(out_dir, chunk_start, superset_idx=ss_idx)
        out_cache = cache_dir_for(out_dir, chunk_start + len(layer_ids), superset_idx=ss_idx)

        # We always re-forward inside a superset (need to accumulate scores).
        # Resume granularity is the superset itself (tracked via supersets_done.json
        # in the outer orchestrator), not chunks within a superset.
        print(f"\n[reap] superset {ss_idx} chunk layers {layer_ids[0]}..{layer_ids[-1]}: "
              f"{sum(score_each)} score, {len(layer_ids) - sum(score_each)} hash",
              flush=True)
        t_chunk = time.time()
        delete_cache_dir(out_cache)
        process_chunk(
            layer_ids=layer_ids, layer_prefix="layers",
            base_args=base_args, idx=idx,
            calib_seqs=ss_seqs, batches=batches,
            in_cache=in_cache, out_cache=out_cache,
            score_each=score_each, n_keep=n_keep,
            external_score_bufs=body_score_bufs,
        )
        # Persist updated cumulative aggregate per layer.
        for L in [L for L, sc in zip(layer_ids, score_each) if sc]:
            (chunks_dir / f"layers.{L}.json").write_text(
                json.dumps(_aggregate_to_artifact(body_score_bufs[L], n_keep), indent=2)
            )
        print(f"[reap] chunk done in {time.time()-t_chunk:.1f}s", flush=True)

    # MTP.
    if process_mtp and base_args.n_mtp_layers > 0:
        assert mtp_score_buf is not None, "orchestrator should have pre-created mtp_score_buf"
        in_cache_mtp = cache_dir_for(out_dir, base_args.n_layers, superset_idx=ss_idx)
        out_cache_mtp = cache_dir_for(out_dir, base_args.n_layers + 1, superset_idx=ss_idx)
        print(f"\n[reap] superset {ss_idx} mtp.0:", flush=True)
        t_layer = time.time()
        delete_cache_dir(out_cache_mtp)
        process_mtp_chunk(
            base_args=base_args, idx=idx,
            calib_seqs=ss_seqs, batches=batches,
            in_cache=in_cache_mtp, out_cache=out_cache_mtp,
            n_keep=n_keep,
            external_score_buf=mtp_score_buf,
        )
        (chunks_dir / "mtp.0.json").write_text(
            json.dumps(_aggregate_to_artifact(mtp_score_buf, n_keep), indent=2)
        )
        print(f"[reap] mtp.0 superset done in {time.time()-t_layer:.1f}s", flush=True)
