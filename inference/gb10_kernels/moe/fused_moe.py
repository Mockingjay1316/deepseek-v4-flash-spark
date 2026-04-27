"""Top-level fused-MoE driver: pre-quantized FP8 act + stacked FP4 expert weights.

Pipeline:
  1. get_fused_mapping(topk_idx) -> mapping tensors
  2. expand_to_fused_with_sf(x_fp8, x_sf) -> expanded act + per-128 SF
  3. grouped_fp4_gemm(expanded_x, W13) -> [N, 2*inter] BF16
  4. swiglu_forward_and_per_token_cast(...) -> [N, inter] FP8 + per-128 SF
  5. grouped_fp4_gemm(y_fp8, W2) -> [N, dim] BF16
  6. reduce_fused(y_expanded, weights) -> [num_tokens, dim] BF16

Caller is responsible for FP8-quantizing x (e.g. via inference/kernel.py:act_quant).
This keeps gb10_kernels self-contained.
"""
from typing import Optional

import torch

from .mapping import get_fused_mapping
from .expand import expand_to_fused_with_sf
from .grouped_fp4_gemm import _build_block_to_expert, grouped_fp4_gemm
from .reduce import reduce_fused
from ..quant.swiglu_quant import swiglu_forward_and_per_token_cast


def fused_moe_fp4(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    w13: torch.Tensor,
    w13_sf: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    num_experts: int,
    swiglu_limit: float,
    alignment: int = 32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """End-to-end fused MoE: routing -> expand -> grouped GEMM W13 ->
    SwiGLU+quant -> grouped GEMM W2 -> weighted reduce.

    Args:
      x_fp8: [num_tokens, dim] FP8 e4m3 (already quantized by caller).
      x_sf: [num_tokens, dim // 128] FP32 act scales.
      topk_weights: [num_tokens, num_topk] FP32 routing weights.
      topk_idx: [num_tokens, num_topk] int64 expert indices (-1 = no-route).
      w13: [E, 2*inter, dim // 2] FP4 packed (gate concat up along dim 1).
      w13_sf: [E, 2*inter, dim // 32] E8M0 weight scales.
      w2: [E, dim, inter // 2] FP4 packed.
      w2_sf: [E, dim, inter // 32] E8M0 weight scales.
      num_experts: total experts in this layer (e.g. 256 or 128).
      swiglu_limit: clamp limit for SwiGLU (matches Expert.forward).
      alignment: per-expert M alignment (>= grouped_fp4_gemm block_M=32).
      out: optional pre-allocated [num_tokens, dim] BF16 output buffer.

    Returns:
      y: [num_tokens, dim] BF16.
    """
    num_tokens, dim = x_fp8.shape
    num_topk = topk_idx.size(1)
    inter = w13.size(1) // 2

    if num_tokens == 0:
        if out is not None:
            out.zero_()
            return out
        return torch.zeros((0, dim), dtype=torch.bfloat16, device=x_fp8.device)

    # get_fused_mapping requires contiguous int64 indices; Gate emits int32 for
    # hash layers (tid2eid lookup) and int64 for score layers (topk).
    if topk_idx.dtype != torch.int64:
        topk_idx = topk_idx.to(torch.int64)
    if not topk_idx.is_contiguous():
        topk_idx = topk_idx.contiguous()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    # Routing.
    pos_to_expert, _, _, token_topk_to_pos, *_ = get_fused_mapping(
        topk_idx, num_experts, num_expanded_tokens=0, alignment=alignment,
    )
    # Compute the per-M-block expert id once and reuse for both grouped GEMMs.
    block_to_expert = _build_block_to_expert(pos_to_expert, block_M=32)

    # Scatter activations + scales into expert-major layout.
    expanded_x, expanded_x_sf = expand_to_fused_with_sf(
        x_fp8, x_sf, num_per_channels=128,
        token_topk_to_pos=token_topk_to_pos, pos_to_expert=pos_to_expert,
        use_tma_aligned_col_major_sf=False,
    )

    # W13: gate concat up.
    gate_up = grouped_fp4_gemm(
        expanded_x, expanded_x_sf, w13, w13_sf, pos_to_expert,
        block_to_expert=block_to_expert,
    )

    # SwiGLU + per-token FP8 cast.
    y_fp8, y_sf = swiglu_forward_and_per_token_cast(
        gate_up, pos_to_expert, swiglu_limit, sf_block=128,
    )

    # W2: down.
    y_expanded = grouped_fp4_gemm(
        y_fp8, y_sf, w2, w2_sf, pos_to_expert,
        block_to_expert=block_to_expert,
    )

    # Weighted reduce back to per-token.
    return reduce_fused(y_expanded, topk_weights, token_topk_to_pos, out=out)
