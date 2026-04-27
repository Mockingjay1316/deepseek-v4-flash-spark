"""reduce_fused: weighted gather of expert outputs back to per-token layout.

Adapted from deepseek-ai/TileKernels tile_kernels/moe/reduce_fused_kernel.py
(MIT, Copyright (c) 2026 DeepSeek). The QuantTensor abstraction is replaced with
plain tensors; we don't need the FP8 output path (V4-Flash MoE outputs BF16).
"""
import os
from typing import Optional

import torch
import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _get_kernel(
    hidden: int,
    num_topk: int,
    in_dtype: T.dtype,
    out_dtype: T.dtype,
    with_weights: bool,
):
    num_threads = 128

    num_tokens = T.dynamic('num_tokens')
    num_expanded_tokens = T.dynamic('num_expanded_tokens')

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_expanded_tokens, hidden), in_dtype],
        topk_weights: T.Tensor[(num_tokens, num_topk), T.float32],
        token_topk_to_pos: T.Tensor[(num_tokens, num_topk), T.int32],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as (pid_token,):
            reduced_fragment = T.alloc_fragment((hidden,), T.float32)
            topk_weights_local = T.alloc_fragment((num_topk,), T.float32)
            topk_to_pos_local = T.alloc_fragment((num_topk,), T.int32)

            T.clear(reduced_fragment)
            if with_weights:
                T.copy(topk_weights[pid_token, :], topk_weights_local)
            T.copy(token_topk_to_pos[pid_token, :], topk_to_pos_local)

            for k in T.unroll(num_topk):
                pos = topk_to_pos_local[k]
                T.assume(pos < num_expanded_tokens)
                if pos >= 0:
                    s = T.alloc_var(T.float32)
                    s = 1
                    if with_weights:
                        s = topk_weights_local[k]
                    for i in T.Parallel(hidden):
                        reduced_fragment[i] += x[pos, i] * s

            for i in T.Parallel(hidden):
                out[pid_token, i] = reduced_fragment[i]

    return kernel


def reduce_fused(
    x: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    token_topk_to_pos: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce expanded expert outputs back to (num_tokens, hidden).

    Args:
      x: [num_expanded_tokens, hidden] expert outputs.
      topk_weights: optional [num_tokens, num_topk] routing weights.
      token_topk_to_pos: [num_tokens, num_topk] from get_fused_mapping.
      out: optional pre-allocated output [num_tokens, hidden].
    """
    num_expanded_tokens, hidden = x.shape
    num_tokens, num_topk = token_topk_to_pos.shape
    assert hidden % 256 == 0

    in_dtype = x.dtype
    out_dtype = in_dtype
    if out is not None:
        assert out.shape == (num_tokens, hidden)
    else:
        out = torch.empty((num_tokens, hidden), dtype=out_dtype, device='cuda')

    kernel = _get_kernel(
        hidden, num_topk,
        T.dtype(in_dtype), T.dtype(out_dtype),
        with_weights=topk_weights is not None,
    )
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(x, topk_weights, token_topk_to_pos, out)
    return out
