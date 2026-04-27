"""mhc_post_fwd: HC residual mixing -> y = post * x + sum_i comb[i, :] @ residual[i, :].

Replaces Block.hc_post in inference/model.py:
  y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

Adapted from deepseek-ai/TileKernels (MIT, Copyright (c) 2026 DeepSeek).
Removed T.pdl_sync() (not present on sm_121) and the bwd kernel (inference-only).
"""
import math

import tilelang
import torch
from tilelang import language as T


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_post_fwd_kernel(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024):
    n = T.dynamic('num_tokens')
    h = hidden
    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],          # comb
        b: T.Tensor[(n, mhc, h), T.bfloat16],           # residual
        c: T.Tensor[(n, mhc), T.float32],               # post
        d: T.Tensor[(n, h), T.bfloat16],                # x (attn or ffn output)
        x: T.Tensor[(n, mhc, h), T.bfloat16],           # output (residual-shaped)
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                T.copy(x_local, x_shared)
                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk], disable_tma=True)

    return kernel


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused HC-post.

    Args:
      x: [..., hidden] BF16. Attention or FFN output (the "new" hidden state).
      residual: [..., mhc_mult, hidden] BF16. Pre-mix HC state.
      post_mix: [..., mhc_mult, 1] FP32 (from mhc_pre_big_fuse).
      comb_mix: [..., mhc_mult, mhc_mult] FP32 (from mhc_pre_big_fuse).
      out: optional pre-allocated [..., mhc_mult, hidden] BF16.

    Returns:
      y: [..., mhc_mult, hidden] BF16.
    """
    mhc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_mix.dtype == torch.float32
    assert comb_mix.dtype == torch.float32
    assert x.shape == (*outer_shape, hidden)
    assert post_mix.shape == (*outer_shape, mhc_mult, 1)
    assert comb_mix.shape == (*outer_shape, mhc_mult, mhc_mult)

    residual_c = residual.contiguous()
    x_c = x.contiguous()
    post_c = post_mix.contiguous()
    comb_c = comb_mix.contiguous()

    if out is None:
        out = torch.empty_like(residual_c)
    kernel = _mhc_post_fwd_kernel(mhc_mult, hidden)
    kernel(
        comb_c.view(-1, mhc_mult, mhc_mult),
        residual_c.view(-1, mhc_mult, hidden),
        post_c.view(-1, mhc_mult, 1).squeeze(-1),
        x_c.view(-1, hidden),
        out.view(-1, mhc_mult, hidden),
    )
    return out
