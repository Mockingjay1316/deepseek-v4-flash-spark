"""mhc_pre_big_fuse: fused RMSNorm + sigmoid mixes + Sinkhorn + apply pre to residual.

Replaces the multi-step Block.hc_pre in inference/model.py:
  flatten -> rsqrt(square.mean) -> F.linear(hc_fn) * rsqrt
    -> hc_split_sinkhorn (kernel) -> sum(pre * x.view(...))

Pipeline (matches deepseek-ai/TileKernels/tile_kernels/modeling/mhc/ops/pre_big_fuse.py):
  1. Round fn to TF32 (round_to_tf32).
  2. fwd_mul kernel: x @ fn^T computed alongside x.square().sum(-1), all in one pass.
  3. fuse kernel: combines partials, computes RMS, sigmoid mixes, Sinkhorn,
     applies pre to residual to produce layer_input.

Adapted from deepseek-ai/TileKernels (MIT, Copyright (c) 2026 DeepSeek). For the
V4-Flash inference contract:
  - residual is BF16 [b, s, hc, dim]
  - hc_fn is FP32 [(2 + hc) * hc, hc * dim]
  - hc_scale FP32 [3]; hc_base FP32 [(2 + hc) * hc]
  - returns (post [n, hc, 1], comb [n, hc, hc], layer_input [n, dim])

n_splits is hardcoded to 1 because TileLang doesn't support split-K gemm.
"""
import math
from typing import Tuple

import tilelang
import torch
from tilelang import language as T


def _round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 to TF32 by zeroing the low 13 mantissa bits via integer add."""
    return (x.view(torch.int32) + 0x1000).view(torch.float32)


_PASS_CONFIGS_GEMM = {
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
}

_PASS_CONFIGS_FUSE = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS_GEMM)
def _mhc_pre_norm_fn_fwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
):
    """Fused gemm + sqrsum: out = x @ fn^T;  sqrsum = sum_k x[:, k]^2.
    rms_group_size = mhc_mult * hidden_size (the full feature dim).
    n_rms_group = 1 for our use (single RMS over the full feature dim).
    mhc_mult3 = (2 + mhc_mult) * mhc_mult; padded to 32 in shared mem.
    """
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic('num_tokens')
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        out: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        _ = mhc_mult3
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_rms_group) as (pid_x, pid_y):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            for pz in T.Pipelined(rms_group_size // hidden_block, num_stages=2):
                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})

                T.copy(x[pid_x * token_block, pid_y * rms_group_size + pz * hidden_block], x_smem_16)
                T.copy(fn[0, pid_y * rms_group_size + pz * hidden_block], fn_smem)

                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem_16, x_frag_16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_frag_16, x_frag)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

                T.gemm(x_frag, fn_smem, out_frag,
                       transpose_A=False, transpose_B=True, clear_accum=False)
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            for i in T.Parallel(token_block):
                sqrsum[pid_x * token_block + i, pid_y] = sqrsum_l[i]
            for i, j in T.Parallel(token_block, 32):
                if j < mhc_mult3:
                    out[pid_x * token_block + i, pid_y, j] = out_frag[i, j]

    return kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS_FUSE)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    mhc_mult: int = 4,
):
    """Post-gemm fusion: combines partials, computes RMS, sigmoid pre/post,
    Sinkhorn-iterated softmax for comb, applies pre to residual."""
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def kernel(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=96) as pid:
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if T.get_thread_binding() < 32:
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = T.sigmoid(
                        mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult]
                    ) * mhc_post_mult_value
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2]
                                + mhc_base[j * mhc_mult + k + mhc_mult * 2])

                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * mhc_scale[0] + mhc_base[j])
                        + mhc_pre_eps
                    )
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return kernel


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult_value: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused HC-pre: returns (post_mix, comb_mix, layer_input).

    Args:
      residual: [..., mhc_mult, hidden_size] BF16. Outer dims may be 2D ([b, s]).
      hc_fn: [(2 + mhc_mult) * mhc_mult, mhc_mult * hidden_size] FP32 weights.
      hc_scale: [3] FP32 scale for {pre, post, comb}.
      hc_base: [(2 + mhc_mult) * mhc_mult] FP32 bias.
      rms_eps: epsilon for RMSNorm (matches Block.norm_eps).
      hc_eps: epsilon for sigmoid pre AND Sinkhorn iterations (matches Block.hc_eps).
      sinkhorn_iters: number of Sinkhorn iterations (matches Block.hc_sinkhorn_iters).
      post_mult_value: multiplier on the post sigmoid (matches the `2 *` in
        inference/kernel.py:hc_split_sinkhorn).

    Returns:
      post_mix [..., mhc_mult, 1] FP32  (matches `post.unsqueeze(-1)` in hc_post).
      comb_mix [..., mhc_mult, mhc_mult] FP32.
      layer_input [..., hidden_size] BF16.
    """
    assert residual.dtype == torch.bfloat16
    assert hc_fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_scale.numel() == 3
    assert hc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    assert hc_fn.shape == (mhc_mult3, mhc_hidden_size)
    assert hc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.contiguous().view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    n_splits = 1  # TileLang doesn't support split-K

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult * mhc_mult, dtype=torch.float32,
                           device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16,
                              device=residual.device)
    gemm_out_mul = torch.empty(n_splits, num_tokens, mhc_mult3, dtype=torch.float32,
                               device=residual.device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32,
                                  device=residual.device)

    fn_tf32 = _round_to_tf32(hc_fn)

    fwd_mul = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
    fwd_mul(residual_flat.view(-1, mhc_hidden_size), fn_tf32,
            gemm_out_mul.view(-1, 1, mhc_mult3), gemm_out_sqrsum.view(-1, 1))

    fuse = _mhc_pre_big_fuse(hidden_size, rms_eps, hc_eps, hc_eps,
                              float(post_mult_value), sinkhorn_iters,
                              n_splits=1, mhc_mult=mhc_mult)
    fuse(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
         residual_flat, post_mix, comb_mix, layer_input)

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input
