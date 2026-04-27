"""grouped_fp4_gemm: per-expert grouped FP8 act x FP4 weight GEMM.

C[m, n] = A_fp8[m, k] @ B_fp4[expert_of(m), n, k]^T,
  for m in [0, total_M), where expert_of(m) = pos_to_expert[m].

This is our extension of inference/kernel.py:fp4_gemm_kernel for the fused-MoE
pipeline (expand_to_fused -> grouped_fp4_gemm -> swiglu_quant -> grouped_fp4_gemm
-> reduce_fused). Each tile-block belongs to exactly one expert because
get_fused_mapping uses alignment >= block_M = 32.

Layout matches fp4_gemm exactly:
  A: [total_M, K]      FP8 e4m3
  A_sf: [total_M, K//128] FP32 (or FE8M0 if scale_dtype=float8_e8m0fnu)
  B: [E, N, K//2]      FP4 e2m1 packed (stored as float4_e2m1fn_x2; logical [E,N,K])
  B_sf: [E, N, K//32]  FE8M0
  C: [total_M, N]      BF16

Padded positions (where pos_to_expert == -1) get C[m, :] = 0.
"""
import torch
import tilelang
import tilelang.language as T

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def grouped_fp4_gemm_kernel(
    N: int, K: int, E: int,
    out_dtype=BF16, accum_dtype=FP32,
    scale_a_dtype=FP32, scale_b_dtype=FE8M0,
):
    """A_fp8 act scales as FP32 (transient, tiny); B_fp4 weight scales as E8M0
    (~200M entries per layer at full 256 experts, so storage matters)."""
    M = T.symbolic("M")  # total expanded rows
    act_group_size = 128
    weight_group_size = 32
    block_M = 32
    block_N = 128
    block_K = 32
    n_sub = act_group_size // block_K  # 4 sub-blocks per act scale group

    num_blocks_M = T.symbolic("num_blocks_M")

    @T.prim_func
    def kernel(
        A: T.Tensor[(M, K), FP8],
        B: T.Tensor[(E, N, K), FP4],
        C: T.Tensor[(M, N), out_dtype],
        scales_a: T.Tensor[(M, T.ceildiv(K, act_group_size)), scale_a_dtype],
        scales_b: T.Tensor[(E, N, T.ceildiv(K, weight_group_size)), scale_b_dtype],
        block_to_expert: T.Tensor[(num_blocks_M,), INT32],
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            expert_id = block_to_expert[by]

            # Padding-only block: zero output and bail.
            if expert_id < 0:
                C_zero = T.alloc_shared((block_M, block_N), out_dtype)
                T.clear(C_zero)
                T.copy(C_zero, C[by * block_M, bx * block_N])
                T.thread_return()

            T.assume(0 <= expert_id < E)

            A_shared = T.alloc_shared((block_M, block_K), FP8)
            B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
            B_shared = T.alloc_shared((block_N, block_K), FP8)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
            scale_a_frag = T.alloc_fragment((block_M,), FP32)
            scale_b_frag = T.alloc_fragment((block_N,), FP32)

            # NB: do NOT enable T.use_swizzle here. The existing
            # inference/kernel.py:fp4_gemm uses panel_size=10 for L2 reordering,
            # but with a dynamic expert_id the swizzle reorders kernel-block
            # scheduling such that B[expert_id, ...] reads point to the wrong
            # expert (rms balloons to ~1.4). Tested empirically.
            T.clear(C_local)
            T.clear(C_local_accum)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[expert_id, bx * block_N, k * block_K], B_fp4_shared)
                # FP4 -> FP8 cast via FP32 to avoid ambiguous overload.
                for i, j in T.Parallel(block_N, block_K):
                    B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))

                # Weight scale: per-32 on K.
                for i in T.Parallel(block_N):
                    scale_b_frag[i] = T.Cast(FP32, scales_b[expert_id, bx * block_N + i, k])

                # Act scale: per-128 on K (k // 4 because block_K=32 and act_group=128).
                for i in T.Parallel(block_M):
                    scale_a_frag[i] = T.Cast(FP32, scales_a[by * block_M + i, k // n_sub])

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                T.clear(C_local)

            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return kernel


def _torch_to_tl_dtype(d: torch.dtype) -> str:
    if d == torch.float32:
        return FP32
    if d == torch.float8_e8m0fnu:
        return FE8M0
    raise ValueError(f"unsupported scale dtype {d}")


def _build_block_to_expert(pos_to_expert: torch.Tensor, block_M: int) -> torch.Tensor:
    """Per-M-block expert id (size = ceildiv(total_M, block_M))."""
    total_M = pos_to_expert.size(0)
    num_blocks = (total_M + block_M - 1) // block_M
    # Take first row of each block; alignment in get_fused_mapping ensures all
    # rows in a block share an expert id (or -1 for full-pad blocks).
    return pos_to_expert[: num_blocks * block_M : block_M].contiguous()


def grouped_fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor,
    b: torch.Tensor, b_s: torch.Tensor,
    pos_to_expert: torch.Tensor,
    block_to_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-expert grouped FP4 GEMM. Mixed-dtype scales: A=FP32, B=E8M0 (default).

    Args:
      a: [total_M, K] FP8 e4m3 activations (expanded).
      a_s: [total_M, K//128] FP32 act scales (transient, tiny).
      b: [E, N, K//2] FP4 weights (float4_e2m1fn_x2 packed).
      b_s: [E, N, K//32] E8M0 weight scales (~200M entries/layer at E=256).
      pos_to_expert: [total_M] int32 from get_fused_mapping (-1 = pad).
      block_to_expert: optional precomputed per-M-block expert id
        (size = ceildiv(total_M, 32)). If None, derived from pos_to_expert.

    Returns:
      c: [total_M, N] BF16 output. Pad rows are zero.
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    assert pos_to_expert.is_contiguous() and pos_to_expert.dtype == torch.int32

    K = a.size(-1)
    total_M = a.numel() // K
    E = b.size(0)
    N = b.size(1)
    assert pos_to_expert.size(0) == total_M
    assert b_s.shape == (E, N, K // 32)
    assert a_s.shape[0] == total_M and a_s.shape[1] == K // 128

    if block_to_expert is None:
        block_to_expert = _build_block_to_expert(pos_to_expert, block_M=32)
    assert block_to_expert.dtype == torch.int32
    assert block_to_expert.is_contiguous()

    c = a.new_empty(total_M, N, dtype=torch.get_default_dtype())
    kernel = grouped_fp4_gemm_kernel(
        N, K, E,
        scale_a_dtype=_torch_to_tl_dtype(a_s.dtype),
        scale_b_dtype=_torch_to_tl_dtype(b_s.dtype),
    )
    kernel(a, b, c, a_s, b_s, block_to_expert)
    return c
