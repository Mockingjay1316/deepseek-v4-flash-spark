"""Fused SwiGLU + per-token FP8 quantization for the MoE inner activation.

Bridges the two grouped GEMMs in fused MoE:
  expanded_x [N, dim] FP8 -> grouped_fp4_gemm(W13 [E, 2*inter, dim])
       -> gate_up [N, 2*inter] BF16
  swiglu_forward_and_per_token_cast(gate_up, pos_to_expert, swiglu_limit)
       -> y_fp8 [N, inter] + y_sf [N, inter // 128] FP32
       -> grouped_fp4_gemm(W2 [E, dim, inter]) -> y_out [N, dim] BF16

Math per row n (when pos_to_expert[n] >= 0):
  gate = clamp(gate_up[n, :inter],         max=swiglu_limit)
  up   = clamp(gate_up[n, inter:2*inter], -swiglu_limit, swiglu_limit)
  y    = silu(gate) * up                   # bf16 -> fp32
  per 128-channel group g:
    amax = max(|y[g*128:(g+1)*128]|)
    sf   = amax / 448.0  (or ceil_pow2 of that, if scale_dtype=E8M0)
    y_fp8[g*128:(g+1)*128] = clamp(y / sf, -448, 448)

Pad rows (pos_to_expert == -1) get y_fp8 = 0, sf = 0 (early-bail for perf).

Default output scale dtype is FP32 to match grouped_fp4_gemm's A-side contract;
pass scale_dtype=torch.float8_e8m0fnu to round-trip through E8M0 (used when the
caller wants to feed an E8M0-only consumer).
"""
import torch
import tilelang
import tilelang.language as T

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
FE8M0 = "float8_e8m0fnu"
INT32 = "int32"

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def swiglu_forward_and_per_token_cast_kernel(
    inter: int, sf_block: int = 128, in_dtype=BF16, out_dtype=FP8,
    scale_dtype=FE8M0, round_scale: bool = True,
):
    M = T.symbolic("M")
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max
    group_size = sf_block
    num_groups = inter // sf_block
    assert inter % sf_block == 0

    @T.prim_func
    def kernel(
        x: T.Tensor[(M, 2 * inter), in_dtype],
        y: T.Tensor[(M, inter), out_dtype],
        sf: T.Tensor[(M, num_groups), scale_dtype],
        pos_to_expert: T.Tensor[(M,), INT32],
        swiglu_limit: T.float32,
    ):
        with T.Kernel(M, T.ceildiv(inter, group_size), threads=128) as (pid_m, pid_g):
            # Pad row early-bail: zero output and one-time zero of the SF row.
            if pos_to_expert[pid_m] < 0:
                for j in T.Parallel(group_size):
                    y[pid_m, pid_g * group_size + j] = T.Cast(out_dtype, 0)
                if pid_g == 0:
                    for g in T.Parallel(num_groups):
                        sf[pid_m, g] = T.Cast(scale_dtype, 0)
                T.thread_return()

            silu_local = T.alloc_fragment((group_size,), FP32)
            amax_local = T.alloc_fragment((1,), FP32)

            # Single fused pass: load gate+up, clamp, swiglu.
            for j in T.Parallel(group_size):
                g = T.Cast(FP32, x[pid_m, pid_g * group_size + j])
                u = T.Cast(FP32, x[pid_m, inter + pid_g * group_size + j])
                g = T.min(g, swiglu_limit)
                u = T.clamp(u, -swiglu_limit, swiglu_limit)
                # silu(g) = g * sigmoid(g) = g / (1 + exp(-g)); compute in fp32.
                silu_local[j] = (g / (1.0 + T.exp(-g))) * u

            T.reduce_absmax(silu_local, amax_local, dim=0)
            # Floor amax so subnormal-range values still produce a representable scale.
            clamped_amax = T.max(amax_local[0], 6 * (2**-126))
            if round_scale:
                # Power-of-2 round (matches act_quant_kernel's UE8M0 path).
                bits = T.reinterpret("uint32", clamped_amax * fp8_max_inv)
                exp_x = (bits >> 23) & 0xFF
                man_bits = bits & ((1 << 23) - 1)
                exp_ceil = T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))
                sf_val = T.reinterpret("float32", T.uint32((exp_ceil + 127) << 23))
                sf_inv = T.reinterpret("float32", T.uint32((127 - exp_ceil) << 23))
            else:
                sf_val = clamped_amax * fp8_max_inv
                sf_inv = fp8_max / clamped_amax
            sf[pid_m, pid_g] = T.Cast(scale_dtype, sf_val)

            for j in T.Parallel(group_size):
                y[pid_m, pid_g * group_size + j] = T.Cast(
                    out_dtype, T.clamp(silu_local[j] * sf_inv, -fp8_max, fp8_max),
                )

    return kernel


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    pos_to_expert: torch.Tensor,
    swiglu_limit: float,
    sf_block: int = 128,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU + FP8 per-token cast (per-128 scale groups).

    Args:
      x: [M, 2*inter] BF16 (gate concat with up).
      pos_to_expert: [M] int32 (-1 = pad row, output is zero).
      swiglu_limit: clamp limit (matches Expert.forward in inference/model.py).
      sf_block: scale group size on the inter dim. Default 128 to match the
        FP8 act-quant block used by grouped_fp4_gemm.
      scale_dtype: torch.float8_e8m0fnu (default, production) or torch.float32.
        E8M0 forces power-of-2 rounding to round-trip cleanly through the dtype.

    Returns:
      (y_fp8 [M, inter], y_sf [M, inter // sf_block] in scale_dtype).
    """
    assert x.is_contiguous() and pos_to_expert.is_contiguous()
    assert x.dim() == 2 and pos_to_expert.dim() == 1
    assert pos_to_expert.dtype == torch.int32
    M2, twoH = x.shape
    assert twoH % 2 == 0
    inter = twoH // 2
    assert inter % sf_block == 0
    assert pos_to_expert.size(0) == M2

    y = x.new_empty(M2, inter, dtype=torch.float8_e4m3fn)
    sf = x.new_empty(M2, inter // sf_block, dtype=scale_dtype)

    if M2 == 0:
        return y, sf

    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32
    round_scale = scale_dtype == torch.float8_e8m0fnu
    kernel = swiglu_forward_and_per_token_cast_kernel(
        inter, sf_block, scale_dtype=tl_dtype, round_scale=round_scale,
    )
    kernel(x, y, sf, pos_to_expert, float(swiglu_limit))
    return y, sf
