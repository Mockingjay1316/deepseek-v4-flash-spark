"""GB10-tailored TileLang kernels for V4-Flash inference.

Some kernels are adapted from deepseek-ai/TileKernels (MIT-licensed,
Copyright (c) 2026 DeepSeek). Adaptations are noted per-file. Our additions
(e.g. grouped_fp4_gemm) target the V4-Flash FP4-weights / FP8-acts / E8M0-scales
contract directly without the BaseCastConfig indirection.

Module layout:
  utils.py           - host-side helpers (align, ceil_div, num_sms)
  moe/
    mapping.py       - get_fused_mapping: build (token,topk)->expanded layout
    expand.py        - expand_to_fused: scatter tokens (with FP8 scales)
    reduce.py        - reduce_fused: weighted gather back to (num_tokens, dim)
    grouped_fp4_gemm.py
                     - OURS: A_fp8[total_m,K] x B_fp4[E,N,K]^T per-segment
  quant/
    swiglu_quant.py  - swiglu_forward_and_per_token_cast (vendored, FP8 quant)
"""
