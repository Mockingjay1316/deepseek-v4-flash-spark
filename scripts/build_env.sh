#!/usr/bin/env bash
# Build recipe for running DeepSeek-V4-Flash on DGX Spark (GB10, sm_121, aarch64).
#
# Reference environment this was validated against:
#   - Python 3.12
#   - PyTorch 2.11.0+cu130  (built with CXX11 ABI)
#   - CUDA 13.0.88          (/usr/local/cuda -> cuda-13.0)
#   - Platform: Linux aarch64, SM_121

set -euo pipefail
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME

# ---------------------------------------------------------------------------
# fast_hadamard_transform
# ---------------------------------------------------------------------------
# Two gotchas on GB10 that the standard `pip install fast_hadamard_transform`
# does not survive:
#
#   1. The package's setup.py imports torch at module scope to detect CUDA arch
#      and to use torch.utils.cpp_extension.{CUDAExtension, BuildExtension}.
#      pip's default PEP-517 build runs in an isolated env where torch is NOT
#      installed, so setup.py blows up with `ModuleNotFoundError: No module
#      named 'torch'` before it ever reports its build requirements. Must use
#      --no-build-isolation so the build sees the active env's torch.
#
#   2. The v1.1.0 sdist on PyPI is broken: it ships without the `csrc/`
#      directory, so even with build isolation disabled, ninja fails with
#      `missing csrc/fast_hadamard_transform.cpp`. This is a packaging bug
#      in the published wheel/sdist; there is no prebuilt wheel for
#      linux_aarch64 + cu130 + cp312 anyway. Install from the Git repo
#      instead (v1.1.0 tag's setup.py already gencodes sm_121 when CUDA>=13).
#
# Build time on GB10: ~3-5 minutes with MAX_JOBS=4; drop to 2 if RAM pressure.
pip install --no-build-isolation \
    "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"

# Smoke test:
python - <<'PY'
import torch
from fast_hadamard_transform import hadamard_transform
x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
y = hadamard_transform(x, scale=x.size(-1) ** -0.5)
assert y.shape == x.shape and torch.isfinite(y).all()
print("fast_hadamard_transform OK")
PY

# ---------------------------------------------------------------------------
# TODO: record the recipes for tilelang==0.1.8 and DeepGEMM (MegaMoE PR #304)
#       after verifying they load on sm_121. The plan's Step 0 depends on
#       these being importable; if upstream wheels don't ship sm_121 yet,
#       build with TORCH_CUDA_ARCH_LIST=12.1.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TileLang 0.1.8 + apache-tvm-ffi pin
# ---------------------------------------------------------------------------
# apache-tvm-ffi 0.1.10 (released April 2026) regressed TileLang 0.1.8's
# PreLowerSemanticCheck / NestedLoopChecker: every JIT compile fails with
#   AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
# See https://github.com/state-spaces/mamba/issues/907 for the cross-repo
# report. Pin apache-tvm-ffi to 0.1.9 until TileLang releases a fix.
pip install 'apache-tvm-ffi==0.1.9'
