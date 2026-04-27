"""Memory-friendly checkpoint loader.

The default `safetensors.torch.load_model` calls `load_file` first, which
materializes the *entire* shard's state_dict into a fresh dict before copying
into the model parameters. On a unified-memory box (e.g. GB10) that doubles
the peak resident set: shard-on-disk size *plus* the already-allocated CUDA
parameters, both backed by the same physical RAM.

`load_streaming` instead opens the file with safe_open and copies one tensor
at a time into the existing model parameter, so the temporary buffer is at
most one tensor (a few MB) regardless of total checkpoint size.

`load_direct` is a mmap-less reader that parses the safetensors header and
issues plain pread() calls per tensor. After each tensor is copied into the
model, posix_fadvise(POSIX_FADV_DONTNEED) tells the kernel to drop the
freshly-read bytes from page cache. This keeps peak page-cache usage bounded
to a few tensors rather than growing to the whole shard — critical on
unified-memory GPUs (GB10 / Jetson) where page cache and CUDA allocations
compete for the same physical RAM.
"""
import json
import os
import re
import struct
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open


# Per-expert tensor keys in the checkpoint look like
#   layers.{L}.ffn.experts.{i}.{w1,w2,w3}.{weight,scale}
#   mtp.{L}.ffn.experts.{i}.{w1,w2,w3}.{weight,scale}
# The model now stores them stacked under the parent module:
#   layers.{L}.ffn.experts_{w13,w13_scale,w2,w2_scale}
# This regex extracts (parent, expert_idx, proj, suffix).
_EXPERT_RE = re.compile(r"^(.+)\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")


def _resolve_expert_target(
    sd: dict, key: str,
) -> Optional[Tuple[torch.Tensor, str]]:
    """If key is a per-expert tensor and the model uses stacked Params, return
    the writable slice into the corresponding stacked Param plus the stacked
    Param's full state_dict key (for missing-tracking). Returns None otherwise.
    """
    m = _EXPERT_RE.match(key)
    if m is None:
        return None
    parent, expert_idx_str, proj, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
    expert_idx = int(expert_idx_str)
    if proj == "w2":
        stacked_name = "experts_w2_scale" if suffix == "scale" else "experts_w2"
        stacked_key = f"{parent}.{stacked_name}"
        if stacked_key not in sd:
            return None
        return sd[stacked_key][expert_idx], stacked_key
    # proj in (w1, w3): both go into experts_w13 along axis 1.
    stacked_name = "experts_w13_scale" if suffix == "scale" else "experts_w13"
    stacked_key = f"{parent}.{stacked_name}"
    if stacked_key not in sd:
        return None
    target_param = sd[stacked_key]
    inter_dim = target_param.size(1) // 2  # 2*inter -> inter
    if proj == "w1":
        return target_param[expert_idx, :inter_dim], stacked_key
    return target_param[expert_idx, inter_dim:], stacked_key


_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "F4": torch.float4_e2m1fn_x2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _parse_header(fd: int) -> tuple[dict, int]:
    """Read safetensors header. Returns (meta_dict, data_start_offset)."""
    hdr_len_bytes = os.pread(fd, 8, 0)
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    hdr_bytes = os.pread(fd, hdr_len, 8)
    meta = json.loads(hdr_bytes.decode("utf-8"))
    data_start = 8 + hdr_len
    return meta, data_start


@torch.no_grad()
def load_streaming(
    model: torch.nn.Module,
    shard_files: Iterable[str],
    device: str = "cuda",
) -> Tuple[List[str], List[str]]:
    """Stream tensors from each shard into the model in-place.

    Returns (missing, unexpected) where missing = model params not seen in any
    shard, unexpected = checkpoint keys not present in the model.
    """
    sd = model.state_dict()
    expected = set(sd.keys())
    seen: set[str] = set()
    unexpected: list[str] = []

    for shard in shard_files:
        with safe_open(str(shard), framework="pt", device=device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in sd:
                    target = sd[key]
                else:
                    resolved = _resolve_expert_target(sd, key)
                    if resolved is None:
                        unexpected.append(key)
                        del tensor
                        continue
                    target, stacked_key = resolved
                    seen.add(stacked_key)
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"shape mismatch for {key}: ckpt {tuple(tensor.shape)} vs model {tuple(target.shape)}"
                    )
                target.copy_(tensor, non_blocking=False)
                del tensor
                if key in sd:
                    seen.add(key)

    missing = sorted(expected - seen)
    return missing, unexpected


@torch.no_grad()
def load_direct(
    model: torch.nn.Module,
    shard_files: Iterable[str],
    drop_cache: bool = True,
    print_every: int = 200,
) -> Tuple[List[str], List[str]]:
    """mmap-free per-tensor loader.

    For each tensor in each shard:
      1. pread() the raw bytes into a freshly-allocated CPU uint8 buffer
      2. Wrap as a CPU tensor with the right dtype/shape
      3. copy_() into the existing model parameter (GPU-resident on GB10 =
         same physical RAM as the CPU buffer, so this is effectively a
         rebind/noop on unified memory)
      4. (optional) posix_fadvise(FADV_DONTNEED) on the byte range to evict
         the just-read pages from page cache

    On GB10 this keeps peak page-cache use to the size of the largest tensor
    instead of the whole shard (~85GB for pruned-128).
    """
    sd = model.state_dict()
    expected = set(sd.keys())
    seen: set[str] = set()
    unexpected: list[str] = []

    for shard in shard_files:
        fd = os.open(str(shard), os.O_RDONLY)
        try:
            # Hint sequential so readahead helps bulk read
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            meta, data_start = _parse_header(fd)
            meta.pop("__metadata__", None)

            count = 0
            for key, spec in meta.items():
                dtype_str = spec["dtype"]
                shape = list(spec["shape"])
                if dtype_str == "F4" and shape:
                    shape[-1] = shape[-1] // 2
                shape = tuple(shape)
                start, end = spec["data_offsets"]
                nbytes = end - start

                # Resolve target: direct sd hit, or stacked-Param slice.
                target = None
                stacked_key = None
                if key in sd:
                    target = sd[key]
                else:
                    resolved = _resolve_expert_target(sd, key)
                    if resolved is not None:
                        target, stacked_key = resolved

                if target is None:
                    unexpected.append(key)
                    if drop_cache:
                        try:
                            os.posix_fadvise(fd, data_start + start, nbytes,
                                             os.POSIX_FADV_DONTNEED)
                        except (AttributeError, OSError):
                            pass
                    continue

                assert dtype_str in _DTYPE_MAP, f"unsupported dtype {dtype_str} for {key}"
                dtype = _DTYPE_MAP[dtype_str]

                buf = os.pread(fd, nbytes, data_start + start)
                if len(buf) != nbytes:
                    raise RuntimeError(f"short read for {key}: {len(buf)} / {nbytes}")
                cpu_tensor = torch.frombuffer(buf, dtype=dtype).reshape(shape)
                if cpu_tensor.shape != target.shape:
                    raise RuntimeError(
                        f"shape mismatch for {key}: ckpt {tuple(cpu_tensor.shape)} vs model {tuple(target.shape)}"
                    )
                target.copy_(cpu_tensor, non_blocking=False)
                del cpu_tensor, buf
                if stacked_key is not None:
                    seen.add(stacked_key)
                else:
                    seen.add(key)

                if drop_cache:
                    try:
                        os.posix_fadvise(fd, data_start + start, nbytes,
                                         os.POSIX_FADV_DONTNEED)
                    except (AttributeError, OSError):
                        pass
                count += 1
                if print_every and count % print_every == 0:
                    print(f"  load_direct: {count}/{len(meta)} tensors", flush=True)
        finally:
            os.close(fd)

    missing = sorted(expected - seen)
    return missing, unexpected
