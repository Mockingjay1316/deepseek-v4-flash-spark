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
import struct
from typing import Iterable, List, Tuple

import numpy as np
import torch
from safetensors import safe_open


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
                if key not in sd:
                    unexpected.append(key)
                    continue
                tensor = f.get_tensor(key)
                target = sd[key]
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"shape mismatch for {key}: ckpt {tuple(tensor.shape)} vs model {tuple(target.shape)}"
                    )
                target.copy_(tensor, non_blocking=False)
                del tensor
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
                if key not in sd:
                    unexpected.append(key)
                    # Still advance the "read-but-drop" cursor logically by
                    # advising this range can be dropped.
                    if drop_cache:
                        start, end = spec["data_offsets"]
                        try:
                            os.posix_fadvise(fd, data_start + start, end - start,
                                             os.POSIX_FADV_DONTNEED)
                        except (AttributeError, OSError):
                            pass
                    continue
                target = sd[key]
                dtype_str = spec["dtype"]
                assert dtype_str in _DTYPE_MAP, f"unsupported dtype {dtype_str} for {key}"
                dtype = _DTYPE_MAP[dtype_str]
                shape = list(spec["shape"])
                # Packed F4 stores 2 values per byte; safetensors/python returns
                # a tensor with the last dim halved to match physical storage.
                if dtype_str == "F4" and shape:
                    shape[-1] = shape[-1] // 2
                shape = tuple(shape)
                start, end = spec["data_offsets"]
                nbytes = end - start

                # Read raw bytes (CPU buffer). For very large reads pread
                # is a single syscall and avoids mmap page caching of the
                # entire file.
                buf = os.pread(fd, nbytes, data_start + start)
                if len(buf) != nbytes:
                    raise RuntimeError(f"short read for {key}: {len(buf)} / {nbytes}")

                # Zero-copy view over the freshly-read bytes.
                cpu_tensor = torch.frombuffer(buf, dtype=dtype).reshape(shape)
                if cpu_tensor.shape != target.shape:
                    raise RuntimeError(
                        f"shape mismatch for {key}: ckpt {tuple(cpu_tensor.shape)} vs model {tuple(target.shape)}"
                    )
                target.copy_(cpu_tensor, non_blocking=False)
                del cpu_tensor, buf
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
