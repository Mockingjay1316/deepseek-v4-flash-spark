"""Host-side helpers. Adapted from deepseek-ai/TileKernels (MIT)."""
import functools

import torch


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@functools.lru_cache(maxsize=None)
def get_num_sms() -> int:
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    return prop.multi_processor_count
