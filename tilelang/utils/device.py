"""Runtime device helpers. Importing this module does not discover devices."""

import os
import warnings

import torch


def _is_available(device_type: str) -> bool:
    if device_type == "npu" and not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return False
    api = torch.backends.mps if device_type == "mps" else getattr(torch, device_type, None)
    return api is not None and api.is_available()


def __getattr__(name):
    # Compatibility only: internal callers pass devices explicitly. Do not cache
    # runtime availability or select an event class as a side effect of import.
    if name in ("IS_CUDA", "IS_NPU", "IS_MPS"):
        warnings.warn(f"{name} is deprecated; pass an explicit device instead", DeprecationWarning, stacklevel=2)
        return _is_available(name.removeprefix("IS_").lower())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_device(device: int | str | torch.device | None = None) -> torch.device:
    """Resolve a runtime device; only omitted devices require availability probes.

    CUDA/HIP, NPU, then MPS is the default discovery order. Indexed devices
    bypass discovery and current-device queries; unindexed CUDA/HIP and NPU
    devices use their runtime's current device.
    """
    if device is None:
        for device_type in ("cuda", "npu", "mps"):
            if _is_available(device_type):
                device = torch.device(device_type)
                break
        else:
            raise RuntimeError("No device is available")
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type not in ("cuda", "npu", "mps"):
        raise ValueError(f"Expected a CUDA/HIP, MPS, or NPU device, got {device}")
    if device.index is None and device.type in ("cuda", "npu"):
        device = torch.device(device.type, getattr(torch, device.type).current_device())
    return device


def Event(*args, device: int | str | torch.device | None = None, **kwargs):
    """Create an event for the selected runtime backend."""
    device = resolve_device(device)
    return getattr(torch, device.type).Event(*args, **kwargs)


def get_current_device():
    """Legacy current-device query, preserving integer CUDA/HIP and NPU results."""
    for device_type in ("cuda", "npu", "mps"):
        if _is_available(device_type):
            return "mps:0" if device_type == "mps" else getattr(torch, device_type).current_device()
    return None


def device_synchronize(device: int | str | torch.device | None = None) -> None:
    """Synchronize the selected runtime backend."""
    if device is None:
        for device_type in ("cuda", "npu", "mps"):
            if _is_available(device_type):
                getattr(torch, device_type).synchronize()
                return
        raise RuntimeError("No device is available")
    device_type = "cuda" if isinstance(device, int) else torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "npu":
        torch.npu.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()
    else:
        raise ValueError(f"device_synchronize only supports CUDA/HIP, MPS, or NPU devices, got {device}")


def get_available_cpu_count() -> int:
    """CPU cores available to this process (cpuset affinity), at least 1.

    Falls back to ``os.cpu_count`` where affinity is unavailable (macOS/Windows).
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return max(1, cpu_count or 1)
