import os

import torch


def _initialize_device():
    global IS_CUDA, IS_NPU, IS_MPS, _Event
    if "_Event" in globals():
        return
    IS_CUDA = torch.cuda.is_available()

    IS_NPU = False
    try:
        if not hasattr(torch, "npu"):
            import torch_npu  # noqa: F401
        IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
    except ImportError:
        IS_NPU = False
    except Exception as e:
        print(f"An unexpected error occurred while checking NPU availability: {e}")

    IS_MPS = False
    try:
        IS_MPS = torch.backends.mps.is_available()
    except AttributeError:
        print("MPS backend is not available in this PyTorch build.")
    except Exception as e:
        print(f"An unexpected error occurred while checking MPS availability: {e}")

    if IS_CUDA:
        _Event = torch.cuda.Event
    elif IS_NPU:
        _Event = torch.npu.Event
    elif IS_MPS:
        _Event = torch.mps.Event
    else:
        _Event = None


def __getattr__(name):
    if name in ("IS_CUDA", "IS_NPU", "IS_MPS"):
        _initialize_device()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def Event(*args, **kwargs):
    """Create an event, resolving the available device on first use."""
    _initialize_device()
    return _Event(*args, **kwargs)


def get_current_device():
    _initialize_device()
    device = None
    if IS_CUDA:
        device = torch.cuda.current_device()
    elif IS_NPU:
        device = torch.npu.current_device()
    elif IS_MPS:
        device = "mps:0"

    return device


def device_synchronize(device: int | torch.device | None = None) -> None:
    """Synchronize CUDA/HIP, MPS, or NPU, preferring CUDA when no device is given."""
    if device is None:
        _initialize_device()
        if IS_CUDA:
            torch.cuda.synchronize()
        elif IS_NPU:
            torch.npu.synchronize()
        elif IS_MPS:
            torch.mps.synchronize()
        else:
            raise RuntimeError("No device is available")
        return

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
