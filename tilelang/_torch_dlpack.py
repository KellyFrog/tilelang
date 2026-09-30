"""Defer TVM-FFI's optional CUDA integration until kernel invocation."""

import contextlib
import os
import threading


_lock = threading.Lock()
_initialized = False


@contextlib.contextmanager
def defer_torch_c_dlpack():
    # TVM-FFI probes CUDA even when Torch already provides the exchange API.
    # Relax also calls the loader directly, bypassing this environment flag.
    flag = "TVM_FFI_DISABLE_TORCH_C_DLPACK"
    previous = os.environ.get(flag)
    os.environ[flag] = "1"
    try:
        from tvm_ffi import _optional_torch_c_dlpack as optional

        loader = optional.load_torch_c_dlpack_extension
        optional.load_torch_c_dlpack_extension = lambda: None
        try:
            yield
        finally:
            optional.load_torch_c_dlpack_extension = loader
    finally:
        if previous is None:
            os.environ.pop(flag, None)
        else:
            os.environ[flag] = previous


def initialize_torch_c_dlpack():
    """Initialize once, keeping the extension alive for its installed capsules."""
    global _initialized
    if _initialized or os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0") != "0":
        return
    with _lock:
        if _initialized:
            return
        from tvm_ffi import _optional_torch_c_dlpack as optional

        # An earlier independent tvm_ffi import may already have loaded it.
        if not hasattr(optional, "_LIB"):
            optional._LIB = optional.load_torch_c_dlpack_extension()
        optional.patch_torch_cuda_stream_protocol()
        _initialized = True
