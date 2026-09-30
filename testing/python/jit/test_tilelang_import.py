import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("disabled", [None, "0", "1"])
def test_import_does_not_initialize_cuda(disabled):
    env = os.environ.copy()
    flag = "TVM_FFI_DISABLE_TORCH_C_DLPACK"
    env.pop(flag, None)
    if disabled is not None:
        env[flag] = disabled
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import os
                import sys
                import torch

                calls = []
                def unexpected_cuda_probe(*args, **kwargs):
                    calls.append(True)
                    raise AssertionError("Import must not probe or initialize CUDA")

                torch.cuda.is_available = unexpected_cuda_probe
                torch._C._cuda_getDeviceCount = unexpected_cuda_probe
                torch._C._cuda_init = unexpected_cuda_probe
                flag = "TVM_FFI_DISABLE_TORCH_C_DLPACK"
                previous = os.environ.get(flag)
                import tilelang
                import tvm_ffi
                assert not calls
                if torch.version.hip is not None or (
                    sys.platform == "darwin" and torch.backends.mps.is_available()
                ):
                    assert os.environ.get(flag) == (previous if previous is not None else "1")
                else:
                    assert os.environ.get(flag) == previous
            """),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dlpack_initialization_is_deferred_and_retained(monkeypatch):
    import torch
    from tilelang import _torch_dlpack
    from tvm_ffi import _optional_torch_c_dlpack as optional
    from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

    calls = []
    library = object()

    def load():
        calls.append("load")
        return library

    monkeypatch.delenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", raising=False)
    monkeypatch.setattr(_torch_dlpack, "_initialized", False)
    monkeypatch.setattr(optional, "load_torch_c_dlpack_extension", load)
    monkeypatch.setattr(optional, "patch_torch_cuda_stream_protocol", lambda: calls.append("stream"))
    # Track _LIB so monkeypatch also restores it after initialization.
    monkeypatch.setattr(optional, "_LIB", None, raising=False)
    monkeypatch.delattr(optional, "_LIB")

    adapter = TVMFFIKernelAdapter.__new__(TVMFFIKernelAdapter)
    adapter._prepare_torch_device(torch.device("cpu"))
    assert calls == []
    for _ in range(2):
        adapter._prepare_torch_device(torch.device("cuda"))
    assert calls == ["load", "stream"]
    assert optional._LIB is library


def test_dlpack_initialization_respects_disable(monkeypatch):
    from tilelang import _torch_dlpack
    from tvm_ffi import _optional_torch_c_dlpack as optional

    monkeypatch.setenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
    monkeypatch.setattr(_torch_dlpack, "_initialized", False)
    monkeypatch.setattr(optional, "load_torch_c_dlpack_extension", lambda: pytest.fail("DLPack extension is disabled"))
    _torch_dlpack.initialize_torch_c_dlpack()
    assert not _torch_dlpack._initialized
