"""TileLang-owned imports must not add CUDA discovery to dependency imports."""

import subprocess
import sys
import textwrap


def test_tilelang_import_does_not_probe_cuda():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import torch
                from tvm_ffi import _optional_torch_c_dlpack as optional

                # Older TVM-FFI versions probe CUDA on import and when Relax
                # calls this loader. Isolate TileLang's contract here; the
                # dependency's import behavior is covered in TVM-FFI tests.
                optional.load_torch_c_dlpack_extension = lambda: None
                calls = []
                def unexpected(*args, **kwargs):
                    calls.append(True)
                    raise AssertionError("TileLang import must not probe CUDA")

                torch.cuda.is_available = unexpected
                torch.cuda.current_device = unexpected
                torch.cuda.get_device_capability = unexpected
                for name in ("_cuda_getDeviceCount", "_cuda_init"):
                    if hasattr(torch._C, name):
                        setattr(torch._C, name, unexpected)
                import tilelang
                assert not calls
            """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
