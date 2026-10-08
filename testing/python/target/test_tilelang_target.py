from __future__ import annotations

import pytest

import tilelang
import tilelang.testing
from tilelang.backend import get_backend
from tvm.target import Target
import tilelang.backend.target as target_registry
from tilelang.backend.target import (
    auto_detect_target,
    determine_target,
    list_target_detectors,
    register_target_detector,
)


def test_tilelang_does_not_export_target_wrapper():
    assert not hasattr(tilelang, "Target")


def test_default_target_env_accepts_json_string(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", '{"kind": "cuda", "arch": "sm_100f", "code": ["sm_100a", "sm_103a"]}')

    assert tilelang.env.get_default_target() == {
        "kind": "cuda",
        "arch": "sm_100f",
        "code": ["sm_100a", "sm_103a"],
    }


def test_default_target_env_rejects_unquoted_json_keys(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", '{kind: "cuda", arch: "sm_100f"}')

    with pytest.raises(ValueError, match="Use JSON syntax"):
        tilelang.env.get_default_target()


def test_default_target_env_keeps_plain_string(monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "cuda")

    assert tilelang.env.get_default_target() == "cuda"


def test_bare_cuda_target_uses_detected_exact_arch(monkeypatch):
    from tilelang.cuda import target as cuda_target

    monkeypatch.setattr(cuda_target, "_detect_torch_cuda_arch", lambda: "sm_90a")

    target = determine_target("cuda", return_object=True)

    assert isinstance(target, Target)
    assert target.kind.name == "cuda"
    assert str(target.attrs["arch"]) == "sm_90a"


def test_cuda_target_code_attr_survives_target_normalization():
    target = determine_target(
        {"kind": "cuda", "arch": "sm_100f", "code": ["sm_100a", "sm_103a"]},
        return_object=True,
    )

    assert isinstance(target, Target)
    assert target.kind.name == "cuda"
    assert str(target.attrs["arch"]) == "sm_100f"
    assert list(target.attrs["code"]) == ["sm_100a", "sm_103a"]
    assert list(target.export()["code"]) == ["sm_100a", "sm_103a"]


def test_cuda_target_code_attr_rejects_string_code():
    with pytest.raises(AssertionError, match="valid target config dict"):
        determine_target({"kind": "cuda", "arch": "sm_100f", "code": "sm_100a"}, return_object=True)


def test_cuda_target_rejects_compute_arch():
    with pytest.raises(AssertionError, match="valid target config dict"):
        determine_target({"kind": "cuda", "arch": "compute_90"}, return_object=True)


def test_auto_target_uses_registered_detectors():
    name = "unit-auto-target"
    old_detectors = dict(target_registry._TARGET_DETECTORS)
    try:
        target_registry._TARGET_DETECTORS.clear()
        register_target_detector(name, lambda: Target({"kind": "llvm", "mcpu": "native"}), override=True)

        target = auto_detect_target()

        assert isinstance(target, Target)
        assert target.kind.name == "llvm"
        assert str(target.attrs["mcpu"]) == "native"
        assert name in list_target_detectors()
    finally:
        target_registry._TARGET_DETECTORS.clear()
        target_registry._TARGET_DETECTORS.update(old_detectors)


def test_auto_target_detector_falls_through_none_result():
    first_name = "unit-auto-none"
    second_name = "unit-auto-fallback"
    old_detectors = dict(target_registry._TARGET_DETECTORS)
    try:
        target_registry._TARGET_DETECTORS.clear()
        register_target_detector(first_name, lambda: None, override=True)
        register_target_detector(second_name, lambda: "llvm", override=True)

        assert auto_detect_target() == "llvm"
        assert list_target_detectors() == (first_name, second_name)
    finally:
        target_registry._TARGET_DETECTORS.clear()
        target_registry._TARGET_DETECTORS.update(old_detectors)


def test_backend_module_resolves_execution_policy():
    backend = get_backend("cpu")
    c_target = Target("c")
    llvm_target = Target("llvm")

    assert backend.allowed_execution_backends(c_target) == ("cython", "tvm_ffi")
    assert backend.allowed_execution_backends(llvm_target) == ("tvm_ffi",)
    assert backend.resolve_execution_backend("auto", c_target).name == "cython"
    assert backend.resolve_execution_backend("auto", llvm_target).name == "tvm_ffi"


def test_execution_backend_registry_rejects_invalid_backend():
    target = Target("llvm")
    backend = get_backend("cpu")

    with pytest.raises(ValueError, match="Invalid execution backend"):
        backend.resolve_execution_backend("nvrtc", target)


def test_execution_backend_registry_rejects_removed_dlpack_backend():
    target = Target("llvm")
    backend = get_backend("cpu")

    with pytest.raises(ValueError, match="Invalid execution backend 'dlpack'"):
        backend.resolve_execution_backend("dlpack", target)


@pytest.mark.parametrize("arch, expected", [("sm_61", False), ("sm_80", True), ("sm_90a", True), ("sm_100f", True)])
def test_cuda_capabilities_use_explicit_arch_without_device_probes(monkeypatch, arch, expected):
    import torch
    from tilelang.contrib import nvcc

    def unexpected(*args, **kwargs):
        pytest.fail("Explicit CUDA targets must not query a runtime device")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    monkeypatch.setattr(tilelang.tvm, "cuda", unexpected)
    target = determine_target({"kind": "cuda", "arch": arch}, return_object=True)
    assert nvcc.have_tensorcore(target=target) is expected
    assert nvcc.get_target_arch_and_code(target)[0] == arch.removeprefix("sm_")
    with target:
        assert nvcc.have_tensorcore() is expected


@tilelang.testing.requires_cuda
def test_explicit_cuda_targets_lower_without_device_probes(monkeypatch):
    import torch
    import tilelang.language as T

    def unexpected(*args, **kwargs):
        pytest.fail("Explicit-target source generation must not query runtime devices")

    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    monkeypatch.setattr(tilelang.tvm, "cuda", unexpected)

    @T.prim_func
    def copy(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(128):
                B[i] = A[i] + 1

    # Resolve and lower multiple targets in one process, without cached device state.
    for arch in ("sm_80", "sm_90a", "sm_100f"):
        with Target({"kind": "cuda", "arch": arch}) as target:
            artifact = tilelang.lower(copy, target=target)
        assert str(artifact.target.attrs["arch"]) == arch
        assert "__global__" in artifact.kernel_source
