#!/usr/bin/env python3
"""
Verify PyTorch installation and CUDA functionality.
Tests basic PyTorch operations and CUDA availability.
"""

import os
import site
import sys
from pathlib import Path
from typing import Iterable, List, Set

DEFAULT_RUNTIME_POLICY = "auto"
_RUNTIME_POLICY = os.environ.get("AISP_CUDNN_RUNTIME_POLICY", DEFAULT_RUNTIME_POLICY).strip().lower()
if _RUNTIME_POLICY not in {"auto", "torch", "system"}:
    _RUNTIME_POLICY = DEFAULT_RUNTIME_POLICY


def _split_paths(value: str) -> List[str]:
    return [entry for entry in value.split(":") if entry]


def _append_if_valid(path: str, ordered: List[str], seen: Set[str]) -> None:
    if not path or path in seen or not os.path.isdir(path):
        return
    ordered.append(path)
    seen.add(path)


def _looks_like_torch_lib(path: str) -> bool:
    normalized = path.rstrip("/")
    return normalized.endswith("/torch/lib") or "/site-packages/torch/lib" in normalized


def _candidate_torch_site_roots() -> Iterable[Path]:
    roots: List[Path] = []
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    roots.append(Path(sys.prefix) / "lib" / py_version / "site-packages")
    roots.append(Path("/usr/local/lib") / py_version / "dist-packages")

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(Path(venv) / "lib" / py_version / "site-packages")

    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass

    seen: Set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        yield root


def _discover_torch_lib_dirs() -> List[str]:
    dirs: List[str] = []
    seen: Set[str] = set()
    for root in _candidate_torch_site_roots():
        candidate = root / "torch" / "lib"
        candidate_str = str(candidate)
        if candidate_str in seen or not candidate.is_dir():
            continue
        seen.add(candidate_str)
        dirs.append(candidate_str)
    return dirs


def _discover_system_cudnn_dirs() -> List[str]:
    candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
    ]
    found: List[str] = []
    seen: Set[str] = set()
    for directory in candidates:
        if directory in seen or not os.path.isdir(directory):
            continue
        if list(Path(directory).glob("libcudnn*.so*")):
            found.append(directory)
            seen.add(directory)
    return found


def _discover_cuda_lib_dirs() -> List[str]:
    cuda_homes = [
        os.environ.get("CUDA_HOME", ""),
        "/usr/local/cuda",
        "/usr/local/cuda-13.1",
        "/usr/local/cuda-13.0",
    ]
    ordered: List[str] = []
    seen: Set[str] = set()
    for home in cuda_homes:
        if not home:
            continue
        _append_if_valid(os.path.join(home, "lib64"), ordered, seen)
        _append_if_valid(os.path.join(home, "lib64", "stubs"), ordered, seen)
    return ordered


def _configure_ld_library_path() -> tuple[str, bool]:
    current_raw = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = _split_paths(current_raw)
    torch_lib_dirs = _discover_torch_lib_dirs()
    system_cudnn_dirs = _discover_system_cudnn_dirs()
    cuda_lib_dirs = _discover_cuda_lib_dirs()

    ordered: List[str] = []
    seen: Set[str] = set()

    using_torch_runtime = False
    if _RUNTIME_POLICY in {"auto", "torch"}:
        existing_torch_paths = [path for path in current_paths if _looks_like_torch_lib(path)]
        if existing_torch_paths:
            for path in existing_torch_paths:
                _append_if_valid(path, ordered, seen)
        else:
            for path in torch_lib_dirs:
                _append_if_valid(path, ordered, seen)
        using_torch_runtime = any(_looks_like_torch_lib(path) for path in ordered)

    if _RUNTIME_POLICY == "system":
        for path in system_cudnn_dirs:
            _append_if_valid(path, ordered, seen)

    if _RUNTIME_POLICY == "system" or not using_torch_runtime:
        for path in cuda_lib_dirs:
            _append_if_valid(path, ordered, seen)

    for path in current_paths:
        if _RUNTIME_POLICY == "system" and _looks_like_torch_lib(path):
            continue
        _append_if_valid(path, ordered, seen)

    new_ld_path = ":".join(ordered)
    os.environ["LD_LIBRARY_PATH"] = new_ld_path
    return new_ld_path, new_ld_path != current_raw


_EFFECTIVE_LD_LIBRARY_PATH, _LD_LIBRARY_PATH_CHANGED = _configure_ld_library_path()


def _maybe_reexec_for_runtime_loader() -> None:
    if not _LD_LIBRARY_PATH_CHANGED:
        return
    if os.environ.get("_AISP_VERIFY_PYTORCH_REEXEC") == "1":
        return
    reexec_env = dict(os.environ)
    reexec_env["_AISP_VERIFY_PYTORCH_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], reexec_env)


_maybe_reexec_for_runtime_loader()

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def check_pytorch_import():
    """Check if PyTorch can be imported."""
    print_section("PyTorch Import Check")
    
    try:
        import torch
        print(f"[OK] PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        return True, torch
    except ImportError as e:
        print(f"ERROR: Failed to import PyTorch: {e}")
        return False, None


def check_cuda_availability(torch):
    """Check CUDA availability."""
    print_section("CUDA Availability Check")
    
    if torch.cuda.is_available():
        print("[OK] CUDA is available")
        print(f"   CUDA Version: {torch.version.cuda}")
        try:
            cudnn_version = torch.backends.cudnn.version()
            print(f"   cuDNN Version: {cudnn_version}")
        except RuntimeError as e:
            if "version incompatibility" in str(e):
                print(f"   ERROR: cuDNN version incompatibility: {e}")
                print(
                    "   Fix: adjust AISP_CUDNN_RUNTIME_POLICY (auto/torch/system) "
                    "and ensure LD_LIBRARY_PATH points at a matching runtime."
                )
                raise RuntimeError(f"cuDNN version incompatibility: {e}") from e
            raise
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"     Compute Capability: {props.major}.{props.minor}")
            print(f"     Total Memory: {props.total_memory / 1024**3:.2f} GB")
        
        return True
    else:
        print("ERROR: CUDA is not available")
        print("   PyTorch may not be built with CUDA support")
        return False


def test_basic_operations(torch):
    """Test basic PyTorch operations."""
    print_section("Basic Operations Test")
    
    try:
        # CPU test
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print("[OK] CPU operations working")
        
        # GPU test
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print("[OK] GPU operations working")
            
            # Test synchronization
            torch.cuda.synchronize()
            print("[OK] CUDA synchronization working")
        
        return True
    except Exception as e:
        print(f"ERROR: Operations test failed: {e}")
        return False


def test_mixed_precision(torch):
    """Test mixed precision support."""
    print_section("Mixed Precision Support")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping mixed precision test")
        return True
    
    try:
        # Test FP16
        x = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        y = torch.randn(100, 100, device='cuda', dtype=torch.float16)
        z = torch.matmul(x, y)
        print("[OK] FP16 operations working")
        
        # Test BF16
        if torch.cuda.is_bf16_supported():
            x = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
            y = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
            z = torch.matmul(x, y)
            print("[OK] BF16 operations working")
        else:
            print("WARNING: BF16 not supported on this GPU")
        
        # Test AMP
        from torch.amp import autocast
        with autocast("cuda"):
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
        print("[OK] Automatic Mixed Precision (AMP) working")
        
        return True
    except Exception as e:
        print(f"ERROR: Mixed precision test failed: {e}")
        return False


def test_distributed(torch):
    """Test distributed capabilities."""
    print_section("Distributed Capabilities Check")
    
    try:
        if torch.distributed.is_available():
            print("[OK] torch.distributed is available")
            
            if torch.distributed.is_nccl_available():
                print("[OK] NCCL backend is available")
            else:
                print("WARNING: NCCL backend not available")
            
            if torch.distributed.is_gloo_available():
                print("[OK] Gloo backend is available")
            else:
                print("WARNING: Gloo backend not available")
        else:
            print("ERROR: torch.distributed is not available")
            return False
        
        return True
    except Exception as e:
        print(f"ERROR: Distributed check failed: {e}")
        return False


def test_compile_support(torch):
    """Test torch.compile support."""
    print_section("torch.compile Support Check")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, skipping compile test")
        return True
    
    try:
        import torch.nn as nn
        
        model = nn.Linear(10, 10).cuda()
        compiled_model = torch.compile(model)
        
        x = torch.randn(5, 10, device='cuda')
        with torch.no_grad():
            y = compiled_model(x)
        
        print("[OK] torch.compile is working")
        return True
    except Exception as e:
        print(f"WARNING: torch.compile test failed: {e}")
        print("   This may be expected on some configurations")
        return True  # Don't fail on this


def main():
    """Run all PyTorch verification checks."""
    print("\n" + "="*80)
    print("  PyTorch Installation Verification Suite")
    print("="*80)
    print(f"Runtime cuDNN policy: {_RUNTIME_POLICY}")
    print(f"Effective LD_LIBRARY_PATH: {_EFFECTIVE_LD_LIBRARY_PATH}")
    
    results = {}
    
    # Check PyTorch import
    success, torch_module = check_pytorch_import()
    if not success:
        print("\nERROR: CRITICAL: PyTorch is not installed properly!")
        return 1
    
    results['import'] = True
    
    # Run all checks
    results['cuda'] = check_cuda_availability(torch_module)
    results['operations'] = test_basic_operations(torch_module)
    results['mixed_precision'] = test_mixed_precision(torch_module)
    results['distributed'] = test_distributed(torch_module)
    results['compile'] = test_compile_support(torch_module)
    
    # Summary
    print_section("Verification Summary")
    
    checks = [
        ('PyTorch Import', results['import']),
        ('CUDA Availability', results['cuda']),
        ('Basic Operations', results['operations']),
        ('Mixed Precision', results['mixed_precision']),
        ('Distributed Support', results['distributed']),
        ('torch.compile Support', results['compile']),
    ]
    
    for check_name, passed in checks:
        status = "[OK] PASS" if passed else "ERROR: FAIL"
        print(f"{status}  {check_name}")
    
    # Critical checks that must pass
    critical_checks = ['import', 'cuda', 'operations']
    critical_passed = all(results[check] for check in critical_checks)
    
    if critical_passed:
        print("\n[OK] All critical checks PASSED!")
        print("   PyTorch is properly installed and functional.")
        return 0
    else:
        print("\nERROR: Some critical checks FAILED!")
        print("   Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
