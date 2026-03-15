"""Benchmark framework core components.

This package provides the core infrastructure for benchmark verification
and correctness enforcement.

Key Modules:
- verification: Data models (InputSignature, ToleranceSpec, etc.)
- quarantine: QuarantineManager for non-compliant benchmarks
- verify_runner: VerifyRunner for verification execution
- contract: BenchmarkContract defining required interface
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ComparisonDetails": ("core.benchmark.verification", "ComparisonDetails"),
    "EnforcementPhase": ("core.benchmark.verification", "EnforcementPhase"),
    "InputSignature": ("core.benchmark.verification", "InputSignature"),
    "PrecisionFlags": ("core.benchmark.verification", "PrecisionFlags"),
    "QuarantineReason": ("core.benchmark.verification", "QuarantineReason"),
    "QuarantineRecord": ("core.benchmark.verification", "QuarantineRecord"),
    "ToleranceSpec": ("core.benchmark.verification", "ToleranceSpec"),
    "VerifyResult": ("core.benchmark.verification", "VerifyResult"),
    "DEFAULT_TOLERANCES": ("core.benchmark.verification", "DEFAULT_TOLERANCES"),
    "get_enforcement_phase": ("core.benchmark.verification", "get_enforcement_phase"),
    "get_tolerance_for_dtype": ("core.benchmark.verification", "get_tolerance_for_dtype"),
    "is_verification_enabled": ("core.benchmark.verification", "is_verification_enabled"),
    "set_deterministic_seeds": ("core.benchmark.verification", "set_deterministic_seeds"),
    "QuarantineManager": ("core.benchmark.quarantine", "QuarantineManager"),
    "detect_skip_flags": ("core.benchmark.quarantine", "detect_skip_flags"),
    "check_benchmark_compliance": ("core.benchmark.quarantine", "check_benchmark_compliance"),
    "GoldenOutput": ("core.benchmark.verify_runner", "GoldenOutput"),
    "GoldenOutputCache": ("core.benchmark.verify_runner", "GoldenOutputCache"),
    "VerifyConfig": ("core.benchmark.verify_runner", "VerifyConfig"),
    "VerifyRunner": ("core.benchmark.verify_runner", "VerifyRunner"),
    "BenchmarkContract": ("core.benchmark.contract", "BenchmarkContract"),
    "check_benchmark_file": ("core.benchmark.contract", "check_benchmark_file"),
    "check_benchmark_file_ast": ("core.benchmark.contract", "check_benchmark_file_ast"),
}

__all__ = [
    "ComparisonDetails",
    "EnforcementPhase",
    "InputSignature",
    "PrecisionFlags",
    "QuarantineReason",
    "QuarantineRecord",
    "ToleranceSpec",
    "VerifyResult",
    "DEFAULT_TOLERANCES",
    "get_enforcement_phase",
    "get_tolerance_for_dtype",
    "is_verification_enabled",
    "set_deterministic_seeds",
    "QuarantineManager",
    "detect_skip_flags",
    "check_benchmark_compliance",
    "GoldenOutput",
    "GoldenOutputCache",
    "VerifyConfig",
    "VerifyRunner",
    "BenchmarkContract",
    "check_benchmark_file",
    "check_benchmark_file_ast",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
