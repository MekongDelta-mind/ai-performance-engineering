"""Shared helpers and data classes used across core and labs."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AsyncInputPipelineBenchmark": (
        "core.common.async_input_pipeline",
        "AsyncInputPipelineBenchmark",
    ),
    "PipelineConfig": ("core.common.async_input_pipeline", "PipelineConfig"),
    "AVAILABLE_SPEC_PRESETS": (
        "core.common.moe_parallelism_plan",
        "AVAILABLE_SPEC_PRESETS",
    ),
    "ClusterSpec": ("core.common.moe_parallelism_plan", "ClusterSpec"),
    "ModelSpec": ("core.common.moe_parallelism_plan", "ModelSpec"),
    "ParallelismPlan": ("core.common.moe_parallelism_plan", "ParallelismPlan"),
    "PlanEvaluator": ("core.common.moe_parallelism_plan", "PlanEvaluator"),
    "SPEC_PRESETS": ("core.common.moe_parallelism_plan", "SPEC_PRESETS"),
    "format_report": ("core.common.moe_parallelism_plan", "format_report"),
    "get_active_spec_preset": (
        "core.common.moe_parallelism_plan",
        "get_active_spec_preset",
    ),
    "get_default_cluster_spec": (
        "core.common.moe_parallelism_plan",
        "get_default_cluster_spec",
    ),
    "get_default_model_spec": (
        "core.common.moe_parallelism_plan",
        "get_default_model_spec",
    ),
    "resolve_specs": ("core.common.moe_parallelism_plan", "resolve_specs"),
    "set_active_spec_preset": (
        "core.common.moe_parallelism_plan",
        "set_active_spec_preset",
    ),
    "get_preferred_device": ("core.common.device_utils", "get_preferred_device"),
    "cuda_supported": ("core.common.device_utils", "cuda_supported"),
}

__all__ = [
    "AsyncInputPipelineBenchmark",
    "PipelineConfig",
    "AVAILABLE_SPEC_PRESETS",
    "ClusterSpec",
    "ModelSpec",
    "ParallelismPlan",
    "PlanEvaluator",
    "SPEC_PRESETS",
    "format_report",
    "get_active_spec_preset",
    "get_default_cluster_spec",
    "get_default_model_spec",
    "resolve_specs",
    "set_active_spec_preset",
    "get_preferred_device",
    "cuda_supported",
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
