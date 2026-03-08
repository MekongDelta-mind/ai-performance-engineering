from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from core.scripts.harness.example_registry import EXAMPLE_BY_NAME, _example
from core.scripts.harness import profile_harness


def test_example_run_command_prefers_module_name_for_python_examples() -> None:
    example = _example(
        name="ch15_tensor_parallel_demo",
        path="ch15/tensor_parallel_demo.py",
        module_name="ch15.tensor_parallel_demo",
        description="demo",
    )

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch15.tensor_parallel_demo"]


def test_base_env_includes_repo_root_on_pythonpath() -> None:
    example = _example(
        name="ch15_tensor_parallel_demo",
        path="ch15/tensor_parallel_demo.py",
        module_name="ch15.tensor_parallel_demo",
        description="demo",
        tags=["ch15"],
    )

    env = profile_harness.base_env(example)

    assert str(REPO_ROOT) in env["PYTHONPATH"].split(os.pathsep)


def test_ch20_example_registry_uses_module_launch_for_ai_kernel_generator() -> None:
    example = EXAMPLE_BY_NAME["ch20_ai_kernel_generator"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch20.ai_kernel_generator"]


def test_ch05_example_registry_uses_module_launch_for_storage_demo() -> None:
    example = EXAMPLE_BY_NAME["ch05_storage_io_optimization"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch05.storage_io_optimization"]


def test_ch06_example_registry_uses_module_launch_for_add_demo() -> None:
    example = EXAMPLE_BY_NAME["ch06_add_parallel"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch06.optimized_add"]


def test_ch07_example_registry_uses_module_launch_for_memory_access_demo() -> None:
    example = EXAMPLE_BY_NAME["ch07_memory_access"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch07.memory_access_pytorch"]


def test_ch08_example_registry_uses_module_launch_for_occupancy_demo() -> None:
    example = EXAMPLE_BY_NAME["ch08_occupancy"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch08.occupancy_pytorch"]


def test_ch09_example_registry_uses_module_launch_for_fusion_demo() -> None:
    example = EXAMPLE_BY_NAME["ch09_fusion"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch09.fusion_pytorch"]


def test_ch13_example_registry_uses_module_launch_for_custom_allocator_demo() -> None:
    example = EXAMPLE_BY_NAME["ch13_custom_allocator"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch13.custom_allocator"]


def test_ch14_example_registry_uses_module_launch_for_torch_compiler_demo() -> None:
    example = EXAMPLE_BY_NAME["ch14_torch_compiler_examples"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch14.torch_compiler_examples"]


def test_ch18_example_registry_uses_module_launch_for_flexdecoding_demo() -> None:
    example = EXAMPLE_BY_NAME["ch18_flexdecoding"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch18.baseline_flexdecoding"]


def test_ch19_example_registry_uses_module_launch_for_adaptive_parallelism_demo() -> None:
    example = EXAMPLE_BY_NAME["ch19_adaptive_parallelism"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch19.adaptive_parallelism_strategy"]


def test_ch01_example_registry_uses_module_launch_for_performance_basics() -> None:
    example = EXAMPLE_BY_NAME["ch01_performance_basics"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch01.baseline_performance"]


def test_ch02_example_registry_uses_module_launch_for_hardware_info() -> None:
    example = EXAMPLE_BY_NAME["ch02_hardware_info"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch02.hardware_info"]


def test_ch03_example_registry_uses_module_launch_for_bind_numa_affinity() -> None:
    example = EXAMPLE_BY_NAME["ch03_bind_numa_affinity"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "ch03.bind_numa_affinity"]


def test_tool_memory_profiler_example_registry_uses_module_launch() -> None:
    example = EXAMPLE_BY_NAME["tool_memory_profiler"]

    cmd = profile_harness.example_run_command(example, REPO_ROOT)

    assert cmd == [sys.executable, "-m", "core.profiling.memory_profiler"]
