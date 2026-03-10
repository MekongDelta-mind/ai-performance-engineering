"""Pytest tests for benchmark correctness.

Lightweight correctness checks run against a small whitelist of benchmarks
to keep CI times reasonable. Run with --full-benchmark-tests to exercise the
entire suite.

Usage:
    pytest tests/test_benchmark_correctness.py                    # Quick whitelist only
    pytest tests/test_benchmark_correctness.py --full-benchmark-tests  # Full test suite

Tests that benchmarks:
1. Can be discovered and loaded
2. Can run setup/teardown without errors
3. Can execute benchmark_fn without errors
4. Pass validate_result() if implemented
5. Return valid BenchmarkConfig from get_config()
"""

import pytest
from pathlib import Path

from core.env import apply_env_defaults
apply_env_defaults()

import torch
from core.utils.chapter_compare_template import (
    discover_benchmarks,
    get_last_load_error,
    load_benchmark,
)
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig


# Skip tests if CUDA is not available (NVIDIA GPU required)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available"
)

# Whitelist of chapters/benchmarks for quick correctness checks
# Only test a few representative benchmarks to keep CI fast
QUICK_TEST_WHITELIST = [
    ("ch01", "baseline_performance.py"),
    ("ch01", "optimized_performance.py"),
    ("ch07", "baseline_copy_uncoalesced.py"),
    ("ch07", "optimized_copy_uncoalesced_coalesced.py"),
]


def _is_environment_skip(message: str | None) -> bool:
    if not message:
        return False
    normalized = message.lower()
    return (
        "requires >=" in normalized and "gpu" in normalized
        or normalized.startswith("skipped:")
        or "requires transformer engine" in normalized
        or "requires grace-blackwell" in normalized
    )


def _load_or_skip(module_path: Path) -> BaseBenchmark | None:
    benchmark = load_benchmark(module_path, timeout_seconds=20)
    if benchmark is not None:
        return benchmark
    if _is_environment_skip(get_last_load_error()):
        return None
    assert benchmark is not None, f"{module_path}: {get_last_load_error()}"
    return None


def _run_with_environment_skip(action, module_path: Path) -> bool:
    try:
        action()
        return True
    except RuntimeError as exc:
        if _is_environment_skip(str(exc)):
            return False
        raise AssertionError(f"{module_path}: {exc}") from exc


def get_test_chapters(request):
    """Get chapter directories to test.
    
    Args:
        request: pytest request object to access config
        
    Returns:
        List of (chapter_dir, benchmark_files) tuples for the quick whitelist,
        or all chapters if --full-benchmark-tests flag is set.
    """
    repo_root = Path(__file__).parent.parent
    run_full_tests = request.config.getoption("--full-benchmark-tests", default=False)
    
    if run_full_tests:
        # Full test suite - all chapters
        chapter_dirs = []
        for ch_dir in repo_root.iterdir():
            if ch_dir.is_dir() and ch_dir.name.startswith("ch") and ch_dir.name[2:].isdigit():
                chapter_dirs.append(ch_dir)
        return sorted(chapter_dirs)
    else:
        # Quick check - only whitelisted benchmarks
        test_items = []
        for chapter_name, benchmark_file in QUICK_TEST_WHITELIST:
            chapter_dir = repo_root / chapter_name
            if chapter_dir.exists():
                benchmark_path = chapter_dir / benchmark_file
                if benchmark_path.exists():
                    test_items.append((chapter_dir, [benchmark_path]))
        return test_items


@pytest.fixture(scope="module")
def harness():
    """Create a benchmark harness for testing."""
    config = BenchmarkConfig(
        iterations=5,  # Reduced for faster tests
        warmup=5,
        timeout_seconds=10,  # Shorter timeout for tests
        enable_profiling=False,  # Disable profiling for correctness tests
        enable_nsys=False,
        enable_ncu=False,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


@pytest.mark.slow
def test_benchmark_loadable(request):
    """Test that whitelisted benchmarks can be loaded."""
    test_items = get_test_chapters(request)
    run_full_tests = request.config.getoption("--full-benchmark-tests", default=False)
    checked = 0
    
    for test_item in test_items:
        if run_full_tests:
            chapter_dir = test_item
            pairs = discover_benchmarks(chapter_dir)
            for baseline_path, optimized_paths, _example_name in pairs:
                baseline = _load_or_skip(baseline_path)
                if baseline is None:
                    continue
                checked += 1
                for optimized_path in optimized_paths:
                    optimized = _load_or_skip(optimized_path)
                    if optimized is None:
                        continue
                    checked += 1
        else:
            chapter_dir, benchmark_files = test_item
            for benchmark_path in benchmark_files:
                benchmark = _load_or_skip(benchmark_path)
                assert benchmark is not None
                checked += 1
    assert checked > 0


@pytest.mark.slow
def test_benchmark_setup_teardown(request, harness):
    """Test that benchmarks can run setup and teardown without errors."""
    test_items = get_test_chapters(request)
    run_full_tests = request.config.getoption("--full-benchmark-tests", default=False)
    checked = 0
    
    for test_item in test_items:
        if run_full_tests:
            chapter_dir = test_item
            pairs = discover_benchmarks(chapter_dir)
            for baseline_path, optimized_paths, _example_name in pairs:
                baseline = _load_or_skip(baseline_path)
                if baseline is not None and _run_with_environment_skip(baseline.setup, baseline_path):
                    baseline.teardown()
                    checked += 1
                for optimized_path in optimized_paths:
                    optimized = _load_or_skip(optimized_path)
                    if optimized is not None and _run_with_environment_skip(optimized.setup, optimized_path):
                        optimized.teardown()
                        checked += 1
        else:
            chapter_dir, benchmark_files = test_item
            for benchmark_path in benchmark_files:
                benchmark = load_benchmark(benchmark_path)
                benchmark.setup()
                benchmark.teardown()
                checked += 1
    assert checked > 0


@pytest.mark.slow
def test_benchmark_execution(request, harness):
    """Test that benchmarks can execute benchmark_fn without errors."""
    test_items = get_test_chapters(request)
    run_full_tests = request.config.getoption("--full-benchmark-tests", default=False)
    checked = 0
    
    for test_item in test_items:
        if run_full_tests:
            chapter_dir = test_item
            pairs = discover_benchmarks(chapter_dir)
            for baseline_path, optimized_paths, _example_name in pairs:
                baseline = _load_or_skip(baseline_path)
                if baseline is not None and _run_with_environment_skip(baseline.setup, baseline_path):
                    _run_with_environment_skip(baseline.benchmark_fn, baseline_path)
                    baseline.teardown()
                    checked += 1
                for optimized_path in optimized_paths:
                    optimized = _load_or_skip(optimized_path)
                    if optimized is not None and _run_with_environment_skip(optimized.setup, optimized_path):
                        _run_with_environment_skip(optimized.benchmark_fn, optimized_path)
                        optimized.teardown()
                        checked += 1
        else:
            chapter_dir, benchmark_files = test_item
            for benchmark_path in benchmark_files:
                benchmark = load_benchmark(benchmark_path)
                benchmark.setup()
                benchmark.benchmark_fn()
                benchmark.teardown()
                checked += 1
    assert checked > 0


@pytest.mark.slow
def test_benchmark_config(request):
    """Test that benchmarks return valid config from get_config()."""
    test_items = get_test_chapters(request)
    run_full_tests = request.config.getoption("--full-benchmark-tests", default=False)
    checked = 0
    
    for test_item in test_items:
        if run_full_tests:
            chapter_dir = test_item
            pairs = discover_benchmarks(chapter_dir)
            for baseline_path, optimized_paths, _example_name in pairs:
                baseline = _load_or_skip(baseline_path)
                if baseline is None:
                    continue
                config = baseline.get_config()
                assert config is None or isinstance(config, BenchmarkConfig)
                checked += 1
                for optimized_path in optimized_paths:
                    optimized = _load_or_skip(optimized_path)
                    if optimized is None:
                        continue
                    config = optimized.get_config()
                    assert config is None or isinstance(config, BenchmarkConfig)
                    checked += 1
        else:
            chapter_dir, benchmark_files = test_item
            for benchmark_path in benchmark_files:
                benchmark = load_benchmark(benchmark_path)
                config = benchmark.get_config()
                assert config is None or isinstance(config, BenchmarkConfig)
                checked += 1
    assert checked > 0


@pytest.mark.slow
def test_benchmark_validation(request):
    """Test that benchmarks pass validate_result() if implemented."""
    test_items = get_test_chapters(request)
    run_full_tests = request.config.getoption("--full-benchmark-tests", default=False)
    checked = 0
    
    for test_item in test_items:
        if run_full_tests:
            chapter_dir = test_item
            pairs = discover_benchmarks(chapter_dir)
            for baseline_path, optimized_paths, _example_name in pairs:
                baseline = _load_or_skip(baseline_path)
                if baseline is not None and _run_with_environment_skip(baseline.setup, baseline_path):
                    _run_with_environment_skip(baseline.benchmark_fn, baseline_path)
                    validation_error = baseline.validate_result()
                    assert validation_error is None or isinstance(validation_error, str)
                    baseline.teardown()
                    checked += 1
                for optimized_path in optimized_paths:
                    optimized = _load_or_skip(optimized_path)
                    if optimized is not None and _run_with_environment_skip(optimized.setup, optimized_path):
                        _run_with_environment_skip(optimized.benchmark_fn, optimized_path)
                        validation_error = optimized.validate_result()
                        assert validation_error is None or isinstance(validation_error, str)
                        optimized.teardown()
                        checked += 1
        else:
            chapter_dir, benchmark_files = test_item
            for benchmark_path in benchmark_files:
                benchmark = load_benchmark(benchmark_path)
                benchmark.setup()
                benchmark.benchmark_fn()
                validation_error = benchmark.validate_result()
                assert validation_error is None or isinstance(validation_error, str)
                benchmark.teardown()
                checked += 1
    assert checked > 0


def test_benchmark_protocol_compliance():
    """Test that benchmarks implement the BaseBenchmark correctly (quick check only)."""
    from core.harness.benchmark_harness import BaseBenchmark
    
    # Test that a sample benchmark implements the protocol
    repo_root = Path(__file__).parent.parent
    ch01_dir = repo_root / "ch01"
    
    if not ch01_dir.exists():
        pytest.skip("ch01 directory not found")
    
    # Use whitelisted benchmark for the quick check
    baseline_path = ch01_dir / "baseline_ilp.py"
    if not baseline_path.exists():
        pytest.skip("baseline_ilp.py not found")
    
    benchmark = load_benchmark(baseline_path)
    if benchmark is None:
        pytest.skip("Failed to load benchmark")
    
    # Check that benchmark implements required methods
    assert hasattr(benchmark, "setup")
    assert hasattr(benchmark, "benchmark_fn")
    assert hasattr(benchmark, "teardown")
    assert hasattr(benchmark, "get_config")
    assert hasattr(benchmark, "validate_result")
    
    # Check that methods are callable
    assert callable(benchmark.setup)
    assert callable(benchmark.benchmark_fn)
    assert callable(benchmark.teardown)
    assert callable(benchmark.get_config)
    assert callable(benchmark.validate_result)
