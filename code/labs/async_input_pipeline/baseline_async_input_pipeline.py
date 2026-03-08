"""Baseline async input pipeline benchmark (blocking copies, no pinning)."""

from __future__ import annotations

from core.common.async_input_pipeline import AsyncInputPipelineBenchmark, PipelineConfig


def get_benchmark() -> AsyncInputPipelineBenchmark:
    cfg = PipelineConfig(
        batch_size=256,
        feature_shape=(3, 256, 256),
        dataset_size=2048,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        non_blocking=False,
        use_copy_stream=False,
    )
    return AsyncInputPipelineBenchmark(cfg, label="baseline_async_input_pipeline")


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
