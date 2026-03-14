from pathlib import Path


def test_fp4_perchannel_sources_do_not_emit_hot_loop_nvtx_ranges() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "ch09" / "baseline_cublas_gemm_fp4_perchannel.cu",
        repo_root / "ch09" / "optimized_cublas_gemm_fp4_perchannel.cu",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert 'NVTX_RANGE("iteration")' not in text
        assert 'NVTX_RANGE("verify")' not in text
