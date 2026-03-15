import torch

from ch04.symmetric_memory_perf_common import build_square_verification_probe


def test_build_square_verification_probe_uses_available_numel_for_small_buffers() -> None:
    tensor = torch.arange(16_384, dtype=torch.float32)

    probe, probe_numel = build_square_verification_probe(tensor)

    assert probe_numel == 16_384
    assert tuple(probe.shape) == (128, 128)


def test_build_square_verification_probe_caps_probe_at_256_square() -> None:
    tensor = torch.arange(200_000, dtype=torch.float32)

    probe, probe_numel = build_square_verification_probe(tensor)

    assert probe_numel == 256 * 256
    assert tuple(probe.shape) == (256, 256)
