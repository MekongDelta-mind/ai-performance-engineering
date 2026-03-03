"""Minimal compatibility helpers for local NVFP4 challenge iteration."""

from __future__ import annotations

from typing import Any, Callable

import torch


def _clone_tree(x: Any) -> Any:
    if isinstance(x, tuple):
        return tuple(_clone_tree(v) for v in x)
    if isinstance(x, list):
        return [_clone_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x


def _assert_close_tensors(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def make_match_reference(
    ref_kernel: Callable[[Any], Any],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Callable[[Callable[[Any], Any], Any], bool]:
    """Return a checker compatible with Popcorn-style `check_implementation`.

    The returned callable accepts `(custom_kernel, data)` and returns `True` on
    success; otherwise it raises an exception with a concrete mismatch reason.
    """

    def _checker(custom_kernel: Callable[[Any], Any], data: Any) -> bool:
        ref_data = _clone_tree(data)
        got_data = _clone_tree(data)

        expected = ref_kernel(ref_data)
        actual = custom_kernel(got_data)

        if expected is None:
            expected = ref_data[-1]
        if actual is None:
            actual = got_data[-1]

        if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
            raise TypeError(
                "check_implementation expected Tensor outputs; "
                f"got actual={type(actual).__name__}, expected={type(expected).__name__}"
            )
        _assert_close_tensors(actual, expected, rtol=rtol, atol=atol)
        return True

    return _checker

