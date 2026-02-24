from __future__ import annotations

import pytest

from core.utils.chapter_compare_template import _parse_compare_cli_args


def test_parse_compare_cli_args_non_compare_invocation_returns_defaults() -> None:
    args = _parse_compare_cli_args(["--validity-profile", "portable"], argv0="pytest")
    assert args.validity_profile is None
    assert args.profile == "none"
    assert args.examples is None


def test_parse_compare_cli_args_absent() -> None:
    args = _parse_compare_cli_args([], argv0="compare.py")
    assert args.validity_profile is None
    assert args.profile == "none"
    assert args.examples is None


def test_parse_compare_cli_args_with_validity_and_profile_and_examples() -> None:
    args = _parse_compare_cli_args(
        [
            "--validity-profile",
            "portable",
            "--profile",
            "minimal",
            "--examples",
            "gemm",
            "--examples",
            "attention,decode",
        ],
        argv0="compare.py",
    )
    assert args.validity_profile == "portable"
    assert args.profile == "minimal"
    assert args.examples == ["gemm", "attention", "decode"]


def test_parse_compare_cli_args_equals_token() -> None:
    args = _parse_compare_cli_args(["--validity-profile=strict"], argv0="compare.py")
    assert args.validity_profile == "strict"


def test_parse_compare_cli_args_unknown_flag_fails_fast() -> None:
    with pytest.raises(SystemExit):
        _parse_compare_cli_args(["--unknown-flag"], argv0="compare.py")


def test_parse_compare_cli_args_help_exits() -> None:
    with pytest.raises(SystemExit):
        _parse_compare_cli_args(["--help"], argv0="compare.py")
