"""Coverage for build-cleanup overrides in run_benchmarks."""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.harness.run_benchmarks import clean_build_directories


@contextmanager
def _env_override(**updates: str | None) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_clean_build_directories_skip_override_via_skip_flag(caplog, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO)
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    with _env_override(AISP_SKIP_BUILD_CLEAN="1", AISP_CLEAN_BUILD_DIRS=None):
        clean_build_directories(chapter_dir)

    assert any("Skipping build directory cleanup" in rec.message for rec in caplog.records)


def test_clean_build_directories_skip_override_via_clean_flag(caplog, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO)
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    with _env_override(AISP_SKIP_BUILD_CLEAN=None, AISP_CLEAN_BUILD_DIRS="0"):
        clean_build_directories(chapter_dir)

    assert any("Skipping build directory cleanup" in rec.message for rec in caplog.records)
