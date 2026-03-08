"""Helpers for local nvfp4_gemm evaluators without sys.path mutation."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from labs.nvfp4_gemm import task as canonical_task


_MISSING = object()


def load_module_from_path(path: Path, module_name: str) -> ModuleType:
    """Load a Python module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@contextmanager
def install_module_aliases(aliases: dict[str, ModuleType]) -> Iterator[None]:
    """Temporarily install top-level module aliases for local eval imports."""
    previous = {name: sys.modules.get(name, _MISSING) for name in aliases}
    try:
        sys.modules.update(aliases)
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_utils_module(utils_file: Path, module_name: str) -> ModuleType:
    """Load the evaluator's utils module from a file path."""
    return load_module_from_path(utils_file.resolve(), module_name)


def load_reference_module(
    reference_file: Path,
    *,
    module_name: str,
    utils_module: ModuleType,
) -> ModuleType:
    """Load the reference module with stable aliases for sibling imports."""
    with install_module_aliases({"task": canonical_task, "utils": utils_module}):
        return load_module_from_path(reference_file.resolve(), module_name)


def load_submission_module(
    submission_file: Path,
    *,
    module_name: str,
    reference_module: ModuleType,
    utils_module: ModuleType,
) -> ModuleType:
    """Load the submission module with stable aliases for sibling imports."""
    aliases = {
        "task": canonical_task,
        "utils": utils_module,
        "reference_submission": reference_module,
    }
    with install_module_aliases(aliases):
        return load_module_from_path(submission_file.resolve(), module_name)
