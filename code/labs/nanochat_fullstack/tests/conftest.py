"""Test configuration for nanochat_fullstack tests."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
importlib.import_module("labs.nanochat_fullstack")


def _ensure_rustbpe_extension() -> None:
    """Ensure `import rustbpe` resolves to the compiled Python extension.

    The repository contains a Rust crate at `labs/nanochat_fullstack/rustbpe/`.
    When `labs/nanochat_fullstack/` is on `sys.path`, that directory can be
    imported as a *namespace package* named `rustbpe`, which shadows the real
    extension module and lacks `Tokenizer`.

    This helper copies the built pyo3 cdylib (`librustbpe.so`) into
    `labs/nanochat_fullstack/rustbpe.so` so Python imports the extension first.
    """
    try:
        module = importlib.import_module("rustbpe")
        if hasattr(module, "Tokenizer"):
            return
    except Exception:
        pass

    crate_dir = PROJECT_ROOT / "rustbpe"
    release_dir = crate_dir / "target" / "release"
    src = release_dir / "librustbpe.so"
    dest = PROJECT_ROOT / "rustbpe.so"

    if not src.exists():
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=str(crate_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to build rustbpe extension via cargo.\n"
                f"cwd: {crate_dir}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
            )

    if not src.exists():
        raise RuntimeError(f"rustbpe build artifact not found: {src}")

    shutil.copyfile(src, dest)

    sys.modules.pop("rustbpe", None)
    spec = importlib.util.spec_from_file_location("rustbpe", dest)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load rustbpe extension from {dest}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["rustbpe"] = module
    if not hasattr(module, "Tokenizer"):
        raise RuntimeError("rustbpe extension loaded but Tokenizer is missing")


_ensure_rustbpe_extension()
