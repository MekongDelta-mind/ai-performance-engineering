from __future__ import annotations

from pathlib import Path

import pytest

from core.harness import serving_stack


def test_site_package_roots_warn_on_discovery_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serving_stack.site, "getusersitepackages", lambda: (_ for _ in ()).throw(RuntimeError("user boom")))
    monkeypatch.setattr(serving_stack.site, "getsitepackages", lambda: (_ for _ in ()).throw(RuntimeError("system boom")))

    with pytest.warns(RuntimeWarning) as record:
        roots = serving_stack._site_package_roots()

    assert roots == []
    messages = [str(w.message) for w in record]
    assert any("user site-packages" in message for message in messages)
    assert any("system site-packages" in message for message in messages)


def test_get_serving_stack_pins_fails_fast_on_unreadable_requirements(tmp_path: Path) -> None:
    requirements_dir = tmp_path / "requirements_latest.txt"
    requirements_dir.mkdir()

    with pytest.raises(RuntimeError, match="Unable to read serving stack requirements file"):
        serving_stack.get_serving_stack_pins(requirements_path=requirements_dir)
