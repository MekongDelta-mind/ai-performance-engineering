"""Full-stack NanoChat implementation (Karpathy's nanochat project)."""

from __future__ import annotations

import sys

from . import nanochat as _nanochat


# Preserve NanoChat's historical absolute import style without mutating sys.path.
sys.modules.setdefault("nanochat", _nanochat)
