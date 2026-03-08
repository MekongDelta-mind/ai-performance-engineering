#!/usr/bin/env python3

"""Subprocess helper for dynamic_memory_allocator."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        # When profilers invoke this script directly, treat it as a no-op so runs don't fail.
        print("allocator worker invoked without arguments; skipping")
        return 0

    factory_path = sys.argv[1]
    request_file = Path(sys.argv[2])

    from ch19 import dynamic_memory_allocator as dma

    factory = dma._resolve_factory(factory_path)
    request = pickle.loads(request_file.read_bytes())
    model = factory()
    result = model.generate(request)
    sys.stdout.buffer.write(pickle.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
