"""Book-aligned wrapper exposing the optimized tensor-add example."""

from ch06.optimized_add import get_benchmark as _get_benchmark


def get_benchmark():
    bench = _get_benchmark()
    bench.name = "add_tensors"
    return bench
