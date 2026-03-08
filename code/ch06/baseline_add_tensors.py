"""Book-aligned wrapper exposing the tensor-add baseline example."""

from ch06.baseline_add import get_benchmark as _get_benchmark


def get_benchmark():
    bench = _get_benchmark()
    bench.name = "add_tensors"
    return bench
