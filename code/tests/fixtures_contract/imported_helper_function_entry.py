"""Wrapper fixture for imported helper-function hot-path checks."""

from __future__ import annotations

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark
from tests.fixtures_contract.imported_helper_common import imported_random_input


class ImportedHelperFunctionBench(VerificationPayloadMixin, BaseBenchmark):
    def setup(self):
        self._x = torch.zeros(8, 8, device=self.device)

    def benchmark_fn(self):
        self._x = imported_random_input(self.device)

    def teardown(self):
        pass

    def validate_result(self):
        return None

    def capture_verification_payload(self):
        self._set_verification_payload(
            inputs={"x": self._x},
            output=self._x,
            batch_size=1,
            parameter_count=0,
        )


def get_benchmark():
    return ImportedHelperFunctionBench()
