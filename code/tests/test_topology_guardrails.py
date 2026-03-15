from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

RUNTIME_TOPOLOGY_FILES = [
    Path("ch02/cpu_gpu_topology_aware.py"),
    Path("ch02/optimized_grace_coherent_memory.py"),
    Path("ch03/bind_numa_affinity.py"),
    Path("ch03/numa_topology_script.sh"),
    Path("ch03/cpu_gpu_numa_optimizations.sh"),
    Path("ch03/kubernetes_topology_pod.yaml"),
    Path("ch04/gb200_grace_numa_optimization.py"),
    Path("core/optimization/parallelism_planner/topology_detector.py"),
    Path("labs/dynamic_router/topology.py"),
]

BANNED_LITERALS = {
    "nvmlDeviceGetNUMANodeId": "Wrong NVML API symbol casing.",
    "topo -m -i": "nvidia-smi topo -m does not support -i.",
}

BANNED_HEURISTIC_SNIPPETS = {
    "numa_node = gpu_id": "Do not infer GPU locality from matching indices.",
    "gpu_id % len(topology)": "Do not infer NUMA locality from topology length modulo.",
    "i // gpus_per_numa": "Do not synthesize GPU->NUMA mappings from GPU counts.",
}


def _read(relpath: Path) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_runtime_topology_files_do_not_reintroduce_banned_literals() -> None:
    for relpath in RUNTIME_TOPOLOGY_FILES:
        text = _read(relpath)
        for literal, reason in BANNED_LITERALS.items():
            assert literal not in text, f"{relpath}: {reason}"


def test_runtime_topology_files_do_not_guess_gpu_numa_locality() -> None:
    for relpath in RUNTIME_TOPOLOGY_FILES:
        text = _read(relpath)
        for snippet, reason in BANNED_HEURISTIC_SNIPPETS.items():
            assert snippet not in text, f"{relpath}: {reason}"


def test_ch03_topology_shell_helpers_are_shell_syntax_clean() -> None:
    for relpath in (
        Path("ch03/numa_topology_script.sh"),
        Path("ch03/cpu_gpu_numa_optimizations.sh"),
    ):
        subprocess.run(
            ["bash", "-n", str(REPO_ROOT / relpath)],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )


def test_kubernetes_topology_manifest_uses_pci_bus_query_not_topo_index_hack() -> None:
    text = _read(Path("ch03/kubernetes_topology_pod.yaml"))
    assert "--query-gpu=pci.bus_id" in text
    assert "topo -m -i" not in text
