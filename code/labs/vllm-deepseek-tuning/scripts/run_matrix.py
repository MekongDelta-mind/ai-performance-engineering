#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

NUM_RE = re.compile(r"([-+]?\d*\.?\d+)")


def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _wait_for_server(base_url: str, timeout_sec: int, poll_interval_sec: int, proc: Optional[subprocess.Popen] = None) -> None:
    import urllib.request

    deadline = time.time() + timeout_sec
    health = f"{base_url}/health"
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            raise RuntimeError(f"Server process exited before becoming healthy (returncode={proc.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=2) as r:  # nosec B310
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(poll_interval_sec)
    raise TimeoutError(f"Timed out waiting for {health}")


def _resolve_path(value: str, base_dir: Path, force: bool = False) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)

    candidate = (base_dir / path).resolve()
    if force or expanded.startswith("./") or expanded.startswith("../") or candidate.exists():
        return str(candidate)

    # Keep non-path identifiers (for example, HF IDs) unchanged.
    return expanded


def _load_vllm_cmd(global_cfg: dict, override: str, base_dir: Path):
    raw = override or global_cfg.get("vllm_cmd", "vllm")
    if isinstance(raw, str):
        cmd = shlex.split(raw)
    else:
        cmd = [str(x) for x in raw]

    if not cmd:
        raise ValueError("vllm_cmd resolved to an empty command")
    cmd = [os.path.expandvars(x) for x in cmd]
    if "/" in cmd[0] or cmd[0].startswith(".") or cmd[0].startswith("~"):
        expanded = Path(os.path.expanduser(cmd[0]))
        if expanded.is_absolute():
            cmd[0] = str(expanded)
        elif expanded.exists():
            cmd[0] = str(expanded.resolve())
        else:
            cmd[0] = _resolve_path(cmd[0], base_dir, force=True)
    return cmd


def _start_server(run_cfg: dict, tokenizer: str, vllm_cmd: list[str]):
    serve = run_cfg.get("serve", {})
    cmd = list(vllm_cmd) + ["serve", run_cfg["model"]]

    if serve.get("tp"):
        cmd += ["-tp", str(serve["tp"])]
    if serve.get("dp"):
        cmd += ["-dp", str(serve["dp"])]
    if serve.get("enable_expert_parallel"):
        cmd += ["--enable-expert-parallel"]

    spec = serve.get("speculative", {})
    if spec:
        if spec.get("method"):
            cmd += ["--speculative-config.method", str(spec["method"])]
        if spec.get("num_speculative_tokens") is not None:
            cmd += ["--speculative-config.num_speculative_tokens", str(spec["num_speculative_tokens"])]

    cmd += ["--tokenizer", tokenizer]
    cmd += [str(x) for x in serve.get("extra_args", [])]

    env = os.environ.copy()
    env.update({k: str(v) for k, v in run_cfg.get("env", {}).items()})

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc, cmd


def _run_bench(
    base_url: str,
    tokenizer: str,
    run_cfg: dict,
    scenario: dict,
    concurrency: int,
    global_cfg: dict,
    vllm_cmd: list[str],
):
    cmd = list(vllm_cmd) + [
        "bench",
        "serve",
        "--model",
        run_cfg["model"],
        "--seed",
        str(global_cfg.get("seed", 42)),
        "--dataset-name",
        "random",
        "--base-url",
        base_url,
        "--tokenizer",
        tokenizer,
        "--num-prompts",
        str(global_cfg.get("num_prompts", 1000)),
        "--max-concurrency",
        str(concurrency),
        "--random-input-len",
        str(scenario["isl"]),
        "--random-output-len",
        str(scenario["osl"]),
    ]
    if global_cfg.get("ignore_eos", True):
        cmd.append("--ignore-eos")

    timeout_sec = int(global_cfg.get("bench_timeout_sec", 3600))
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    return cmd, out.returncode, (out.stdout or "") + "\n" + (out.stderr or "")


def _num_from_line(line: str) -> Optional[float]:
    nums = NUM_RE.findall(line)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def _extract_metrics(raw: str) -> dict:
    metrics = {
        "prefill_lines": [],
        "decode_lines": [],
        "ttft_lines": [],
        "tpot_lines": [],
    }
    for line in raw.splitlines():
        lower = line.lower()
        if "throughput" in lower and "tok" in lower:
            if "output" in lower or "decode" in lower:
                metrics["decode_lines"].append(line.strip())
            else:
                metrics["prefill_lines"].append(line.strip())
        if "ttft" in lower:
            metrics["ttft_lines"].append(line.strip())
        if "tpot" in lower:
            metrics["tpot_lines"].append(line.strip())

    metrics["prefill_toks_per_s"] = _num_from_line(metrics["prefill_lines"][-1]) if metrics["prefill_lines"] else None
    metrics["decode_toks_per_s"] = _num_from_line(metrics["decode_lines"][-1]) if metrics["decode_lines"] else None
    return metrics


def _terminate(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def _load_concurrency(global_cfg: dict, profile: str):
    profiles = global_cfg.get("concurrency_profiles", {})
    if profile and profile in profiles:
        return profiles[profile]
    return global_cfg.get("max_concurrency_values", [8, 16, 32, 64, 128, 256])


def _capture_server_log_head(proc: subprocess.Popen, max_lines: int = 100, max_wait_sec: float = 3.0):
    if not proc.stdout:
        return []

    lines = []
    deadline = time.time() + max_wait_sec
    while len(lines) < max_lines and time.time() < deadline:
        if proc.poll() is not None:
            remaining = proc.stdout.read() or ""
            lines.extend([ln.rstrip() for ln in remaining.splitlines()])
            break
        try:
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
        except Exception:
            break
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())

    return lines[:max_lines]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--run-filter", default="", help="Substring filter for run names")
    parser.add_argument("--scenario-filter", default="", help="Substring filter for scenario names")
    parser.add_argument("--results-dir", default="", help="Override results directory")
    parser.add_argument("--tokenizer", default="", help="Override tokenizer path/id")
    parser.add_argument("--base-url", default="", help="Override vLLM serve base URL")
    parser.add_argument("--vllm-cmd", default="", help="Override vLLM command prefix")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text())
    global_cfg = cfg["global"]
    tokenizer = args.tokenizer or cfg["paths"]["tokenizer"]
    tokenizer = _resolve_path(tokenizer, config_path.parent)
    base_url = args.base_url or global_cfg.get("base_url", "http://127.0.0.1:8000")
    vllm_cmd = _load_vllm_cmd(global_cfg, args.vllm_cmd, config_path.parent)
    scenarios = cfg["scenarios"]
    conc_vals = [int(x) for x in _load_concurrency(global_cfg, args.profile)]

    results_dir_value = args.results_dir or global_cfg.get("results_dir", "results")
    results_dir = Path(_resolve_path(results_dir_value, config_path.parent, force=True))
    results_dir.mkdir(parents=True, exist_ok=True)

    for run_cfg in cfg["runs"]:
        if args.run_filter and args.run_filter not in run_cfg["name"]:
            continue

        print(f"\n=== Starting run: {run_cfg['name']} ({run_cfg['model']}) ===")
        proc, serve_cmd = _start_server(run_cfg, tokenizer, vllm_cmd=vllm_cmd)
        server_log = []
        try:
            try:
                _wait_for_server(
                    base_url,
                    timeout_sec=int(global_cfg.get("startup_timeout_sec", 900)),
                    poll_interval_sec=int(global_cfg.get("poll_interval_sec", 2)),
                    proc=proc,
                )
            except Exception as e:
                server_log = _capture_server_log_head(proc, max_lines=200, max_wait_sec=2.0)
                print(f"ERROR: server failed to become healthy for run '{run_cfg['name']}': {e}", file=sys.stderr)
                for ln in server_log[:20]:
                    print(f"[server] {ln}", file=sys.stderr)
                continue
            print("Server healthy.")
            server_log = _capture_server_log_head(proc)

            for scenario_name in run_cfg.get("scenarios", []):
                if args.scenario_filter and args.scenario_filter not in scenario_name:
                    continue
                if scenario_name not in scenarios:
                    print(f"WARNING: scenario '{scenario_name}' not found in config; skipping.", file=sys.stderr)
                    continue
                sc = scenarios[scenario_name]
                for c in conc_vals:
                    print(f"Running {run_cfg['name']} | {scenario_name} | concurrency={c}")
                    try:
                        bench_cmd, rc, raw = _run_bench(
                            base_url=base_url,
                            tokenizer=tokenizer,
                            run_cfg=run_cfg,
                            scenario=sc,
                            concurrency=c,
                            global_cfg=global_cfg,
                            vllm_cmd=vllm_cmd,
                        )
                    except subprocess.TimeoutExpired as e:
                        bench_cmd = e.cmd
                        rc = 124
                        raw = f"TIMEOUT after {e.timeout}s\n" + ((e.stdout or "") + "\n" + (e.stderr or ""))

                    stamp = _ts()
                    stem = f"{stamp}_{run_cfg['name']}_{scenario_name}_c{c}"
                    raw_path = results_dir / f"{stem}.raw.txt"
                    json_path = results_dir / f"{stem}.json"
                    raw_path.write_text(raw)

                    payload = {
                        "timestamp": stamp,
                        "run_name": run_cfg["name"],
                        "model": run_cfg["model"],
                        "scenario": scenario_name,
                        "isl": sc["isl"],
                        "osl": sc["osl"],
                        "max_concurrency": c,
                        "base_url": base_url,
                        "vllm_cmd": vllm_cmd,
                        "returncode": rc,
                        "serve_cmd": serve_cmd,
                        "bench_cmd": bench_cmd,
                        "server_log_head": server_log,
                        "metrics": _extract_metrics(raw),
                        "raw_file": str(raw_path),
                    }
                    json_path.write_text(json.dumps(payload, indent=2))

        finally:
            _terminate(proc)
            print(f"Stopped run: {run_cfg['name']}")

    print("\nDone. Results saved under", results_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
