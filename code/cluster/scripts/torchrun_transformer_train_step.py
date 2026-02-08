#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from core.harness.benchmark_harness import (  # type: ignore
    _resolve_physical_device_index,
    lock_gpu_clocks,
    ramp_gpu_clocks,
)
from core.utils.logger import get_logger, setup_logging


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _app_clock_snapshot(device_index: int) -> Dict[str, Any]:
    try:
        import pynvml
    except ImportError as exc:
        return {"error": f"pynvml import failed: {exc}"}
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM))
        cur_sm = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        cur_mem = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
        return {
            "applications_sm_mhz": app_sm,
            "applications_mem_mhz": app_mem,
            "current_sm_mhz": cur_sm,
            "current_mem_mhz": cur_mem,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads

        self.ln1 = nn.LayerNorm(hidden, elementwise_affine=True)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

        self.ln2 = nn.LayerNorm(hidden, elementwise_affine=True)
        self.fc1 = nn.Linear(hidden, hidden * mlp_ratio, bias=False)
        self.fc2 = nn.Linear(hidden * mlp_ratio, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        y = self.ln1(x)
        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.heads, self.head_dim).transpose(1, 2)  # (b, nh, s, hd)
        k = k.view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, s, h)
        x = x + self.proj(attn)

        y = self.ln2(x)
        y = self.fc2(F.gelu(self.fc1(y), approximate="tanh"))
        x = x + y
        return x


class ToyTransformer(nn.Module):
    def __init__(self, hidden: int, layers: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(hidden, heads, mlp_ratio) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(hidden, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x)


def _percentiles(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"p50": None, "p99": None}
    ys = sorted(xs)
    p50 = ys[len(ys) // 2]
    p99 = ys[max(0, int(0.99 * len(ys)) - 1)]
    return {"p50": p50, "p99": p99}


def _try_make_optimizer(params, lr: float) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    meta: Dict[str, Any] = {"name": "AdamW", "fused": False, "lr": lr}
    try:
        opt = torch.optim.AdamW(params, lr=lr, fused=True)
        meta["fused"] = True
        return opt, meta
    except TypeError:
        pass
    except RuntimeError:
        pass
    opt = torch.optim.AdamW(params, lr=lr)
    return opt, meta


def main() -> int:
    setup_logging(level="INFO")
    logger = get_logger("torchrun_train_step")

    ap = argparse.ArgumentParser(description="Tiny transformer train-step benchmark (bf16/fp16), multi-node via torchrun.")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    ap.add_argument("--label", default=os.environ.get("LABEL", ""))
    ap.add_argument("--output-json", default="", help="Rank0 output JSON path (recommended: results/structured/...).")
    ap.add_argument("--steps", type=int, default=30, help="Measured steps.")
    ap.add_argument("--warmup-steps", type=int, default=5)

    ap.add_argument("--batch-size", type=int, default=2, help="Per-rank batch size.")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--mlp-ratio", type=int, default=4)

    ap.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--fsdp", type=int, default=1, help="1 to run FSDP FULL_SHARD (reduce-scatter/all-gather).")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    local_rank = _env_int("LOCAL_RANK", 0)
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    node = socket.gethostname()

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        require_lock = True
        ramp_requested = _env_bool("RAMP_GPU_CLOCKS", default=True)

        # Lock clocks per-rank and enforce "all ranks locked" collectively.
        physical_index = int(_resolve_physical_device_index(local_rank))
        with lock_gpu_clocks(device=local_rank) as locked:
            theoretical_tflops, theoretical_mem_gbps = locked
            lock_meta = {
                "locked": bool(theoretical_tflops) or bool(theoretical_mem_gbps),
                "theoretical_tflops_fp16": theoretical_tflops,
                "theoretical_mem_gbps": theoretical_mem_gbps,
            }
            clocks = _app_clock_snapshot(physical_index)
            clock_payload = {
                "global_rank": rank,
                "local_rank": local_rank,
                "node": node,
                "physical_gpu": physical_index,
                "lock": lock_meta,
                "clocks": clocks,
            }
            print(f"APP_CLOCKS {json.dumps(clock_payload, sort_keys=True)}", flush=True)

            ok = 1 if lock_meta["locked"] else 0
            ok_t = torch.tensor([ok], device="cuda", dtype=torch.int32)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            lock_ok_all = int(ok_t.item())
            if require_lock and lock_ok_all != 1:
                if rank == 0:
                    logger.error("Clock lock required but at least one rank failed to lock.")
                return 3

            gathered_clocks: List[Dict[str, Any]] = [None] * world_size
            dist.all_gather_object(gathered_clocks, clock_payload)

            if ramp_requested:
                ramp_gpu_clocks(device=local_rank)

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

            model = ToyTransformer(hidden=args.hidden, layers=args.layers, heads=args.heads, mlp_ratio=args.mlp_ratio).cuda()
            fsdp_enabled = bool(args.fsdp)
            fsdp_meta: Dict[str, Any] = {"enabled": fsdp_enabled}
            if fsdp_enabled:
                from torch.distributed.fsdp import (  # type: ignore
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision,
                    ShardingStrategy,
                )

                mp = MixedPrecision(param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype)
                model = FSDP(
                    model,
                    mixed_precision=mp,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=torch.cuda.current_device(),
                    use_orig_params=True,
                )
                fsdp_meta.update({"sharding": "FULL_SHARD", "mixed_precision": args.precision})

            opt, opt_meta = _try_make_optimizer(model.parameters(), lr=args.lr)

            # Fixed inputs to avoid measuring host RNG and allocations.
            x = torch.randn((args.batch_size, args.seq_len, args.hidden), device="cuda", dtype=amp_dtype)
            target = torch.randn_like(x)

            # A small sync point to align ranks before warmup.
            dist.barrier()

            # Warmup (not recorded).
            for _ in range(args.warmup_steps):
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(x)
                    loss = F.mse_loss(out, target)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            dist.barrier()

            # Measure.
            step_times: List[float] = []
            for _ in range(args.steps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(x)
                    loss = F.mse_loss(out, target)
                loss.backward()
                opt.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                # Record the max step time across ranks (tail latency proxy).
                t = torch.tensor([elapsed], device="cuda", dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                if rank == 0:
                    step_times.append(float(t.item()))

            if rank == 0:
                avg_s = mean(step_times) if step_times else 0.0
                pct = _percentiles(step_times)
                tokens_per_step = int(args.batch_size) * int(args.seq_len) * int(world_size)
                tok_s = (tokens_per_step / avg_s) if avg_s > 0 else 0.0

                fp8_supported = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")
                te_available = False
                try:
                    import transformer_engine  # type: ignore

                    _ = transformer_engine
                    te_available = True
                except Exception:
                    te_available = False

                payload = {
                    "run_id": args.run_id,
                    "label": args.label,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "hosts": sorted(set([str(x.get("node", "")) for x in gathered_clocks if isinstance(x, dict)])),
                    "world_size": world_size,
                    "local_rank_count": _env_int("LOCAL_WORLD_SIZE", 0),
                    "backend": "nccl",
                    "precision": args.precision,
                    "config": {
                        "batch_size_per_rank": args.batch_size,
                        "seq_len": args.seq_len,
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "heads": args.heads,
                        "mlp_ratio": args.mlp_ratio,
                        "warmup_steps": args.warmup_steps,
                        "steps": args.steps,
                    },
                    "optimizer": opt_meta,
                    "fsdp": fsdp_meta,
                    "results": {
                        "tokens_per_step": tokens_per_step,
                        "avg_step_s": avg_s,
                        "p50_step_s": pct["p50"],
                        "p99_step_s": pct["p99"],
                        "tokens_per_s": tok_s,
                        "step_times_s": step_times,
                    },
                    "app_clocks": gathered_clocks,
                    "fp8": {
                        "torch_float8_supported": bool(fp8_supported),
                        "transformer_engine_available": bool(te_available),
                        "note": "This benchmark currently runs bf16/fp16 only; use FP8 training stacks (e.g. TransformerEngine) separately.",
                    },
                    "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
                }

                if args.output_json:
                    out_path = Path(args.output_json)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    logger.info("Wrote %s", out_path)
                else:
                    print(json.dumps(payload, indent=2, sort_keys=True))

            return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

