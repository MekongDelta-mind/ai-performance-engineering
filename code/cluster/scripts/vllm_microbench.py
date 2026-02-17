#!/usr/bin/env python3
import argparse
import os
import random
import time
from statistics import mean

import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.tokenizers import get_tokenizer


def build_prompts(tokenizer, prompt_len, concurrency):
    vocab_size = getattr(tokenizer, "vocab_size", 32000)
    prompts = []
    for _ in range(concurrency):
        token_ids = [random.randint(0, vocab_size - 1) for _ in range(prompt_len)]
        prompts.append(TokensPrompt(prompt_token_ids=token_ids))
    return prompts


def run_once(llm, prompts, gen_len):
    params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=gen_len,
        detokenize=False,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    end = time.perf_counter()
    return end - start, outputs


def count_tokens(outputs):
    total_prompt = 0
    total_output = 0
    for out in outputs:
        if out.prompt_token_ids:
            total_prompt += len(out.prompt_token_ids)
        if out.outputs:
            total_output += sum(len(o.token_ids) for o in out.outputs if o)
    return total_prompt, total_output


def main():
    parser = argparse.ArgumentParser(description="Simple vLLM microbench")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    if os.environ.get("AISP_CLOCK_LOCKED") != "1":
        raise SystemExit(
            "ERROR: GPU clock lock is required for this benchmark.\n"
            "\n"
            "Run via:\n"
            "  scripts/run_with_gpu_clocks.sh -- env/venv/bin/python scripts/vllm_microbench.py ...\n"
        )

    conc_list = [int(x) for x in args.concurrency.split(",") if x.strip()]

    tokenizer = get_tokenizer(args.model)
    llm = LLM(model=args.model)

    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write("model,prompt_len,gen_len,concurrency,tok_per_s,p50_ms,p99_ms\n")

    for conc in conc_list:
        prompts = build_prompts(tokenizer, args.prompt_len, conc)

        # Warmup
        for _ in range(args.warmup):
            run_once(llm, prompts, args.gen_len)

        latencies = []
        tok_per_s = []
        for _ in range(args.iters):
            elapsed, outputs = run_once(llm, prompts, args.gen_len)
            total_prompt, total_output = count_tokens(outputs)
            total_tokens = total_prompt + total_output
            tok_per_s.append(total_tokens / elapsed if elapsed > 0 else 0.0)
            latencies.append(elapsed * 1000.0)

        p50 = float(np.percentile(latencies, 50))
        p99 = float(np.percentile(latencies, 99))
        avg_tok = mean(tok_per_s)

        with open(args.output_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{args.model},{args.prompt_len},{args.gen_len},{conc},{avg_tok:.6f},{p50:.6f},{p99:.6f}\n"
            )

        print(
            f"conc={conc} tok/s={avg_tok:.2f} p50_ms={p50:.2f} p99_ms={p99:.2f}"
        )


if __name__ == "__main__":
    main()
