#!/usr/bin/env python3
"""
Case4: TTFT (Time To First Token) Benchmark using IPC pipeline and CUDA graphs
"""

import os
import sys
import time
import json
import argparse
import statistics
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from case4_jit_client_ipc import jit_client_process
from case4_graph_generator_server_ipc import graph_generator_process


def _start_processes(model_name: str):
    """Start JIT and Generator once and wait for ready."""
    jit_request_queue = mp.Queue()
    jit_response_queue = mp.Queue()
    gen_request_queue = mp.Queue()
    gen_response_queue = mp.Queue()

    jit_proc = mp.Process(target=jit_client_process, args=(jit_request_queue, jit_response_queue, model_name))
    gen_proc = mp.Process(target=graph_generator_process, args=(gen_request_queue, gen_response_queue, model_name))
    jit_proc.start()
    gen_proc.start()

    jit_ready = False
    gen_ready = False
    start_wait = time.time()
    while not (jit_ready and gen_ready):
        if time.time() - start_wait > 300:
            raise RuntimeError("Timeout waiting for Case4 processes to be ready")
        if not jit_response_queue.empty():
            msg = jit_response_queue.get()
            if msg.get("status") == "ready":
                jit_ready = True
        if not gen_response_queue.empty():
            msg = gen_response_queue.get()
            if msg.get("status") == "ready":
                gen_ready = True
        time.sleep(0.05)

    return (jit_proc, gen_proc, jit_request_queue, jit_response_queue, gen_request_queue, gen_response_queue)


def _ensure_graph(gen_request_queue, gen_response_queue, seq_len: int):
    gen_request_queue.put({
        "cmd": "prep_next",
        "prev_seq_len": None,
        "next_seq_len": seq_len,
    })
    prep_resp = gen_response_queue.get()
    if prep_resp.get("status") != "success":
        raise RuntimeError(f"prep_next failed for seq_len={seq_len}: {prep_resp}")


def _warmup_token(gen_request_queue, gen_response_queue, jit_request_queue, jit_response_queue, tokens):
    seq_len = len(tokens)
    gen_request_queue.put({
        "cmd": "generate",
        "seq_len": seq_len,
        "input_ids": torch.tensor([tokens], dtype=torch.long),
    })
    gen_out = gen_response_queue.get()
    if gen_out.get("status") != "success":
        raise RuntimeError(f"Warmup generate failed: {gen_out}")
    logits = gen_out["logits"]
    jit_request_queue.put({
        "cmd": "sample",
        "logits": logits,
        "temperature": 1.0,
        "do_sample": False,
    })
    sample_out = jit_response_queue.get()
    if sample_out.get("status") != "success":
        raise RuntimeError(f"Warmup sample failed: {sample_out}")


def measure_ttft_for_prompt_once(model_name: str, prompt_text: str,
                                 jit_request_queue, jit_response_queue,
                                 gen_request_queue, gen_response_queue) -> float:
    """Measure TTFT using already running processes; includes a warmup replay to avoid first-use overhead."""
    # Tokenize via JIT
    jit_request_queue.put({
        "cmd": "tokenize",
        "text": prompt_text,
        "add_special_tokens": True,
    })
    resp = jit_response_queue.get()
    if resp.get("status") != "success":
        raise RuntimeError(f"Tokenization failed: {resp}")
    tokens = resp["input_ids"][0].tolist()
    seq_len = len(tokens)

    # Ensure graph exists and do a warmup token generation (excluded from timing)
    _ensure_graph(gen_request_queue, gen_response_queue, seq_len)
    _warmup_token(gen_request_queue, gen_response_queue, jit_request_queue, jit_response_queue, tokens)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed first-token generation
    start = time.time()
    gen_request_queue.put({
        "cmd": "generate",
        "seq_len": seq_len,
        "input_ids": torch.tensor([tokens], dtype=torch.long),
    })
    gen_out = gen_response_queue.get()
    if gen_out.get("status") != "success":
        raise RuntimeError(f"Generate failed: {gen_out}")
    logits = gen_out["logits"]
    jit_request_queue.put({
        "cmd": "sample",
        "logits": logits,
        "temperature": 1.0,
        "do_sample": False,
    })
    sample_out = jit_response_queue.get()
    if sample_out.get("status") != "success":
        raise RuntimeError(f"Sample failed: {sample_out}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - start) * 1000.0


def main():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may have been set already; continue
        pass
    parser = argparse.ArgumentParser(description="Case4 TTFT Benchmark (IPC)")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompts-file", type=str, required=True, help="JSON mapping length->prompt")
    parser.add_argument("--iterations", type=int, default=1, help="Number of TTFT measurements per prompt length (avg reported)")
    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        data = json.load(f)
    # Normalize to int keys, keep stable order by sorted lengths
    prompts_map = {int(k): v for k, v in data.items()}
    lengths = sorted(prompts_map.keys())

    print("=" * 50)
    print("Case4 TTFT Benchmark")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Prompt lengths: {lengths}")
    print("=" * 50)

    # Start processes once and reuse across all prompts
    jit_proc, gen_proc, jit_request_queue, jit_response_queue, gen_request_queue, gen_response_queue = _start_processes(args.model_name)

    results = []
    try:
        for L in lengths:
            prompt = prompts_map[L]
            print(f"Running TTFT for length={L} (iterations={args.iterations})...", flush=True)
            try:
                vals = []
                for _ in range(args.iterations):
                    ttft = measure_ttft_for_prompt_once(
                        args.model_name,
                        prompt,
                        jit_request_queue,
                        jit_response_queue,
                        gen_request_queue,
                        gen_response_queue,
                    )
                    vals.append(ttft)
                avg_ttft = sum(vals) / len(vals)
                std_ttft = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"TTFT avg/std: {avg_ttft:.2f}ms / {std_ttft:.2f}ms (length={L})", flush=True)
                results.append((L, avg_ttft, std_ttft))
            except Exception as e:
                print(f"  ❌ Failed for length={L}: {e}", flush=True)
    finally:
        # Cleanup processes
        try:
            jit_request_queue.put({"cmd": "stop"})
            gen_request_queue.put({"cmd": "stop"})
        except Exception:
            pass
        jit_proc.join()
        gen_proc.join()

    # Save CSV under Case4 output
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "case4_ttft_benchmark.csv")
    with open(csv_path, "w") as f:
        f.write("prompt_length,ttft_mean_ms,ttft_std_ms\n")
        for item in results:
            L = item[0]
            mean_ms = item[1]
            std_ms = item[2]
            f.write(f"{L},{mean_ms:.2f},{std_ms:.2f}\n")
    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"{'Prompt Length':<15} {'TTFT Mean (ms)':<18} {'TTFT Std (ms)':<15}")
    print("-" * 50)
    for item in results:
        L = item[0]
        mean_ms = item[1]
        std_ms = item[2]
        print(f"{L:<15} {mean_ms:<18.2f} {std_ms:<15.2f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())


