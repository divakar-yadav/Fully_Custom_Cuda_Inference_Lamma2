#!/usr/bin/env python3
"""
Case4 Simple P99 Benchmark - Just max_tokens and P99
"""

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import time
import argparse
import sys
import os
import statistics
from typing import List

# Add path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add both script dir and project root to path
sys.path.insert(0, script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from case4_jit_client_ipc import jit_client_process
from case4_graph_generator_server_ipc import graph_generator_process


def calculate_p99(latencies: List[float]) -> float:
    """Calculate P99 latency"""
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    index = int(len(sorted_latencies) * 0.99)
    index = min(index, len(sorted_latencies) - 1)
    return sorted_latencies[index]


def run_p99_for_length(model_name: str, max_tokens: int, iterations: int = 100):
    """Run P99 benchmark for one token length"""
    # Create IPC queues
    jit_request_queue = mp.Queue()
    jit_response_queue = mp.Queue()
    gen_request_queue = mp.Queue()
    gen_response_queue = mp.Queue()
    
    # Start processes
    jit_process = mp.Process(
        target=jit_client_process,
        args=(jit_request_queue, jit_response_queue, model_name)
    )
    jit_process.start()
    
    gen_process = mp.Process(
        target=graph_generator_process,
        args=(gen_request_queue, gen_response_queue, model_name)
    )
    gen_process.start()
    
    # Wait for ready with timeout
    jit_ready = False
    gen_ready = False
    timeout = 300  # 5 minutes max wait
    start_time = time.time()
    
    print("  Waiting for processes to be ready...", flush=True)
    
    while not (jit_ready and gen_ready):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"  ❌ Timeout waiting for processes (>{timeout}s)", flush=True)
            jit_process.terminate()
            gen_process.terminate()
            return None
        
        if not jit_response_queue.empty():
            msg = jit_response_queue.get()
            if msg.get("status") == "ready":
                jit_ready = True
                print("  ✅ JIT Client ready", flush=True)
        
        if not gen_response_queue.empty():
            msg = gen_response_queue.get()
            if msg.get("status") == "ready":
                gen_ready = True
                print("  ✅ Graph Generator ready", flush=True)
        
        time.sleep(0.5)
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            print(f"  Waiting... ({int(elapsed)}s elapsed)", flush=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompt
    test_prompt = "The future of artificial intelligence"
    
    # Tokenize initial prompt
    jit_request_queue.put({
        "cmd": "tokenize",
        "text": test_prompt,
        "add_special_tokens": True
    })
    
    response = jit_response_queue.get()
    if response["status"] != "success":
        return None
    
    initial_input_ids = response["input_ids"]
    initial_tokens = initial_input_ids[0].tolist()
    
    # Collect per-token latencies
    all_token_latencies = []
    
    for iteration in range(iterations):
        try:
            current_tokens = initial_tokens.copy()
            
            for step in range(max_tokens):
                current_len = len(current_tokens)
                
                # Generate token
                step_start = time.time()
                
                gen_request_queue.put({
                    "cmd": "generate",
                    "seq_len": current_len,
                    "input_ids": torch.tensor([current_tokens], dtype=torch.long)
                })
                
                response = gen_response_queue.get()
                if response["status"] != "success":
                    raise RuntimeError(f"Generation failed at step {step}: {response.get('error', 'unknown error')}")
                
                logits = response["logits"]
                gen_time = (time.time() - step_start) * 1000.0
                all_token_latencies.append(gen_time)
                
                # Prepare next graph and free the one we just used.
                # Do this immediately after we have the logits so it can overlap with sampling on the JIT side.
                try:
                    prev_seq = current_len
                    next_seq = current_len + 1
                    gen_request_queue.put({
                        "cmd": "prep_next",
                        "prev_seq_len": prev_seq,
                        "next_seq_len": next_seq
                    })
                    # Consume the prep_next response to keep the queue clean
                    prep_resp = gen_response_queue.get()
                    if prep_resp.get("status") != "success":
                        raise RuntimeError(f"prep_next failed: {prep_resp.get('error', 'unknown error')}")
                except Exception as prep_err:
                    raise RuntimeError(f"prep_next error at step {step}: {prep_err}")
                
                # Sample next token
                jit_request_queue.put({
                    "cmd": "sample",
                    "logits": logits,
                    "temperature": 1.0,
                    "do_sample": False
                })
                
                response = jit_response_queue.get()
                if response["status"] != "success":
                    raise RuntimeError(f"Sampling failed at step {step}: {response.get('error', 'unknown error')}")
                
                next_token = response["next_token"]
                current_tokens.append(next_token[0].item())
                
                # Check for EOS
                if next_token[0].item() == tokenizer.eos_token_id:
                    break
        
        except Exception as e:
            print(f"  ❌ Error in iteration {iteration}: {e}", flush=True)
            # Fail the benchmark if we can't generate - no silent skipping
            raise
    
    # Stop processes
    jit_request_queue.put({"cmd": "stop"})
    gen_request_queue.put({"cmd": "stop"})
    
    jit_process.join()
    gen_process.join()
    
    if not all_token_latencies:
        return None
    
    p99 = calculate_p99(all_token_latencies)
    return p99


def main():
    parser = argparse.ArgumentParser(description="Case4 P99 Benchmark")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--lengths", type=int, nargs="+", default=[10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations (defaults to 1)")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    token_lengths = args.lengths
    iterations = args.iterations
    
    print("="*50)
    print("Case4 P99 Benchmark")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Token Lengths: {token_lengths}")
    print(f"Iterations: {iterations}")
    print("="*50)
    print()
    
    results = []
    
    for max_tokens in token_lengths:
        print(f"Running max_tokens={max_tokens}...", end=" ", flush=True)
        p99 = run_p99_for_length(model_name, max_tokens, iterations)
        
        if p99:
            results.append((max_tokens, p99))
            print(f"P99: {p99:.2f}ms")
        else:
            print("FAILED")
    
    print()
    print("="*50)
    print("RESULTS")
    print("="*50)
    print(f"{'Max Tokens':<12} {'P99 (ms)':<12}")
    print("-"*50)
    
    for max_tokens, p99 in results:
        print(f"{max_tokens:<12} {p99:<12.2f}")
    
    print("="*50)
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, "case4_simple_p99.csv")
    with open(csv_file, 'w') as f:
        f.write("max_tokens,p99_ms\n")
        for max_tokens, p99 in results:
            f.write(f"{max_tokens},{p99:.2f}\n")
    
    print(f"\nResults saved to: {csv_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
