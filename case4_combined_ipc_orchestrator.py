#!/usr/bin/env python3
"""
Case4: Combined IPC Orchestrator
Coordinates JIT client and Graph Generator server processes
"""

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import time
import argparse


# Import the process functions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from case4_jit_client_ipc import jit_client_process
from case4_graph_generator_server_ipc import graph_generator_process


def run_combined_inference(model_name, prompt, max_new_tokens=10):
    """
    Run combined inference using JIT client + Graph generator server
    """
    print("="*80)
    print("CASE4: COMBINED IPC INFERENCE")
    print("="*80)
    
    # Create IPC queues
    # JIT client queues
    jit_request_queue = mp.Queue()
    jit_response_queue = mp.Queue()
    
    # Graph generator queues
    gen_request_queue = mp.Queue()
    gen_response_queue = mp.Queue()
    
    # Start JIT client process
    print("\n[Main] Starting JIT client process...")
    jit_process = mp.Process(
        target=jit_client_process,
        args=(jit_request_queue, jit_response_queue, model_name)
    )
    jit_process.start()
    
    # Start Graph generator process
    print("[Main] Starting graph generator process...")
    gen_process = mp.Process(
        target=graph_generator_process,
        args=(gen_request_queue, gen_response_queue, model_name)
    )
    gen_process.start()
    
    # Wait for both to be ready
    print("[Main] Waiting for processes to initialize...")
    
    jit_ready = False
    gen_ready = False
    
    while not (jit_ready and gen_ready):
        # Check JIT client
        if not jit_response_queue.empty():
            msg = jit_response_queue.get()
            if msg.get("status") == "ready":
                jit_ready = True
                print("[Main] ✅ JIT client ready")
        
        # Check generator
        if not gen_response_queue.empty():
            msg = gen_response_queue.get()
            if msg.get("status") == "ready":
                gen_ready = True
                print("[Main] ✅ Graph generator ready")
        
        time.sleep(0.1)
    
    # Load tokenizer locally for decoding
    print("\n[Main] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run inference
    print("\n[Main] Running combined inference...")
    print(f"[Main] Prompt: '{prompt}'")
    
    # Step 1: Tokenize via JIT client
    jit_request_queue.put({
        "cmd": "tokenize",
        "text": prompt,
        "add_special_tokens": True
    })
    
    response = jit_response_queue.get()
    if response["status"] != "success":
        print(f"❌ Tokenization failed: {response['error']}")
        return
    
    input_ids = response["input_ids"]
    print(f"[Main] Tokenized: {input_ids.shape}")
    
    # Generate tokens
    all_input_ids = input_ids.clone()
    times = []
    
    for step in range(max_new_tokens):
        current_len = all_input_ids.size(1)
        
        # Step 2: Generate via Graph Generator
        step_start = time.time()
        
        gen_request_queue.put({
            "cmd": "generate",
            "seq_len": current_len,
            "input_ids": all_input_ids
        })
        
        response = gen_response_queue.get()
        if response["status"] != "success":
            print(f"❌ Generation failed at step {step}: {response['error']}")
            break
        
        logits = response["logits"]
        gen_time = (time.time() - step_start) * 1000.0
        times.append(gen_time)
        
        # Step 3: Sample via JIT client
        jit_request_queue.put({
            "cmd": "sample",
            "logits": logits,
            "temperature": 1.0,
            "do_sample": False
        })
        
        response = jit_response_queue.get()
        if response["status"] != "success":
            print(f"❌ Sampling failed at step {step}: {response['error']}")
            break
        
        next_token = response["next_token"]
        
        # Step 4: Append token
        all_input_ids = torch.cat([all_input_ids, next_token], dim=1)
        
        # Decode and print
        next_word = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(f"[Step {step+1}] Time: {gen_time:.2f}ms | Token: '{next_word}'")
        
        if next_token[0].item() == tokenizer.eos_token_id:
            print("[Main] EOS token generated")
            break
    
    # Stop processes
    print("\n[Main] Stopping processes...")
    jit_request_queue.put({"cmd": "stop"})
    gen_request_queue.put({"cmd": "stop"})
    
    jit_process.join()
    gen_process.join()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated {len(times)} tokens")
    print(f"Avg time: {sum(times)/len(times):.2f}ms/token")
    print(f"Total time: {sum(times):.2f}ms")
    
    # Decode full sequence
    full_text = tokenizer.decode(all_input_ids[0], skip_special_tokens=True)
    print(f"\nFull output: '{full_text}'")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Case4 Combined IPC Orchestrator")
    parser.add_argument("--model-name", type=str, required=True, help="Model name/path")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    try:
        run_combined_inference(args.model_name, args.prompt, args.max_tokens)
        return 0
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
