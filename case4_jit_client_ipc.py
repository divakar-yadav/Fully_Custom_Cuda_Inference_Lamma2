#!/usr/bin/env python3
"""
Case4: JIT Preprocessing Client (IPC-based)
Handles JIT-compiled dynamic operations: preprocessing and sampling
"""

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import argparse


# JIT-compiled preprocessing for dynamic shapes
@torch.jit.script
def jit_preprocessing(input_ids: torch.Tensor, target_len: int, pad_token_id: int) -> torch.Tensor:
    """JIT-compiled preprocessing for dynamic shapes"""
    current_len = input_ids.size(1)
    
    if current_len > target_len:
        return input_ids[:, :target_len]
    elif current_len < target_len:
        batch_size = input_ids.size(0)
        padding = torch.full((batch_size, target_len - current_len), pad_token_id, 
                           device=input_ids.device, dtype=input_ids.dtype)
        return torch.cat([input_ids, padding], dim=1)
    else:
        return input_ids


@torch.jit.script
def jit_sampling(logits: torch.Tensor, temperature: float, do_sample: bool) -> torch.Tensor:
    """JIT-compiled sampling"""
    if logits.dim() == 3:
        next_token_logits = logits[0, -1, :]
    elif logits.dim() == 2:
        next_token_logits = logits[-1, :]
    else:
        next_token_logits = logits
    
    if do_sample:
        scaled_logits = next_token_logits / temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
    else:
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
    
    # Ensure 2D output for consistency
    if next_token.dim() == 1:
        next_token = next_token.unsqueeze(0)
    
    return next_token


def jit_client_process(request_queue, response_queue, model_name):
    """
    JIT Preprocessing Client Process
    
    Receives preprocessing/sampling requests and handles them with JIT-compiled functions
    """
    try:
        print("[JIT Client] ===== JIT PREPROCESSING CLIENT PROCESS =====", flush=True)
        
        # Load tokenizer
        print(f"[JIT Client] Loading tokenizer: {model_name}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = tokenizer.pad_token_id
        
        print("[JIT Client] Ready!", flush=True)
        
        # Signal ready
        response_queue.put({"status": "ready", "process": "jit_client"})
        
        while True:
            request = request_queue.get()
            
            if request.get("cmd") == "stop":
                print("[JIT Client] Stopping...", flush=True)
                break
            
            elif request.get("cmd") == "preprocess":
                # Preprocess input_ids to target length
                input_ids = request["input_ids"]
                target_len = request["target_len"]
                
                try:
                    # Apply JIT-compiled preprocessing
                    processed_ids = jit_preprocessing(input_ids, target_len, pad_token_id)
                    
                    response_queue.put({
                        "status": "success",
                        "processed_ids": processed_ids
                    })
                except Exception as e:
                    response_queue.put({
                        "status": "error",
                        "error": str(e)
                    })
            
            elif request.get("cmd") == "sample":
                # Sample next token from logits
                logits = request["logits"]
                temperature = request.get("temperature", 1.0)
                do_sample = request.get("do_sample", False)
                
                try:
                    # Apply JIT-compiled sampling
                    next_token = jit_sampling(logits, temperature, do_sample)
                    
                    response_queue.put({
                        "status": "success",
                        "next_token": next_token
                    })
                except Exception as e:
                    response_queue.put({
                        "status": "error",
                        "error": str(e)
                    })
            
            elif request.get("cmd") == "tokenize":
                # Tokenize text prompt
                text = request["text"]
                add_special_tokens = request.get("add_special_tokens", True)
                
                try:
                    input_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt")
                    
                    response_queue.put({
                        "status": "success",
                        "input_ids": input_ids
                    })
                except Exception as e:
                    response_queue.put({
                        "status": "error",
                        "error": str(e)
                    })
        
        print("[JIT Client] Stopped", flush=True)
    
    except Exception as e:
        print(f"[JIT Client] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        response_queue.put({"status": "error", "error": str(e), "process": "jit_client"})


def main():
    parser = argparse.ArgumentParser(description="Case4 JIT Preprocessing Client")
    parser.add_argument("--model-name", type=str, required=True, help="Model name/path")
    
    args = parser.parse_args()
    
    # Create IPC queues
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    
    # Start JIT client process
    jit_process = mp.Process(
        target=jit_client_process,
        args=(request_queue, response_queue, args.model_name)
    )
    jit_process.start()
    
    print("[Main] JIT client process started")
    print("[Main] Waiting for ready signal...")
    
    # Wait for ready
    ready_msg = response_queue.get()
    print(f"[Main] Received: {ready_msg}")
    
    # Simple test
    print("\n[Main] Testing JIT client...")
    
    # Test 1: Tokenize
    test_prompt = "The future of AI is"
    request_queue.put({
        "cmd": "tokenize",
        "text": test_prompt
    })
    
    response = response_queue.get()
    if response["status"] == "success":
        print(f"[Test] Tokenization successful: {response['input_ids'].shape}")
        input_ids = response['input_ids']
    else:
        print(f"[Test] Tokenization failed: {response['error']}")
        return 1
    
    # Test 2: Preprocess
    request_queue.put({
        "cmd": "preprocess",
        "input_ids": input_ids,
        "target_len": 50
    })
    
    response = response_queue.get()
    if response["status"] == "success":
        print(f"[Test] Preprocessing successful: {response['processed_ids'].shape}")
    else:
        print(f"[Test] Preprocessing failed: {response['error']}")
        return 1
    
    # Test 3: Sample (dummy logits) - ensure 2D shape
    dummy_logits = torch.randn(1, 32000)
    request_queue.put({
        "cmd": "sample",
        "logits": dummy_logits,
        "temperature": 1.0,
        "do_sample": False
    })
    
    response = response_queue.get()
    if response["status"] == "success":
        print(f"[Test] Sampling successful: shape={response['next_token'].shape}, value={response['next_token']}")
    else:
        print(f"[Test] Sampling failed: {response['error']}")
        return 1
    
    # Stop
    print("\n[Main] Stopping JIT client...")
    request_queue.put({"cmd": "stop"})
    jit_process.join()
    
    print("[Main] ✅ All tests passed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
