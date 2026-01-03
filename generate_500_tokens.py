#!/usr/bin/env python3
"""
Case4: Generate exactly N (default 500) tokens using CUDA Graphs + JIT IPC
Measures total generation time (excluding process/model load), prints and saves output.
"""

import os
import sys
import time
import argparse
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

# Local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, script_dir)

from case4_jit_client_ipc import jit_client_process
from case4_graph_generator_server_ipc import graph_generator_process


def ensure_real_cuda():
    tmp_dir = "/tmp/cuda_libs"
    os.makedirs(tmp_dir, exist_ok=True)
    target = "/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05"
    link = os.path.join(tmp_dir, "libcuda.so.1")
    try:
        if not os.path.exists(link):
            os.symlink(target, link)
    except FileExistsError:
        pass
    os.environ["LD_LIBRARY_PATH"] = f"{tmp_dir}:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")


def wait_ready(jit_resp_q, gen_resp_q, timeout_s: float = 300.0) -> bool:
    jit_ready = False
    gen_ready = False
    start = time.time()
    while not (jit_ready and gen_ready):
        if (time.time() - start) > timeout_s:
            return False
        if not jit_ready and not jit_resp_q.empty():
            msg = jit_resp_q.get()
            if msg.get("status") == "ready":
                jit_ready = True
        if not gen_ready and not gen_resp_q.empty():
            msg = gen_resp_q.get()
            if msg.get("status") == "ready":
                gen_ready = True
        time.sleep(0.05)
    return True


def generate_tokens(model_name: str, text: str, max_new_tokens: int):
    ensure_real_cuda()
    # Prefer KV-only fast path in generator for long runs; also enable bg capture for overlap
    use_kv_path = os.environ.get("CASE4_USE_KV_PATH", "1") == "1"
    if use_kv_path:
    os.environ["CASE4_SKIP_PRECAPTURE"] = "1"
    os.environ["CASE4_KV_ONLY"] = "1"
    else:
        # Use CUDA Graphs mode with pre-capture and coordinated prep_next between steps
        os.environ["CASE4_SKIP_PRECAPTURE"] = "0"   # pre-capture enabled (150 graphs)
        os.environ["CASE4_KV_ONLY"] = "0"           # use CUDA graphs, not KV-only
    # Enable background capture overlap when graphs are used
    os.environ.setdefault("CASE4_BG_CAPTURE", "1")
    # Safe defaults
    os.environ.setdefault("CASE4_QUANT", "none")
    os.environ.setdefault("CASE4_ATTN", "sdpa")

    jit_req_q = mp.Queue()
    jit_resp_q = mp.Queue()
    gen_req_q = mp.Queue()
    gen_resp_q = mp.Queue()

    jit_p = mp.Process(target=jit_client_process, args=(jit_req_q, jit_resp_q, model_name))
    gen_p = mp.Process(target=graph_generator_process, args=(gen_req_q, gen_resp_q, model_name))
    jit_p.start()
    gen_p.start()

    ok = wait_ready(jit_resp_q, gen_resp_q)
    if not ok:
        jit_p.terminate(); gen_p.terminate()
        raise RuntimeError("Timeout waiting for case4 processes to be ready")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize initial prompt (with special tokens to match typical usage)
    jit_req_q.put({"cmd": "tokenize", "text": text, "add_special_tokens": True})
    resp = jit_resp_q.get()
    if resp.get("status") != "success":
        jit_p.terminate(); gen_p.terminate()
        raise RuntimeError(f"Tokenization failed: {resp.get('error')}")

    current_tokens = resp["input_ids"][0].tolist()

    # Ensure KV init to build arena and (optionally) capture custom graph
    gen_req_q.put({"cmd": "kv_init", "session": "s1", "input_ids": torch.tensor([current_tokens], dtype=torch.long)})
    init_resp = gen_resp_q.get()
    if init_resp.get("status") != "success":
        jit_p.terminate(); gen_p.terminate()
        raise RuntimeError(f"kv_init failed: {init_resp.get('error')}")

    # If KV path is selected, ask generator to run decode loop internally (single IPC)
    if use_kv_path:
        # Prefill to initialize KV (excluded from timing)
        gen_req_q.put({"cmd": "kv_init", "session": "s1", "input_ids": torch.tensor([current_tokens], dtype=torch.long)})
        init_resp = gen_resp_q.get()
        if init_resp.get("status") != "success":
            jit_p.terminate(); gen_p.terminate()
            raise RuntimeError(f"kv_init failed: {init_resp.get('error')}")
    steps = max_new_tokens
    start_token = current_tokens[-1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
    gen_req_q.put({"cmd": "kv_generate", "session": "s1", "steps": steps, "do_sample": False, "start_token": int(start_token)})
    gresp = gen_resp_q.get()
    if gresp.get("status") != "success":
        jit_p.terminate(); gen_p.terminate()
        raise RuntimeError(f"kv_generate failed: {gresp.get('error')}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = float(gresp.get("elapsed_ms", 0.0))
        current_tokens.extend(gresp["tokens"])
        total_ms = elapsed_ms
    else:
        # Timed per-token loop using CUDA graphs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        use_prep_next = os.environ.get("CASE4_PREP_NEXT", "1") == "1"
        use_custom = os.environ.get("CASE4_USE_CUSTOM", "0") == "1"
        if use_custom:
            # Use custom dynamic-length graph path (single graph replay)
            steps = max_new_tokens
            start_token = current_tokens[-1]
            gen_req_q.put({
                "cmd": "custom_generate",
                "session": "s1",
                "steps": steps,
                "start_token": int(start_token)
            })
            gresp = gen_resp_q.get()
            if gresp.get("status") != "success":
                jit_p.terminate(); gen_p.terminate()
                raise RuntimeError(f"custom_generate failed: {gresp.get('error')}")
            gen_tokens = gresp["tokens"]
            current_tokens.extend(gen_tokens)
        else:
            for step in range(max_new_tokens):
                seq_len = len(current_tokens)
                # Request replay for current seq_len
                gen_req_q.put({
                    "cmd": "generate",
                    "seq_len": int(seq_len),
                    "input_ids": torch.tensor([current_tokens], dtype=torch.long)
                })
                gresp = gen_resp_q.get()
                if gresp.get("status") != "success":
                    jit_p.terminate(); gen_p.terminate()
                    raise RuntimeError(f"generate failed at seq_len={seq_len}: {gresp.get('error')}")
                logits = gresp["logits"]  # CPU tensor (vocab)

                # Overlap: while we sample on client, ask generator to delete prev and capture next
                if use_prep_next:
                    gen_req_q.put({
                        "cmd": "prep_next",
                        "prev_seq_len": int(seq_len),
                        "next_seq_len": int(seq_len + 1)
                    })

                # Sample greedily on CPU (fast)
                jit_req_q.put({
                    "cmd": "sample",
                    "logits": logits,
                    "temperature": 1.0,
                    "do_sample": False
                })
                sresp = jit_resp_q.get()
                if sresp.get("status") != "success":
                    jit_p.terminate(); gen_p.terminate()
                    raise RuntimeError(f"sampling failed at step {step}: {sresp.get('error')}")
                next_token = int(sresp["next_token"][0, 0].item())

                # Wait for prep_next completion to ensure no capture/replay collision next step
                if use_prep_next:
                    prep_resp = gen_resp_q.get()
                    if prep_resp.get("status") != "success":
                        jit_p.terminate(); gen_p.terminate()
                        raise RuntimeError(f"prep_next failed: {prep_resp.get('error')}")

                # Append token
                current_tokens.append(next_token)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms = (time.time() - t0) * 1000.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    use_custom = os.environ.get("CASE4_USE_CUSTOM", "0") == "1"
    if use_custom:
        # Use custom dynamic-length graph path (single graph replay)
        steps = max_new_tokens
        start_token = current_tokens[-1]
        gen_req_q.put({
            "cmd": "custom_generate",
            "session": "s1",
            "steps": steps,
            "start_token": int(start_token)
        })
        gresp = gen_resp_q.get()
        if gresp.get("status") != "success":
            jit_p.terminate(); gen_p.terminate()
            raise RuntimeError(f"custom_generate failed: {gresp.get('error')}")
    gen_tokens = gresp["tokens"]
    current_tokens.extend(gen_tokens)
    else:
        for step in range(max_new_tokens):
            seq_len = len(current_tokens)
            # Request replay for current seq_len
            gen_req_q.put({
                "cmd": "generate",
                "seq_len": int(seq_len),
                "input_ids": torch.tensor([current_tokens], dtype=torch.long)
            })
            gresp = gen_resp_q.get()
            if gresp.get("status") != "success":
                jit_p.terminate(); gen_p.terminate()
                raise RuntimeError(f"generate failed at seq_len={seq_len}: {gresp.get('error')}")
            logits = gresp["logits"]  # CPU tensor (vocab)

            # Overlap: while we sample on client, ask generator to delete prev and capture next
            if use_prep_next:
                gen_req_q.put({
                    "cmd": "prep_next",
                    "prev_seq_len": int(seq_len),
                    "next_seq_len": int(seq_len + 1)
                })

            # Sample greedily on CPU (fast)
            jit_req_q.put({
                "cmd": "sample",
                "logits": logits,
                "temperature": 1.0,
                "do_sample": False
            })
            sresp = jit_resp_q.get()
            if sresp.get("status") != "success":
                jit_p.terminate(); gen_p.terminate()
                raise RuntimeError(f"sampling failed at step {step}: {sresp.get('error')}")
            next_token = int(sresp["next_token"][0, 0].item())

            # Wait for prep_next completion to ensure no capture/replay collision next step
            if use_prep_next:
                prep_resp = gen_resp_q.get()
                if prep_resp.get("status") != "success":
                    jit_p.terminate(); gen_p.terminate()
                    raise RuntimeError(f"prep_next failed: {prep_resp.get('error')}")

            # Append token
            current_tokens.append(next_token)

    # Decode final text
    generated_text = tokenizer.decode(current_tokens, skip_special_tokens=True)

    # Shutdown
    jit_req_q.put({"cmd": "stop"})
    gen_req_q.put({"cmd": "stop"})
    jit_p.join(); gen_p.join()

    return total_ms, max_new_tokens / (total_ms / 1000.0), generated_text


def main():
    # Ensure safe CUDA multiprocessing (avoid fork-related CUDA init errors)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Case4: generate 500 tokens and time it")
    parser.add_argument("--model-name", type=str, default=os.path.join(project_root, "latest_case5", "llama2_hf_local"))
    parser.add_argument("--input-text", type=str, default="The future of artificial intelligence")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()

    total_ms, tps, text = generate_tokens(args.model_name, args.input_text, args.max_new_tokens)

    print("=" * 80)
    print("CASE 4 (" + ("Custom decode graph" if (os.environ.get("CASE4_USE_CUSTOM","0")=="1") else "CUDA Graphs + JIT IPC") + "): generation timing")
    print("=" * 80)
    print(f"Total time for {args.max_new_tokens} token generation: {total_ms:.2f} ms")
    print(f"Tokens/sec: {tps:.2f}")
    print("-" * 80)
    print("Generated output:")
    print(text)

    out_dir = os.path.join(script_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "case4_generate_500_tokens.csv")
    out_txt = os.path.join(out_dir, "case4_generate_500_tokens_output.txt")
    with open(out_txt, "w") as f:
        f.write(text)
    header = "max_new_tokens,total_ms,tokens_per_second,output_file\n"
    row = f"{args.max_new_tokens},{total_ms},{tps},{out_txt}\n"
    if not os.path.exists(out_csv):
        with open(out_csv, "w") as f:
            f.write(header)
            f.write(row)
    else:
        with open(out_csv, "a") as f:
            f.write(row)


if __name__ == "__main__":
    sys.exit(main())


