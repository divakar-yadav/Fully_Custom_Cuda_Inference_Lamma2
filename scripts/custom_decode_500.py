import os
import time
import multiprocessing as mp
import torch
from transformers import AutoTokenizer
from case4_ipc_20250130.case4_graph_generator_server_ipc import graph_generator_process

MODEL = "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local"
TOKENIZER = MODEL
PROMPT = "The future of AI is"
OUT_DIR = "/home/azureuser/divakar_projects/cuda_graph_sharing/case4_ipc_20250130/output"
TOK_FILE = os.path.join(OUT_DIR, "custom_decode_500_tokens.txt")
TXT_FILE = os.path.join(OUT_DIR, "custom_decode_500_text.txt")


def main():
    mp.set_start_method('spawn', force=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    rq, rs = mp.Queue(), mp.Queue()
    p = mp.Process(target=graph_generator_process, args=(rq, rs, MODEL))
    p.start()

    ready = rs.get(timeout=180)
    print("READY:", ready, flush=True)
    if ready.get('status') != 'ready':
        raise SystemExit(ready)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(PROMPT, return_tensors='pt')
    input_ids = enc['input_ids']

    # Initialize KV arena
    rq.put({"cmd": "kv_init", "session": "default", "input_ids": input_ids})
    resp = rs.get(timeout=300)
    print("KV_INIT:", type(resp), resp.get('status'), flush=True)
    if resp.get('status') != 'success':
        raise SystemExit(resp)

    # Replay 500 tokens (custom exact-seq path); graphs will be captured on-demand per step
    start_token = int(input_ids[0, -1].item())
    rq.put({
        "cmd": "devgraph_replay_seq_custom",
        "session": "default",
        "steps": 500,
        "start_token": start_token,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    })
    resp = rs.get(timeout=1200)
    print("REPLAY500:", resp.get('status'), "elapsed_ms=", resp.get('elapsed_ms'), flush=True)
    if resp.get('status') != 'success':
        raise SystemExit(resp)
    ids = resp.get('tokens', [])
    print("TOKENS(500):", ids, flush=True)
    text = tok.decode(ids, skip_special_tokens=True)
    print("TEXT(500):", text[:500].replace('\n', ' ') + ("..." if len(text) > 500 else ""), flush=True)

    with open(TOK_FILE, 'w') as f:
        f.write(" ".join(str(x) for x in ids))
    with open(TXT_FILE, 'w') as f:
        f.write(text)
    print("Saved:", TOK_FILE, TXT_FILE, flush=True)

    # Shutdown
    rq.put({"cmd": "stop"})
    p.join(timeout=60)


if __name__ == '__main__':
    os.environ.setdefault('CASE4_USE_CUSTOM', '1')
    os.environ.setdefault('CASE4_KV_ONLY', '1')
    os.environ.setdefault('CASE4_SKIP_PRECAPTURE', '1')
    os.environ.setdefault('CASE4_BG_CAPTURE', '0')
    os.environ.setdefault('CASE4_DISABLE_EXT', '0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()

