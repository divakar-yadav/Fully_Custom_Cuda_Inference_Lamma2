import os
import time
import multiprocessing as mp
import torch
from transformers import AutoTokenizer
from case4_ipc_20250130.case4_graph_generator_server_ipc import graph_generator_process

MODEL = "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local"
TOKENIZER = MODEL
PROMPT = "The future of AI is"
OUT_TXT = "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/output/case4_custom_exactseq_50.txt"


def main():
    mp.set_start_method('spawn', force=True)
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

    rq.put({"cmd": "kv_init", "session": "default", "input_ids": input_ids})
    resp = rs.get(timeout=300)
    print("KV_INIT:", type(resp), resp.get('status'), flush=True)
    if resp.get('status') != 'success':
        raise SystemExit(resp)

    # Start background capture (custom exact-seq) and pre-capture at least 150 graphs
    rq.put({"cmd": "devgraph_bg_capture_start_custom", "session": "default", "ahead": 200})
    resp = rs.get(timeout=120)
    print("BG_CAPTURE_START:", resp, flush=True)
    if resp.get('status') != 'success':
        raise SystemExit(resp)
    # Wait until at least 150 graphs are captured beyond current seq_len
    rq.put({"cmd": "devgraph_wait_precapture_custom", "session": "default", "min_ahead": 150, "timeout_s": 180})
    resp = rs.get(timeout=240)
    print("PRECATURE_READY:", resp, flush=True)
    if resp.get('status') != 'success':
        raise SystemExit(resp)
    # Stop BG capture during replay to avoid allocator/capture conflicts
    rq.put({"cmd": "devgraph_bg_capture_stop", "session": "default"})
    resp = rs.get(timeout=60)
    print("BG_CAPTURE_STOP:", resp, flush=True)

    # Replay 50 tokens in one go (graphs are captured in background)
    start_token = int(input_ids[0, -1].item())
    rq.put({
        "cmd": "devgraph_replay_seq_custom",
        "session": "default",
        "steps": 50,
        "start_token": start_token,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    })
    resp = rs.get(timeout=600)
    print("REPLAY50:", resp.get('status'), "elapsed_ms=", resp.get('elapsed_ms'), flush=True)
    if resp.get('status') != 'success':
        raise SystemExit(resp)
    ids = resp.get('tokens', [])
    text = tok.decode(ids, skip_special_tokens=True)
    preview = (text[:200].replace('\n', ' ') + '...') if len(text) > 200 else text
    print("TEXT(50):", preview, flush=True)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, 'w') as f:
        f.write(text)
    print("Saved:", OUT_TXT, flush=True)

    rq.put({"cmd": "stop"})
    p.join(timeout=60)


if __name__ == '__main__':
    os.environ.setdefault('CASE4_USE_CUSTOM', '1')
    os.environ.setdefault('CASE4_KV_ONLY', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()

