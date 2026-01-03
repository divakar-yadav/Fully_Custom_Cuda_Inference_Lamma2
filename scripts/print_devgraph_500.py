import os, multiprocessing as mp, torch
from transformers import AutoTokenizer
from case4_ipc_20250130.case4_graph_generator_server_ipc import graph_generator_process

MODEL = os.environ.get("SMOKE_MODEL", "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local")
PROMPT = os.environ.get("SMOKE_PROMPT", "The future of AI is")
SEED = int(os.environ.get("SMOKE_SEED", "123"))
PARAMS = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4, seed=SEED)

def main():
    mp.set_start_method('spawn', force=True)
    rq, rs = mp.Queue(), mp.Queue()
    p = mp.Process(target=graph_generator_process, args=(rq, rs, MODEL))
    p.start()
    rs.get(timeout=120)
    tok=AutoTokenizer.from_pretrained(MODEL)
    enc=tok(PROMPT, return_tensors='pt')
    input_ids=enc.input_ids
    # kv_init timing
    import time, os
    t_kv0 = time.time()
    rq.put({'cmd':'kv_init','session':'s','input_ids':input_ids})
    kv = rs.get(timeout=120)
    kv_ms = (time.time() - t_kv0) * 1000.0
    # capture timing (exact-seq initial graph via custom kernel)
    t_cap0 = time.time()
    rq.put({'cmd':'devgraph_capture_exact_custom','session':'s'})
    cap=rs.get(timeout=120)
    cap_ms = (time.time() - t_cap0) * 1000.0
    if cap.get('status')!='success':
        rq.put({'cmd':'stop'}); p.join(timeout=60); raise SystemExit(cap)
    # start background exact-seq capture (custom)
    rq.put({'cmd':'devgraph_bg_capture_start_custom','session':'s','ahead':150}); rs.get(timeout=120)
    steps=500
    rq.put({'cmd':'devgraph_replay_seq_custom','session':'s','steps':steps,'start_token': int(input_ids[0,-1]), **PARAMS})
    resp=rs.get(timeout=1800)
    tokens=resp.get('tokens',[])
    gen_ids=torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    full_ids=torch.cat([input_ids, gen_ids], dim=1)
    text=tok.decode(full_ids[0], skip_special_tokens=True)
    # Print timings
    replay_ms = float(resp.get('elapsed_ms', 0.0))
    tok_s = steps / (replay_ms / 1000.0) if replay_ms > 0 else 0.0
    print(f"KV init: {kv_ms:.2f} ms")
    print(f"Capture (initial exact graph): {cap_ms:.2f} ms")
    print(f"Replay {steps} tokens (exact-seq graphs): {replay_ms:.2f} ms ({tok_s:.2f} tok/s)")
    print("Graphs captured: multiple (exact-seq, background)")
    print(text)
    # Save timings to CSV
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "devgraph_500_timing.csv")
    header = "kv_init_ms,capture_ms,replay_ms,tok_s,graphs_mode,steps,prompt\n"
    row = f"{kv_ms:.2f},{cap_ms:.2f},{replay_ms:.2f},{tok_s:.2f},exact-seq-bg,{steps},\"{PROMPT}\"\n"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(header)
            f.write(row)
    else:
        with open(csv_path, "a") as f:
            f.write(row)
    rq.put({'cmd':'stop'}); p.join(timeout=60)

if __name__ == "__main__":
    main()

