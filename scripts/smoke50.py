import os, time, multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from case4_ipc_20250130.case4_graph_generator_server_ipc import graph_generator_process

MODEL = os.environ.get("SMOKE_MODEL", "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local")
PROMPT = os.environ.get("SMOKE_PROMPT", "The future of AI is")
SEED = int(os.environ.get("SMOKE_SEED", "123"))

def run_devgraph_tokens(n_tokens: int = 50):
    rq, rs = mp.Queue(), mp.Queue()
    p = mp.Process(target=graph_generator_process, args=(rq, rs, MODEL))
    p.start()
    ready = rs.get(timeout=120)
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(PROMPT, return_tensors='pt')
    input_ids = enc.input_ids
    rq.put({'cmd': 'kv_init', 'session': 's', 'input_ids': input_ids}); rs.get(timeout=120)
    rq.put({'cmd': 'devgraph_capture', 'session': 's'}); cap = rs.get(timeout=120)
    if cap.get('status') != 'success':
        rq.put({'cmd': 'stop'}); p.join(timeout=60); raise SystemExit(cap)
    rq.put({'cmd': 'devgraph_replay', 'session': 's', 'steps': n_tokens, 'start_token': int(input_ids[0, -1]),
            'do_sample': True, 'temperature': 0.7, 'top_p': 0.9, 'repetition_penalty': 1.15,
            'no_repeat_ngram_size': 4, 'seed': SEED})
    resp = rs.get(timeout=600)
    ms = float(resp.get('elapsed_ms', 0.0))
    rq.put({'cmd': 'stop'}); p.join(timeout=60)
    return resp.get('tokens', []), ms

def run_hf_tokens(n_tokens: int = 50):
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype='auto', device_map={'': 0}).eval()
    enc = tok(PROMPT, return_tensors='pt').to('cuda')
    out = model.generate(**enc, max_new_tokens=n_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                         repetition_penalty=1.15, no_repeat_ngram_size=4)
    return out[0].tolist()[len(enc.input_ids[0]):]

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    dv, ms = run_devgraph_tokens(50)
    hf = run_hf_tokens(50)
    print(f"Devgraph 50: {ms:.2f} ms ({50/(ms/1000.0):.2f} tok/s)")
    print("Equal@50:", dv == hf)
    if dv != hf:
        for i, (a, b) in enumerate(zip(dv, hf)):
            if a != b:
                print("First mismatch at", i, a, b)
                break

