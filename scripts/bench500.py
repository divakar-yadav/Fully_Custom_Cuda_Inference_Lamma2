import os, time, multiprocessing as mp, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from case4_ipc_20250130.case4_graph_generator_server_ipc import graph_generator_process

MODEL = os.environ.get("SMOKE_MODEL", "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local")
PROMPT = os.environ.get("SMOKE_PROMPT", "The future of AI is")
SEED = int(os.environ.get("SMOKE_SEED", "123"))
PARAMS = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4, seed=SEED)

def run_devgraph(steps=500):
    rq, rs = mp.Queue(), mp.Queue()
    p = mp.Process(target=graph_generator_process, args=(rq, rs, MODEL))
    p.start()
    rs.get(timeout=120)
    tok=AutoTokenizer.from_pretrained(MODEL)
    enc=tok(PROMPT, return_tensors='pt')
    input_ids=enc.input_ids
    rq.put({'cmd':'kv_init','session':'s','input_ids':input_ids}); rs.get(timeout=120)
    rq.put({'cmd':'devgraph_capture','session':'s'}); cap=rs.get(timeout=120)
    if cap.get('status')!='success':
        rq.put({'cmd':'stop'}); p.join(timeout=60); raise SystemExit(cap)
    rq.put({'cmd':'devgraph_replay','session':'s','steps':steps,'start_token': int(input_ids[0,-1]), **PARAMS})
    resp=rs.get(timeout=1800)
    ms=float(resp.get('elapsed_ms',0.0))
    tokens=resp.get('tokens',[])
    rq.put({'cmd':'stop'}); p.join(timeout=60)
    return ms, tokens

def run_hf(steps=500):
    set_seed(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype='auto', device_map={'':0}).eval()
    enc=tok(PROMPT, return_tensors='pt').to('cuda')
    t0=time.time()
    gen_params = {k: v for k, v in PARAMS.items() if k != 'seed'}
    out=model.generate(**enc, max_new_tokens=steps, **gen_params)
    ms=(time.time()-t0)*1000.0
    tokens=out[0].tolist()[len(enc.input_ids[0]):]
    return ms, tokens

if __name__=='__main__':
    mp.set_start_method('spawn', force=True)
    d_ms, d_tokens = run_devgraph(500)
    h_ms, h_tokens = run_hf(500)
    print(f'Devgraph: {d_ms:.2f} ms ({500/(d_ms/1000.0):.2f} tok/s)')
    print(f'HF      : {h_ms:.2f} ms ({500/(h_ms/1000.0):.2f} tok/s)')
    print('Token parity:', d_tokens==h_tokens)
    # Save Devgraph tokens to txt
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "devgraph_tokens.txt")
    with open(out_path, "w") as f:
        f.write(" ".join(map(str, d_tokens)) + "\n")

