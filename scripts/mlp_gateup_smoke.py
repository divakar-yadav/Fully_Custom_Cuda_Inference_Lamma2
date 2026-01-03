import os
import multiprocessing as mp
from case4_ipc_20250130.case4_graph_generator_server_ipc import graph_generator_process

MODEL = os.environ.get("SMOKE_MODEL", "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local")

def main():
    mp.set_start_method('spawn', force=True)
    rq, rs = mp.Queue(), mp.Queue()
    p = mp.Process(target=graph_generator_process, args=(rq, rs, MODEL))
    p.start()
    ready = rs.get(timeout=120)
    print("READY:", ready, flush=True)
    rq.put({'cmd': 'mlp_gateup_smoke', 'layer_idx': 0})
    resp = rs.get(timeout=120)
    print(resp, flush=True)
    rq.put({'cmd': 'stop'})
    p.join(timeout=60)

if __name__ == "__main__":
    os.environ.setdefault("CASE4_KV_ONLY", "1")
    os.environ.setdefault("CASE4_USE_CUSTOM", "1")
    os.environ.setdefault("CASE4_SKIP_PRECAPTURE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()

