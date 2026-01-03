import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import case4_ipc_20250130.custom_kernels.capture_decode_step as cds
import case4_ipc_20250130.custom_kernels.d2d_row_copy as d2d
from case4_ipc_20250130.custom_kernels.weight_loader import load_and_pack

MODEL = "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local"

def layer_weights(model, l):
    lyr = model.model.layers[l]
    Wq = lyr.self_attn.q_proj.weight.detach().to(torch.float16).cuda()  # [H,H]
    Wk = lyr.self_attn.k_proj.weight.detach().to(torch.float16).cuda()
    Wv = lyr.self_attn.v_proj.weight.detach().to(torch.float16).cuda()
    Wo = lyr.self_attn.o_proj.weight.detach().to(torch.float16).cuda()  # [H,H]
    Wg = lyr.mlp.gate_proj.weight.detach().to(torch.float16).cuda()     # [I,H]
    Wu = lyr.mlp.up_proj.weight.detach().to(torch.float16).cuda()       # [I,H]
    Wd = lyr.mlp.down_proj.weight.detach().to(torch.float16).cuda()     # [H,I]
    rin = lyr.input_layernorm.weight.detach().to(torch.float16).cuda()  # [H]
    rpost = lyr.post_attention_layernorm.weight.detach().to(torch.float16).cuda()
    return Wq, Wk, Wv, Wo, Wg, Wu, Wd, rin, rpost

def rmsnorm(x, w, eps):
    var = (x.float() * x.float()).mean()
    xhat = x * torch.rsqrt(var + eps)
    return (xhat * w).to(x.dtype)

def swiglu(g, u):
    return torch.nn.functional.silu(g) * u

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok("The future of AI is", return_tensors="pt")
    last_token = int(enc["input_ids"][0, -1].item())

    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0})
    model.eval()

    H = int(getattr(cfg, "hidden_size", 4096))
    I = int(getattr(cfg, "intermediate_size", H * 2))
    V = int(getattr(cfg, "vocab_size", 32000))
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))

    # Packed pointers (device) for our kernels
    pw = load_and_pack(MODEL)

    # Build x from embedding row
    emb_w = model.get_input_embeddings().weight.detach().to(torch.float16).cuda()  # [V,H]
    x_pt = emb_w[last_token].contiguous()  # torch path
    x_k = torch.empty_like(x_pt)           # kernel path
    d2d.copy_emb_row(int(pw["emb_ptr"]), V, H, last_token, x_k)

    # Choose layer 0 for parity
    l = 0
    Wq, Wk, Wv, Wo, Wg, Wu, Wd, rin, rpost = layer_weights(model, l)

    # Torch path (T=1 attention -> ctx = V; RoPE/softmax irrelevant)
    x_in = rmsnorm(x_pt, rin, eps)                  # [H]
    q_pt = (Wq @ x_in)                              # [H]
    k_pt = (Wk @ x_in)
    v_pt = (Wv @ x_in)
    ctx_pt = v_pt                                   # T=1 => attention returns V
    y_attn_pt = (Wo @ ctx_pt) + x_pt                # residual
    x_mlp_in = rmsnorm(y_attn_pt, rpost, eps)
    g_pt = (Wg @ x_mlp_in)
    u_pt = (Wu @ x_mlp_in)
    act_pt = swiglu(g_pt, u_pt)
    mlp_pt = (Wd @ act_pt)
    y_pt = y_attn_pt + mlp_pt

    # Kernel path: use our GEMMs/rmsnorm
    # Buffers
    x_norm = torch.empty_like(x_k)
    q = torch.empty_like(x_k)
    k = torch.empty_like(x_k)
    v = torch.empty_like(x_k)
    ctx = torch.empty_like(x_k)
    y_attn = torch.empty_like(x_k)
    x_mlp_n = torch.empty_like(x_k)
    g = torch.empty((I,), dtype=torch.float16, device=device)
    u = torch.empty((I,), dtype=torch.float16, device=device)
    act = torch.empty((I,), dtype=torch.float16, device=device)
    mlp = torch.empty_like(x_k)

    cds.rmsnorm(x_k, rin, x_norm, eps)                              # input norm
    cds.down_gemm(x_norm, q, int(Wq.data_ptr()), H, H)              # q = Wq * x_norm
    cds.down_gemm(x_norm, k, int(Wk.data_ptr()), H, H)              # k = Wk * x_norm
    cds.down_gemm(x_norm, v, int(Wv.data_ptr()), H, H)              # v = Wv * x_norm
    ctx.copy_(v)                                                    # T=1
    cds.down_gemm(ctx, y_attn, int(Wo.data_ptr()), H, H)            # Wo * ctx
    y_attn.add_(x_k)                                                # + residual
    cds.rmsnorm(y_attn, rpost, x_mlp_n, eps)                        # post-attn norm
    cds.gate_up_gemm(x_mlp_n, g, u, int(Wg.data_ptr()), int(Wu.data_ptr()), H, I)
    cds.swiglu(g, u, act)
    cds.down_gemm(act, mlp, int(Wd.data_ptr()), I, H)
    y_k = y_attn + mlp

    # Report diffs
    def diff(a, b, name):
        da = (a - b)
        print(f"{name}: L2={da.float().norm().item():.6f}  max={da.abs().max().item():.6f}")

    torch.cuda.synchronize()
    print("== Parity (layer 0, T=1) ==")
    diff(q_pt, q, "Q")
    diff(k_pt, k, "K")
    diff(v_pt, v, "V")
    diff(ctx_pt, ctx, "ctx")
    diff(y_attn_pt, y_attn, "attn_out+res")
    diff(mlp_pt, mlp, "mlp")
    diff(y_pt, y_k, "final_y")

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()


