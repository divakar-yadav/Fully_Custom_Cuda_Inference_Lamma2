import os
import math
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import case4_ipc_20250130.custom_kernels.capture_decode_step as cds
import case4_ipc_20250130.custom_kernels.attn_varlen as attn_varlen
import case4_ipc_20250130.custom_kernels.d2d_row_copy as d2d
from case4_ipc_20250130.custom_kernels.weight_loader import load_and_pack

MODEL = "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local"
PROMPT = "The future of AI is"
STEPS = 8

def rmsnorm_host(x, w, eps):
    var = (x.float() * x.float()).mean()
    xhat = x * torch.rsqrt(var + eps)
    return (xhat * w).to(x.dtype)

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(PROMPT, return_tensors="pt")

    cfg = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map={"":0})
    model.eval()

    H = int(getattr(cfg, "hidden_size", 4096))
    L = int(getattr(cfg, "num_hidden_layers", 32))
    Hh = int(getattr(cfg, "num_attention_heads", 32))
    Kh = int(getattr(cfg, "num_key_value_heads", Hh))
    D = H // Hh
    V = int(getattr(cfg, "vocab_size", 32000))
    eps = float(getattr(cfg, "rms_norm_eps", 1e-6))
    I = int(getattr(cfg, "intermediate_size", H*2))
    max_len = 1024

    # Load packed weights (device pointers)
    pw = load_and_pack(MODEL)

    # Arena (our path)
    arena_k = torch.empty((L, 1, Kh, max_len, D), dtype=torch.float16, device=device)
    arena_v = torch.empty_like(arena_k)
    seq_len_dev = torch.tensor([0], dtype=torch.int32, device=device)
    pos_dev = torch.tensor([0], dtype=torch.int32, device=device)

    # Scratch
    x = torch.empty((H,), dtype=torch.float16, device=device)
    x_norm = torch.empty((H,), dtype=torch.float16, device=device)
    attn_ctx = torch.empty((H,), dtype=torch.float16, device=device)
    mlp_out = torch.empty((H,), dtype=torch.float16, device=device)
    q_vec = torch.empty((H,), dtype=torch.float16, device=device)
    k_vec = torch.empty((Kh * D,), dtype=torch.float16, device=device)
    v_vec = torch.empty((Kh * D,), dtype=torch.float16, device=device)
    gate = torch.empty((I,), dtype=torch.float16, device=device)
    up = torch.empty((I,), dtype=torch.float16, device=device)
    act = torch.empty((I,), dtype=torch.float16, device=device)
    logits = torch.empty((1, V), dtype=torch.float32, device=device)

    # Prefill from full prompt for both paths
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, return_dict=True)
    hf_cache = out.past_key_values  # HF cache at end of prompt
    prompt_ids = ids[0].tolist()
    # RoPE params
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    rotary_dim = int(getattr(cfg, "rotary_dim", D))
    rotary_dim = max(2, min(rotary_dim, D))

    def apply_rope_(hd_mat: torch.Tensor, pos: int, d_rot: int, base: float, device: torch.device):
        # hd_mat: [num_heads, D], rotate first d_rot dims using HF rotate_half convention
        d = int(d_rot)
        if d <= 0:
            return hd_mat
        rot = hd_mat[:, :d]
        half = d // 2
        idx = torch.arange(half, device=device, dtype=torch.float32)
        inv = (base ** (-idx / float(half))).view(1, -1)  # [1, half]
        angle = pos * inv
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        x1 = rot[:, :half].float()
        x2 = rot[:, half:d].float()
        r_first = x1 * cos - x2 * sin
        r_second = x2 * cos + x1 * sin
        rot[:, :half] = r_first.to(rot.dtype)
        rot[:, half:d] = r_second.to(rot.dtype)
        hd_mat[:, :d] = rot
        return hd_mat
    # Our prefill to build arena_k/v up to len(prompt_ids)
    seq_len_dev.fill_(0)
    pos_dev.fill_(0)
    for pos, tok_id in enumerate(prompt_ids):
        # x from embedding
        d2d.copy_emb_row(int(pw["emb_ptr"]), V, H, int(tok_id), x)
        # Walk all layers to produce K/V at this position
        for l in range(L):
            rin = int(pw.get("rms_in_ptrs", [0]*L)[l]) if pw.get("rms_in_ptrs") else 0
            if rin:
                cds.rmsnorm_ptr(x, rin, x_norm, eps)
            else:
                x_norm.copy_(x)
            wq = int(pw.get("q_ptrs", [0]*L)[l]) if pw.get("q_ptrs") else 0
            wk = int(pw.get("k_ptrs", [0]*L)[l]) if pw.get("k_ptrs") else 0
            wv = int(pw.get("v_ptrs", [0]*L)[l]) if pw.get("v_ptrs") else 0
            cds.down_gemm(x_norm, q_vec, wq, H, H)
            cds.down_gemm(x_norm, k_vec, wk, H, Kh * D)
            cds.down_gemm(x_norm, v_vec, wv, H, Kh * D)
            qh = q_vec.view(Hh, D).contiguous()
            kh_kv = k_vec.view(Kh, D).contiguous()
            # Apply RoPE manually (HF-equivalent) at current pos
            pos = int(seq_len_dev.item())
            qh = apply_rope_(qh, pos, rotary_dim, rope_theta, device)
            kh_kv = apply_rope_(kh_kv, pos, rotary_dim, rope_theta, device)
            vh_kv = v_vec.view(Kh, D).contiguous()
            T = int(seq_len_dev.item())
            if T < max_len:
                arena_k[l, 0, :, T, :].copy_(kh_kv)
                arena_v[l, 0, :, T, :].copy_(vh_kv)
            Ks_kv = arena_k[l, 0, :, :T+1, :].contiguous()
            Vs_kv = arena_v[l, 0, :, :T+1, :].contiguous()
            group = max(1, Hh // Kh)
            Ks = Ks_kv.repeat_interleave(group, dim=0)
            Vs = Vs_kv.repeat_interleave(group, dim=0)
            ctx = attn_varlen.forward(qh, Ks, Vs, T + 1, 1.0 / math.sqrt(max(1, D)))
            attn_ctx.copy_(ctx.view(-1).contiguous())
            wo = int(pw.get("o_ptrs", [0]*L)[l]) if pw.get("o_ptrs") else 0
            cds.down_gemm(attn_ctx, mlp_out, wo, H, H)
            mlp_out.add_(x)
            x.copy_(mlp_out)
            rpost = int(pw.get("rms_post_ptrs", [0]*L)[l]) if pw.get("rms_post_ptrs") else 0
            if rpost:
                cds.rmsnorm_ptr(x, rpost, x_norm, eps)
            else:
                x_norm.copy_(x)
            wg = int(pw.get("mlp_gate_ptrs", [0]*L)[l]) if pw.get("mlp_gate_ptrs") else 0
            wu = int(pw.get("mlp_up_ptrs", [0]*L)[l]) if pw.get("mlp_up_ptrs") else 0
            wd = int(pw.get("mlp_down_ptrs", [0]*L)[l]) if pw.get("mlp_down_ptrs") else 0
            cds.gate_up_gemm(x_norm, gate, up, wg, wu, H, I)
            cds.swiglu(gate, up, act)
            cds.down_gemm(act, mlp_out, wd, I, H)
            mlp_out.add_(x)
            x.copy_(mlp_out)
        if int(seq_len_dev.item()) + 1 < max_len:
            seq_len_dev.add_(1)
            pos_dev.fill_(int(seq_len_dev.item()))
    last_token = int(prompt_ids[-1])

    # Run STEPS parity iterations; use HF to pick next token so both paths share the same inputs
    for step in range(STEPS):
        # HF next-token to keep inputs aligned
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[last_token]], dtype=torch.long, device=device),
                        use_cache=True, past_key_values=hf_cache, return_dict=True)
            hf_cache = out.past_key_values
            hf_logits = out.logits[0, -1, :].float()
            next_token = int(torch.argmax(hf_logits).item())

        # Our path: build x from embedding(last_token) and compute layer-0 K/V; full layer stack for better x
        d2d.copy_emb_row(int(pw["emb_ptr"]), V, H, last_token, x)
        for l in range(L):
            # input LN
            rin = int(pw.get("rms_in_ptrs", [0]*L)[l]) if pw.get("rms_in_ptrs") else 0
            if rin:
                cds.rmsnorm_ptr(x, rin, x_norm, eps)
            else:
                x_norm.copy_(x)
            # QKV
            wq = int(pw.get("q_ptrs", [0]*L)[l]) if pw.get("q_ptrs") else 0
            wk = int(pw.get("k_ptrs", [0]*L)[l]) if pw.get("k_ptrs") else 0
            wv = int(pw.get("v_ptrs", [0]*L)[l]) if pw.get("v_ptrs") else 0
            cds.down_gemm(x_norm, q_vec, wq, H, H)
            cds.down_gemm(x_norm, k_vec, wk, H, Kh * D)
            cds.down_gemm(x_norm, v_vec, wv, H, Kh * D)
            qh = q_vec.view(Hh, D).contiguous()
            kh = k_vec.view(Kh, D).contiguous()
            vh = v_vec.view(Kh, D).contiguous()
            T = int(seq_len_dev.item())
            # Apply RoPE at current T (manual HF-equivalent)
            qh = apply_rope_(qh, T, rotary_dim, rope_theta, device)
            kh = apply_rope_(kh, T, rotary_dim, rope_theta, device)
            if T < max_len:
                arena_k[l, 0, :, T, :].copy_(kh)
                arena_v[l, 0, :, T, :].copy_(vh)
            Ks = arena_k[l, 0, :, :T+1, :].contiguous()
            Vs = arena_v[l, 0, :, :T+1, :].contiguous()
            ctx = attn_varlen.forward(qh, Ks, Vs, T + 1, 1.0 / math.sqrt(max(1, D)))
            attn_ctx.copy_(ctx.view(-1).contiguous())
            # O-proj + residual
            wo = int(pw.get("o_ptrs", [0]*L)[l]) if pw.get("o_ptrs") else 0
            cds.down_gemm(attn_ctx, mlp_out, wo, H, H)
            mlp_out.add_(x)
            x.copy_(mlp_out)
            # post-attn LN
            rpost = int(pw.get("rms_post_ptrs", [0]*L)[l]) if pw.get("rms_post_ptrs") else 0
            if rpost:
                cds.rmsnorm_ptr(x, rpost, x_norm, eps)
            else:
                x_norm.copy_(x)
            # MLP
            wg = int(pw.get("mlp_gate_ptrs", [0]*L)[l]) if pw.get("mlp_gate_ptrs") else 0
            wu = int(pw.get("mlp_up_ptrs", [0]*L)[l]) if pw.get("mlp_up_ptrs") else 0
            wd = int(pw.get("mlp_down_ptrs", [0]*L)[l]) if pw.get("mlp_down_ptrs") else 0
            cds.gate_up_gemm(x_norm, gate, up, wg, wu, H, I)
            cds.swiglu(gate, up, act)
            cds.down_gemm(act, mlp_out, wd, I, H)
            mlp_out.add_(x)
            x.copy_(mlp_out)
        # advance T
        if T + 1 < max_len:
            seq_len_dev.add_(1)
            pos_dev.fill_(int(seq_len_dev.item()))
        # Compare layer-0 last K/V with HF pkv
        hf_k0 = hf_cache[0][0][0]  # [n_head, T+1, D]
        hf_v0 = hf_cache[0][1][0]
        our_k0 = arena_k[0, 0, :, :T+1, :].contiguous()  # already RoPE-applied in our path
        our_v0 = arena_v[0, 0, :, :T+1, :].contiguous()
        dk = (our_k0 - hf_k0).float().norm().item()
        dv = (our_v0 - hf_v0).float().norm().item()
        print(f"step {step+1}: L0 K L2={dk:.4f}  V L2={dv:.4f}")
        last_token = next_token

if __name__ == "__main__":
    main()

