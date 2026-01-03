import os
import torch
import math
from transformers import AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import case4_ipc_20250130.custom_kernels.capture_decode_step as cds
from case4_ipc_20250130.custom_kernels.weight_loader import load_and_pack
import case4_ipc_20250130.custom_kernels.d2d_row_copy as d2d
import case4_ipc_20250130.custom_kernels.attn_varlen as attn_varlen

MODEL = "/home/azureuser/divakar_projects/cuda_graph_sharing/latest_case5/llama2_hf_local"
PROMPT = "The future of AI is"
OUT_DIR = "/home/azureuser/divakar_projects/cuda_graph_sharing/case4_ipc_20250130/output"
TOK_FILE = os.path.join(OUT_DIR, "direct_custom_decode_500_tokens.txt")
TXT_FILE = os.path.join(OUT_DIR, "direct_custom_decode_500_text.txt")
DO_SAMPLE = True
TEMPERATURE = 0.8
TOP_P = 0.95
REPETITION_PENALTY = 1.1
REPETITION_WINDOW = 128

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(PROMPT, return_tensors="pt")
    start_token = int(enc["input_ids"][0, -1].item())

    cfg = AutoConfig.from_pretrained(MODEL)
    hidden = int(getattr(cfg, "hidden_size", 4096))
    num_layers = int(getattr(cfg, "num_hidden_layers", 32))
    num_heads = int(getattr(cfg, "num_attention_heads", 32))
    kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = hidden // num_heads
    vocab = int(getattr(cfg, "vocab_size", 32000))
    max_len = 1024

    # Load and pack weights to obtain device pointers
    pw = load_and_pack(MODEL)
    gate_shapes = pw.get("mlp_gate_shapes", [])
    up_shapes = pw.get("mlp_up_shapes", [])
    emb_shape = pw.get("emb_shape", (vocab, hidden))
    lm_head_shape = pw.get("lm_head_shape", (0, 0))
    # Infer intermediate size
    I = int(max(gate_shapes[0][0] if gate_shapes else 0, up_shapes[0][0] if up_shapes else 0) or (hidden * 2))
    V = int(lm_head_shape[0]) if isinstance(lm_head_shape, (list, tuple)) and lm_head_shape and lm_head_shape[0] else \
        int(emb_shape[0]) if isinstance(emb_shape, (list, tuple)) and emb_shape and emb_shape[0] else vocab
    # Pointers
    rms_in_ptr = int(pw.get("rms_in_ptrs", [0])[0]) if pw.get("rms_in_ptrs") else 0
    rms_post_ptr = int(pw.get("rms_post_ptrs", [0])[0]) if pw.get("rms_post_ptrs") else 0
    rms_final_ptr = int(pw.get("rms_final_ptr", 0))
    w_gate_ptr = int(pw.get("mlp_gate_ptrs", [0])[0]) if pw.get("mlp_gate_ptrs") else 0
    w_up_ptr = int(pw.get("mlp_up_ptrs", [0])[0]) if pw.get("mlp_up_ptrs") else 0
    w_down_ptr = int(pw.get("mlp_down_ptrs", [0])[0]) if pw.get("mlp_down_ptrs") else 0
    emb_ptr = int(pw.get("emb_ptr", 0))
    lm_head_ptr = int(pw.get("lm_head_ptr", 0))
    # Prefer separate lm_head for logits; fallback to embedding only if tied
    w_vocab_ptr = lm_head_ptr or emb_ptr
    rms_eps = float(getattr(cfg, "rms_norm_eps", 1e-6))

    # Allocate KV arena (grouped-query attention: store kv_heads; broadcast at use)
    arena_k = torch.empty((num_layers, 1, kv_heads, max_len, head_dim), dtype=torch.float16, device=device)
    arena_v = torch.empty_like(arena_k)
    seq_len_dev = torch.tensor([enc["input_ids"].size(1) - 1], dtype=torch.int32, device=device)
    pos_dev = torch.tensor([int(seq_len_dev.item())], dtype=torch.int32, device=device)
    logits = torch.empty((1, V), dtype=torch.float32, device=device)
    ctrl = torch.empty((1, 1), dtype=torch.long, device=device)

    # Scratch buffers
    x = torch.empty((hidden,), dtype=torch.float16, device=device)
    x_norm = torch.empty((hidden,), dtype=torch.float16, device=device)
    gate = torch.empty((I,), dtype=torch.float16, device=device)
    up = torch.empty((I,), dtype=torch.float16, device=device)
    act = torch.empty((I,), dtype=torch.float16, device=device)
    mlp_out = torch.empty((hidden,), dtype=torch.float16, device=device)
    attn_ctx = torch.empty((hidden,), dtype=torch.float16, device=device)
    q_vec = torch.empty((hidden,), dtype=torch.float16, device=device)
    k_vec = torch.empty((kv_heads * head_dim,), dtype=torch.float16, device=device)
    v_vec = torch.empty((kv_heads * head_dim,), dtype=torch.float16, device=device)

    # HF rotary module per layer for exact RoPE
    # We will fetch at use-time from model (requires HF model for shapes only); fallback to math if unavailable
    # Build RoPE cache helper for fallback
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    rotary_dim = int(getattr(cfg, "rotary_dim", head_dim))
    rotary_dim = max(2, min(rotary_dim, head_dim))
    def apply_rope_(hd_mat: torch.Tensor, pos: int, d_rot: int, base: float, device: torch.device):
        d = int(d_rot)
        if d <= 0:
            return hd_mat
        rot = hd_mat[:, :d]
        half = d // 2
        idx = torch.arange(half, device=device, dtype=torch.float32)
        inv = (base ** (-idx / float(half))).view(1, -1)
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

    tokens = []
    def apply_repetition_penalty_(logits_cpu: torch.Tensor, recent_ids, penalty: float):
        if penalty is None or penalty <= 1.0 or not recent_ids:
            return
        # HF-style repetition penalty
        uniq = set(int(t) for t in recent_ids)
        for tid in uniq:
            val = logits_cpu[tid].item()
            if val > 0:
                logits_cpu[tid] = val / penalty
            else:
                logits_cpu[tid] = val * penalty

    def sample_top_p_(logits_cpu: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature is None or temperature <= 0:
            temperature = 1.0
        scaled = logits_cpu / float(temperature)
        probs = torch.softmax(scaled, dim=-1)
        # Nucleus
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > float(top_p)
        # Always keep at least one
        if mask.numel() > 0:
            mask[0] = False
        filtered_probs = sorted_probs.masked_fill(mask, 0.0)
        filtered_probs = filtered_probs / filtered_probs.sum()
        choice = torch.multinomial(filtered_probs, num_samples=1).item()
        return int(sorted_idx[choice].item())
    # Prefill KV cache with the full prompt except we will start generation after it
    prompt_ids = enc["input_ids"][0].tolist()
    seq_len_dev.fill_(0)
    pos_dev.fill_(0)
    for pos in range(len(prompt_ids)):
        tok_id = int(prompt_ids[pos])
        d2d.copy_emb_row(emb_ptr, V, hidden, tok_id, x)
        for l in range(num_layers):
            # Attn block
            rin = int(pw.get("rms_in_ptrs", [0]*num_layers)[l]) if pw.get("rms_in_ptrs") else 0
            if rin != 0:
                cds.rmsnorm_ptr(x, rin, x_norm, rms_eps)
            else:
                x_norm.copy_(x)
            wq_ptr = int(pw.get("q_ptrs", [0]*num_layers)[l]) if pw.get("q_ptrs") else 0
            wk_ptr = int(pw.get("k_ptrs", [0]*num_layers)[l]) if pw.get("k_ptrs") else 0
            wv_ptr = int(pw.get("v_ptrs", [0]*num_layers)[l]) if pw.get("v_ptrs") else 0
            if wq_ptr and wk_ptr and wv_ptr:
                cds.down_gemm(x_norm, q_vec, wq_ptr, hidden, hidden)
                cds.down_gemm(x_norm, k_vec, wk_ptr, hidden, kv_heads * head_dim)
                cds.down_gemm(x_norm, v_vec, wv_ptr, hidden, kv_heads * head_dim)
                q_heads = q_vec.view(num_heads, head_dim).contiguous()
                k_heads = k_vec.view(kv_heads, head_dim).contiguous()
                v_heads = v_vec.view(kv_heads, head_dim).contiguous()
                T = int(seq_len_dev.item())
                # Apply RoPE to q and k at current position before caching
                q_heads = apply_rope_(q_heads, T, rotary_dim, rope_theta, device)
                k_heads = apply_rope_(k_heads, T, rotary_dim, rope_theta, device)
                if T < max_len:
                    arena_k[l, 0, :, T, :].copy_(k_heads)
                    arena_v[l, 0, :, T, :].copy_(v_heads)
                Ks = arena_k[l, 0, :, :T+1, :].contiguous()
                Vs = arena_v[l, 0, :, :T+1, :].contiguous()
                group = max(1, num_heads // kv_heads)
                Ks = Ks.repeat_interleave(group, dim=0)
                Vs = Vs.repeat_interleave(group, dim=0)
                scale = 1.0 / math.sqrt(max(1, head_dim))
                ctx = attn_varlen.forward(q_heads, Ks, Vs, T + 1, scale)
                attn_ctx.copy_(ctx.view(-1).contiguous())
                o_ptr = int(pw.get("o_ptrs", [0]*num_layers)[l]) if pw.get("o_ptrs") else 0
                if o_ptr:
                    cds.down_gemm(attn_ctx, mlp_out, o_ptr, hidden, hidden)
                    mlp_out.add_(x)
                    x.copy_(mlp_out)
                else:
                    x.add_(attn_ctx)
            # MLP block
            lp = int(pw.get("rms_post_ptrs", [0]*num_layers)[l]) if pw.get("rms_post_ptrs") else 0
            if lp != 0:
                cds.rmsnorm_ptr(x, lp, x_norm, rms_eps)
            else:
                x_norm.copy_(x)
            wg = int(pw.get("mlp_gate_ptrs", [0]*num_layers)[l]) if pw.get("mlp_gate_ptrs") else 0
            wu = int(pw.get("mlp_up_ptrs", [0]*num_layers)[l]) if pw.get("mlp_up_ptrs") else 0
            wd = int(pw.get("mlp_down_ptrs", [0]*num_layers)[l]) if pw.get("mlp_down_ptrs") else 0
            cds.gate_up_gemm(x_norm, gate, up, wg, wu, hidden, I)
            cds.swiglu(gate, up, act)
            cds.down_gemm(act, mlp_out, wd, I, hidden)
            mlp_out.add_(x)
            x.copy_(mlp_out)
        # advance T after all layers for this position
        if int(seq_len_dev.item()) + 1 < max_len:
            seq_len_dev.add_(1)
            pos_dev.fill_(int(seq_len_dev.item()))
    # Start generation after prompt
    last_token = int(prompt_ids[-1])
    d2d.copy_emb_row(emb_ptr, V, hidden, int(last_token), x)

    # No background capture; invoke per-op path (no CUDA graph)
    for _ in range(500):
        ctrl.fill_(int(last_token))
        # Forward through ATTENTION + MLP stack across layers (no RoPE in this pass)
        for l in range(num_layers):
            # ----- Attention -----
            # RMSNorm (input_layernorm) before attention
            rin = int(pw.get("rms_in_ptrs", [0]*num_layers)[l]) if pw.get("rms_in_ptrs") else 0
            if rin != 0:
                cds.rmsnorm_ptr(x, rin, x_norm, rms_eps)
            else:
                x_norm.copy_(x)
            # QKV weights (separate pointers)
            wq_ptr = int(pw.get("q_ptrs", [0]*num_layers)[l]) if pw.get("q_ptrs") else 0
            wk_ptr = int(pw.get("k_ptrs", [0]*num_layers)[l]) if pw.get("k_ptrs") else 0
            wv_ptr = int(pw.get("v_ptrs", [0]*num_layers)[l]) if pw.get("v_ptrs") else 0
            if wq_ptr != 0 and wk_ptr != 0 and wv_ptr != 0:
                cds.down_gemm(x_norm, q_vec, wq_ptr, hidden, hidden)
                # Compute K,V with separate HxI GEMMs (I = kv_heads*head_dim)
                cds.down_gemm(x_norm, k_vec, wk_ptr, hidden, kv_heads * head_dim)
                cds.down_gemm(x_norm, v_vec, wv_ptr, hidden, kv_heads * head_dim)
                # Reshape
                q_heads = q_vec.view(num_heads, head_dim).contiguous()
                k_kv = k_vec.view(kv_heads, head_dim).contiguous()
                v_kv = v_vec.view(kv_heads, head_dim).contiguous()
                # Apply RoPE to q/k before caching/attention (HF-accurate if model available)
                T = int(seq_len_dev.item())
                # Manual HF-accurate RoPE on first rotary_dim dims
                q_heads = apply_rope_(q_heads, T, rotary_dim, rope_theta, device)
                k_kv = apply_rope_(k_kv, T, rotary_dim, rope_theta, device)
                # Append K/V to cache at position T
                if T < max_len:
                    arena_k[l, 0, :, T, :].copy_(k_kv)
                    arena_v[l, 0, :, T, :].copy_(v_kv)
                # Build K,V slices up to T (inclusive)
                Ks_kv = arena_k[l, 0, :, :T+1, :].contiguous()
                Vs_kv = arena_v[l, 0, :, :T+1, :].contiguous()
                # Broadcast KV heads to query heads
                group = max(1, num_heads // kv_heads)
                Ks = Ks_kv.repeat_interleave(group, dim=0)  # [num_heads, T+1, D]
                Vs = Vs_kv.repeat_interleave(group, dim=0)
                scale = 1.0 / math.sqrt(max(1, head_dim))
                ctx = attn_varlen.forward(q_heads, Ks, Vs, T + 1, scale)  # [H, D]
                attn_ctx.copy_(ctx.view(-1).contiguous())
                # O projection and residual
                o_ptr = int(pw.get("o_ptrs", [0]*num_layers)[l]) if pw.get("o_ptrs") else 0
                if o_ptr != 0:
                    cds.down_gemm(attn_ctx, mlp_out, o_ptr, hidden, hidden)
                    mlp_out.add_(x)  # residual
                    x.copy_(mlp_out)
                else:
                    x.add_(attn_ctx)
            # ----- MLP -----
            # RMSNorm (post-attn) before MLP, or fall back to identity
            lp = int(pw.get("rms_post_ptrs", [0]*num_layers)[l]) if pw.get("rms_post_ptrs") else 0
            if lp != 0:
                cds.rmsnorm_ptr(x, lp, x_norm, rms_eps)
            else:
                x_norm.copy_(x)
            # MLP gate/up
            wg = int(pw.get("mlp_gate_ptrs", [0]*num_layers)[l]) if pw.get("mlp_gate_ptrs") else 0
            wu = int(pw.get("mlp_up_ptrs", [0]*num_layers)[l]) if pw.get("mlp_up_ptrs") else 0
            wd = int(pw.get("mlp_down_ptrs", [0]*num_layers)[l]) if pw.get("mlp_down_ptrs") else 0
            cds.gate_up_gemm(x_norm, gate, up, wg, wu, hidden, I)
            cds.swiglu(gate, up, act)
            cds.down_gemm(act, mlp_out, wd, I, hidden)
            # Residual add: x = x + mlp(x_norm)
            mlp_out.add_(x)
            x.copy_(mlp_out)
        # Final RMSNorm and logits on last x
        y_final = x
        if rms_final_ptr != 0:
            cds.rmsnorm_ptr(x, rms_final_ptr, x_norm, rms_eps)
            y_final = x_norm
        cds.logits_gemm_vocab(y_final, logits[0], w_vocab_ptr, hidden, V)
        # Sample on CPU with repetition penalty and nucleus sampling
        logits_cpu = logits[0].to("cpu")
        if DO_SAMPLE:
            apply_repetition_penalty_(logits_cpu, tokens[-REPETITION_WINDOW:], REPETITION_PENALTY)
            next_token = sample_top_p_(logits_cpu, TEMPERATURE, TOP_P)
        else:
            next_token = int(torch.argmax(logits_cpu).item())
        tokens.append(next_token)
        last_token = next_token
        # Update x as embedding of new token for next step's first layer input
        d2d.copy_emb_row(emb_ptr, V, hidden, int(last_token), x)
        # Advance sequence length once per generated token
        if int(seq_len_dev.item()) + 1 < max_len:
            seq_len_dev.add_(1)
            pos_dev.fill_(int(seq_len_dev.item()))

    with open(TOK_FILE, "w") as f:
        f.write(" ".join(str(x) for x in tokens))
    text = tok.decode(tokens, skip_special_tokens=True)
    with open(TXT_FILE, "w") as f:
        f.write(text)
    print("TOKENS(500):", tokens[:50], "...")
    print("Saved:", TOK_FILE, TXT_FILE)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    main()

