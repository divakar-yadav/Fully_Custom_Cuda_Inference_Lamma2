import torch
import torch.nn.functional as F


def _flash_sdp_call(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # Force Flash/efficient SDPA kernels
    try:
        ctx = torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
    except Exception:
        # New API name in future PyTorch
        from torch.nn.attention import sdpa_kernel
        ctx = sdpa_kernel(math=False, flash=True, mem_efficient=True)
    with ctx:
        if scale is not None:
            query = query * float(scale)
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


def enable_flash_sdp_monkeypatch():
    # Monkey-patch torch.nn.functional.scaled_dot_product_attention
    F.scaled_dot_product_attention = _flash_sdp_call


