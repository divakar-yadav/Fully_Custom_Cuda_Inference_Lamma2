import os
import torch
from torch.utils.cpp_extension import load

_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        src_dir = os.path.join(os.path.dirname(__file__), 'flash_attn_ext')
        _ext = load(name='flash_attn_ext', sources=[
            os.path.join(src_dir, 'binding.cpp'),
            os.path.join(src_dir, 'flash_attn_kernel.cu')
        ], extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)
    return _ext

def sdp_single_q(q, k, v, scale=None):
    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)
    ext = _load_ext()
    dtype = q.dtype
    qf = q.contiguous().to(torch.float32)
    kf = k.contiguous().to(torch.float32)
    vf = v.contiguous().to(torch.float32)
    out = ext.sdp_single_q(qf, kf, vf, float(scale))
    return out.to(dtype)

def enable_flash_sdp_monkeypatch():
    import torch.nn.functional as F
    _orig = F.scaled_dot_product_attention
    def _fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if query.size(2) == 1 and attn_mask is None:
            return sdp_single_q(query, key, value, scale)
        # Fallback to original for prefill/masked cases
        return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    F.scaled_dot_product_attention = _fn


