"""
Lightweight weight loader/packer (no PyTorch allocations during replay).

Loads Hugging Face safetensors from a LLaMA-style model dir, packs fused QKV
per layer, and uploads to persistent cudaMalloc device buffers. Exposes raw
device pointers and shapes/strides for usage from custom kernels.

Note: This is a minimal scaffold for Milestone 1; it focuses on persistent
GPU buffers and pointer metadata. Packing completeness and model variant
coverage can be extended incrementally.
"""

import os
import glob
import ctypes
from typing import Dict, Any, Tuple

import numpy as np

try:
    from safetensors.numpy import load_file as load_safetensors
except Exception as e:
    load_safetensors = None  # type: ignore


# cudaRuntime API via ctypes
_libcudart = ctypes.CDLL("libcudart.so")
cudaMalloc = _libcudart.cudaMalloc
cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudaMalloc.restype = ctypes.c_int
cudaFree = _libcudart.cudaFree
cudaFree.argtypes = [ctypes.c_void_p]
cudaFree.restype = ctypes.c_int
cudaMemcpy = _libcudart.cudaMemcpy
cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudaMemcpy.restype = ctypes.c_int
cudaMemcpyHostToDevice = 1


def _cuda_check(code: int, msg: str):
    if code != 0:
        raise RuntimeError(f"CUDA error ({code}) in {msg}")


def cuda_malloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _cuda_check(cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return int(ctypes.cast(ptr, ctypes.c_void_p).value)


def cuda_memcpy_h2d(dst_ptr: int, src: np.ndarray):
    assert src.flags['C_CONTIGUOUS']
    _cuda_check(
        cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            src.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(src.nbytes),
            cudaMemcpyHostToDevice,
        ),
        "cudaMemcpy H2D",
    )


def cuda_free(ptr: int):
    if ptr:
        _cuda_check(cudaFree(ctypes.c_void_p(ptr)), "cudaFree")


def _concat_qkv(weights: Dict[str, np.ndarray], layer_idx: int) -> np.ndarray:
    # Try common LLaMA key patterns
    keys = [
        (f"model.layers.{layer_idx}.self_attn.q_proj.weight",
         f"model.layers.{layer_idx}.self_attn.k_proj.weight",
         f"model.layers.{layer_idx}.self_attn.v_proj.weight"),
        (f"layers.{layer_idx}.attention.wq.weight",
         f"layers.{layer_idx}.attention.wk.weight",
         f"layers.{layer_idx}.attention.wv.weight"),
    ]
    for qk, kk, vk in keys:
        if qk in weights and kk in weights and vk in weights:
            q = weights[qk]
            k = weights[kk]
            v = weights[vk]
            # Expect [out, in]; pack along out dim
            return np.concatenate([q, k, v], axis=0)
    raise KeyError(f"QKV weights not found for layer {layer_idx}")


def _find_num_layers(weights: Dict[str, np.ndarray]) -> int:
    max_idx = -1
    for k in weights.keys():
        if ".layers." in k:
            try:
                idx = int(k.split(".layers.")[1].split(".")[0])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
        elif k.startswith("layers."):
            try:
                idx = int(k.split(".")[1])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
    if max_idx < 0:
        raise RuntimeError("Could not infer num_layers from safetensors keys")
    return max_idx + 1


def _find_weight(weights: Dict[str, np.ndarray], candidates) -> np.ndarray:
    for k in candidates:
        if k in weights:
            return weights[k]
    raise KeyError(f"None of the keys found: {candidates}")


def load_and_pack(model_dir: str) -> Dict[str, Any]:
    if load_safetensors is None:
        raise RuntimeError("safetensors is required: pip install safetensors")

    shard_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shard_paths:
        raise RuntimeError(f"No safetensors found in {model_dir}")

    # Merge all shards into a key->array dict (CPU, float16)
    weights: Dict[str, np.ndarray] = {}
    for p in shard_paths:
        shard = load_safetensors(p)
        for k, arr in shard.items():
            if isinstance(arr, np.ndarray):
                weights[k] = arr.astype(np.float16, copy=False)

    num_layers = _find_num_layers(weights)

    # Find token embedding (common LLaMA keys)
    emb_keys = [
        "model.embed_tokens.weight",
        "tok_embeddings.weight",
    ]
    emb_arr = None
    for ek in emb_keys:
        if ek in weights:
            emb_arr = weights[ek]
            break
    if emb_arr is None:
        # Try AutoTokenizer JSON maps or alternate shards is out of scope here; proceed without emb
        pass
    emb_ptr = 0
    emb_shape: Tuple[int, int] = (0, 0)
    if emb_arr is not None:
        emb_c = np.ascontiguousarray(emb_arr)
        emb_ptr = cuda_malloc(emb_c.nbytes)
        cuda_memcpy_h2d(emb_ptr, emb_c)
        emb_shape = emb_c.shape  # [vocab, hidden]

    # LM head (may be untied for some LLaMA variants)
    lm_head_ptr = 0
    lm_head_shape: Tuple[int, int] = (0, 0)
    try:
        lm_keys = [
            "lm_head.weight",
            "output.weight",
            "model.lm_head.weight",
        ]
        lm = _find_weight(weights, lm_keys)
        lm_c = np.ascontiguousarray(lm)
        lm_head_ptr = cuda_malloc(lm_c.nbytes)
        cuda_memcpy_h2d(lm_head_ptr, lm_c)
        lm_head_shape = lm_c.shape
    except KeyError:
        pass

    # Pack Q, K, V per layer and upload (as separate device buffers)
    qkv_ptrs = []
    qkv_shapes = []
    q_ptrs = []
    q_shapes = []
    k_ptrs = []
    k_shapes = []
    v_ptrs = []
    v_shapes = []
    # Output projection O per layer
    o_ptrs = []
    o_shapes = []
    # MLP weights per layer: gate/up/down
    mlp_gate_ptrs = []
    mlp_gate_shapes = []
    mlp_up_ptrs = []
    mlp_up_shapes = []
    mlp_down_ptrs = []
    mlp_down_shapes = []
    for i in range(num_layers):
        # Q
        try:
            q_keys = [
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"layers.{i}.attention.wq.weight",
            ]
            qw = _find_weight(weights, q_keys)
            qc = np.ascontiguousarray(qw)
            qp = cuda_malloc(qc.nbytes)
            cuda_memcpy_h2d(qp, qc)
            q_ptrs.append(qp)
            q_shapes.append(qc.shape)
        except KeyError:
            q_ptrs.append(0)
            q_shapes.append((0, 0))
        # K
        try:
            k_keys = [
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"layers.{i}.attention.wk.weight",
            ]
            kw = _find_weight(weights, k_keys)
            kc = np.ascontiguousarray(kw)
            kp = cuda_malloc(kc.nbytes)
            cuda_memcpy_h2d(kp, kc)
            k_ptrs.append(kp)
            k_shapes.append(kc.shape)
        except KeyError:
            k_ptrs.append(0)
            k_shapes.append((0, 0))
        # V
        try:
            v_keys = [
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"layers.{i}.attention.wv.weight",
            ]
            vw = _find_weight(weights, v_keys)
            vc = np.ascontiguousarray(vw)
            vp = cuda_malloc(vc.nbytes)
            cuda_memcpy_h2d(vp, vc)
            v_ptrs.append(vp)
            v_shapes.append(vc.shape)
        except KeyError:
            v_ptrs.append(0)
            v_shapes.append((0, 0))
        # Also keep fused QKV if available for compatibility
        try:
            qkv = _concat_qkv(weights, i)  # [3*hidden, hidden]
            qkv_c = np.ascontiguousarray(qkv)
            dptr = cuda_malloc(qkv_c.nbytes)
            cuda_memcpy_h2d(dptr, qkv_c)
            qkv_ptrs.append(dptr)
            qkv_shapes.append(qkv_c.shape)
        except Exception:
            qkv_ptrs.append(0)
            qkv_shapes.append((0, 0))

        # O projection
        o_keys = [
            f"model.layers.{i}.self_attn.o_proj.weight",
            f"layers.{i}.attention.wo.weight",
        ]
        try:
            o_w = _find_weight(weights, o_keys)
            o_c = np.ascontiguousarray(o_w)
            o_ptr = cuda_malloc(o_c.nbytes)
            cuda_memcpy_h2d(o_ptr, o_c)
            o_ptrs.append(o_ptr)
            o_shapes.append(o_c.shape)
        except KeyError:
            o_ptrs.append(0)
            o_shapes.append((0, 0))

        # MLP: gate/up/down
        gate_keys = [
            f"model.layers.{i}.mlp.gate_proj.weight",
            f"layers.{i}.feed_forward.w1.weight",
        ]
        up_keys = [
            f"model.layers.{i}.mlp.up_proj.weight",
            f"layers.{i}.feed_forward.w3.weight",
        ]
        down_keys = [
            f"model.layers.{i}.mlp.down_proj.weight",
            f"layers.{i}.feed_forward.w2.weight",
        ]
        # gate
        try:
            g_w = _find_weight(weights, gate_keys)
            g_c = np.ascontiguousarray(g_w)
            g_ptr = cuda_malloc(g_c.nbytes)
            cuda_memcpy_h2d(g_ptr, g_c)
            mlp_gate_ptrs.append(g_ptr)
            mlp_gate_shapes.append(g_c.shape)
        except KeyError:
            mlp_gate_ptrs.append(0)
            mlp_gate_shapes.append((0, 0))
        # up
        try:
            u_w = _find_weight(weights, up_keys)
            u_c = np.ascontiguousarray(u_w)
            u_ptr = cuda_malloc(u_c.nbytes)
            cuda_memcpy_h2d(u_ptr, u_c)
            mlp_up_ptrs.append(u_ptr)
            mlp_up_shapes.append(u_c.shape)
        except KeyError:
            mlp_up_ptrs.append(0)
            mlp_up_shapes.append((0, 0))
        # down
        try:
            d_w = _find_weight(weights, down_keys)
            d_c = np.ascontiguousarray(d_w)
            d_ptr2 = cuda_malloc(d_c.nbytes)
            cuda_memcpy_h2d(d_ptr2, d_c)
            mlp_down_ptrs.append(d_ptr2)
            mlp_down_shapes.append(d_c.shape)
        except KeyError:
            mlp_down_ptrs.append(0)
            mlp_down_shapes.append((0, 0))

    # RMSNorm per layer: input and post-attention norms
    rms_in_ptrs = []
    rms_in_shapes = []
    rms_post_ptrs = []
    rms_post_shapes = []
    for i in range(num_layers):
        in_keys = [
            f"model.layers.{i}.input_layernorm.weight",
            f"layers.{i}.attention_norm.weight",
        ]
        post_keys = [
            f"model.layers.{i}.post_attention_layernorm.weight",
            f"layers.{i}.ffn_norm.weight",
        ]
        # input rms
        try:
            w = _find_weight(weights, in_keys)
            c = np.ascontiguousarray(w)
            p = cuda_malloc(c.nbytes)
            cuda_memcpy_h2d(p, c)
            rms_in_ptrs.append(p)
            rms_in_shapes.append(c.shape)
        except KeyError:
            rms_in_ptrs.append(0)
            rms_in_shapes.append((0,))
        # post-attn rms
        try:
            w = _find_weight(weights, post_keys)
            c = np.ascontiguousarray(w)
            p = cuda_malloc(c.nbytes)
            cuda_memcpy_h2d(p, c)
            rms_post_ptrs.append(p)
            rms_post_shapes.append(c.shape)
        except KeyError:
            rms_post_ptrs.append(0)
            rms_post_shapes.append((0,))

    # Final RMSNorm
    final_rms_ptr = 0
    final_rms_shape: Tuple[int] = (0,)
    try:
        final_keys = [
            "model.norm.weight",
            "norm.weight",
        ]
        w = _find_weight(weights, final_keys)
        c = np.ascontiguousarray(w)
        final_rms_ptr = cuda_malloc(c.nbytes)
        cuda_memcpy_h2d(final_rms_ptr, c)
        final_rms_shape = c.shape
    except KeyError:
        pass

    return {
        'num_layers': num_layers,
        'qkv_ptrs': qkv_ptrs,
        'qkv_shapes': qkv_shapes,
        'q_ptrs': q_ptrs,
        'q_shapes': q_shapes,
        'k_ptrs': k_ptrs,
        'k_shapes': k_shapes,
        'v_ptrs': v_ptrs,
        'v_shapes': v_shapes,
        'o_ptrs': o_ptrs,
        'o_shapes': o_shapes,
        'mlp_gate_ptrs': mlp_gate_ptrs,
        'mlp_gate_shapes': mlp_gate_shapes,
        'mlp_up_ptrs': mlp_up_ptrs,
        'mlp_up_shapes': mlp_up_shapes,
        'mlp_down_ptrs': mlp_down_ptrs,
        'mlp_down_shapes': mlp_down_shapes,
        'rms_in_ptrs': rms_in_ptrs,
        'rms_in_shapes': rms_in_shapes,
        'rms_post_ptrs': rms_post_ptrs,
        'rms_post_shapes': rms_post_shapes,
        'rms_final_ptr': final_rms_ptr,
        'rms_final_shape': final_rms_shape,
        'emb_ptr': emb_ptr,
        'emb_shape': emb_shape,
        'lm_head_ptr': lm_head_ptr,
        'lm_head_shape': lm_head_shape,
        'dtype': 'fp16',
    }


