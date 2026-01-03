#!/usr/bin/env python3
"""
Case4: CUDA Graph Generator Server (IPC-based)
Handles CUDA graph capture, replay, and inference via IPC
"""

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from collections import OrderedDict
import threading
import gc
import argparse
import sys
import os
import math

# Add project root to path to find the C++ extension
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import concurrent_capture_replay_simple as cpp_streams
# Add custom decode extension path (case4 custom_kernels) first; fallback to case6 scaffold
custom_ext_dir = os.path.join(script_dir, 'custom_kernels')
if os.path.isdir(custom_ext_dir):
    if custom_ext_dir not in sys.path:
        sys.path.insert(0, custom_ext_dir)
    build_dir = os.path.join(custom_ext_dir, 'build')
    if os.path.isdir(build_dir):
        # Add direct build dir and any build/lib.* directories
        if build_dir not in sys.path:
            sys.path.insert(0, build_dir)
        try:
            for d in os.listdir(build_dir):
                full = os.path.join(build_dir, d)
                if os.path.isdir(full) and d.startswith('lib.') and full not in sys.path:
                    sys.path.insert(0, full)
        except Exception:
            pass
try:
    import importlib
    custom_decode_step = importlib.import_module('capture_decode_step')
except Exception:
    custom_decode_step = None
try:
    # Optional: QKV GEMM (Milestone 2)
    qkv_gemm = importlib.import_module('qkv_gemm')
except Exception:
    qkv_gemm = None
try:
    # Varlen attention kernel
    attn_varlen = importlib.import_module('attn_varlen')
except Exception:
    # Fallback: load torch libs globally then retry
    try:
        import ctypes, torch as _torch_for_load
        torch_lib_dir = os.path.join(os.path.dirname(_torch_for_load.__file__), 'lib')
        for so in os.listdir(torch_lib_dir):
            if so.endswith('.so'):
                p = os.path.join(torch_lib_dir, so)
                try:
                    ctypes.CDLL(p, mode=getattr(ctypes, 'RTLD_GLOBAL', None) or 0)
                except Exception:
                    pass
        import importlib as _imp
        attn_varlen = _imp.import_module('attn_varlen')
    except Exception:
        attn_varlen = None
try:
    # D2D embedding row copy (Milestone 2 - Blocker 1)
    d2d_row_copy = importlib.import_module('d2d_row_copy')
except Exception:
    d2d_row_copy = None

class FusedQKVShim(torch.nn.Module):
    """
    Lightweight shim to replace q_proj/k_proj/v_proj with a fused QKV GEMM call.
    Note: For smoke testing, each of q/k/v calls computes all three and returns the requested one.
    """
    def __init__(self, fused_qkv_weight: torch.Tensor, which: str, hidden_size: int, replay_stream: torch.cuda.Stream):
        super().__init__()
        self.fused_qkv_weight = fused_qkv_weight
        self.which = which  # 'q' | 'k' | 'v'
        self.hidden_size = int(hidden_size)
        self.replay_stream = replay_stream
        # Preallocate scratch
        device = fused_qkv_weight.device
        self._in = torch.empty((self.hidden_size,), dtype=torch.float16, device=device)
        self._q = torch.empty_like(self._in)
        self._k = torch.empty_like(self._in)
        self._v = torch.empty_like(self._in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shapes: [1, 1, H] during decode or [1, H]
        if x.dim() == 3:
            assert x.size(0) == 1 and x.size(1) == 1 and x.size(2) == self.hidden_size, "FusedQKVShim expects [1,1,H]"
            src = x[0, 0].contiguous().to(dtype=torch.float16, device=self._in.device)
        elif x.dim() == 2:
            assert x.size(0) == 1 and x.size(1) == self.hidden_size, "FusedQKVShim expects [1,H]"
            src = x[0].contiguous().to(dtype=torch.float16, device=self._in.device)
        else:
            raise RuntimeError(f"Unexpected x shape for FusedQKVShim: {tuple(x.shape)}")
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            # Copy into scratch
            self._in.copy_(src, non_blocking=True)
            # Fused GEMM
            qkv_gemm.qkv_project(self._in, self.fused_qkv_weight, self._q, self._k, self._v)
            out_vec = {'q': self._q, 'k': self._k, 'v': self._v}[self.which]
            # Return as [1,1,H] to match expected projection output
            return out_vec.view(1, 1, self.hidden_size)
try:
    # Optional: weight loader/packer (Milestone 1)
    from case4_ipc_20250130.custom_kernels.weight_loader import load_and_pack as load_and_pack_weights
except Exception:
    load_and_pack_weights = None


class StaticModelForward:
    """Model forward wrapper for CUDA graph capture."""
    
    def __init__(self, model, seq_len, device, stream):
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.stream = stream
        
        with torch.cuda.stream(self.stream):
            self.static_input_ids = torch.zeros((1, seq_len), device=device, dtype=torch.long)
            self.static_position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            self.static_logits = None
    
    def forward(self):
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=self.static_input_ids,
                    position_ids=self.static_position_ids,
                    use_cache=False,
                    return_dict=True
                )
                self.static_logits = outputs.logits
        return self.static_logits


class GraphGenerator:
    """
    CUDA Graph Generator - handles graph capture and replay
    (No JIT operations here - those are handled by the JIT client)
    """
    
    def __init__(self, model_name, ahead_buffer=150, pre_capture_size=150, max_seq_len=1024):
        self.device = torch.device("cuda:0")
        self.ahead_buffer = ahead_buffer
        self.max_seq_len = max_seq_len
        self.kv_only = os.environ.get("CASE4_KV_ONLY", "0") == "1"
        
        print(f"[Generator] ===== CUDA GRAPH GENERATOR SERVER =====")
        print(f"[Generator] Strategy: {'KV-only' if self.kv_only else 'CUDA graphs (no JIT)'}")
        if not self.kv_only:
            print(f"[Generator] Pre-capture: {pre_capture_size} graphs")
            print(f"[Generator] Ahead buffer: {ahead_buffer} graphs")
        
        print(f"[Generator] Loading model: {model_name}")
        # Optional quantization / attention backends
        quant_mode = os.environ.get("CASE4_QUANT", "none").lower()
        attn_mode = os.environ.get("CASE4_ATTN", "none").lower()
        from transformers import AutoConfig
        load_kwargs = {"low_cpu_mem_usage": True}
        if quant_mode == "int8":
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise RuntimeError("CASE4_QUANT=int8 requires transformers[bitsandbytes] installed")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs.update({
                "quantization_config": bnb_config,
                "device_map": {"": 0},
            })
        else:
            load_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": {"": 0},
            })
        # Attention implementation hint
        if attn_mode in ("flash2", "xformers", "sdpa"):
            impl = {
                "flash2": "flash_attention_2",
                "xformers": "xformers",
                "sdpa": "sdpa",
            }[attn_mode]
            load_kwargs["attn_implementation"] = impl
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        # Force Flash SDPA via custom kernel scaffold if requested
        if os.environ.get("CASE4_FORCE_FLASH", "0") == "1":
            try:
                # Try absolute package import first (works under multiprocessing spawn)
                from case4_ipc_20250130.custom_kernels.flash_attn import enable_flash_sdp_monkeypatch
                enable_flash_sdp_monkeypatch()
            except Exception:
                try:
                    # Fallback to local import if run as script
                    from custom_kernels.flash_attn import enable_flash_sdp_monkeypatch  # type: ignore
                    enable_flash_sdp_monkeypatch()
                except Exception as e:
                    raise RuntimeError(f"CASE4_FORCE_FLASH=1 requested but enabling flash SDPA failed: {e}")
        # Post-load attention backend setup
        if attn_mode == "flash2":
            # Prefer flash SDP kernels
            try:
                torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
            except Exception:
                pass
            if hasattr(self.model, "config") and hasattr(self.model.config, "use_flash_attention_2"):
                self.model.config.use_flash_attention_2 = True
        elif attn_mode == "sdpa":
            try:
                torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
            except Exception:
                pass
        elif attn_mode == "xformers":
            # Ensure xformers is available; if not, fail explicitly (no silent fallback)
            try:
                import xformers  # noqa: F401
            except Exception:
                raise RuntimeError("CASE4_ATTN=xformers requires xformers installed")
        self.model.eval()
        print(f"[Generator] Model loaded")
        # Derive dims early
        cfg = self.model.config
        self.hidden_size = int(getattr(cfg, 'hidden_size', 0))
        self.num_layers_cfg = int(getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 0)))
        self.num_heads_cfg = int(getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', 0)))
        # Milestone 1: load/pack fused weights once (no PyTorch tensors used during replay)
        self.packed_weights = None
        if os.environ.get("CASE4_USE_CUSTOM", "0") == "1" and load_and_pack_weights is not None:
            try:
                print("[Generator] Packing fused weights (QKV) to persistent device buffers...", flush=True)
                self.packed_weights = load_and_pack_weights(model_name)
                nl = self.packed_weights.get('num_layers', 0)
                print(f"[Generator] Packed weights: num_layers={nl}", flush=True)
            except Exception as e:
                print(f"[Generator] Weight pack failed: {e}", flush=True)
        print(f"[Generator] Custom decode module: {'loaded' if custom_decode_step is not None else 'not found'}")
        # Precompute core dims from packed weights for custom path
        self.intermediate_size = 0
        self.vocab_size_cfg = int(getattr(cfg, 'vocab_size', 0))
        if self.packed_weights:
            try:
                # hidden_size from QKV shapes [3H, H]
                if self.hidden_size == 0 and self.packed_weights['qkv_shapes']:
                    self.hidden_size = int(self.packed_weights['qkv_shapes'][0][1])
                # intermediate_size from gate/up shapes [I, H]
                gate0 = self.packed_weights['mlp_gate_shapes'][0] if self.packed_weights['mlp_gate_shapes'] else (0, 0)
                up0 = self.packed_weights['mlp_up_shapes'][0] if self.packed_weights['mlp_up_shapes'] else (0, 0)
                self.intermediate_size = int(max(gate0[0], up0[0]))
                # vocab size from embedding if available
                if self.vocab_size_cfg == 0 and self.packed_weights.get('emb_shape', (0, 0)) != (0, 0):
                    self.vocab_size_cfg = int(self.packed_weights['emb_shape'][0])
                # Expose packed pointer lists for fast access
                self.W_gate_ptrs = list(self.packed_weights.get('mlp_gate_ptrs', []))
                self.W_up_ptrs = list(self.packed_weights.get('mlp_up_ptrs', []))
                self.W_down_ptrs = list(self.packed_weights.get('mlp_down_ptrs', []))
                self.RMS_in_ptrs = list(self.packed_weights.get('rms_in_ptrs', []))
                self.RMS_post_ptrs = list(self.packed_weights.get('rms_post_ptrs', []))
                self.RMS_final_ptr = int(self.packed_weights.get('rms_final_ptr', 0))
                # Tied lm_head: use embedding weight
                self.LM_HEAD_ptr = int(self.packed_weights.get('emb_ptr', 0))
            except Exception:
                pass
        # Scratch buffers for custom decode (capture-safe) - allocated on replay_stream
        self.custom_scratch = {}
        if os.environ.get("CASE4_USE_CUSTOM", "0") == "1" and self.hidden_size > 0:
            H = int(self.hidden_size)
            I = int(self.intermediate_size) if self.intermediate_size else max(H * 2, 4096)
            V = int(self.vocab_size_cfg) if self.vocab_size_cfg else 32000
            with torch.cuda.stream(torch.cuda.Stream(device=self.device) if not hasattr(self, 'replay_stream') else self.replay_stream):
                # FP16 feature vectors
                self.custom_scratch['x_norm'] = torch.empty((H,), dtype=torch.float16, device=self.device)
                self.custom_scratch['gate'] = torch.empty((I,), dtype=torch.float16, device=self.device)
                self.custom_scratch['up'] = torch.empty((I,), dtype=torch.float16, device=self.device)
                self.custom_scratch['act'] = torch.empty((I,), dtype=torch.float16, device=self.device)
                self.custom_scratch['mlp_out'] = torch.empty((H,), dtype=torch.float16, device=self.device)
                # FP32 logits buffer (device)
                self.custom_scratch['logits'] = torch.empty((1, V), dtype=torch.float32, device=self.device)
            # Expose scratch for custom launch paths
            self.logits_buf_custom = self.custom_scratch['logits']
        
        if not self.kv_only:
            # Initialize C++ stream manager
            print(f"[Generator] Initializing C++ streams...")
            self.cpp_mgr = cpp_streams.SimpleConcurrentManager(device_id=0)
            replay_h = self.cpp_mgr.get_replay_stream()
            capture_h = self.cpp_mgr.get_capture_stream()
            transfer_h = self.cpp_mgr.get_transfer_stream()
            self.replay_stream = torch.cuda.ExternalStream(replay_h)
            self.capture_stream = torch.cuda.ExternalStream(capture_h)
            self.transfer_stream = torch.cuda.ExternalStream(transfer_h)
            print(f"[Generator] C++ streams ready")
            # Pre-allocate buffers
            with torch.cuda.stream(self.transfer_stream):
                self.input_buffer_gpu = torch.zeros((1, max_seq_len), device=self.device, dtype=torch.long)
                self.output_buffer_cpu = torch.zeros((1, 32000), dtype=torch.float16)
                self.input_pinned = torch.zeros((1, max_seq_len), dtype=torch.long).pin_memory()
                self.output_pinned = torch.zeros((1, 32000), dtype=torch.float16).pin_memory()
            self.cpp_mgr.sync_transfer()
            # Graph storage
            self.graphs = OrderedDict()
            self.graphs_lock = threading.Lock()
        else:
            # Minimal state for KV-only mode
            self.cpp_mgr = None
            # Create dedicated streams in KV-only mode for custom ops
            self.replay_stream = torch.cuda.Stream(device=self.device)
            self.capture_stream = torch.cuda.Stream(device=self.device)
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self.graphs = OrderedDict()
            self.graphs_lock = threading.Lock()
        # Prepare cuBLAS handle for custom kernels (must be created before capture)
        try:
            if custom_decode_step is not None:
                with torch.cuda.stream(self.capture_stream):
                    custom_decode_step.prepare_cublas()
        except Exception:
            pass
        # Preallocate QKV scratch buffers (stream-safe, no per-step allocs)
        if self.hidden_size > 0:
            with torch.cuda.stream(self.replay_stream):
                self.qkv_scratch_in_h = torch.empty((self.hidden_size,), dtype=torch.float16, device=self.device)
                self.q_out = torch.empty_like(self.qkv_scratch_in_h)
                self.k_out = torch.empty_like(self.qkv_scratch_in_h)
                self.v_out = torch.empty_like(self.qkv_scratch_in_h)
                # Attention context scratch (per-step)
                self.attn_ctx = torch.empty_like(self.qkv_scratch_in_h)
        # Build fused QKV tensors per layer once (avoid per-step cat/contiguous)
        self.fused_qkv_tensors = []
        try:
            for li in range(self.num_layers_cfg):
                layer = self.model.model.layers[li]
                q_w = layer.self_attn.q_proj.weight
                k_w = layer.self_attn.k_proj.weight
                v_w = layer.self_attn.v_proj.weight
                fused = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
                self.fused_qkv_tensors.append(fused)
        except Exception:
            # If model layout differs, leave empty; smoke paths will fallback
            self.fused_qkv_tensors = []
        # Optional: install QKV shim into attention projections for smoke decode
        if os.environ.get("CASE4_QKV_SHIM", "0") == "1" and self.fused_qkv_tensors:
            try:
                for li in range(min(self.num_layers_cfg, len(self.fused_qkv_tensors))):
                    fused = self.fused_qkv_tensors[li]
                    attn = self.model.model.layers[li].self_attn
                    attn.q_proj = FusedQKVShim(fused, 'q', self.hidden_size, self.replay_stream)
                    attn.k_proj = FusedQKVShim(fused, 'k', self.hidden_size, self.replay_stream)
                    attn.v_proj = FusedQKVShim(fused, 'v', self.hidden_size, self.replay_stream)
                print("[Generator] QKV shim installed into LlamaAttention projections", flush=True)
            except Exception as e:
                print(f"[Generator] QKV shim install failed: {e}", flush=True)
        # KV cache store for sessioned incremental decoding (no graphs)
        self.kv_store = {}
        # Paged/in-place KV arena for stable storage
        self.kv_arena = {}
        # Captured single-token decode graphs per session
        self.decode_graphs = {}
        # Device-control decode graphs per session (position_ids + ctrl_input on device)
        self.dev_graphs = {}
        # Custom dynamic-length decode state per session
        self.custom_state = {}
        # Exact-seq custom graphs and background capture control (KV-only custom path)
        self.dev_seq_graphs = {}  # session -> { seq_len: state }
        self.dev_bg_thread = None
        self.dev_bg_stop = False
        self.dev_bg_ahead = 0
        self.dev_bg_lock = threading.Lock()
        
        # Tracking
        self.max_captured = 0
        self.current_seq_len = 0
        self.skipped_graphs = set()  # Track sequence lengths that failed to capture
        
        # Background capture
        self.stop_capture = False
        self.capturing = False  # Flag indicating critical capture phase
        self.capture_phase_lock = threading.Lock()  # Lock for capture phase
        self.capture_thread = None  # Background ahead-capture thread
        self.suspend_bg = False  # Pause background capture during prep_next
        
        print(f"[Generator] Ready!\n")
    
    def start_background_capture(self):
        """Start continuous ahead capture in a background thread (idempotent)."""
        if self.kv_only:
            return
        if self.capture_thread is not None and self.capture_thread.is_alive():
            return
        self.stop_capture = False
        self.capture_thread = threading.Thread(target=self._continuous_capture_worker, daemon=True)
        self.capture_thread.start()
    
    def _capture_graph_on_cpp_stream(self, seq_len):
        """
        Capture graph on C++ capture_stream.
        
        Note: The actual graph capture (inside torch.cuda.graph context) is very brief.
        Most of the time is spent on warmup, which can run in parallel with replay.
        """
        wrapper = StaticModelForward(self.model, seq_len, self.device, self.capture_stream)
        
        # Warmup - can run in parallel with replay (not in capture mode yet)
        for _ in range(3):
            wrapper.forward()
        self.cpp_mgr.sync_capture()
        
        # Actual graph capture (disjoint tensors; no serialization with replay)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.capture_stream):
            wrapper.forward()
        self.cpp_mgr.sync_capture()  # Ensure capture fully completes
        # No device-wide synchronize; keep streams independent
        
        return graph, wrapper
    
    def _continuous_capture_worker(self):
        """Background worker: CONTINUOUSLY captures ahead on capture_stream (parallel with replay)."""
        torch.cuda.set_device(self.device)
        
        print("[Capture Thread] Started - continuous ahead capture (parallel)", flush=True)
        
        while not self.stop_capture:
            try:
                # Allow client-directed prep_next to take exclusive control
                if self.suspend_bg:
                    time.sleep(0.005)
                    continue
                current = self.current_seq_len
                target = current + self.ahead_buffer
                
                # Capture up to target
                if self.max_captured < target and self.max_captured < self.max_seq_len:
                    # Find next sequence to capture, skipping any that were previously skipped
                    with self.graphs_lock:
                        next_seq = self.max_captured + 1
                        # Skip over any previously skipped sequences
                        while next_seq <= self.max_seq_len and (next_seq in self.skipped_graphs or next_seq in self.graphs):
                            if next_seq in self.graphs:
                                # Already captured, update max_captured
                                self.max_captured = next_seq
                                next_seq += 1
                            elif next_seq in self.skipped_graphs:
                                # Was skipped, try next one
                                next_seq += 1
                            else:
                                break
                        
                        # If we've skipped past the target or max_seq_len, we're done
                        if next_seq > target or next_seq > self.max_seq_len:
                            time.sleep(0.01)
                            continue
                        
                        # Double-check this seq isn't already captured
                        if next_seq in self.graphs:
                            self.max_captured = next_seq
                            continue
                    
                    # Try capturing this graph with retry logic for transient errors
                    retry_count = 0
                    max_retries = 5  # More retries for CUDA allocator recovery
                    captured_successfully = False
                    
                    while retry_count < max_retries and not captured_successfully:
                        # Capture next graph on capture_stream
                        # Warmup phase can run in parallel with replay
                        # Only the brief graph capture window needs coordination
                        try:
                            graph, wrapper = self._capture_graph_on_cpp_stream(next_seq)
                            
                            with self.graphs_lock:
                                self.graphs[next_seq] = (graph, wrapper)
                                self.max_captured = next_seq
                                # Remove from skipped if it was there (in case of retry after skip)
                                self.skipped_graphs.discard(next_seq)
                            
                            captured_successfully = True
                            
                            if next_seq % 20 == 0:
                                buffer_ahead = self.max_captured - current
                                print(f"[Capture] ✓ seq_len={next_seq} (ahead={buffer_ahead})", flush=True)
                        
                        except Exception as e:
                            # Check if we should stop before handling error
                            if self.stop_capture:
                                break
                            
                            retry_count += 1
                            
                            # If capture fails due to CUDA graph restrictions, retry
                            error_str = str(e).lower()
                            if "offset increment" in error_str or "operation not permitted" in error_str or "capturing" in error_str:
                                # CUDA graph capture conflict - brief retry delay
                                # This happens when replay modifies captured graph tensors during capture
                                if retry_count < max_retries:
                                    time.sleep(0.01)
                                    continue
                            elif "context is destroyed" in error_str or "destroyed" in error_str:
                                # Process is shutting down
                                break
                            elif "captures_underway" in error_str or "internal assert" in error_str or "operation failed due to a previous error" in error_str:
                                # ⚠️ Internal CUDA allocator state corruption - try to recover
                                # This typically happens after capturing many graphs (150+) due to CUDA allocator state issues
                                print(f"\n[Capture] ⚠️  WARNING: CUDA allocator corruption at seq_len={next_seq} (retry {retry_count}/{max_retries})", flush=True)
                                print(f"[Capture]    Error type: CUDA internal assert/captures_underway - attempting recovery...", flush=True)
                                # Try to recover CUDA state with aggressive cleanup
                                try:
                                    # Avoid any device-wide sync/reset while capture may be active
                                    # Back off and let in-flight capture/replay finish
                                        time.sleep(0.5 + retry_count * 0.2)
                                except Exception as recovery_error:
                                    # Only log non-CUDA-allocator errors
                                    if "captures_underway" not in str(recovery_error) and "INTERNAL ASSERT" not in str(recovery_error):
                                        print(f"[Capture] Recovery error: {recovery_error}", flush=True)
                                
                                if retry_count < max_retries:
                                    continue
                                else:
                                    # After max retries, try one more aggressive recovery before giving up
                                    print(f"[Capture] 🔄 Final recovery backoff for seq_len={next_seq}...", flush=True)
                                    time.sleep(1.0)
                                    
                                    # After max retries, skip this seq_len and try next
                                    print(f"\n[Capture] ❌ SKIP seq_len={next_seq} after {max_retries} retries (CUDA allocator corruption persists)", flush=True)
                                    print(f"[Capture]    Note: This graph cannot be used. Generation will fail if this sequence length is requested.", flush=True)
                                    break
                            else:
                                print(f"[Capture] ERROR at seq_len={next_seq} (retry {retry_count}/{max_retries}): {e}", flush=True)
                                if retry_count < max_retries:
                                    time.sleep(0.1)
                                    continue
                                else:
                                    time.sleep(0.5)
                                    break
                    
                    # If we couldn't capture after retries, skip this seq_len to avoid infinite loop
                    # Note: Replay will fail immediately if it requests a skipped graph
                    if not captured_successfully:
                        # Mark as skipped (but not captured) to move forward
                        print(f"[Capture] ⚠️  FINAL: Skipping seq_len={next_seq} after {max_retries} failed attempts (added to skipped_graphs)", flush=True)
                        skipped_count = len(self.skipped_graphs)
                        print(f"[Capture]    Total skipped graphs so far: {skipped_count}", flush=True)
                        # Track skipped graph - do NOT update max_captured (only update on successful capture)
                        with self.graphs_lock:
                            self.skipped_graphs.add(next_seq)
                        time.sleep(0.01)
                        continue
                
                else:
                    # We're ahead enough, sleep briefly
                    time.sleep(0.01)
            
            except Exception as e:
                # Check if we should stop
                if self.stop_capture:
                    break
                error_str = str(e).lower()
                if "context is destroyed" in error_str or "destroyed" in error_str:
                    # Process is shutting down
                    break
                print(f"[Capture] ERROR: {e}", flush=True)
                time.sleep(0.1)
        
        print("[Capture Thread] Stopped", flush=True)
    
    def pre_capture(self, num=150):
        """Pre-capture LARGE initial buffer."""
        print(f"[Generator] Pre-capturing {num} graphs...")
        start = time.time()
        
        for seq_len in range(1, num + 1):
            graph, wrapper = self._capture_graph_on_cpp_stream(seq_len)
            with self.graphs_lock:
                self.graphs[seq_len] = (graph, wrapper)
                self.max_captured = seq_len
            
            if seq_len % 50 == 0:
                elapsed = time.time() - start
                rate = seq_len / elapsed
                remaining = (num - seq_len) / rate
                print(f"[Generator] Progress: {seq_len}/{num} ({elapsed:.1f}s, ETA: {remaining:.1f}s)")
        
        elapsed = time.time() - start
        print(f"[Generator] ✓ {num} graphs ready ({elapsed:.1f}s)")
        # Background capture is coordinated per step via 'prep_next' to overlap with sampling
        # and reduce allocator pressure. No continuous background thread is started here.

    def delete_graph(self, seq_len: int):
        """Delete a previously captured graph to free memory."""
        with self.graphs_lock:
            if seq_len in self.graphs:
                graph, wrapper = self.graphs.pop(seq_len)
                # Explicitly drop references
                del graph
                del wrapper
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def capture_one(self, seq_len: int):
        """Capture a single graph for the given sequence length, if missing."""
        with self.graphs_lock:
            if seq_len in self.graphs or seq_len in self.skipped_graphs:
                return
        # Attempt capture
        graph, wrapper = self._capture_graph_on_cpp_stream(seq_len)
        with self.graphs_lock:
            self.graphs[seq_len] = (graph, wrapper)
            if seq_len > self.max_captured:
                self.max_captured = seq_len
    
    def replay_fast(self, seq_len, input_ids_cpu, pause_capture=False):
        """
        Fast replay on replay_stream.
        
        Args:
            pause_capture: If True, coordinate with background capture to avoid conflicts
                          by waiting for any ongoing capture to finish before modifying
                          captured graph tensors.
        
        Returns:
            CPU tensor with logits for next token prediction
        """
        self.current_seq_len = seq_len
        
        # Check if graph was skipped - fail immediately
        with self.graphs_lock:
            if seq_len in self.skipped_graphs:
                raise RuntimeError(f"Graph {seq_len} was skipped due to capture failures and cannot be replayed! max_captured={self.max_captured}")
        
        # Check if graph exists - wait for external capture; no on-demand fallback
        if seq_len not in self.graphs:
            max_wait = 30.0  # Wait up to 30 seconds for graph to be captured
            wait_start = time.time()
            
            if seq_len > self.max_captured:
                print(f"[Replay] Waiting for seq_len={seq_len} (max_captured={self.max_captured})...", flush=True)
            
            while seq_len not in self.graphs:
                # Check again if it was skipped during wait
                with self.graphs_lock:
                    if seq_len in self.skipped_graphs:
                        raise RuntimeError(f"Graph {seq_len} was skipped during wait! max_captured={self.max_captured}")
                
                if (time.time() - wait_start) > max_wait:
                    raise RuntimeError(f"Graph {seq_len} never captured! max_captured={self.max_captured}")
                time.sleep(0.001)
        
        # Get graph - must exist at this point
        if seq_len in self.graphs:
            # Get graph
            with self.graphs_lock:
                graph, wrapper = self.graphs[seq_len]
            
            # CRITICAL SECTION: Copy into static tensor and replay
            # This is where we modify tensors that are part of captured graphs
            # We must ensure no capture is active during this operation
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    # Replay on replay_stream
                    with torch.cuda.stream(self.replay_stream):
                        # Copy input_ids - seq_len should match the graph's captured sequence length
                        self.input_pinned[:, :seq_len].copy_(input_ids_cpu[:, :seq_len])
                        self.input_buffer_gpu[:, :seq_len].copy_(self.input_pinned[:, :seq_len], non_blocking=True)
                        wrapper.static_input_ids[:, :seq_len].copy_(self.input_buffer_gpu[:, :seq_len], non_blocking=False)
                        graph.replay()
                    break
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if ("offset increment" in error_str or "capturing" in error_str) and retry_count < max_retries:
                        # Capture conflict - wait and retry
                        retry_count += 1
                        try:
                            self.cpp_mgr.sync_capture()
                            # keep streams independent; no device-wide sync
                        except:
                            pass
                        time.sleep(0.01)
                        continue
                    raise
            
            self.cpp_mgr.sync_replay()
            
            # Copy output via transfer stream
            # Extract last token logits from (batch, seq_len, vocab_size) -> (vocab_size,)
            last_token_logits = wrapper.static_logits[0, -1, :]  # Shape: (vocab_size,)
            
            with torch.cuda.stream(self.transfer_stream):
                # Copy to pinned memory - take only the vocab_size portion
                vocab_size = last_token_logits.shape[0]
                self.output_pinned[0, :vocab_size].copy_(last_token_logits, non_blocking=True)
            self.cpp_mgr.sync_transfer()
            
            # Copy to CPU buffer and return 1D logits
            self.output_buffer_cpu[0, :vocab_size].copy_(self.output_pinned[0, :vocab_size])
            
            # Return 1D logits (vocab_size,) - sampler expects this shape
            result_logits = self.output_buffer_cpu[0, :vocab_size].clone()
            
            # Reduced logging - only log every 50 tokens to avoid spam
            if seq_len % 50 == 0:
                buffer_ahead = self.max_captured - seq_len
                with self.graphs_lock:
                    num_graphs = len(self.graphs)
                print(f"[Replay] ✓ seq_len={seq_len} (ahead={buffer_ahead}, buffer={num_graphs})", flush=True)
            
            return result_logits
        else:
            # Graph should exist - this should never happen if wait logic works correctly
            raise RuntimeError(f"Graph {seq_len} not found in graphs dict even after waiting! This is a bug.")
    
    def cleanup(self):
        """Stop background thread."""
        self.stop_capture = True
        if hasattr(self, 'capture_thread') and self.capture_thread is not None:
            # Wait for thread to finish, but don't wait too long
            self.capture_thread.join(timeout=1.0)
            if self.capture_thread.is_alive():
                # Thread didn't finish gracefully, but that's okay during shutdown
                pass

    def qkv_smoke(self, token_id: int, layer_idx: int = 0):
        """
        Milestone 2 wiring: run qkv_gemm on a single token embedding and fused QKV
        using the extension. This is a shape/exec smoke test (not end-to-end decode).
        """
        if qkv_gemm is None:
            raise RuntimeError("qkv_gemm extension not available")
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight  # [vocab, hidden], fp16, cuda
            hidden = int(emb.size(1))
            tok = int(token_id)
            if tok < 0 or tok >= int(emb.size(0)):
                tok = 0
            input_h = emb[tok].contiguous()  # [H], fp16
            # Build fused [3H, H] from model weights for smoke (packing used later)
            layer = self.model.model.layers[layer_idx]
            q_w = layer.self_attn.q_proj.weight
            k_w = layer.self_attn.k_proj.weight
            v_w = layer.self_attn.v_proj.weight
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()
            # Outputs
            q_out = torch.empty((hidden,), dtype=torch.float16, device=input_h.device)
            k_out = torch.empty_like(q_out)
            v_out = torch.empty_like(q_out)
            # Call extension
            qkv_gemm.qkv_project(input_h, qkv_w, q_out, k_out, v_out)
            # Return simple stats
            return {
                "hidden": hidden,
                "q_mean": float(q_out.float().mean().item()),
                "k_mean": float(k_out.float().mean().item()),
                "v_mean": float(v_out.float().mean().item()),
            }

    def emb_row_smoke(self, token_id: int):
        """
        Milestone 2 - Blocker 1: copy embedding row from packed buffer into a scratch tensor
        and compare with HF embedding row. Returns mean absolute diff.
        """
        if d2d_row_copy is None:
            raise RuntimeError("d2d_row_copy extension not available")
        if not self.packed_weights:
            raise RuntimeError("packed_weights not available; set CASE4_USE_CUSTOM=1")
        emb_ptr = int(self.packed_weights.get('emb_ptr', 0))
        rows, hidden = self.packed_weights.get('emb_shape', (0, 0))
        if emb_ptr == 0 or rows == 0 or hidden == 0:
            raise RuntimeError("packed embedding not available")
        tok = int(token_id)
        if tok < 0 or tok >= int(rows):
            tok = 0
        with torch.no_grad():
            # scratch
            out = torch.empty((hidden,), dtype=torch.float16, device=self.device)
            d2d_row_copy.copy_emb_row(emb_ptr, int(rows), int(hidden), tok, out)
            # reference
            ref = self.model.get_input_embeddings().weight[tok].contiguous()
            diff = torch.mean(torch.abs(out.float() - ref.float())).item()
            return {"rows": int(rows), "hidden": int(hidden), "token_id": tok, "mae": float(diff)}

    def qkv_heads_smoke(self, token_id: int, layer_idx: int = 0):
        """
        Milestone 2 - Blocker 2: Verify fused [3H,H] layout and per-head split.
        Does Q/K/V GEMM and reshapes to [n_heads, head_dim] on the replay stream.
        """
        if qkv_gemm is None:
            raise RuntimeError("qkv_gemm extension not available")
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            # dims
            cfg = self.model.config
            n_layers = getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 0))
            n_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', 0))
            hidden = getattr(cfg, 'hidden_size', 0)
            if hidden == 0 or n_heads == 0:
                raise RuntimeError("invalid model dims")
            head_dim = hidden // n_heads
            # input embedding row (use HF weight for now; D2D packed path exists above)
            emb = self.model.get_input_embeddings().weight  # [vocab, hidden]
            tok = max(0, min(int(token_id), int(emb.size(0)) - 1))
            input_h = emb[tok].contiguous()  # [H], fp16
            # fused [3H,H] from HF weights (packed device ptr test is later)
            layer = self.model.model.layers[layer_idx]
            q_w = layer.self_attn.q_proj.weight
            k_w = layer.self_attn.k_proj.weight
            v_w = layer.self_attn.v_proj.weight
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0).contiguous()  # [3H,H]
            # outputs
            q_out = torch.empty((hidden,), dtype=torch.float16, device=input_h.device)
            k_out = torch.empty_like(q_out)
            v_out = torch.empty_like(q_out)
            qkv_gemm.qkv_project(input_h, qkv_w, q_out, k_out, v_out)
            # per-head reshape checks
            q_heads = q_out.view(n_heads, head_dim)
            k_heads = k_out.view(n_heads, head_dim)
            v_heads = v_out.view(n_heads, head_dim)
            stats = {
                "hidden": int(hidden),
                "n_heads": int(n_heads),
                "head_dim": int(head_dim),
                "q_h0_mean": float(q_heads[0].float().mean().item()),
                "k_h0_mean": float(k_heads[0].float().mean().item()),
                "v_h0_mean": float(v_heads[0].float().mean().item()),
            }
            return stats

    def qkv_stream_smoke(self, token_id: int, layer_idx: int = 0, iters: int = 50):
        """
        Milestone 2 - Blocker 3: Run QKV on replay_stream only with preallocated buffers
        without any device-wide synchronizations. Uses packed embedding D2D row copy.
        """
        if qkv_gemm is None or d2d_row_copy is None:
            raise RuntimeError("extensions not available")
        if not self.packed_weights:
            raise RuntimeError("packed_weights not available; set CASE4_USE_CUSTOM=1")
        emb_ptr = int(self.packed_weights.get('emb_ptr', 0))
        rows, hidden = self.packed_weights.get('emb_shape', (0, 0))
        if emb_ptr == 0 or hidden != self.hidden_size:
            raise RuntimeError("invalid packed embedding or hidden size")
        # Fused weight tensor
        if not self.fused_qkv_tensors or layer_idx >= len(self.fused_qkv_tensors):
            raise RuntimeError("fused_qkv_tensors unavailable")
        fused_w = self.fused_qkv_tensors[layer_idx]
        tok = max(0, min(int(token_id), int(rows) - 1))
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            for _ in range(int(iters)):
                d2d_row_copy.copy_emb_row(emb_ptr, int(rows), int(hidden), tok, self.qkv_scratch_in_h)
                qkv_gemm.qkv_project(self.qkv_scratch_in_h, fused_w, self.q_out, self.k_out, self.v_out)
            # Record an event only; do not synchronize device
            ev = torch.cuda.Event(blocking=False)
            torch.cuda.current_stream().record_event(ev)
        # Return metadata only (no .item() reads that cause sync)
        return {"iters": int(iters), "hidden": int(hidden), "layer_idx": int(layer_idx)}

    # ===== Milestone 3: RoPE + causal mask (smoke for RoPE) =====
    def _init_rope_cache(self, max_len: int):
        if getattr(self, "_rope_ready", False):
            return
        device = self.device
        head_dim = self.hidden_size // max(1, self.num_heads_cfg)
        self.rope_head_dim = head_dim
        theta = getattr(self.model.config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_len, head_dim/2]
        self.rope_cos = torch.cos(freqs)  # [T, D/2]
        self.rope_sin = torch.sin(freqs)  # [T, D/2]
        self._rope_ready = True

    @staticmethod
    def _apply_rope_inplace_heads(q_heads: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: int):
        """
        q_heads: [n_heads, head_dim] fp16/fp32
        cos/sin: [T, head_dim/2]
        """
        d = q_heads.size(1)
        d2 = d // 2
        q_even = q_heads[:, 0:d2]
        q_odd  = q_heads[:, d2:d]
        c = cos[pos].unsqueeze(0)  # [1, d/2]
        s = sin[pos].unsqueeze(0)
        # rotate pairs: [x1, x2] -> [x1*c - x2*s, x1*s + x2*c]
        q_heads[:, 0:d2] = q_even * c - q_odd * s
        q_heads[:, d2:d] = q_even * s + q_odd * c

    def rope_smoke(self, token_id: int, layer_idx: int = 0, pos: int = 0, max_len: int = 4096):
        """
        Compute Q via fused GEMM, apply our custom RoPE in-place on heads, and return stats.
        """
        if qkv_gemm is None:
            raise RuntimeError("qkv_gemm extension not available")
        if layer_idx != 0:
            layer_idx = 0
        self._init_rope_cache(max_len=max_len)
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight  # [vocab, hidden]
            tok = max(0, min(int(token_id), int(emb.size(0)) - 1))
            x = emb[tok].contiguous()  # [H], fp16
            fused = self.fused_qkv_tensors[layer_idx]
            q = torch.empty_like(x)
            k = torch.empty_like(x)
            v = torch.empty_like(x)
            qkv_gemm.qkv_project(x, fused, q, k, v)
            n_heads = self.num_heads_cfg
            head_dim = self.rope_head_dim
            qh = q.view(n_heads, head_dim).float()
            self._apply_rope_inplace_heads(qh, self.rope_cos, self.rope_sin, pos)
            return {
                "status": "ok",
                "n_heads": int(n_heads),
                "head_dim": int(head_dim),
                "pos": int(pos),
                "q_mean": float(qh.mean().item()),
            }

    # ===== Milestone 3: Causal mask (precompute and smoke) =====
    def _init_causal_mask(self, max_len: int):
        if getattr(self, "_mask_ready", False) and getattr(self, "_mask_max_len", 0) >= max_len:
            return
        with torch.no_grad():
            m = torch.full((max_len, max_len), float("-inf"), device=self.device, dtype=torch.float32)
            m = torch.triu(m, diagonal=1)  # upper triangle = -inf, diag/lower kept (-inf on upper only)
            m[m != float("-inf")] = 0.0    # set allowed positions to 0
            self._causal_mask_2d = m       # [T, T]
            self._mask_ready = True
            self._mask_max_len = max_len

    def mask_smoke(self, seq_len: int, max_len: int = 4096):
        """
        Build a [1,1,1,seq_len] mask view for 1-token decode over past length=seq_len.
        Returns shape and finite/inf counts.
        """
        seq_len = int(seq_len)
        max_len = int(max_len)
        if seq_len <= 0 or seq_len > max_len:
            raise RuntimeError("invalid seq_len for mask_smoke")
        self._init_causal_mask(max_len=max_len)
        with torch.no_grad():
            # For 1-token query at position seq_len-1, keys are 0..(seq_len-1): all allowed (0.0)
            row = self._causal_mask_2d[seq_len-1:seq_len, :seq_len]  # [1, seq_len]
            mask = row.view(1, 1, 1, seq_len).contiguous()  # [1,1,1,seq_len]
            finite = int(torch.isfinite(mask).sum().item())
            numel = int(mask.numel())
            infs = numel - finite
            return {
                "shape": list(mask.shape),
                "finite": finite,
                "infs": infs,
                "seq_len": seq_len
            }

    # ===== Milestone 4: Attention smoke (Q·K^T -> softmax -> context = P·V) =====
    def attn_smoke(self, session: str, token_id: int, layer_idx: int = 0):
        """
        Use arena K/V (prefill) and our fused Q (with RoPE) to compute a single-step attention
        context for one layer. Returns shape checks and probability sum ~ 1.0.
        """
        arena = self.kv_arena.get(session, None)
        if arena is None:
            raise RuntimeError("KV session not initialized")
        seq_len = int(arena['seq_len'])
        if seq_len <= 0:
            raise RuntimeError("empty arena seq_len")
        n_heads = int(arena['num_heads'])
        head_dim = int(arena['head_dim'])
        hidden = int(self.hidden_size)
        if n_heads * head_dim != hidden:
            raise RuntimeError("hidden mismatch")
        self._init_rope_cache(max_len=arena['max_len'])
        # Build Q from embedding row for provided token_id (layer 0 input smoke)
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            emb = self.model.get_input_embeddings().weight
            tok = max(0, min(int(token_id), int(emb.size(0)) - 1))
            x = emb[tok].contiguous()
            fused = self.fused_qkv_tensors[layer_idx]
            q = torch.empty((hidden,), dtype=torch.float16, device=self.device)
            ktmp = torch.empty_like(q)
            vtmp = torch.empty_like(q)
            qkv_gemm.qkv_project(x, fused, q, ktmp, vtmp)
            qh = q.view(n_heads, head_dim).float()
            # Apply RoPE to Q at current pos (seq_len-1)
            self._apply_rope_inplace_heads(qh, self.rope_cos, self.rope_sin, pos=seq_len-1)
            # K,V from arena are already RoPE'd by HF prefill; shapes [H, T, D]
            K = arena['k'][layer_idx, 0, :, :seq_len, :].contiguous().float()  # [H,T,D]
            V = arena['v'][layer_idx, 0, :, :seq_len, :].contiguous().float()  # [H,T,D]
            scale = 1.0 / (head_dim ** 0.5)
            # scores_h: [H, T] = [H, D]·[H, D, T]
            scores = torch.einsum("hd,htd->ht", qh, K) * scale
            # For single query at pos T-1, all keys 0..T-1 are valid; softmax along T
            probs = torch.softmax(scores, dim=-1)  # [H, T]
            # context per head: [H, D] = [H, T]·[H, T, D]
            ctx = torch.einsum("ht,htd->hd", probs, V)  # [H, D]
            ctx_flat = ctx.reshape(hidden)  # [H*D]
            p_sum = float(probs[0].sum().item())
            return {
                "seq_len": seq_len,
                "n_heads": n_heads,
                "head_dim": head_dim,
                "hidden": hidden,
                "probs_sum_h0": p_sum,
                "ctx_mean": float(ctx_flat.mean().item()),
            }
    def attn_varlen_smoke(self, session: str, token_id: int, layer_idx: int = 0):
        """
        Varlen attention using custom kernel: q from qkv_gemm (+RoPE), K/V from arena up to seq_len.
        """
        if qkv_gemm is None or attn_varlen is None:
            raise RuntimeError("attn_varlen or qkv_gemm extension not available")
        arena = self.kv_arena.get(session, None)
        if arena is None:
            raise RuntimeError("KV session not initialized")
        seq_len = int(arena['seq_len'])
        if seq_len <= 0:
            raise RuntimeError("empty arena seq_len")
        n_heads = int(arena['num_heads'])
        head_dim = int(arena['head_dim'])
        hidden = int(self.hidden_size)
        if n_heads * head_dim != hidden:
            raise RuntimeError("hidden mismatch")
        self._init_rope_cache(max_len=arena['max_len'])
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            # input hidden from embedding row
            emb = self.model.get_input_embeddings().weight
            tok = max(0, min(int(token_id), int(emb.size(0)) - 1))
            x = emb[tok].contiguous()
            fused = self.fused_qkv_tensors[layer_idx]
            # qkv on replay stream (stream-bound handle)
            qkv_gemm.qkv_project(x, fused, self.q_out, self.k_out, self.v_out)
            qh = self.q_out.view(n_heads, head_dim).float()
            # RoPE on q
            self._apply_rope_inplace_heads(qh, self.rope_cos, self.rope_sin, pos=seq_len-1)
            # K,V slices from arena up to seq_len
            K = arena['k'][layer_idx, 0, :, :seq_len, :].contiguous()
            V = arena['v'][layer_idx, 0, :, :seq_len, :].contiguous()
            # varlen attention
            scale = 1.0 / (head_dim ** 0.5)
            ctx = attn_varlen.forward(qh.to(dtype=torch.float16), K, V, int(seq_len), float(scale))
            ctx = ctx.view(hidden).float()
            return {
                "seq_len": seq_len,
                "hidden": hidden,
                "ctx_mean": float(ctx.mean().item()),
            }

    # ===== Milestone 6: KV append device-side (smoke) =====
    def kv_append_smoke(self, session: str, last_token: int):
        """
        Runs one HF decode step using arena as past, appends last K/V in-place to arena,
        and bumps device-side seq_len scalar (and host mirror) without device-wide syncs.
        """
        arena = self.kv_arena.get(session, None)
        if arena is None:
            raise RuntimeError("KV session not initialized")
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            # Build past views up to current seq_len
            seq_len = int(arena['seq_len'])
            past_view = []
            for l in range(arena['num_layers']):
                k_view = arena['k'][l, 0, :, :seq_len, :].unsqueeze(0)
                v_view = arena['v'][l, 0, :, :seq_len, :].unsqueeze(0)
                past_view.append((k_view, v_view))
            # One-step decode from last_token
            input_ids = torch.tensor([[int(last_token)]], dtype=torch.long, device=self.device)
            try:
                from transformers.cache_utils import DynamicCache
                pkv = DynamicCache.from_legacy_cache(tuple(past_view))
            except Exception:
                pkv = tuple(past_view)
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=pkv,
                return_dict=True
            )
            # Append last position into arena in-place
            new_seq = seq_len + 1
            if new_seq > arena['max_len']:
                raise RuntimeError("Exceeded arena max_len during kv_append_smoke")
            for l, (k_new, v_new) in enumerate(outputs.past_key_values):
                arena['k'][l, 0, :, seq_len:new_seq, :].copy_(k_new[0, :, seq_len:new_seq, :])
                arena['v'][l, 0, :, seq_len:new_seq, :].copy_(v_new[0, :, seq_len:new_seq, :])
            arena['seq_len'] = new_seq
            # Device-side bump (no host readback)
            if 'seq_len_dev' in arena and isinstance(arena['seq_len_dev'], torch.Tensor):
                arena['seq_len_dev'].add_(1)
            # Return new seq_len (host mirror) and next token (greedy)
            next_token = int(torch.argmax(outputs.logits[0, -1, :]).item())
            return {
                "seq_len": int(new_seq),
                "next_token": next_token,
            }
    # ===== Milestone 5: MLP + logits smoke (SwiGLU with cuBLAS via torch for now) =====
    def mlp_logits_smoke(self, token_id: int, layer_idx: int = 0):
        """
        Runs LLaMA MLP (SwiGLU) for a single hidden vector and projects to logits via lm_head.
        Uses HF weights for GEMV but runs on replay_stream with no device-wide syncs in the path.
        """
        with torch.no_grad(), torch.cuda.stream(self.replay_stream):
            cfg = self.model.config
            hidden = int(getattr(cfg, 'hidden_size', 0))
            eps = float(getattr(cfg, 'rms_norm_eps', 1e-6))
            if hidden <= 0:
                raise RuntimeError("invalid hidden size")
            # Input hidden from embedding row (smoke)
            emb = self.model.get_input_embeddings().weight  # [V,H]
            tok = max(0, min(int(token_id), int(emb.size(0)) - 1))
            x = emb[tok].contiguous().float()  # [H]
            # RMSNorm (input_layernorm)
            ln_w = None
            try:
                ln_w = self.model.model.layers[layer_idx].input_layernorm.weight
            except Exception:
                ln_w = None
            if ln_w is not None:
                var = (x.float() * x.float()).mean()
                x = x * torch.rsqrt(var + eps)
                x = x * ln_w.float()
            # MLP weights
            layer = self.model.model.layers[layer_idx]
            w_gate = layer.mlp.gate_proj.weight.float()   # [I,H]
            w_up   = layer.mlp.up_proj.weight.float()     # [I,H]
            w_down = layer.mlp.down_proj.weight.float()   # [H,I]
            # GEMV: [I] = [I,H] @ [H]
            gate = torch.matmul(w_gate, x)    # [I]
            up   = torch.matmul(w_up, x)      # [I]
            act  = torch.nn.functional.silu(gate) * up    # [I]
            y    = torch.matmul(w_down, act)  # [H]
            # Residual add (smoke): y = x + y
            y = x + y
            # Final RMSNorm + lm_head projection
            try:
                final_ln_w = self.model.model.norm.weight.float()
                var = (y.float() * y.float()).mean()
                y = y * torch.rsqrt(var + eps)
                y = y * final_ln_w
            except Exception:
                pass
            # lm_head
            lm = self.model.lm_head.weight.float()  # [V,H] tied with embedding
            logits = torch.matmul(lm, y)            # [V]
            top1 = int(torch.argmax(logits).item())
            return {
                "hidden": hidden,
                "vocab": int(logits.numel()),
                "top1": top1,
                "logits_mean": float(logits.mean().item()),
            }


def graph_generator_process(request_queue, response_queue, model_name):
    """
    Graph Generator Server Process - handles CUDA graphs via IPC
    """
    try:
        print("[Generator] ===== CUDA GRAPH GENERATOR SERVER PROCESS =====", flush=True)
        
        generator = GraphGenerator(
            model_name,
            ahead_buffer=150,
            pre_capture_size=150
        )

        # Optionally skip pre-capture when using KV-only mode
        skip_precapture = os.environ.get("CASE4_SKIP_PRECAPTURE", "0") == "1"
        if not skip_precapture:
            # Pre-capture for fast startup (150 graphs)
            generator.pre_capture(num=150)
            # Optionally start continuous background capture to avoid starvation during replay
            if os.environ.get("CASE4_BG_CAPTURE", "1") == "1":
                print("[Generator] Starting background ahead-capture thread...", flush=True)
                generator.start_background_capture()
        
        # Ready signal
        response_queue.put({"status": "ready", "process": "graph_generator"})
        print("[Generator] Ready! Graph generator enabled!\n", flush=True)
        
        while True:
            request = request_queue.get()
            
            if request.get("cmd") == "stop":
                generator.cleanup()
                break
            
            elif request.get("cmd") == "generate":
                seq_len = request["seq_len"]
                input_ids_cpu = request["input_ids"]
                
                # Fast replay - returns raw logits
                # Use pause_capture=False - replay flag will coordinate with background capture
                # Temporarily pause capture during replay to avoid CUDA graph conflicts
                # This ensures true parallelism (warmup runs in parallel) with safety during critical operations
                logits_cpu = generator.replay_fast(seq_len, input_ids_cpu, pause_capture=True)
                
                response_queue.put({
                    "status": "success",
                    "logits": logits_cpu,
                    "seq_len": seq_len
                })
            elif request.get("cmd") == "kv_init":
                # Prefill to build KV cache for a session
                session = request.get("session", "default")
                input_ids_cpu = request["input_ids"]
                try:
                    # Store prompt ids for HF-compatible sampling processors
                    try:
                        prompt_ids_list = input_ids_cpu[0].tolist()
                    except Exception:
                        prompt_ids_list = []
                    # Allocate arena if missing
                    if session not in generator.kv_arena:
                        cfg = generator.model.config
                        num_layers = getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 0))
                        num_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', 0))
                        head_dim = getattr(cfg, 'hidden_size', 0) // max(1, num_heads)
                        max_len = int(os.environ.get('CASE4_MAX_SEQ_LEN', '4096'))
                        k_store = torch.empty((num_layers, 1, num_heads, max_len, head_dim), dtype=torch.float16, device=generator.device)
                        v_store = torch.empty_like(k_store)
                        generator.kv_arena[session] = {
                            'k': k_store,
                            'v': v_store,
                            'seq_len': 0,
                            'num_layers': num_layers,
                            'num_heads': num_heads,
                            'head_dim': head_dim,
                            'max_len': max_len,
                        }
                    with torch.no_grad():
                        outputs = generator.model(
                            input_ids=input_ids_cpu.to(generator.device),
                            use_cache=True,
                            return_dict=True
                        )
                    # Copy prefill into arena and store views
                    arena = generator.kv_arena[session]
                    seq_len = outputs.past_key_values[0][0].size(2)
                    if seq_len > arena['max_len']:
                        raise RuntimeError(f"Prefill seq_len {seq_len} exceeds arena max_len {arena['max_len']}")
                    past_views = []
                    for l, (k, v) in enumerate(outputs.past_key_values):
                        # k,v: [1,H,T,D] -> store to arena [H,T,D]
                        arena['k'][l, 0, :, :seq_len, :].copy_(k[0])
                        arena['v'][l, 0, :, :seq_len, :].copy_(v[0])
                        k_view = arena['k'][l, 0, :, :seq_len, :].unsqueeze(0)
                        v_view = arena['v'][l, 0, :, :seq_len, :].unsqueeze(0)
                        past_views.append((k_view, v_view))
                    arena['seq_len'] = int(seq_len)
                    # Device-side seq_len scalar for device control
                    arena['seq_len_dev'] = torch.tensor([int(seq_len)], dtype=torch.int32, device=generator.device)
                    generator.kv_store[session] = tuple(past_views)
                    # Store prompt ids in arena state for processors
                    arena['prompt_ids'] = prompt_ids_list
                    # Return logits for the last position
                    logits = outputs.logits[0, -1, :].detach().cpu()
                    # Optional: capture custom dynamic-length decode graph once
                    if os.environ.get("CASE4_USE_CUSTOM", "0") == "1" and not generator.kv_only:
                        try:
                            mod = custom_decode_step
                            if mod is None:
                                # Try to pre-load torch libs and import again
                                try:
                                    import ctypes
                                    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
                                    for so in os.listdir(torch_lib_dir):
                                        if so.endswith('.so'):
                                            p = os.path.join(torch_lib_dir, so)
                                            try:
                                                ctypes.CDLL(p, mode=getattr(ctypes, 'RTLD_GLOBAL', None) or 0)
                                            except Exception:
                                                pass
                                    import importlib
                                    mod = importlib.import_module('custom_decode_step')
                                except Exception as _imp_e:
                                    print(f"[Generator] Custom module import failed: {_imp_e}")
                                    mod = None
                            if mod is None:
                                raise RuntimeError("custom_decode_step not available")
                            device = generator.device
                            vocab_size = int(logits.shape[0])
                            ctrl_input = torch.empty((1, 1), dtype=torch.long, device=device)
                            seq_len_dev = torch.tensor([arena['seq_len']], dtype=torch.int32, device=device)
                            pos_dev = torch.tensor([arena['seq_len']], dtype=torch.int32, device=device)
                            logits_buf = torch.empty((1, vocab_size), dtype=torch.float32, device=device)
                            g = torch.cuda.CUDAGraph()
                            cap_stream = torch.cuda.default_stream(device)
                            with torch.cuda.stream(cap_stream):
                                with torch.cuda.graph(g):
                                    mod.capture_decode(ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf)
                            generator.custom_state[session] = {
                                'graph': g,
                                'ctrl_input': ctrl_input,
                                'seq_len_dev': seq_len_dev,
                                'pos_dev': pos_dev,
                                'logits': logits_buf,
                            }
                            print(f"[Generator] Custom decode graph captured for session {session} at seq_len={arena['seq_len']}")
                        except Exception as _e:
                            # Fall back silently; client may choose standard path
                            print(f"[Generator] Custom capture failed: {_e}")
                    response_queue.put({
                        "status": "success",
                        "logits": logits,
                        "session": session
                    })
                except Exception as e:
                    response_queue.put({
                        "status": "error",
                        "error": str(e)
                    })
            elif request.get("cmd") == "qkv_smoke":
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                try:
                    stats = generator.qkv_smoke(token_id=token_id, layer_idx=layer_idx)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "mlp_gateup_smoke":
                # Live test: run gate/up GEMMs + SwiGLU (+ optional down) using packed weights for a layer
                try:
                    if custom_decode_step is None or generator.packed_weights is None:
                        raise RuntimeError("custom module or packed weights not available")
                    layer = int(request.get("layer_idx", 0))
                    H = int(generator.hidden_size)
                    # Infer I from packed shapes
                    gate_shape = generator.packed_weights['mlp_gate_shapes'][layer]
                    up_shape = generator.packed_weights['mlp_up_shapes'][layer]
                    I = int(max(gate_shape[0], up_shape[0]))
                    w_gate_ptr = int(generator.packed_weights['mlp_gate_ptrs'][layer])
                    w_up_ptr = int(generator.packed_weights['mlp_up_ptrs'][layer])
                    w_down_ptr = int(generator.packed_weights['mlp_down_ptrs'][layer])
                    if w_gate_ptr == 0 or w_up_ptr == 0:
                        raise RuntimeError("missing gate/up packed weights for layer")
                    # Create dummy normalized input and outputs on device
                    x_norm = torch.ones((H,), dtype=torch.float16, device=generator.device)
                    gate_out = torch.empty((I,), dtype=torch.float16, device=generator.device)
                    up_out = torch.empty((I,), dtype=torch.float16, device=generator.device)
                    act = torch.empty((I,), dtype=torch.float16, device=generator.device)
                    mlp_out = torch.empty((H,), dtype=torch.float16, device=generator.device)
                    # Run GEMMs and SwiGLU on current stream
                    custom_decode_step.gate_up_gemm(x_norm, gate_out, up_out,
                                                    int(w_gate_ptr), int(w_up_ptr), H, I)
                    custom_decode_step.swiglu(gate_out, up_out, act)
                    if w_down_ptr != 0:
                        custom_decode_step.down_gemm(act, mlp_out, int(w_down_ptr), I, H)
                        mlp_sum = float(mlp_out.float().sum().item())
                    else:
                        mlp_sum = 0.0
                    # Return simple checksums
                    gate_sum = float(gate_out.float().sum().item())
                    up_sum = float(up_out.float().sum().item())
                    act_sum = float(act.float().sum().item())
                    response_queue.put({
                        "status": "success",
                        "layer": layer,
                        "H": H,
                        "I": I,
                        "gate_sum": gate_sum,
                        "up_sum": up_sum,
                        "act_sum": act_sum,
                        "mlp_sum": mlp_sum
                    })
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "devgraph_capture_async":
                # Start capture on capture_stream in a background thread to overlap with replay
                session = request.get("session", "default")
                def _bg():
                    try:
                        # Simulate internal call by directly invoking capture logic
                        arena = generator.kv_arena.get(session, None)
                        if arena is None:
                            return
                        seq_len = int(arena['seq_len'])
                        if seq_len <= 0:
                            return
                        past_view = []
                        for l in range(arena['num_layers']):
                            k_view = arena['k'][l, 0, :, :seq_len, :].unsqueeze(0)
                            v_view = arena['v'][l, 0, :, :seq_len, :].unsqueeze(0)
                            past_view.append((k_view, v_view))
                        try:
                            from transformers.cache_utils import DynamicCache
                            pkv = DynamicCache.from_legacy_cache(tuple(past_view))
                        except Exception:
                            pkv = tuple(past_view)
                        ctrl_input = torch.empty((1, 1), dtype=torch.long, device=generator.device)
                        pos_dev = torch.tensor([[seq_len]], dtype=torch.long, device=generator.device)
                        static = {}
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=generator.capture_stream):
                            out = generator.model(
                                input_ids=ctrl_input,
                                position_ids=pos_dev,
                                use_cache=True,
                                past_key_values=pkv,
                                return_dict=True
                            )
                            static['logits'] = out.logits
                            static['past'] = out.past_key_values
                        # Store under a temp key to avoid race in lookup
                        generator.dev_graphs[session + "_next"] = {
                            'graph': g,
                            'ctrl_input': ctrl_input,
                            'pos_dev': pos_dev,
                            'outputs': static,
                        }
                        # Atomically promote to active if not present
                        generator.dev_graphs[session] = generator.dev_graphs.get(session, generator.dev_graphs[session + "_next"])
                    except Exception:
                        pass
                try:
                    t = threading.Thread(target=_bg, daemon=True)
                    t.start()
                    response_queue.put({"status": "success", "started": True})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "devgraph_ready":
                session = request.get("session", "default")
                state = generator.dev_graphs.get(session) or generator.dev_graphs.get(session + "_next")
                response_queue.put({"status": "success", "ready": bool(state)})
            elif request.get("cmd") == "emb_row_smoke":
                token_id = int(request.get("token_id", 0))
                try:
                    stats = generator.emb_row_smoke(token_id=token_id)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "qkv_heads_smoke":
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                try:
                    stats = generator.qkv_heads_smoke(token_id=token_id, layer_idx=layer_idx)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "qkv_stream_smoke":
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                iters = int(request.get("iters", 50))
                try:
                    stats = generator.qkv_stream_smoke(token_id=token_id, layer_idx=layer_idx, iters=iters)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "rope_smoke":
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                pos = int(request.get("pos", 0))
                try:
                    stats = generator.rope_smoke(token_id=token_id, layer_idx=layer_idx, pos=pos)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "mask_smoke":
                seq_len = int(request.get("seq_len", 1))
                try:
                    stats = generator.mask_smoke(seq_len=seq_len)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "attn_smoke":
                session = request.get("session", "default")
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                try:
                    stats = generator.attn_smoke(session=session, token_id=token_id, layer_idx=layer_idx)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "attn_varlen_smoke":
                session = request.get("session", "default")
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                try:
                    stats = generator.attn_varlen_smoke(session=session, token_id=token_id, layer_idx=layer_idx)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "mlp_logits_smoke":
                token_id = int(request.get("token_id", 0))
                layer_idx = int(request.get("layer_idx", 0))
                try:
                    stats = generator.mlp_logits_smoke(token_id=token_id, layer_idx=layer_idx)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "kv_append_smoke":
                session = request.get("session", "default")
                last_token = int(request.get("last_token", 0))
                try:
                    stats = generator.kv_append_smoke(session=session, last_token=last_token)
                    response_queue.put({"status": "success", "stats": stats})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "kv_step":
                # One-token decode step using stored KV
                session = request.get("session", "default")
                last_token = request["last_token"]
                try:
                    past = generator.kv_store.get(session, None)
                    if past is None:
                        raise RuntimeError("KV session not initialized")
                    input_ids = torch.tensor([[last_token]], dtype=torch.long, device=generator.device)
                    with torch.no_grad():
                        outputs = generator.model(
                            input_ids=input_ids,
                            use_cache=True,
                            past_key_values=past,
                            return_dict=True
                        )
                    generator.kv_store[session] = outputs.past_key_values
                    logits = outputs.logits[0, -1, :].detach().cpu()
                    response_queue.put({
                        "status": "success",
                        "logits": logits,
                        "session": session
                    })
                except Exception as e:
                    response_queue.put({
                        "status": "error",
                        "error": str(e)
                    })
            elif request.get("cmd") == "kv_generate":
                # Run N token generation on GPU using KV cache (no per-token IPC)
                session = request.get("session", "default")
                steps = int(request.get("steps", 1))
                temperature = float(request.get("temperature", 1.0))
                do_sample = bool(request.get("do_sample", False))
                top_p = float(request.get("top_p", 1.0))
                repetition_penalty = float(request.get("repetition_penalty", 1.0))
                no_repeat_ngram_size = int(request.get("no_repeat_ngram_size", 0))
                seed = request.get("seed", None)
                last_token = request.get("start_token", None)
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    if last_token is None:
                        raise RuntimeError("start_token must be provided for kv_generate")
                    generated_tokens = []
                    start = time.time()
                    # Optional RNG for reproducibility
                    rng = None
                    if seed is not None:
                        try:
                            # Align with HF: set global RNGs
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                            rng = torch.Generator(device=generator.device)
                            rng.manual_seed(int(seed))
                        except Exception:
                            rng = None
                    # HF-compatible processors/warpers
                    processors = None
                    try:
                        from transformers.generation.logits_process import (
                            LogitsProcessorList,
                            TopPLogitsWarper,
                            RepetitionPenaltyLogitsProcessor,
                            NoRepeatNGramLogitsProcessor,
                        )
                        processors = LogitsProcessorList()
                        if repetition_penalty and repetition_penalty != 1.0:
                            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
                        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
                        if top_p < 1.0:
                            processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
                    except Exception:
                        processors = None
                    with torch.no_grad():
                        for _ in range(steps):
                            # Build views into arena up to current seq_len (storage is stable)
                            seq_len = arena['seq_len']
                            past_view = []
                            for l in range(arena['num_layers']):
                                k_view = arena['k'][l, 0, :, :seq_len, :].unsqueeze(0)
                                v_view = arena['v'][l, 0, :, :seq_len, :].unsqueeze(0)
                                past_view.append((k_view, v_view))
                            # Feed the previous token to advance one step using KV
                            input_ids = torch.tensor([[int(last_token)]], dtype=torch.long, device=generator.device)
                            # Build cache object if required by transformers version
                            try:
                                from transformers.cache_utils import DynamicCache
                                cache_obj = DynamicCache.from_legacy_cache(tuple(past_view))
                                pkv = cache_obj
                            except Exception:
                                pkv = tuple(past_view)
                            outputs = generator.model(
                                input_ids=input_ids,
                                use_cache=True,
                                past_key_values=pkv,
                                return_dict=True
                            )
                            logits = outputs.logits[0, -1, :].float()
                            if do_sample:
                                # Compose input_ids for processors: prompt + generated so far
                                prompt_ids = arena.get('prompt_ids', [])
                                seq_ids = prompt_ids + generated_tokens
                                input_ids_for_proc = torch.tensor(seq_ids, dtype=torch.long, device=logits.device).view(1, -1)
                                # Temperature first
                                logits = logits / max(temperature, 1e-6)
                                if processors is not None:
                                    logits_for_proc = logits.view(1, -1)
                                    logits_for_proc = processors(input_ids_for_proc, logits_for_proc)
                                    logits = logits_for_proc.view(-1)
                                probs = torch.softmax(logits, dim=-1)
                                if rng is not None:
                                    next_token = int(torch.multinomial(probs, 1, generator=rng).item())
                                else:
                                    next_token = int(torch.multinomial(probs, 1).item())
                            else:
                                next_token = int(torch.argmax(logits).item())
                            generated_tokens.append(next_token)
                            last_token = next_token
                            # Append K/V for this position into arena in-place
                            new_seq = seq_len + 1
                            if new_seq > arena['max_len']:
                                raise RuntimeError("Exceeded arena max_len during kv_generate")
                            for l, (k_new, v_new) in enumerate(outputs.past_key_values):
                                # Copy only the last time step to avoid shape/broadcast issues
                                arena['k'][l, 0, :, seq_len:new_seq, :].copy_(k_new[0, :, -1:, :])
                                arena['v'][l, 0, :, seq_len:new_seq, :].copy_(v_new[0, :, -1:, :])
                            arena['seq_len'] = new_seq
                    torch.cuda.synchronize(device=generator.device)
                    elapsed_ms = (time.time() - start) * 1000.0
                    # Update kv_store views
                    past_views = []
                    for l in range(arena['num_layers']):
                        k_view = arena['k'][l, 0, :, :arena['seq_len'], :].unsqueeze(0)
                        v_view = arena['v'][l, 0, :, :arena['seq_len'], :].unsqueeze(0)
                        past_views.append((k_view, v_view))
                    generator.kv_store[session] = tuple(past_views)
                    response_queue.put({
                        "status": "success",
                        "tokens": generated_tokens,
                        "elapsed_ms": elapsed_ms,
                        "session": session
                    })
                except Exception as e:
                    response_queue.put({
                        "status": "error",
                        "error": str(e)
                    })
            elif request.get("cmd") == "devgraph_capture":
                # Capture single-token decode graph with device-side controls (ctrl_input, position_ids)
                session = request.get("session", "default")
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    seq_len = int(arena['seq_len'])
                    max_len = int(arena['max_len'])
                    # Build pkv views up to MAX length (constant shape) for dynamic-length replay
                    past_view = []
                    for l in range(arena['num_layers']):
                        # Use max_len-1 so that after one-step decode, KV length becomes max_len
                        k_view = arena['k'][l, 0, :, :max_len-1, :].unsqueeze(0)
                        v_view = arena['v'][l, 0, :, :max_len-1, :].unsqueeze(0)
                        past_view.append((k_view, v_view))
                    try:
                        from transformers.cache_utils import DynamicCache
                        pkv = DynamicCache.from_legacy_cache(tuple(past_view))
                    except Exception:
                        pkv = tuple(past_view)
                    # Device controls
                    ctrl_input = torch.empty((1, 1), dtype=torch.long, device=generator.device)
                    pos_dev = torch.tensor([[seq_len]], dtype=torch.long, device=generator.device)
                    # Dynamic attention mask buffers (constant shape; values rebuilt per replay)
                    # For single-token decode with past_len=L (max_len-1), mask length must be L+1 (= max_len)
                    attn_mask = torch.empty((1, 1, 1, max_len), dtype=torch.float16, device=generator.device)
                    # Pre-allocate indices and boolean workspace to avoid allocations during capture
                    t_idx = torch.arange(max_len, device=generator.device, dtype=torch.int32)
                    valid = torch.empty((max_len,), dtype=torch.bool, device=generator.device)
                    seq_len_dev = arena.get('seq_len_dev', torch.tensor([seq_len], dtype=torch.int32, device=generator.device))
                    # Static outputs captured
                    static = {}
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=generator.capture_stream):
                        # Build dynamic mask inside the graph using seq_len_dev without allocating
                        attn_mask.fill_(float("-inf"))
                        # valid if t < (seq_len + 1) to account for current query position
                        torch.lt(t_idx, (seq_len_dev.view(()).to(dtype=torch.int32) + 1), out=valid)
                        attn_mask.masked_fill_(valid.view(1, 1, 1, max_len), 0.0)
                        out = generator.model(
                            input_ids=ctrl_input,
                            position_ids=pos_dev,
                            attention_mask=attn_mask,
                            use_cache=True,
                            past_key_values=pkv,
                            return_dict=True
                        )
                        static['logits'] = out.logits
                        static['past'] = out.past_key_values
                    generator.dev_graphs[session] = {
                        'graph': g,
                        'ctrl_input': ctrl_input,
                        'pos_dev': pos_dev,
                        'seq_len_dev': seq_len_dev,
                        't_idx': t_idx,
                        'valid': valid,
                        'attn_mask': attn_mask,
                        'outputs': static,
                    }
                    response_queue.put({"status": "success"})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "devgraph_replay":
                # Replay device-control decode graph N times; append K/V and bump seq_len_dev
                session = request.get("session", "default")
                steps = int(request.get("steps", 1))
                last_token = request.get("start_token", None)
                temperature = float(request.get("temperature", 1.0))
                do_sample = bool(request.get("do_sample", False))
                top_p = float(request.get("top_p", 1.0))
                repetition_penalty = float(request.get("repetition_penalty", 1.0))
                no_repeat_ngram_size = int(request.get("no_repeat_ngram_size", 0))
                seed = request.get("seed", None)
                try:
                    arena = generator.kv_arena.get(session, None)
                    state = generator.dev_graphs.get(session, None) or generator.dev_graphs.get(session + "_next", None)
                    if arena is None or state is None:
                        raise RuntimeError("devgraph not captured or KV not initialized")
                    if last_token is None:
                        raise RuntimeError("start_token must be provided for devgraph_replay")
                    g = state['graph']
                    ctrl = state['ctrl_input']
                    pos_dev = state['pos_dev']
                    seq_len_dev = state.get('seq_len_dev', None)
                    out_refs = state['outputs']
                    generated = []
                    start_t = time.time()
                    # Optional RNG for reproducibility
                    rng = None
                    if seed is not None:
                        try:
                            # Align with HF: set global RNGs
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                            rng = torch.Generator(device=generator.device)
                            rng.manual_seed(int(seed))
                        except Exception:
                            rng = None
                    # Try to import HF logits processors for parity
                    processors = None
                    try:
                        from transformers.generation.logits_process import (
                            LogitsProcessorList,
                            TopPLogitsWarper,
                            RepetitionPenaltyLogitsProcessor,
                            NoRepeatNGramLogitsProcessor,
                        )
                        processors = LogitsProcessorList()
                        if repetition_penalty and repetition_penalty != 1.0:
                            processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
                        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
                        if top_p < 1.0:
                            processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))
                    except Exception:
                        processors = None
                    with torch.no_grad(), torch.cuda.stream(generator.replay_stream):
                        for _ in range(steps):
                            # Update device controls
                            pos = int(arena['seq_len'])
                            pos_dev.fill_(pos)
                            if seq_len_dev is not None:
                                # Mirror host seq_len into device scalar for mask build inside graph
                                seq_len_dev.fill_(pos)
                            ctrl.fill_(int(last_token))
                            g.replay()
                            logits = out_refs['logits'][0, -1, :].float()
                            if do_sample:
                                # Build input_ids for processors: prompt_ids + generated
                                try:
                                    prompt_ids = arena.get('prompt_ids', [])
                                except Exception:
                                    prompt_ids = []
                                seq_ids = prompt_ids + generated
                                input_ids_for_proc = torch.tensor(seq_ids, dtype=torch.long, device=logits.device).view(1, -1)
                                # Temperature first
                                logits = logits / max(temperature, 1e-6)
                                if processors is not None:
                                    logits_for_proc = logits.view(1, -1)
                                    logits_for_proc = processors(input_ids_for_proc, logits_for_proc)
                                    logits = logits_for_proc.view(-1)
                                probs = torch.softmax(logits, dim=-1)
                                if rng is not None:
                                    next_token = int(torch.multinomial(probs, 1, generator=rng).item())
                                else:
                                    next_token = int(torch.multinomial(probs, 1).item())
                            else:
                                next_token = int(torch.argmax(logits).item())
                            generated.append(next_token)
                            # Append last position from out_refs['past']
                            seq_len = arena['seq_len']
                            new_seq = seq_len + 1
                            if new_seq > arena['max_len']:
                                raise RuntimeError("Exceeded arena max_len during devgraph_replay")
                            for l, (k_new, v_new) in enumerate(out_refs['past']):
                                arena['k'][l, 0, :, seq_len:new_seq, :].copy_(k_new[0, :, -1:, :])
                                arena['v'][l, 0, :, seq_len:new_seq, :].copy_(v_new[0, :, -1:, :])
                            arena['seq_len'] = new_seq
                            if 'seq_len_dev' in arena and isinstance(arena['seq_len_dev'], torch.Tensor):
                                arena['seq_len_dev'].add_(1)
                            last_token = next_token
                    torch.cuda.synchronize(device=generator.device)
                    elapsed_ms = (time.time() - start_t) * 1000.0
                    response_queue.put({"status": "success", "tokens": generated, "elapsed_ms": elapsed_ms, "session": session})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})

            elif request.get("cmd") == "devgraph_capture_exact":
                # Capture a per-seq-length custom graph (exact shapes) at current or specified seq_len
                session = request.get("session", "default")
                seq_len_req = request.get("seq_len", None)
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    seq_len = int(arena['seq_len']) if seq_len_req is None else int(seq_len_req)
                    num_layers = int(arena['num_layers'])
                    # Build pkv views up to seq_len (exact)
                    past_view = []
                    for l in range(num_layers):
                        k_view = arena['k'][l, 0, :, :seq_len, :].unsqueeze(0)
                        v_view = arena['v'][l, 0, :, :seq_len, :].unsqueeze(0)
                        past_view.append((k_view, v_view))
                    try:
                        from transformers.cache_utils import DynamicCache
                        pkv = DynamicCache.from_legacy_cache(tuple(past_view))
                    except Exception:
                        pkv = tuple(past_view)
                    ctrl_input = torch.empty((1, 1), dtype=torch.long, device=generator.device)
                    pos_dev = torch.tensor([[seq_len]], dtype=torch.long, device=generator.device)
                    attn_mask = torch.empty((1, 1, 1, seq_len + 1), dtype=torch.float16, device=generator.device)
                    t_idx = torch.arange(seq_len + 1, device=generator.device, dtype=torch.int32)
                    valid = torch.empty((seq_len + 1,), dtype=torch.bool, device=generator.device)
                    g = torch.cuda.CUDAGraph()
                    static = {}
                    with torch.cuda.graph(g, stream=generator.capture_stream):
                        attn_mask.fill_(float("-inf"))
                        torch.lt(t_idx, (torch.tensor(seq_len, dtype=torch.int32, device=generator.device) + 1), out=valid)
                        attn_mask.masked_fill_(valid.view(1, 1, 1, seq_len + 1), 0.0)
                        out = generator.model(
                            input_ids=ctrl_input,
                            position_ids=pos_dev,
                            attention_mask=attn_mask,
                            use_cache=True,
                            past_key_values=pkv,
                            return_dict=True
                        )
                        static['logits'] = out.logits
                        static['past'] = out.past_key_values
                    sess_map = generator.dev_seq_graphs.setdefault(session, {})
                    sess_map[seq_len] = {
                        'graph': g,
                        'ctrl_input': ctrl_input,
                        'pos_dev': pos_dev,
                        'attn_mask': attn_mask,
                        'outputs': static,
                        'seq_len': seq_len
                    }
                    response_queue.put({"status": "success", "seq_len": seq_len})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})

            elif request.get("cmd") == "devgraph_bg_capture_start":
                # Start background exact-seq graph capture to keep ahead
                session = request.get("session", "default")
                ahead = int(request.get("ahead", 150))
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    with generator.dev_bg_lock:
                        generator.dev_bg_stop = False
                        generator.dev_bg_ahead = ahead
                        if generator.dev_bg_thread is None or not generator.dev_bg_thread.is_alive():
                            def _bg():
                                torch.cuda.set_device(generator.device)
                                print("[DevGraph Capture Thread] Started (exact-seq)", flush=True)
                                while True:
                                    with generator.dev_bg_lock:
                                        if generator.dev_bg_stop:
                                            break
                                        # Pause background capture if requested by replay
                                        if getattr(generator, 'dev_bg_paused', False):
                                            pass
                                        ahead_loc = generator.dev_bg_ahead
                                    if getattr(generator, 'dev_bg_paused', False):
                                        time.sleep(0.001)
                                        continue
                                    cur = int(generator.kv_arena[session]['seq_len'])
                                    target = min(cur + ahead_loc, int(generator.kv_arena[session]['max_len']))
                                    sess_map = generator.dev_seq_graphs.setdefault(session, {})
                                    for s in range(cur, target + 1):
                                        if s in sess_map:
                                            continue
                                        try:
                                            # Inline capture (same as devgraph_capture_exact)
                                            num_layers = int(generator.kv_arena[session]['num_layers'])
                                            past_view = []
                                            for l in range(num_layers):
                                                k_view = generator.kv_arena[session]['k'][l, 0, :, :s, :].unsqueeze(0)
                                                v_view = generator.kv_arena[session]['v'][l, 0, :, :s, :].unsqueeze(0)
                                                past_view.append((k_view, v_view))
                                            try:
                                                from transformers.cache_utils import DynamicCache
                                                pkv = DynamicCache.from_legacy_cache(tuple(past_view))
                                            except Exception:
                                                pkv = tuple(past_view)
                                            ctrl_input = torch.empty((1, 1), dtype=torch.long, device=generator.device)
                                            pos_dev = torch.tensor([[s]], dtype=torch.long, device=generator.device)
                                            attn_mask = torch.empty((1, 1, 1, s + 1), dtype=torch.float16, device=generator.device)
                                            t_idx = torch.arange(s + 1, device=generator.device, dtype=torch.int32)
                                            valid = torch.empty((s + 1,), dtype=torch.bool, device=generator.device)
                                            g = torch.cuda.CUDAGraph()
                                            pool = torch.cuda.graphs.graph_pool_handle()
                                            static = {}
                                            with torch.cuda.graph(g, stream=generator.capture_stream, pool=pool):
                                                attn_mask.fill_(float("-inf"))
                                                torch.lt(t_idx, (torch.tensor(s, dtype=torch.int32, device=generator.device) + 1), out=valid)
                                                attn_mask.masked_fill_(valid.view(1, 1, 1, s + 1), 0.0)
                                                out = generator.model(
                                                    input_ids=ctrl_input,
                                                    position_ids=pos_dev,
                                                    attention_mask=attn_mask,
                                                    use_cache=True,
                                                    past_key_values=pkv,
                                                    return_dict=True
                                                )
                                                static['logits'] = out.logits
                                                static['past'] = out.past_key_values
                                            sess_map[s] = {
                                                'graph': g,
                                                'ctrl_input': ctrl_input,
                                                'pos_dev': pos_dev,
                                                'attn_mask': attn_mask,
                                                'outputs': static,
                                                'seq_len': s
                                            }
                                            if s % 50 == 0:
                                                print(f"[DevGraph Capture] ✓ seq_len={s}", flush=True)
                                        except Exception:
                                            pass
                                    time.sleep(0.002)
                                print("[DevGraph Capture Thread] Stopped", flush=True)
                            generator.dev_bg_thread = threading.Thread(target=_bg, daemon=True)
                            generator.dev_bg_thread.start()
                    response_queue.put({"status": "success"})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})

            elif request.get("cmd") == "devgraph_bg_capture_stop":
                try:
                    with generator.dev_bg_lock:
                        generator.dev_bg_stop = True
                    t = generator.dev_bg_thread
                    if t is not None:
                        t.join(timeout=1.0)
                    response_queue.put({"status": "success"})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})

            elif request.get("cmd") == "devgraph_replay_seq":
                # Replay using exact-seq custom graphs; assumes BG capture keeps ahead
                session = request.get("session", "default")
                steps = int(request.get("steps", 1))
                last_token = request.get("start_token", None)
                temperature = float(request.get("temperature", 1.0))
                do_sample = bool(request.get("do_sample", False))
                top_p = float(request.get("top_p", 1.0))
                repetition_penalty = float(request.get("repetition_penalty", 1.0))
                no_repeat_ngram_size = int(request.get("no_repeat_ngram_size", 0))
                seed = request.get("seed", None)
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    if last_token is None:
                        raise RuntimeError("start_token must be provided for devgraph_replay_seq")
                    generated = []
                    start = time.time()
                    if seed is not None:
                        try:
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                        except Exception:
                            pass
                    for _ in range(steps):
                        seq_len = int(arena['seq_len'])
                        sess_map = generator.dev_seq_graphs.setdefault(session, {})
                        state = sess_map.get(seq_len, None)
                        if state is None:
                            raise RuntimeError(f"graph for seq_len={seq_len} not available yet")
                        g = state['graph']
                        ctrl = state['ctrl_input']
                        pos_dev = state['pos_dev']
                        logits_buf = state['outputs']['logits']
                        ctrl.fill_(int(last_token))
                        with torch.cuda.stream(generator.replay_stream):
                            g.replay()
                        generator.cpp_mgr.sync_replay() if generator.cpp_mgr is not None else None
                        if do_sample and top_p < 1.0:
                            probs = torch.softmax(logits_buf[0] / max(1e-6, temperature), dim=-1).float().cpu()
                            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                            cdf = torch.cumsum(sorted_probs, dim=-1)
                            cutoff = torch.searchsorted(cdf, torch.tensor(top_p)).item()
                            cutoff = max(1, cutoff)
                            top_idx = sorted_idx[:cutoff]
                            top_probs = sorted_probs[:cutoff]
                            top_probs = top_probs / top_probs.sum()
                            next_token = int(top_idx[torch.multinomial(top_probs, 1)].item())
                        else:
                            next_token = int(torch.argmax(logits_buf[0]).item())
                        generated.append(next_token)
                        last_token = next_token
                        # Append new K/V at last position and bump seq_len
                        out_past = state['outputs']['past']
                        new_seq = seq_len + 1
                        if new_seq > arena['max_len']:
                            raise RuntimeError("Exceeded arena max_len during devgraph_replay_seq")
                        for l, (k_new, v_new) in enumerate(out_past):
                            arena['k'][l, 0, :, seq_len:new_seq, :].copy_(k_new[0, :, -1:, :])
                            arena['v'][l, 0, :, seq_len:new_seq, :].copy_(v_new[0, :, -1:, :])
                        arena['seq_len'] = new_seq
                    torch.cuda.synchronize(device=generator.device)
                    elapsed_ms = (time.time() - start) * 1000.0
                    response_queue.put({"status": "success", "tokens": generated, "elapsed_ms": elapsed_ms, "session": session})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            elif request.get("cmd") == "graph_kv_capture":
                # Capture single-token decode graph for a session at current seq_len
                session = request.get("session", "default")
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    # Build views up to current seq_len
                    seq_len = arena['seq_len']
                    past_view = []
                    for l in range(arena['num_layers']):
                        k_view = arena['k'][l, 0, :, :seq_len, :].unsqueeze(0)
                        v_view = arena['v'][l, 0, :, :seq_len, :].unsqueeze(0)
                        past_view.append((k_view, v_view))
                    # Cache object if required
                    try:
                        from transformers.cache_utils import DynamicCache
                        pkv = DynamicCache.from_legacy_cache(tuple(past_view))
                    except Exception:
                        pkv = tuple(past_view)
                    # Control input (host-updated between replays)
                    ctrl_input = torch.empty((1, 1), dtype=torch.long, device=generator.device)
                    static_outputs = {}
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        out = generator.model(
                            input_ids=ctrl_input,
                            use_cache=True,
                            past_key_values=pkv,
                            return_dict=True
                        )
                        static_outputs['logits'] = out.logits
                        static_outputs['past'] = out.past_key_values
                    generator.decode_graphs[session] = {
                        'graph': g,
                        'ctrl_input': ctrl_input,
                        'outputs': static_outputs,
                    }
                    response_queue.put({"status": "success"})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})

            elif request.get("cmd") == "graph_kv_generate":
                # Replay captured graph N times with host-updated control token, append K/V in-place
                session = request.get("session", "default")
                steps = int(request.get("steps", 1))
                temperature = float(request.get("temperature", 1.0))
                do_sample = bool(request.get("do_sample", False))
                last_token = request.get("start_token", None)
                try:
                    arena = generator.kv_arena.get(session, None)
                    state = generator.decode_graphs.get(session, None)
                    if arena is None or state is None:
                        raise RuntimeError("graph not captured or KV not initialized")
                    if last_token is None:
                        raise RuntimeError("start_token must be provided for graph_kv_generate")
                    g = state['graph']
                    ctrl = state['ctrl_input']
                    out_refs = state['outputs']
                    generated = []
                    start = time.time()
                    with torch.no_grad():
                        for _ in range(steps):
                            # Update last token
                            ctrl.fill_(int(last_token))
                            g.replay()
                            logits = out_refs['logits'][0, -1, :]
                            if do_sample:
                                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                                next_token = torch.multinomial(probs, 1).item()
                            else:
                                next_token = torch.argmax(logits).item()
                            generated.append(next_token)
                            # Append only the last position from out_refs['past'] into arena
                            seq_len = arena['seq_len']
                            new_seq = seq_len + 1
                            if new_seq > arena['max_len']:
                                raise RuntimeError("Exceeded arena max_len during graph replay")
                            for l, (k_new, v_new) in enumerate(out_refs['past']):
                                arena['k'][l, 0, :, seq_len:new_seq, :].copy_(k_new[0, :, seq_len:new_seq, :])
                                arena['v'][l, 0, :, seq_len:new_seq, :].copy_(v_new[0, :, seq_len:new_seq, :])
                            arena['seq_len'] = new_seq
                            last_token = next_token
                    torch.cuda.synchronize(device=generator.device)
                    elapsed_ms = (time.time() - start) * 1000.0
                    response_queue.put({
                        "status": "success",
                        "tokens": generated,
                        "elapsed_ms": elapsed_ms,
                        "session": session
                    })
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})

            elif request.get("cmd") == "custom_generate":
                # Replay captured custom decode graph N times using device controls
                session = request.get("session", "default")
                steps = int(request.get("steps", 1))
                start_token = request.get("start_token", None)
                try:
                    state = generator.custom_state.get(session, None)
                    arena = generator.kv_arena.get(session, None)
                    if state is None or arena is None:
                        raise RuntimeError("custom graph not captured or KV not initialized")
                    if start_token is None:
                        raise RuntimeError("start_token must be provided for custom_generate")
                    g = state['graph']
                    ctrl = state['ctrl_input']
                    seq_len_dev = state['seq_len_dev']
                    pos_dev = state['pos_dev']
                    logits_buf = state['logits']
                    generated = []
                    last_token = int(start_token)
                    start = time.time()
                    with torch.no_grad():
                        for _ in range(steps):
                            ctrl.fill_(last_token)
                            pos_dev.fill_(arena['seq_len'])
                            with torch.cuda.stream(generator.replay_stream):
                                g.replay()
                            generator.cpp_mgr.sync_replay()
                            # Greedy from device logits
                            next_token = int(torch.argmax(logits_buf[0]).item())
                            generated.append(next_token)
                            last_token = next_token
                    torch.cuda.synchronize(device=generator.device)
                    elapsed_ms = (time.time() - start) * 1000.0
                    response_queue.put({"status": "success", "tokens": generated, "elapsed_ms": elapsed_ms, "session": session})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            
            elif request.get("cmd") == "prep_next":
                # Overlap with sampling: delete previous graph and capture the next one
                prev_seq = request.get("prev_seq_len")
                next_seq = request.get("next_seq_len")
                error_msgs = []
                # Temporarily pause background ahead-capture during explicit prep
                generator.suspend_bg = True
                try:
                    if prev_seq is not None and prev_seq >= 1:
                        generator.delete_graph(prev_seq)
                except Exception as e:
                    error_msgs.append(f"delete prev_seq={prev_seq} failed: {e}")
                try:
                    if next_seq is not None and next_seq >= 1:
                        generator.capture_one(next_seq)
                except Exception as e:
                    error_msgs.append(f"capture next_seq={next_seq} failed: {e}")
                finally:
                    generator.suspend_bg = False
                if error_msgs:
                    response_queue.put({"status": "error", "error": "; ".join(error_msgs)})
                else:
                    response_queue.put({"status": "success", "prepared": next_seq, "deleted": prev_seq})
            
            elif request.get("cmd") == "status":
                with generator.graphs_lock:
                    seqs = list(generator.graphs.keys())
                response_queue.put({
                    "status": "success",
                    "num_graphs": len(generator.graphs),
                    "min_seq": min(seqs) if seqs else 0,
                    "max_seq": max(seqs) if seqs else 0,
                    "max_captured": generator.max_captured
                })
            
            # === Custom exact-seq capture/replay (kernel-only) ===
            elif request.get("cmd") == "devgraph_capture_exact_custom":
                # Capture a per-seq custom graph using custom_decode_step (no PyTorch forward)
                session = request.get("session", "default")
                seq_len_req = request.get("seq_len", None)
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    if custom_decode_step is None:
                        raise RuntimeError("custom_decode_step extension not available")
                    seq_len = int(arena['seq_len']) if seq_len_req is None else int(seq_len_req)
                    device = generator.device
                    vocab_size = int(getattr(generator.model.config, 'vocab_size', 32000))
                    ctrl_input = torch.empty((1, 1), dtype=torch.long, device=device)
                    # Device seq_len scalar mirrors host arena['seq_len']; kernel updates it in-place
                    seq_len_dev = torch.tensor([seq_len], dtype=torch.int32, device=device)
                    pos_dev = torch.tensor([seq_len], dtype=torch.int32, device=device)
                    logits_buf = torch.empty((1, vocab_size), dtype=torch.float32, device=device)
                    g = torch.cuda.CUDAGraph()
                    # Ensure current stream is the capture stream for extension to bind correctly
                    with torch.cuda.stream(generator.capture_stream):
                        # Optional warmup to materialize kernel; prefer extended only if all required args are valid
                        try:
                            use_ext = (
                                hasattr(generator, "custom_scratch") and bool(generator.custom_scratch) and
                                getattr(generator, "W_gate_ptrs", None) is not None and
                                len(generator.W_gate_ptrs) > 0 and
                                int(generator.W_gate_ptrs[0]) != 0 and
                                getattr(generator, "W_up_ptrs", None) is not None and
                                len(generator.W_up_ptrs) > 0 and
                                int(generator.W_up_ptrs[0]) != 0 and
                                getattr(generator, "W_down_ptrs", None) is not None and
                                len(generator.W_down_ptrs) > 0 and
                                int(generator.W_down_ptrs[0]) != 0 and
                                int(getattr(generator, "LM_HEAD_ptr", 0)) != 0 and
                                int(getattr(generator, "hidden_size", 0)) > 0 and
                                int(getattr(generator, "intermediate_size", 0)) > 0 and
                                os.environ.get("CASE4_DISABLE_EXT", "0") != "1"
                            )
                            if use_ext:
                                H_hidden = int(generator.hidden_size)
                                I = int(generator.intermediate_size)
                                rms_eps = float(getattr(generator.model.config, "rms_norm_eps", 1e-6))
                                layer0 = 0
                                custom_decode_step.capture_decode_ext(
                                    ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf,
                                    generator.custom_scratch['x_norm'],
                                    generator.custom_scratch['gate'],
                                    generator.custom_scratch['up'],
                                    generator.custom_scratch['act'],
                                    generator.custom_scratch['mlp_out'],
                                    int(generator.RMS_in_ptrs[layer0]) if generator.RMS_in_ptrs else 0,
                                    int(generator.RMS_post_ptrs[layer0]) if generator.RMS_post_ptrs else 0,
                                    int(generator.RMS_final_ptr) if getattr(generator, "RMS_final_ptr", 0) else 0,
                                    int(generator.W_gate_ptrs[layer0]) if generator.W_gate_ptrs else 0,
                                    int(generator.W_up_ptrs[layer0]) if generator.W_up_ptrs else 0,
                                    int(generator.W_down_ptrs[layer0]) if generator.W_down_ptrs else 0,
                                    int(generator.LM_HEAD_ptr) if getattr(generator, "LM_HEAD_ptr", 0) else 0,
                                    int(generator.num_layers_cfg), int(H_hidden), int(arena['k'].size(3)), int(arena['k'].size(4)), int(vocab_size),
                                    int(I),
                                    float(rms_eps)
                                )
                            else:
                                custom_decode_step.capture_decode(ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf)
                        except Exception:
                            # Fallback to base capture on any error
                            custom_decode_step.capture_decode(ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf)
                        with torch.cuda.graph(g, stream=generator.capture_stream):
                            try:
                                if use_ext:
                                    H_hidden = int(generator.hidden_size)
                                    I = int(generator.intermediate_size)
                                    rms_eps = float(getattr(generator.model.config, "rms_norm_eps", 1e-6))
                                    layer0 = 0
                                    custom_decode_step.capture_decode_ext(
                                        ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf,
                                        generator.custom_scratch['x_norm'],
                                        generator.custom_scratch['gate'],
                                        generator.custom_scratch['up'],
                                        generator.custom_scratch['act'],
                                        generator.custom_scratch['mlp_out'],
                                        int(generator.RMS_in_ptrs[layer0]) if generator.RMS_in_ptrs else 0,
                                        int(generator.RMS_post_ptrs[layer0]) if generator.RMS_post_ptrs else 0,
                                        int(generator.RMS_final_ptr) if getattr(generator, "RMS_final_ptr", 0) else 0,
                                        int(generator.W_gate_ptrs[layer0]) if generator.W_gate_ptrs else 0,
                                        int(generator.W_up_ptrs[layer0]) if generator.W_up_ptrs else 0,
                                        int(generator.W_down_ptrs[layer0]) if generator.W_down_ptrs else 0,
                                        int(generator.LM_HEAD_ptr) if getattr(generator, "LM_HEAD_ptr", 0) else 0,
                                        int(generator.num_layers_cfg), int(H_hidden), int(arena['k'].size(3)), int(arena['k'].size(4)), int(vocab_size),
                                        int(I),
                                        float(rms_eps)
                                    )
                                else:
                                    custom_decode_step.capture_decode(
                                        ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf
                                    )
                            except Exception:
                                custom_decode_step.capture_decode(
                                    ctrl_input, arena['k'], arena['v'], seq_len_dev, pos_dev, logits_buf
                                )
                    sess_map = generator.dev_seq_graphs.setdefault(session, {})
                    sess_map[seq_len] = {
                        'graph': g,
                        'ctrl_input': ctrl_input,
                        'seq_len_dev': seq_len_dev,
                        'pos_dev': pos_dev,
                        'logits': logits_buf,
                        'seq_len': seq_len,
                        # packed weight pointers and dims for custom decode wiring
                        'H': int(generator.hidden_size),
                        'I': int(generator.intermediate_size) if generator.intermediate_size else 0,
                        'V': int(generator.vocab_size_cfg) if generator.vocab_size_cfg else 0,
                        'W_gate_ptrs': getattr(generator, 'W_gate_ptrs', []),
                        'W_up_ptrs': getattr(generator, 'W_up_ptrs', []),
                        'W_down_ptrs': getattr(generator, 'W_down_ptrs', []),
                        'RMS_in_ptrs': getattr(generator, 'RMS_in_ptrs', []),
                        'RMS_post_ptrs': getattr(generator, 'RMS_post_ptrs', []),
                        'RMS_final_ptr': getattr(generator, 'RMS_final_ptr', 0),
                        'LM_HEAD_ptr': getattr(generator, 'LM_HEAD_ptr', 0),
                    }
                    response_queue.put({"status": "success", "seq_len": seq_len})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            
            elif request.get("cmd") == "devgraph_bg_capture_start_custom":
                # Start background exact-seq capture using custom kernels only
                session = request.get("session", "default")
                ahead = int(request.get("ahead", 150))
                try:
                    if custom_decode_step is None:
                        raise RuntimeError("custom_decode_step extension not available")
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    with generator.dev_bg_lock:
                        generator.dev_bg_stop = False
                        generator.dev_bg_ahead = ahead
                        if generator.dev_bg_thread is None or not generator.dev_bg_thread.is_alive():
                            def _bg_custom():
                                torch.cuda.set_device(generator.device)
                                print("[DevGraph Capture Thread] Started (custom exact-seq)", flush=True)
                                while True:
                                    with generator.dev_bg_lock:
                                        if generator.dev_bg_stop:
                                            break
                                        if getattr(generator, 'dev_bg_paused', False):
                                            pass
                                        ahead_loc = generator.dev_bg_ahead
                                    if getattr(generator, 'dev_bg_paused', False):
                                        time.sleep(0.001)
                                        continue
                                    cur = int(generator.kv_arena[session]['seq_len'])
                                    target = min(cur + ahead_loc, int(generator.kv_arena[session]['max_len']))
                                    sess_map = generator.dev_seq_graphs.setdefault(session, {})
                                    for s in range(cur, target + 1):
                                        if s in sess_map:
                                            continue
                                        try:
                                            device = generator.device
                                            vocab_size = int(getattr(generator.model.config, 'vocab_size', 32000))
                                            ctrl_input = torch.empty((1, 1), dtype=torch.long, device=device)
                                            seq_len_dev = torch.tensor([s], dtype=torch.int32, device=device)
                                            pos_dev = torch.tensor([s], dtype=torch.int32, device=device)
                                            logits_buf = torch.empty((1, vocab_size), dtype=torch.float32, device=device)
                                            g = torch.cuda.CUDAGraph()
                                            pool = torch.cuda.graphs.graph_pool_handle()
                                            with torch.cuda.stream(generator.capture_stream):
                                                try:
                                                    use_ext = (
                                                        hasattr(generator, "custom_scratch") and bool(generator.custom_scratch) and
                                                        getattr(generator, "W_gate_ptrs", None) is not None and
                                                        len(generator.W_gate_ptrs) > 0 and int(generator.W_gate_ptrs[0]) != 0 and
                                                        getattr(generator, "W_up_ptrs", None) is not None and
                                                        len(generator.W_up_ptrs) > 0 and int(generator.W_up_ptrs[0]) != 0 and
                                                        getattr(generator, "W_down_ptrs", None) is not None and
                                                        len(generator.W_down_ptrs) > 0 and int(generator.W_down_ptrs[0]) != 0 and
                                                        int(getattr(generator, "LM_HEAD_ptr", 0)) != 0 and
                                                        int(getattr(generator, "hidden_size", 0)) > 0 and
                                                        int(getattr(generator, "intermediate_size", 0)) > 0 and
                                                        os.environ.get("CASE4_DISABLE_EXT", "0") != "1"
                                                    )
                                                    captured_ok = False
                                                    with torch.cuda.graph(g, stream=generator.capture_stream, pool=pool):
                                                        if use_ext:
                                                            H_hidden = int(generator.hidden_size)
                                                            I = int(generator.intermediate_size)
                                                            rms_eps = float(getattr(generator.model.config, "rms_norm_eps", 1e-6))
                                                            layer0 = 0
                                                            custom_decode_step.capture_decode_ext(
                                                                ctrl_input,
                                                                generator.kv_arena[session]['k'],
                                                                generator.kv_arena[session]['v'],
                                                                seq_len_dev,
                                                                pos_dev,
                                                                logits_buf,
                                                                generator.custom_scratch['x_norm'],
                                                                generator.custom_scratch['gate'],
                                                                generator.custom_scratch['up'],
                                                                generator.custom_scratch['act'],
                                                                generator.custom_scratch['mlp_out'],
                                                                int(generator.RMS_in_ptrs[layer0]) if generator.RMS_in_ptrs else 0,
                                                                int(generator.RMS_post_ptrs[layer0]) if generator.RMS_post_ptrs else 0,
                                                                int(generator.RMS_final_ptr) if getattr(generator, "RMS_final_ptr", 0) else 0,
                                                                int(generator.W_gate_ptrs[layer0]) if generator.W_gate_ptrs else 0,
                                                                int(generator.W_up_ptrs[layer0]) if generator.W_up_ptrs else 0,
                                                                int(generator.W_down_ptrs[layer0]) if generator.W_down_ptrs else 0,
                                                                int(generator.LM_HEAD_ptr) if getattr(generator, "LM_HEAD_ptr", 0) else 0,
                                                                int(generator.num_layers_cfg), int(H_hidden), int(generator.kv_arena[session]['k'].size(3)), int(generator.kv_arena[session]['k'].size(4)), int(vocab_size),
                                                                int(I),
                                                                float(rms_eps)
                                                            )
                                                        else:
                                                            custom_decode_step.capture_decode(
                                                                ctrl_input,
                                                                generator.kv_arena[session]['k'],
                                                                generator.kv_arena[session]['v'],
                                                                seq_len_dev,
                                                                pos_dev,
                                                                logits_buf
                                                            )
                                                    captured_ok = True
                                                except Exception:
                                                    pass  # do not store uncaptured graph; warmup already executed
                                            if captured_ok:
                                                sess_map[s] = {
                                                    'graph': g,
                                                    'ctrl_input': ctrl_input,
                                                    'seq_len_dev': seq_len_dev,
                                                    'pos_dev': pos_dev,
                                                    'logits': logits_buf,
                                                    'seq_len': s
                                                }
                                                if s % 50 == 0:
                                                    print(f"[DevGraph Custom Capture] ✓ seq_len={s}", flush=True)
                                        except Exception:
                                            pass
                                    time.sleep(0.002)
                                print("[DevGraph Capture Thread] Stopped", flush=True)
                            generator.dev_bg_thread = threading.Thread(target=_bg_custom, daemon=True)
                            generator.dev_bg_thread.start()
                    response_queue.put({"status": "success"})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            
            elif request.get("cmd") == "devgraph_replay_seq_custom":
                # Replay using custom exact-seq graphs; kernel updates arena seq_len and logits
                session = request.get("session", "default")
                steps = int(request.get("steps", 1))
                last_token = request.get("start_token", None)
                temperature = float(request.get("temperature", 1.0))
                do_sample = bool(request.get("do_sample", False))
                top_p = float(request.get("top_p", 1.0))
                seed = request.get("seed", None)
                try:
                    if custom_decode_step is None:
                        raise RuntimeError("custom_decode_step extension not available")
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    if last_token is None:
                        raise RuntimeError("start_token must be provided for devgraph_replay_seq_custom")
                    generated = []
                    start = time.time()
                    # Pause background capture during replay to avoid allocator/capture conflicts
                    try:
                        generator.dev_bg_paused = True
                    except Exception:
                        pass
                    if seed is not None:
                        try:
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                        except Exception:
                            pass
                    for _ in range(steps):
                        seq_len = int(arena['seq_len'])
                        # Try to fetch existing exact-seq graph; if missing, capture it synchronously
                        state = generator.dev_seq_graphs.get(session, {}).get(seq_len, None)
                        if state is None:
                            try:
                                device = generator.device
                                vocab_size = int(getattr(generator.model.config, 'vocab_size', 32000))
                                ctrl_input = torch.empty((1, 1), dtype=torch.long, device=device)
                                seq_len_dev = torch.tensor([seq_len], dtype=torch.int32, device=device)
                                pos_dev = torch.tensor([seq_len], dtype=torch.int32, device=device)
                                logits_buf = torch.empty((1, vocab_size), dtype=torch.float32, device=device)
                                g = torch.cuda.CUDAGraph()
                                pool = torch.cuda.graphs.graph_pool_handle()
                                with torch.cuda.stream(generator.capture_stream):
                                    use_ext = (
                                        hasattr(generator, "custom_scratch") and bool(generator.custom_scratch) and
                                        getattr(generator, "W_gate_ptrs", None) is not None and
                                        len(generator.W_gate_ptrs) > 0 and int(generator.W_gate_ptrs[0]) != 0 and
                                        getattr(generator, "W_up_ptrs", None) is not None and
                                        len(generator.W_up_ptrs) > 0 and int(generator.W_up_ptrs[0]) != 0 and
                                        getattr(generator, "W_down_ptrs", None) is not None and
                                        len(generator.W_down_ptrs) > 0 and int(generator.W_down_ptrs[0]) != 0 and
                                        int(getattr(generator, "LM_HEAD_ptr", 0)) != 0 and
                                        int(getattr(generator, "hidden_size", 0)) > 0 and
                                        int(getattr(generator, "intermediate_size", 0)) > 0 and
                                        os.environ.get("CASE4_DISABLE_EXT", "0") != "1"
                                    )
                                    with torch.cuda.graph(g, stream=generator.capture_stream, pool=pool):
                                        if use_ext:
                                            H_hidden = int(generator.hidden_size)
                                            I = int(generator.intermediate_size)
                                            rms_eps = float(getattr(generator.model.config, "rms_norm_eps", 1e-6))
                                            layer0 = 0
                                            custom_decode_step.capture_decode_ext(
                                                ctrl_input,
                                                generator.kv_arena[session]['k'],
                                                generator.kv_arena[session]['v'],
                                                seq_len_dev,
                                                pos_dev,
                                                logits_buf,
                                                generator.custom_scratch['x_norm'],
                                                generator.custom_scratch['gate'],
                                                generator.custom_scratch['up'],
                                                generator.custom_scratch['act'],
                                                generator.custom_scratch['mlp_out'],
                                                int(generator.RMS_in_ptrs[layer0]) if generator.RMS_in_ptrs else 0,
                                                int(generator.RMS_post_ptrs[layer0]) if generator.RMS_post_ptrs else 0,
                                                int(generator.RMS_final_ptr) if getattr(generator, "RMS_final_ptr", 0) else 0,
                                                int(generator.W_gate_ptrs[layer0]) if generator.W_gate_ptrs else 0,
                                                int(generator.W_up_ptrs[layer0]) if generator.W_up_ptrs else 0,
                                                int(generator.W_down_ptrs[layer0]) if generator.W_down_ptrs else 0,
                                                int(generator.LM_HEAD_ptr) if getattr(generator, "LM_HEAD_ptr", 0) else 0,
                                                int(generator.num_layers_cfg), int(H_hidden), int(generator.kv_arena[session]['k'].size(3)), int(generator.kv_arena[session]['k'].size(4)), int(vocab_size),
                                                int(I),
                                                float(rms_eps)
                                            )
                                        else:
                                            custom_decode_step.capture_decode(
                                                ctrl_input,
                                                generator.kv_arena[session]['k'],
                                                generator.kv_arena[session]['v'],
                                                seq_len_dev,
                                                pos_dev,
                                                logits_buf
                                            )
                                sess_map = generator.dev_seq_graphs.setdefault(session, {})
                                sess_map[seq_len] = {
                                    'graph': g,
                                    'ctrl_input': ctrl_input,
                                    'seq_len_dev': seq_len_dev,
                                    'pos_dev': pos_dev,
                                    'logits': logits_buf,
                                    'seq_len': seq_len
                                }
                                state = sess_map[seq_len]
                            except Exception as e:
                                raise RuntimeError(f"failed to capture graph for seq_len={seq_len}: {e}")
                        g = state['graph']
                        ctrl = state['ctrl_input']
                        logits_buf = state['logits']
                        seq_len_dev = state['seq_len_dev']
                        ctrl.fill_(int(last_token))
                        with torch.cuda.stream(generator.replay_stream):
                            g.replay()
                        # Sample (move to CPU to avoid allocations on replay_stream outside capture)
                        logits_cpu = logits_buf[0].to('cpu', dtype=torch.float32)
                        if do_sample and top_p < 1.0:
                            probs = torch.softmax(logits_cpu / max(1e-6, temperature), dim=-1)
                            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                            cdf = torch.cumsum(sorted_probs, dim=-1)
                            cutoff = torch.searchsorted(cdf, torch.tensor(top_p)).item()
                            cutoff = max(1, cutoff)
                            top_idx = sorted_idx[:cutoff]
                            top_probs = sorted_probs[:cutoff]
                            top_probs = top_probs / top_probs.sum()
                            next_token = int(top_idx[torch.multinomial(top_probs, 1)].item())
                        else:
                            next_token = int(torch.argmax(logits_cpu).item())
                        generated.append(next_token)
                        last_token = next_token
                        # Mirror device seq_len to host arena
                        try:
                            arena['seq_len'] = int(seq_len_dev.item())
                        except Exception:
                            arena['seq_len'] = seq_len + 1
                    torch.cuda.synchronize(device=generator.device)
                    elapsed_ms = (time.time() - start) * 1000.0
                    response_queue.put({"status": "success", "tokens": generated, "elapsed_ms": elapsed_ms, "session": session})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
                finally:
                    try:
                        generator.dev_bg_paused = False
                    except Exception:
                        pass
            
            elif request.get("cmd") == "devgraph_wait_precapture_custom":
                # Block until at least min_ahead graphs are captured beyond current seq_len
                session = request.get("session", "default")
                min_ahead = int(request.get("min_ahead", 150))
                timeout_s = float(request.get("timeout_s", 120.0))
                try:
                    arena = generator.kv_arena.get(session, None)
                    if arena is None:
                        raise RuntimeError("KV session not initialized")
                    start_seq = int(arena['seq_len'])
                    max_len = int(arena['max_len'])
                    target = min(start_seq + min_ahead, max_len)
                    start_t = time.time()
                    while True:
                        sess_map = generator.dev_seq_graphs.get(session, {})
                        ready = (target in sess_map)
                        if ready:
                            break
                        if (time.time() - start_t) > timeout_s:
                            raise TimeoutError(f"precapture wait timed out for seq_len={target}")
                        time.sleep(0.01)
                    response_queue.put({"status": "success", "start_seq": start_seq, "target_seq": target})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
            
            elif request.get("cmd") == "devgraph_bg_capture_stop":
                # Stop background custom exact-seq capture thread
                session = request.get("session", "default")
                try:
                    with generator.dev_bg_lock:
                        generator.dev_bg_stop = True
                    t = getattr(generator, "dev_bg_thread", None)
                    if t is not None:
                        t.join(timeout=1.0)
                    response_queue.put({"status": "success", "stopped": True})
                except Exception as e:
                    response_queue.put({"status": "error", "error": str(e)})
    
    except Exception as e:
        print(f"[Generator] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        response_queue.put({"status": "error", "error": str(e), "process": "graph_generator"})


def main():
    parser = argparse.ArgumentParser(description="Case4 CUDA Graph Generator Server")
    parser.add_argument("--model-name", type=str, required=True, help="Model name/path")
    
    args = parser.parse_args()
    
    # Create IPC queues
    request_queue = mp.Queue()
    response_queue = mp.Queue()
    
    # Start generator process
    gen_process = mp.Process(
        target=graph_generator_process,
        args=(request_queue, response_queue, args.model_name)
    )
    gen_process.start()
    
    print("[Main] Graph generator process started")
    print("[Main] Waiting for ready signal...")
    
    # Wait for ready
    ready_msg = response_queue.get()
    print(f"[Main] Received: {ready_msg}")
    
    # Simple test
    print("\n[Main] Testing graph generator...")
    
    # Test: Generate (dummy input)
    dummy_input = torch.randint(0, 1000, (1, 10), dtype=torch.long)
    request_queue.put({
        "cmd": "generate",
        "seq_len": 10,
        "input_ids": dummy_input
    })
    
    response = response_queue.get()
    if response["status"] == "success":
        print(f"[Test] Generate successful: logits shape = {response['logits'].shape}")
    else:
        print(f"[Test] Generate failed: {response['error']}")
        return 1
    
    # Status check
    request_queue.put({"cmd": "status"})
    response = response_queue.get()
    if response["status"] == "success":
        print(f"[Test] Status: {response['num_graphs']} graphs captured")
        print(f"[Test] Range: {response['min_seq']} - {response['max_seq']}, max_captured: {response['max_captured']}")
    
    # Stop
    print("\n[Main] Stopping generator...")
    request_queue.put({"cmd": "stop"})
    gen_process.join()
    
    print("[Main] ✅ Test passed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
