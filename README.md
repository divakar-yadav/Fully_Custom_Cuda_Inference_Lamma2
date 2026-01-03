# Fully_Custom_Cuda_Inference_Lamma2

This repository contains a fully custom CUDA inference implementation for LLaMA2 using custom kernels with minimal Python dependencies. The inference computation is 100% custom (no HuggingFace model forward pass), using direct CUDA kernels for all operations.

## Case4: Custom CUDA Decode + Concurrent CUDA Graph Capture/Replay

### Overview
This module implements a graph‑safe custom decode pipeline for LLaMA‑style models and wires it into a concurrent CUDA Graph capture/replay server. The goal is exact‑seq CUDA graphs captured on a dedicated stream while replaying previously captured graphs on another stream, with no device‑wide syncs and no PyTorch allocations inside the captured regions.

Two modes are provided:
- Standalone (no IPC): direct Python driver using the custom CUDA extensions to validate numerical correctness and generate text.
- IPC (concurrent capture/replay): a server process that captures exact‑seq graphs in the background and replays them on demand via simple RPCs.

### Custom decode (what runs inside/alongside capture)
Per token and per layer:
1) RMSNorm (input) → x_norm
2) GEMMs: Q = Wq·x_norm, K = Wk·x_norm, V = Wv·x_norm
3) RoPE: apply LLaMA rotary embeddings to Q/K (head‑wise, first rotary_dim)
4) Cache K/V at time T (GQA: store kv_heads; broadcast to query heads on use)
5) Varlen attention: ctx = softmax(QK^T/√D)·V (no allocations; stream‑local)
6) O‑proj: y_attn = Wo·ctx + residual x
7) RMSNorm (post‑attn) → MLP gate/up (GEMMs) → SwiGLU → down (GEMM) + residual
8) Final RMSNorm (optional) → logits = lm_head·y
9) seq_len_dev += 1 (device‑side), host samples next token on CPU

All weights/scratch buffers are persistent device allocations (cudaMalloc); GEMMs use cuBLAS with stream‑bound handles; graph capture uses a dedicated pool.

### Repository layout (key files)
- custom_kernels/
  - capture_decode_step.cpp, capture_decode_step_kernel.cu
    - cuBLAS GEMMs (1xH·HxI, 1xI·IxH, logits GEMM), pybind API (`capture_decode`, `capture_decode_ext`), fused RMSNorm, SwiGLU, seq_len++.
  - attn_varlen.cpp, attn_varlen_kernel.cu
    - Varlen attention (Q·K^T → softmax → P·V) with stream‑friendly reductions.
  - qkv_gemm.cpp (+ qkv_gemm_kernel.cu)
    - Utilities for Q/K/V projections (stream‑bound handle).
  - d2d_row_copy.cpp
    - Device‑to‑device copy of one embedding row (tied `lm_head`).
  - weight_loader.py
    - Loads safetensors, cudaMalloc + H2D; exposes device pointers and shapes:
      - q_ptrs/k_ptrs/v_ptrs, o_ptrs, mlp_gate/up/down_ptrs
      - rms_in/post/final_ptrs, emb_ptr
      - hidden/intermediate/vocab sizes, num_layers
- case4_graph_generator_server_ipc.py
  - IPC server: capture/replay streams, per‑seq exact‑shape graph cache, background capture thread, CPU sampling. RPCs:
    - `kv_init`, `devgraph_bg_capture_start_custom`, `devgraph_wait_precapture_custom`, `devgraph_bg_capture_stop`, `devgraph_replay_seq_custom`.
- scripts/
  - direct_custom_decode_500.py
    - Standalone custom decode driver (no IPC). Generates 500 tokens and saves tokens/text.
  - debug_parity_one_step.py
    - T=1 parity vs HF (Q/K/V, ctx, O‑proj+residual, MLP, final y).
  - debug_parity_seq8.py
    - T=8 parity vs HF (build our KV cache vs HF; validates RoPE/GQA/head mapping).
  - run_custom_50.py
    - IPC flow: start server, pre‑capture, replay 50 tokens; saves output.

### Build (extensions)
From this directory:
```
source ../../venv/bin/activate
cd custom_kernels
python setup.py build_ext --inplace -q
```
Artifacts (*.so) land in `custom_kernels/`.

### Standalone: quick tests
1) One‑step parity (sanity on math/layout):
```
python -u scripts/debug_parity_one_step.py
```
Expect small L2/max diffs (fp16 noise).

2) Sequence parity (T=8; checks K/V across steps):
```
python -u scripts/debug_parity_seq8.py
```
Use this to verify RoPE/head mapping/GQA correctness.

3) Generate 500 tokens (custom path only):
```
python -u scripts/direct_custom_decode_500.py
```
Outputs:
- `output/direct_custom_decode_500_tokens.txt`
- `output/direct_custom_decode_500_text.txt`

### IPC: concurrent capture + replay
Run a 50‑token exact‑seq flow with pre‑capture gate:
```
export CASE4_USE_CUSTOM=1
export CASE4_KV_ONLY=1
export CASE4_SKIP_PRECAPTURE=1
export CASE4_BG_CAPTURE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u scripts/run_custom_50.py
```
What happens:
- `kv_init` allocates K/V arena and scalars.
- Background capture (custom path) builds exact‑seq graphs on `capture_stream` using a graph pool; client can block on `devgraph_wait_precapture_custom`.
- Replay uses `devgraph_replay_seq_custom` on `replay_stream`. Sampling is CPU‑side to avoid replay‑time allocations.

To avoid allocator contention during replay, you can stop background capture via `devgraph_bg_capture_stop` just before replay.

### Environment flags
- `CASE4_USE_CUSTOM=1` use custom kernels for decode.
- `CASE4_DISABLE_EXT=1` force base path inside capture (for A/B).
- `CASE4_KV_ONLY=1` minimal server state; no JIT clients.
- `CASE4_SKIP_PRECAPTURE=1` skip non‑custom pre‑capture.
- `CASE4_BG_CAPTURE=1` enable background capture thread.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduce fragmentation.

### Notes on correctness
- RoPE: apply rotary to Q/K (first `rotary_dim`) with `rope_theta` from config; ensure head reshape/order matches HF.
- GQA: store K/V with `num_key_value_heads` and broadcast KV to query heads during attention.
- GEMM orientation: HF weights are [out, in] row‑major; use transposed GEMM in cuBLAS (OP_T) with correct lda/ldb/ldc.
- Sampling: kept on CPU to avoid replay‑time CUDA allocations.

### Known pitfalls
- Any allocation inside capture (“offset increment outside graph capture”) will invalidate the graph. Keep all buffers persistent.
- Avoid device‑wide synchronizes; use stream‑local fences and external streams only.
- If parity diverges, first validate T=1 parity, then T=8 (K/V) to isolate RoPE/GQA issues.

### Profiling
Use Nsight Systems to confirm no device‑wide syncs and overlapping capture/replay:
```
nsys profile -o output/nsys_varlen_smoke50 --force-overwrite=true \
  python -u scripts/run_custom_50.py
```

### Outputs
- IPC runs: see `output/case4_generate_custom_50.log`, token/text files.
- Standalone runs: see `output/direct_custom_decode_500_*`.

### Status
- Graph‑safe decode path implemented (RMSNorm, Q/K/V GEMMs, RoPE, varlen attention, O‑proj, MLP, logits).
- Concurrent capture/replay wired with graph pool and disjoint streams.
- Parity tooling included; use it when tuning RoPE/GQA/head mapping. 
# Case 4 – CUDA Graphs + IPC (with KV-only fast path)

## Overview
Case4 runs inference via two cooperating processes connected with Python multiprocessing queues:
- Graph Generator Server (`case4_graph_generator_server_ipc.py`): hosts the HF model on GPU and executes the decode step.
- JIT Client (`case4_jit_client_ipc.py`): provides lightweight tokenizer and (optional) sampling utilities.

Two execution strategies are available:
- CUDA Graph mode (legacy): capture many sequence-length graphs and replay per token.
- KV-only fast path (current default): no graph capture; use HF model with past_key_values for single-token decode.

## Processes and responsibilities
- Graph Generator Server
  - Loads the model (FP16 by default, optional INT8 via bitsandbytes).
  - In CUDA Graph mode: initializes C++ stream manager, captures/replays graphs (replay/capture coordination).
  - In KV-only mode: skips capture/streams, maintains a per-session KV cache and runs one-token decode on GPU.
- JIT Client
  - Tokenizes the initial prompt.
  - (Legacy path) Can sample next token on CPU; in KV-only fast path, sampling is fused on GPU.

## IPC API (queue messages)
From the client to the generator:
- `tokenize {text, add_special_tokens}` (to JIT client): returns `input_ids`.
- CUDA Graph mode:
  - `generate {seq_len, input_ids}` → returns `logits` for last position.
  - `prep_next {prev_seq_len, next_seq_len}` → capture/delete graphs around current length.
- KV-only fast path:
  - `kv_init {session, input_ids}` → runs prefill once, returns `logits` and stores `past_key_values`.
  - `kv_step {session, last_token}` → single-token decode step using stored KV; returns `logits`.
  - `kv_generate {session, steps, do_sample, start_token}` → fuses N decode steps on GPU with on-device sampling; returns tokens and elapsed time.
- Service:
  - `status` → graph capture status (graph mode only).
  - `stop` → graceful shutdown.

## Modes and environment flags
- `CASE4_KV_ONLY=1` – enable KV-only fast path (no C++ streams/graphs).
- `CASE4_SKIP_PRECAPTURE=1` – skip CUDA graph pre-capture (used in KV-only mode).
- `CASE4_ATTN=sdpa|flash2|xformers` – preferred attention backend when loading the model:
  - `sdpa` uses PyTorch fused Flash SDPA where available (no extra install).
  - `flash2` requires `flash-attn` package; otherwise raises an error (no silent fallback).
  - `xformers` requires `xformers` installed.
- `CASE4_QUANT=int8|none` – INT8 via bitsandbytes (requires `bitsandbytes`).
- `CASE4_FORCE_FLASH=1` – opt-in monkeypatch to direct single-token SDPA to a custom op (see below). Prefill still uses the original SDPA.

## Custom flash-attention scaffold (optional)
Location: `custom_kernels/`
- `flash_attn_ext/` – minimal C++/CUDA extension exposing `sdp_single_q(q,k,v,scale)` for single-query decode; compiled just-in-time.
- `flash_attn.py` – Python loader and conditional monkeypatch wrapper that:
  - For prefill or masked/multi-token cases: defers to the original `scaled_dot_product_attention`.
  - For single-token decode: calls the custom CUDA op.
Notes:
- The shipped kernel is a naive, correctness-first implementation to demonstrate integration; it is not tuned like FlashAttention v2.
- Keep `CASE4_FORCE_FLASH=0` for best latency unless you intend to iterate on the kernel.

## Data flow (KV-only fast path)
1) JIT client tokenizes the input → `input_ids`.
2) Generator prefill: `kv_init(session, input_ids)` builds KV once (excluded from timing).
3) Generator decode: `kv_generate(session, steps, do_sample/start_token)` runs N steps on GPU with on-device sampling and returns tokens and elapsed ms.
4) Caller decodes tokens to text and records metrics.

## Entry points
- Generate and time: `case4_ipc_20250130/generate_500_tokens.py`
  - Example (400 tokens, Flash SDPA, KV-only):
    ```bash
    CASE4_QUANT=none CASE4_ATTN=sdpa CASE4_KV_ONLY=1 CASE4_SKIP_PRECAPTURE=1 \
    /home/azureuser/divakar_projects/cuda_graph_sharing/.venv/bin/python \
      case4_ipc_20250130/generate_500_tokens.py \
      --input-text "The future of artificial intelligence" --max-new-tokens 400
    ```
  - Outputs:
    - CSV: `output/case4_generate_500_tokens.csv`
    - Text: `output/case4_generate_500_tokens_output.txt`

## Performance notes
- KV-only fast path eliminates replay/capture syncs; timing measures decode loop only.
- SDPA Flash generally provides performant fused attention kernels without extra packages.
- Optional INT8 quantization reduces memory bandwidth use during decode; install `bitsandbytes` first.
- The custom CUDA op is a scaffold; for peak performance, use SDPA flash or FlashAttention v2.

## Key files
- Server: `case4_graph_generator_server_ipc.py`
- Client: `case4_jit_client_ipc.py`
- Runner: `generate_500_tokens.py`
- Custom kernels: `custom_kernels/`

# Case4 IPC Implementation - January 30, 2025

This folder contains the IPC-based implementation of Case4 that separates JIT preprocessing and CUDA graph operations into two separate processes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Process                      │
│                  (case4_combined_ipc_orchestrator.py)       │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────────┐       ┌──────────────────────┐
│   JIT Client Process │       │ Graph Generator      │
│ (case4_jit_client)   │◄─────►│ Server Process       │
│                      │  IPC  │ (graph_generator)    │
│ • Tokenization       │       │                      │
│ • Preprocessing      │       │ • CUDA Graph Capture │
│ • Sampling           │       │ • Graph Replay       │
└──────────────────────┘       └──────────────────────┘
```

## Files

- **`case4_jit_client_ipc.py`**: JIT preprocessing client process
- **`case4_graph_generator_server_ipc.py`**: CUDA graph generator server process
- **`case4_combined_ipc_orchestrator.py`**: Main orchestrator coordinating both processes
- **`run_case4_combined_ipc.sh`**: Bash script to run the full system

## Usage

```bash
# Run the combined IPC system
bash case4_ipc_20250130/run_case4_combined_ipc.sh meta-llama/Llama-2-7b-hf

# Or run the orchestrator directly
cd case4_ipc_20250130
python3 case4_combined_ipc_orchestrator.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --prompt "Hello" \
    --max-tokens 20
```

## Key Features

1. **Separate Processes**: JIT operations and CUDA graphs run in isolated processes
2. **IPC Communication**: Fast queue-based communication between processes
3. **JIT Preprocessing**: Tokenization, dynamic shape preprocessing, sampling
4. **CUDA Graphs**: Graph capture/replay for fast inference
5. **Background Capture**: Continuous graph capture to stay ahead of inference

## Benefits

- Process isolation prevents one component from affecting the other
- Can leverage separate GPU contexts if needed
- Easier debugging and profiling of individual components
- Scales to more complex distributed setups

## Dependencies

- Python 3.x
- PyTorch with CUDA
- transformers (only for tokenizer and config reading)
- concurrent_capture_replay_simple (C++ CUDA extension)
