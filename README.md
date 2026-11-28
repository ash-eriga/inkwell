# Inkwell

**Inkwell** is a standalone Rust + CUDA inference engine for **Inkling-2** MoE models in **EXL3 (Trellis + Hadamard)** and GGUF formats, engineered from scratch to run fast on consumer GPUs without Tensor Cores (including NVIDIA Pascal / GTX 1060 6 GB).

## Highlights (v0.1.1)

- **Custom EXL3 CUDA kernels** (`crates/inkwell-cuda`):
  - Exact-track dense and 8-wide MoE GEMV kernels with fail-closed Pascal SASS NOP-stripping (`ch2nopless`), C++/PTX `ch2_mul16_lo`, batched dense attention projections (`gemv_batchf`), vectorized Hadamard (`HADVEC`), and bit-exact micro-kernel launch fusion (`decode_fuse.cuh`).
  - Vectorized chunked prefill driver with centered INT8 `dp4a` dense and grouped MoE GEMMs (`gpu_caps.cuh` auto-detects native `dp4a` on `sm_61` and `sm_70+`, falling back cleanly to bit-exact FP32 on non-`dp4a` GPUs; `INKWELL_PF_INT8=0` forces FP32).
  - Quantized KV cache (`K8V4` + `INT8`/`FP8`/`K8V6`/`K4V2`) with single-pass GQA3 decode attention (`attn_kvq_gqa3.cuh`) and 16-row paired head-major `K8V4` layout across decode and flash prefill. Auto-defaults keep `F16` up to `8,192` tokens and switch to `K8V4` + `dp4a` above `8,192` tokens.
  - Jointly placed resident MoE layers + DP-tuned size-aware per-layer expert VRAM cache (`kslot`, up to `1,264 MiB` ladder) with overlapped split side-stream PCIe staging.
  - Multi-arch CUDA fatbin (`sm_61` primary/tuned; `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` + `compute_90` PTX experimental).
- **Fast Rust host pipeline**: AVX2-accelerated sampler (up to 8.3× faster at 131K vocab), parallelized BPE tokenizer, and overlapped direct `mmap → pinned` model startup (~21.1 s warm).
- **OpenAI-compatible HTTP/SSE server** (`inkwell serve`): `/health`, `/v1/models`, `/v1/chat/completions` (streaming & non-streaming, `reasoning_effort`, `reasoning_content`), and `/v1/completions`.
- **Interactive terminal chat & one-shot CLI** (`inkwell`, `inkwell chat`, `inkwell run`) with live UTF-8 streaming, incremental KV prefix reuse, and thinking-channel formatting.

## Measured Performance (GTX 1060 6 GB · Pascal `sm_61`, Inkling-2-Mini EXL3 2.26 bpw)

All Inkwell v0.1.1 numbers measured with default settings (zero KV/DP4A/INT8 flags):

| Context | Prefill (`chunk=512`) | Decode | Peak VRAM | Notes |
|---|---:|---:|---:|---|
| Short (`base4`) | — | **40.5 tok/s** (24.7 ms/tok) | 5,024 MiB | Auto-F16 KV, bit-exact `md5 5cbe8e06` |
| 512 tok | **250.5 tok/s** | — | 4,848 MiB (post-cache) | Auto-INT8 `dp4a` (FP32 fallback: 174.8 tok/s) |
| 4,096 tok | **232.8 tok/s** | — | 4,952 MiB (post-cache) | Auto-F16 KV + auto-INT8 `dp4a` |
| 65,536 tok (64K) | — | **25.8 tok/s** (38.7 ms/tok) | 5,156 MiB | Auto-`K8V4` paired head-major + `dp4a` |

### vs. Inkwell v0.1 and Upstream `llama.cpp` (`e087c6c`, GTX 1060 6 GB)

Comparing **Inkwell v0.1.1** (`EXL3 2.26 bpw`, default settings) against **Inkwell v0.1** (`2adcf77`) and upstream **`llama.cpp`** (`Inkling-2-Mini-UD-IQ2_XXS.gguf`, `2.06 bpw`, tuned `-ngl 99 -ot ".ffn_.*_exps.=CPU" -fa 1 -b 512 -ub 512`; default `-b 2048` and `-ncmoe` OOM on 6 GB VRAM):

| Metric | Inkwell v0.1.1 (`2.26 bpw`) | Inkwell v0.1 (`2.26 bpw`) | `llama.cpp` (`2.06 bpw`) | v0.1.1 vs `llama.cpp` |
|---|---:|---:|---:|---:|
| Short decode | **40.5 tok/s** | 36.9 tok/s | 7.5 tok/s | **5.4×** |
| Decode @ 64K ctx | **25.8 tok/s** | OOM *(max 16K: 19.1)* | OOM | fits in 5,156 MiB |
| Prefill 512 (`pp512`) | **250.5 tok/s** | 116.5 tok/s | 108.1 tok/s | **2.3×** |
| Prefill 4,096 (`pp4096`) | **232.8 tok/s** | 119.5 tok/s | 164.3 tok/s | **1.42×** |

*(Validated up to 64K context on 6 GB VRAM in v0.1.1 with automatic `K8V4` quantization above 8,192 tokens.)*

## Quick Install (Linux x86_64 · CUDA 12)

```bash
curl -fsSL https://raw.githubusercontent.com/ash-eriga/inkwell/master/install.sh | sh
```

Requires an NVIDIA driver + CUDA 12 runtime (`libcudart.so.12`, `libcublas.so.12`). Updates and rollbacks are built in:

```bash
inkwell update              # download latest release, verify SHA-256, swap atomically
inkwell update --check      # check for updates without installing
inkwell update --rollback   # switch back to the previous installed version
```

## Build from Source

**Requirements:** Linux x86_64, NVIDIA GPU (`sm_61` Pascal tuned; `sm_75+` Turing/Ampere/Ada/Hopper experimental), CUDA Toolkit 12.x, Rust toolchain (`cargo`), and `cmake`.

1. Build the patched `ggml` dependency (see [third_party/ggml/README.md](third_party/ggml/README.md) for full details):

```bash
git clone https://github.com/ggml-org/llama.cpp.git third_party/ggml/ggml-src
cd third_party/ggml/ggml-src
git checkout cbb1c1c50bba81929387dea473873d1df87a6212
git am ../patches/000[1-8]-*.patch
git apply ../patches/0009-uncommitted-ggml-cuda.patch
cmake -B build-inkwell-dp4a -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="61;75;80;86;89;90" -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON
cmake --build build-inkwell-dp4a -j --target ggml ggml-base ggml-cuda ggml-cpu
cd ../../..
```

2. Build Inkwell:

```bash
cargo build --release
# Optional: compile only for your GPU arch for faster builds (e.g. sm_61 or sm_86)
# INKWELL_CUDA_ARCHS=61 cargo build --release
```

The compiled binary is at `target/release/inkwell`. Add `third_party/ggml/ggml-src/build-inkwell-dp4a/bin` (and CUDA `lib64` if non-standard) to `LD_LIBRARY_PATH`, or install a small wrapper script in `~/.local/bin/inkwell`.

## Quick Start

### Interactive Chat & Model Picker

```bash
inkwell chat                    # opens EXL3 local + Hugging Face preset picker
inkwell chat ~/models/inkling --ctx 4096 --effort none
```

### One-Shot Generation

```bash
inkwell run ~/models/inkling -p "Explain trellis quantization in 3 bullet points." --max-tokens 256
```

### OpenAI-Compatible Server

```bash
inkwell serve                   # opens EXL3 model picker & auto-downloader
inkwell serve ~/models/inkling --host 127.0.0.1 --port 8080 --ctx 4096
```

Test with `curl`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 128
  }'
```

## Workspace Layout

- `inkwell-cli/` — CLI (`serve`, `chat`, `run`, benchmark harness)
- `crates/inkwell-cuda/` — Custom CUDA kernels (`kernels.cu`, `prefill.cu`) and FFI bindings
- `crates/inkwell-run/` — Model loader, graph execution, MoE cache/pager, chunked prefill driver
- `crates/inkwell-tokenizer/` — Byte-level GPT-4o BPE tokenizer with special-token support
- `crates/inkwell-sampler/` — Penalty & top-k/top-p/min-p/temperature sampler
- `crates/inkwell-{core,gguf,st,model,kv,pager,graph,spec,ggml}/` — Core runtime crates
- `docs/` — Architecture and porting notes
- `opt/` — Kernel microbenchmarks, optimization ledger, and measurement scripts

## License

MIT — see [LICENSE](LICENSE).
