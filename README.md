# Inkwell

**Inkwell** is a standalone Rust + CUDA inference engine for **Inkling-2** MoE models in **EXL3 (Trellis + Hadamard)** and GGUF formats, engineered from scratch to run fast on consumer GPUs without Tensor Cores (including NVIDIA Pascal / GTX 1060 6 GB).

## Highlights (v0.1)

- **Custom EXL3 CUDA kernels** (`crates/inkwell-cuda`):
  - Exact-track dense and 8-wide MoE GEMV kernels with fail-closed Pascal SASS NOP-stripping (`ch2nopless`, +18% decode) and vectorized Hadamard (`HADVEC`).
  - Chunked prefill driver with grouped MoE GEMM, adaptive decode expert-cache preservation, and BF16 relative-bias expansion.
  - Size-aware per-layer expert VRAM cache (`kslot`) with asynchronous PCIe staging for overflow layers.
  - Multi-arch CUDA fatbin (`sm_61` primary/tuned; `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` + `compute_90` PTX experimental).
- **Fast Rust host pipeline**: AVX2-accelerated sampler (up to 8.3× faster at 131K vocab), parallelized BPE tokenizer, and overlapped direct `mmap → pinned` model startup (~21.1 s warm).
- **OpenAI-compatible HTTP/SSE server** (`inkwell serve`): `/health`, `/v1/models`, `/v1/chat/completions` (streaming & non-streaming, `reasoning_effort`, `reasoning_content`), and `/v1/completions`.
- **Interactive terminal chat & one-shot CLI** (`inkwell`, `inkwell chat`, `inkwell run`) with live UTF-8 streaming, incremental KV prefix reuse, and thinking-channel formatting.

## Measured Performance (GTX 1060 6 GB · Pascal `sm_61`, Inkling-2-Mini EXL3 2.26 bpw)

| Context | Prefill (chunk=512) | Decode (64 tok) | Peak VRAM |
|---|---:|---:|---:|
| Short (`base4`) | — | **36.9 tok/s** (27.1 ms/tok) | 5,048 MiB |
| 512 tok | **116.5 tok/s** | **29.8 tok/s** (33.6 ms/tok) | 5,094 MiB |
| 4,096 tok | **119.5 tok/s** | **26.8 tok/s** (37.3 ms/tok) | 5,206 MiB |
| 16,384 tok | **97.4 tok/s** | **19.1 tok/s** (52.4 ms/tok) | 5,314 MiB |

### vs. Upstream `llama.cpp` (`e087c6c`, GTX 1060 6 GB)

Comparing **Inkwell v0.1** (`EXL3 2.26 bpw`, zero-config) against upstream **`llama.cpp`** (`Inkling-2-Mini-UD-IQ2_XXS.gguf`, `2.06 bpw`, tuned `-ngl 99 -ot ".ffn_.*_exps.=CPU" -fa 1 -b 512 -ub 512`; default `-b 2048` and `-ncmoe` OOM on 6 GB VRAM):

| Metric | Inkwell v0.1 (`2.26 bpw`) | `llama.cpp` (`2.06 bpw`) | Speedup |
|---|---:|---:|---:|
| Short decode | **36.9 tok/s** | 7.5 tok/s | **4.9×** |
| Decode @ 512 ctx | **29.8 tok/s** | 8.3 tok/s | **3.6×** |
| Decode @ 4,096 ctx | **26.8 tok/s** | 7.7 tok/s | **3.5×** |
| Prefill 512 (`pp512`) | **116.5 tok/s** | 108.1 tok/s | **1.08×** |
| Prefill 4,096 (`pp4096`) | 119.5 tok/s | **164.3 tok/s** | 0.73× |

*(Validated up to 16k on 6 GB VRAM in v0.1.)*

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
