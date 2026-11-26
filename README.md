# Inkwell

**Inkwell** is a standalone Rust + CUDA inference engine for **Inkling-2** MoE models in **EXL3 (Trellis + Hadamard)** and GGUF formats, engineered from scratch to run fast on consumer GPUs without Tensor Cores (including NVIDIA Pascal / GTX 1060 6 GB).

## Highlights (v0.1)

- **Custom EXL3 CUDA kernels** (`crates/inkwell-cuda`):
  - Exact-track dense and 8-wide MoE GEMV kernels with fail-closed SASS NOP-stripping (`ch2nopless`) and vectorized Hadamard (`HADVEC`).
  - Chunked prefill driver with grouped MoE GEMM, adaptive decode expert-cache preservation, and BF16 relative-bias expansion.
  - Size-aware per-layer expert VRAM cache (`kslot`) with asynchronous PCIe staging for overflow layers.
- **OpenAI-compatible HTTP/SSE server** (`inkwell serve`): `/health`, `/v1/models`, `/v1/chat/completions` (streaming & non-streaming, `reasoning_effort`, `reasoning_content`), and `/v1/completions`.
- **Interactive terminal chat & one-shot CLI** (`inkwell chat`, `inkwell run`) with live UTF-8 streaming and thinking-channel formatting.

## Measured Performance (GTX 1060 6 GB, Inkling-2-Mini EXL3)

| Context | Prefill (chunk=512) | Decode (64 tok) | Peak VRAM |
|---|---:|---:|---:|
| 4 tok (short) | — | **36.6 tok/s** (27.3 ms/tok) | 5,048 MiB |
| 512 tok | **107.4 tok/s** (4.8 s) | **29.7 tok/s** (33.6 ms/tok) | 5,322 MiB |
| 4,096 tok | **104.9 tok/s** (39.0 s) | **25.8 tok/s** (38.7 ms/tok) | 5,194 MiB |
| 16,384 tok | **84.8 tok/s** (193.3 s) | **19.6 tok/s** (50.9 ms/tok) | 5,298 MiB |

*(Validated up to 16k on 6 GB VRAM in v0.1; 32k/64k compressed KV cache is slated for v0.1.5.)*

## Build

```bash
cargo build --release
```

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

Pass `--verbose` (or `INKWELL_VERBOSE=1`) to display internal VRAM, cache, and prefill diagnostics.

## Roadmap

- **v0.1.5**: Compressed KV cache (32k / 64k context on 6 GB VRAM), further kernel refinement toward sub-25 ms/tok short-context decode.
- **v0.2**: Self-contained **8 MB** runtime target (drop the external 129 MB `libggml-cuda.so` dependency), built-in Vercel `fx` integration for an embedded interactive chat experience alongside the local model, plus standalone EXL3 Pascal kernels on the Hugging Face Kernel Hub.

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
