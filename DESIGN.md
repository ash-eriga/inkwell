# Inkwell

Rust-first inference engine for Inkling-2-Mini on a 16 GB RAM / 6 GB VRAM (GTX 1060, sm_61) rig.
Target: 50 tok/s decode.

Style: follows BeautifulRust skills (rust-best-practices / rust-port / rust-review). Lints in `lints/` (from the skill) applied to the workspace. Builds cached by `kache`.

## Layers (llama.cpp -> Rust mapping)

| Piece | Source | Inkwell home | Status |
|---|---|---|---|
| GGUF parse | gguf.cpp | `inkwell-gguf` (mmap, no copies) | done: 744 tensors, 8081 MiB |
| Arch config | inkling2.cpp | `inkwell-core::InklingConfig` | done |
| Tokenizer | llama-vocab (BPE) | `inkwell-tokenizer` | TODO |
| Host buffer mgmt | llama-memory | `inkwell-core` | TODO |
| Expert RAM<->VRAM pager | llama-moe-cache (dp4a-pascal) | `inkwell-pager` | TODO — port |
| Spec-decode loop | ingine / mtp.safetensors | `inkwell-core` | TODO |
| Graph build | llama-graph | `inkwell-graph` | TODO |
| Kernels (sm_61) | ggml-cuda *.cu | CUDA C++ via build.rs FFI | TODO — port only what we need |

## Kernels to port (minimum)

1. `mmvq` multi-column GEMV for IQ2_XXS/Q4_K (dp4a path is broken on Pascal for these — fall back to scalar, it's memory-bound anyway)
2. `fattn-tile` with `kv_rows` (already ours — batch verify + chunked mask, kills the O(n_ctx) materialized bias from upstream)
3. MoE expert GEMV over host-pinned pointers (`strata_devptr` zero-copy)
4. rmsnorm / rope / swiglu elementwise ops (trivial)

## Performance walls (measured baseline, llama.cpp)

- decode 5.8 t/s = CPU-experts bound (4 threads) -> zero-copy GEMV + expert cache
- OOM >4k ctx = materialized rel-pos bias -> kv_rows FA + chunked mask (PR #32752 partial, #32740 root)
- encode 18-30 t/s @32k
- Target path: ~12 t/s PCIe-bound plain -> spec decode (DSpark mtp.safetensors) + draft-window prefetch -> 50 t/s

## Rules

- Host orchestration 100% Rust: scheduler, pager, spec loop, server.
- Kernels CUDA C++ compiled by nvcc (sm_61), called via FFI. No cudarc/cust.
- Golden tests: bit-lock hashes of kernel outputs vs llama.cpp reference.
- Benchmarks: `inkwell bench` vs `llama-bench` same flags; encode/decode @4k and @32k.

## Quant census (UD-IQ2_XXS, measured 2026-11-21)

| ggml type | n | MiB | likely layer use |
|---|---|---|---|
| IQ2_XXS (16) | 7 | 1169 | biggest experts |
| IQ2_XS (17) | 6 | 1123 | experts |
| IQ3_XXS (19) | 6 | 759 | experts |
| Q2_0 (42) | 17 | 1549 | experts (new llama.cpp type) |
| NVFP4 (40) | 7 | 1275 | experts (moe.fused) |
| MXFP4 (39?) | — | — | (none in this file) |
| IQ4_XS (23) | 2 | 469 | embed/head? |
| IQ3_S (22) | 5 | 1037 | dense rows |
| Q4_K/Q5_K/Q6_K | 56 | 178 | attn/norm misc |
| BF16 (30) | 32 | 3 | gates/biases |
| IQ3_XXS(18)/other | 3 | 7 | misc |

Kernel port list = dequant+dot for the ~10 formats above.
