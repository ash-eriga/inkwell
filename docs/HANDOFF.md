# Inkwell — Handoff para nova LLM (2026-11-22)

## O que é

Inkwell: engine de inferência em **Rust** para o `Inkling-2-Mini` (thinkingmachines, arch `inkling2_loop`, 26.4B total / 3.4B ativo, MoE 128 experts/8 usados, loop×3 = 48 visits/token) rodando em **16 GB RAM + 6 GB VRAM (GTX 1060, sm_61)**. Host todo em Rust (`crates/inkwell-run`, `inkwell-cuda`, `inkwell-ggml`, `inkwell-cli`, `inkwell-tokenizer`, `inkwell-pager`, `inkwell-spec`); kernels CUDA via shim `libggml-cuda` (fork `~/lfm-gtx1060-bench/src-v2`, build em `~/lfm-gtx1060-bench/build-inkwell-dp4a`) + kernels próprios em `crates/inkwell-cuda/src/kernels.cu`.

## Estado atual (commit `49893ae`)

- **Decode: ~8.3-10.7 t/s** sob apps desktop ligados (~95-120ms/tok; upstream llama.cpp recipe = 6.15 t/s, default = 5.71). Meta: **15 t/s** (66ms/tok).
- Texto dourado: `" Paris.\n\nA.\n\nB.\n\nC.\n\nD.\n\n正确答案：A\n\n答案解析..."` — chinês é o golden; variantes sinalizam corrupção.
- Run: `cd ~/inkwell && cargo build -p inkwell-cli --release && LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin GGML_PASCAL_GRAPHS=1 ./target/release/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run`
- Bench kernels: `cargo build --release --example bench_moe -p inkwell-cuda && LD_LIBRARY_PATH=... ./target/release/examples/bench_moe` (gates: `GOLD_ATTN=1`, `GOLD_Q80=1`).
- nsys: `~/lfm-gtx1060-bench/cuda-12.6/bin/nsys` (atenção: graphs só contam launches na captura — profile sem `GGML_PASCAL_GRAPHS` pra per-kernel).
- Ledger completo: `E2E_DEBUG.md` (rounds 1-6c — TODA a arqueologia de decisões).

## Já feito (não repetir)

- CUDA graphs reais: ctx persistente + n_kv padded a 64 + máscara -inf (launch overhead morto)
- Backend splits zerados (todos os tensores do grafo em device)
- Device-side MoE dispatch: resolve/alloc/gather kernels + repoint pra arena (zero split CPU)
- Expert cache VRAM LRU (~100% hit steady-state; misses = warmup only)
- dp4a i8 em TODOS os MoE GEMVs: q2_0 +43%, nvfp4 +17%, iq2_xs +30%, iq2_s +50%, iq2_xxs; iq1_s rejeitado (precisão)
- Atenção de decode fundida (`inkwell_attn_decode`): ~2.3ms/tok (era ~8-9)
- lm_head iq4_xs própria (217 GB/s — o kernel mais rápido)
- madvise(DONTNEED) nos experts pinados: ~4GB RAM liberados

## Armadilhas conhecidas (CRÍTICO)

1. `ggml_view_1d` offset é em **BYTES**, não elementos.
2. Sem loads u32 em offsets ímpares dentro de blocos quant (66B/176B/210B não são 4-aligned → pares u16).
3. Custom kernels num ggml custom op DEVEM rodar em `cuda_ctx->stream()` (export `ggml_backend_cuda_stream`); stream default → sticky illegal access.
4. **Reordenar acumulação float drifta ~0.4% de logit e flipa tokens** — golden text muda sem bug. Antes de qualquer reorder: gold test CPU-vs-kernel dedicado (padrão `GOLD_ATTN`/`GOLD_Q80`).
5. kernels.cu já se auto-duplicou 8× numa escrita falha — conferir `wc -l` depois de edits grandes (~1460 linhas esperado).
6. Bench micro-bench esconde bugs: kernel "rápido" pode só tocar menos memória — sempre diff vs referência.
7. Medir sempre COM apps de desktop ligados (decisão do usuário).

## Decomposição atual (~95ms/tok)

| componente | ms/tok |
|---|---|
| MoE GEMVs dp4a | ~40 ← o muro |
| iq4_xs moe | ~10 |
| gather (warmup only) | ~11 |
| q8_0/q5_k/q6_k densos i8 | ~10 |
| swarm cpy/concat/bcast | ~8 |
| attn fundida | ~2.3 |
| gaps/graph | ~14 |

## Próximos passos rankeados

1. **SAGE-REXL3-2.26bpw** (`vcruz305/Inkling-2-Mini-SAGE-REXL3-2.26bpw`): EXL3 safetensors+trellis, NÃO GGUF. 8.00 GiB, 94.66% top-1. Porte = loader safetensors novo + família de kernels EXL3 (exllamav3 não suporta sm_61 — reescrever tudo). Ganho de bytes/tok ~10-12%. **Alternativa barata**: quant GGUF menor existente (IQ1_M?) — verificar quais quants do Inkling-2-Mini há no HF; iq1_s já está portado.
2. q5_k/q6_k densos (12.9+5.4ms agregado): unpack de nibbles pesado — escrever gold tests (padrão GOLD_Q80) antes de mexer.
3. mix() depth-mix = atenção sobre states (K=V=states, n_kv=n_blk, hd=2304, sem mask/sink) — generalizável do attn_decode, ~1.5ms.
4. swarm bin_bcast/concat ~8ms — fusões por-op.
5. iq1_s i8 (único MoE fp restante, 514µs/launch — precisa gold test próprio).

## Regras do usuário

- SEM megakernels ("ainda não vamos pros megakernels")
- Rust para todo host-path; kernels CUDA via FFI ok
- Tabelas de perf em PT-BR, formato "encode/decode"
- Medir com apps ligados
- Ledger em E2E_DEBUG.md a cada rodada; um commit por mudança validada
- Skill: `inference-optimisation` (0xSero, instalada como plugin Devin)

## Lista de estudo pendente (dada pelo usuário)

- Strata v0.2.18+ (adicionou Inkling, +40% em qwens) — `github.com/Niko1221/Strata`
- `github.com/jesdga95/basalt` (engine Blackwell sm_120 com dspark spec-decode do Inkling2)
- `github.com/alesha-pro/maya-inkling-mini`
- `github.com/nahcmon/inkling-fast-llama.cpp` (~75 tok/s numa 12GB via GGUF "fast" próprio)
- mtplx.com, post X rachpradhan/status/2124455964992840069
- Spec decode: `~/basalt/src/models/inkling2/dspark.cpp` é a referência do drafter DSpark; `inkwell-spec` já tem verify/rollback
