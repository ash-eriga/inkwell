- [spec] DSpark draft VERIFICADO por oráculo numpy (dequant unit-vector + draft.py replicado): todos os estágios dentro de ruído; aceitação ~14%/2tok/round é real, não bug. ngram 1.2 tok/round < mtp.
- [spec] verify ~470ms dominado por gemv_moe_ntok (~2.6ms×60): união de experts deduplica pouco → spec 3x pior que serial nesta GPU. kd-adaptive e verify variável não implementados.
- [serial] INKWELL_MOE_NKY=2,4 → 119/127ms piores que ~97ms. Descartado.
- [teoria] gemv_moe serial ~12GB/s efetivo vs ~150GB/s VRAM — latency-bound, headroom teórico ~4x mas kernel já tem prefetch; difícil.
- [kernel] uint4 batch-parallel loads no gemv_moe (64thr=8b×8chunk): ~990ms vs ~925ms controle = -6% → REVERTIDO. MLP maior não ajuda; kernel já saturado em latency/issue.
- [spec] PROMPT-CTX FIX (commit 5a07b27): draft ctx-K/V agora cobre o prompt (replay step_block em blocos de 8). accept 14%→24-43%, 2.7-4.25 tok/round, 4.2-7.6 tok/s. Reference: api.py prefill ctx_append — confirmado.
- [spec] probabilistic acceptance implementado (INKWELL_SPEC_PROB) — referência usa argmax-match (greedy default). Rejeitado como default (drift de output).
- [spec] blk_compute ~500-540ms ≈ wall verify — gap é dentro do compute (instantiate+launches sem reuse de grafo por chamada). GPU-busy ~210ms.
- [bug] Mtp::new ANTES do primeiro step_block → block_feats_ptr null → ctx_append crash. Ordem correta: blk0 → rollback → Mtp::new → prompt replay → blk0 → ctx_append.

## 2026-11-23 (spec economics deep-dive, commits 5a07b27..3728ab0)

### Fixes landed
- Prompt ctx hole fixed: replay bases now include base 0 (`bc<=0` skip removed;
  ctx_append(0) no longer crashes). Draft ctx-K/V covers full prompt.
- INKWELL_SPEC_DLEN=k: pad vtok rows >k with anchor (shrinks expert union;
  ~80ms verify saving at dlen=4, tok/s ~neutral → keep 7).
- moe_ntok/resolve/ccopy take device `pnexp` (graph-capture safe; removed
  capture-illegal memcpy+sync). union_expand clears tail slots.

### Verify cost structure (nsys specprof3, GENS=4+SPEC_GENS=32)
- blk_compute ~537-649ms; block graph = 26928 nodes, ONE CUDA split.
- CUDA graphs on Pascal DO work for block path (GGML_PASCAL_GRAPHS):
  warmup 2 stable calls → capture → replay. BUT verify stays ~570ms:
  ~93% GPU-busy — it's real kernel time, not launch gaps.
- Kernel mix per verify window: ~18k kernel execs; moe_ntok ~465ms
  in ~900ms window (~70%); dp4a_ntok ~61ms; swarm (had_in/out ~2700ea,
  conv_step 1600, sgemm 700, misc) ~50ms total.
- moe_ntok ~4.5ms/call × ~31 layers; reads expert union (~60/64 for 8
  tokens) ≈ same bytes as 8 serial tokens — MoE does NOT amortize.
  ~2MB/expert → ~120MB/layer → ~3.7GB/verify at ~465ms ≈ 8-12GB/s
  → LATENCY-bound, ~10× under VRAM bandwidth. THE spec lever.
- Serial plateau unchanged: ~97ms/tok (~10.3 TPS) golden intact.

### Spec economics
- spec = (verify+draft)/emit; verify ~570 + draft ~50 → needs emit ≥ 6.4
  to beat 97ms serial. Ref vLLM nat_b8 accept_len = 4.34 → still loses.
  Path: moe_ntok 2-3× faster (latency-bound headroom) + emit ≥4.

### Environment gotchas
- GGML_GDBG=1 prints per-call graph decision (use_graph/upd_req/nodes).
- GDBG shows serial=6958 nodes use_graph=1; block=26928 nodes.
