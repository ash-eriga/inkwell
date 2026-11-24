# HANDOFF — Inkwell REXL3 port (overnight session)

## ★ RESUME AQUI (2026-11-24, serial DP4A/plateau)
=================================================## ★ RESUME AQUI (2026-11-24, plateau declared)
=================================================
Serial ~97ms/tok ≈ 10.3 TPS, golden. BEST ENV:
  GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 \
  INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
  ./target/release/inkwell ~/models/inkling --run

Per-callback decomposition (GGML_OP_PROF=1, per token):
  moe_cb_exl3  n=48   ~53.3ms  ← 55%: GU/D gemvs + resolve + ccopy + had + swiglu
  exl3_mz_cb   n=48   ~19.4ms  ← 20%: fused 5-proj attn (wq/gate/rel/wk/wv) multi-stream
  exl3_cb      n=49   ~12.2ms  ← 13%: wo ×48 (~140us) + output lm-head ×1
  attn cb      n=192  ~1.9ms
  clip_l2_cb   n=48   ~1.4ms  (fused L2 clip — was ~9ms of elementwise ops)
  conv cb      n=144  ~1.1ms
  op=34 (elem) n=511  ~5.2ms  ← fusible? unexplored
  ggml misc           ~3-5ms

PLATEAU declared in ledger: ~10 levers rejected on the gemv/moe structure.
gemv is instruction-throughput-bound (~90% SM occupancy, ~4-5 instr/weight for
trellis decode); only fewer instr/weight (dp4a — dead on x-int8 accuracy) or a
re-quant (vetoed by user: REXL3-2.26bpw only) would move it materially.

Remaining unexplored, in rough expected-value order:
1. op=34 fusion (~5.2ms pool): identify which ggml op id 34 is (elementwise)
   and fuse the 511 calls or cut their count. Upside ≤ ~3ms.
2. Warp-autonomous gemv rewrite (each warp owns a window-stream, syncwarp
   instead of syncthreads): uncertain — kernel is near instruction floor
   already; maybe -20% on gemv pools → ~8ms. Big effort.
3. had_in_moe fused into gemv prologue (~1.5ms pool).

Serial ~99ms/tok ≈ 10.1 TPS, golden. BEST ENV:
  GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 \
  INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
  ./target/release/inkwell ~/models/inkling --run
(NKY=2 é o ganho grande; v2_ok desarma com nky>1 → kernel antigo com split próprio.)

Breakdown: compute ~97ms = non-MoE ~53ms (MOE_SKIP mede) + MoE ~44ms.
PROF: build ~2ms, compute ~97ms, logits ~0.3ms por token.
~5000 ops no grafo ggml (CUDA graph ativo); MoE roda dentro de CUSTOM callbacks.

Rejeitados com dados (opt/ledger.md): MoE dp4a inteiro (GU int8-x ~10% err/janela;
down correto mas 10% mais lento), MOE_PREFETCH (só cached), v2 split-K port (degenerate),
stage 512thr (neutro), K>=24 OOM.

Próximos levers rankeados:
1. Decompor os ~53ms non-MoE — suspeito: lm head 201k×2304 (type==23 → out_head mmvq_iq4_xs,
   ver se tensor é 23; se EXL3 vai por MUL_MAT dentro do compute). Sem nsys no box.
2. ~5000 ops/grafos — kernels de elementwise (RMS_NORM×514, MUL×650...) talvez fusíveis.
3. gemv_moe_n occupancy/ILP (rebuttal #2) — mas o path atual usa kernel ANTIGO com nky=2.
4. CUDA-graph explícito do callback MoE — provavelmente já capturado via ggml graph.
