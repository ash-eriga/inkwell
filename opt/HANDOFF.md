# HANDOFF — Inkwell REXL3 port (overnight session)

## ★ RESUME AQUI (2026-11-24, serial DP4A/plateau)
=================================================
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
