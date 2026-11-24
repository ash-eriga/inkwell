# PLAN — hill-climb Inkwell (restart 2026-11-22)

## Alvos e restrições

- Métrica: decode c1 tok/s, greedy 64 tok, desktop apps ligados (baseline aceito).
- Meta: **15 t/s** (66.7 ms/tok). Baseline: ~8.7 t/s (115 ms/tok).
- Hardware: GTX 1060 6GB (sm_61), 16 GB RAM, driver 580.196.07, CUDA 12.6 local.
- Modelo: Inkling-2-Mini UD-IQ2_XXS GGUF; única troca permitida → `vcruz305/Inkling-2-Mini-SAGE-REXL3-2.26bpw`.
- Qualidade: golden text chinês `"正确答案：A"` deve se manter; reorders fp exigem gold test dedicado.
- SEM megakernels. Rust host-path; kernels CUDA via FFI.
- Ledger narrativo continua em ../E2E_DEBUG.md; este dir = experimentos da skill.

## Inventário

- SM clock 1506 MHz, mem 4252 MHz, P0, 120W. VRAM 6144 MiB (899 usado idle).
- Run: `cargo build -p inkwell-cli --release && LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin GGML_PASCAL_GRAPHS=1 ./target/release/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run`
- nvcc precisa `-ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++` (gcc-16 do sistema é incompatível).
- bench: `opt/bench.sh <config> <N>` → ledger.jsonl.

## Log de experimentos

### 2026-11-22 — baseline-iq2xxs
- 6 amostras (bench.sh): 96.6/102.2/111.0/116.5/101.4/104.0 ms/tok → **mediana 103.1 ms/tok = 9.7 t/s**, golden ✓ em 6/6.
- +2 manuais: 115.3, 112.4 ms/tok, golden ✓.
- Bandwidth medida: D2D 163 GB/s best; kernel grande 217 GB/s; PCIe zc ~8.6 GB/s.
- Bytes/token (THEORY.md): GGUF 2686 MiB; REXL3 ~1830 MiB (−32%; attn dense Q8_0→trellis, ctrl F32→BF16, experts −5%).
- REXL3 traz `mtp.safetensors` (drafter MTP: markov_rank 256, block_size 8, target layers 5/23/41/48) → spec decode nativo.
- Ceiling: GGUF ~22 t/s @60GB/s efetivo; REXL3 ~33 t/s. Meta 15 t/s cabe nos dois; REXL3 domina em bytes × EAR (94.66% top-1 vs 90.6%).
- Custo do porte: loader safetensors + kernels EXL3 trellis sm_61 (exllamav3 não suporta Pascal).

## 2026-11-22 — graph reuse + VRAM experts (176 ms/tok)

- CUDA-graph key era `nodes[0]` (fresh ptr/step → warmup nunca completava → eval eager, ~140ms CPU dispatch/tok). Key agora = FNV-1a sobre (op, type, src->data ptr, node->data ptr) + n_nodes; prop memcmp exclui src[]/view_src/name/userdata (CUSTOM). Warmup completa, replay ativo: eval=2ms, drain=140ms/tok.
- Experts: reserve VRAM 1150→1050MiB; ~1GB de experts agora residentes. `on_dev` pula moe_stage; pinned layers ainda fazem stage PCIe (~50ms/tok restante).
- gemv_dp4a split-K: 585→229ms/run.
- gpu_kern_sum decode: gemv_moe ~640ms/run (56%), dp4a 229ms (20%), had_in/out ~60ms.
- Regressões revertidas: __ldg sem shared (22s), warp-per-tile, TB=8 batching, moe split-K com memset.
- Próximo: (a) INT8 xh + dp4a bytesum (mode-1 error-feedback) no inner loop; (b) predictor de experts pra prefetch do restante pinned; (c) fusão had_out+wsum.

## STATUS 2026-11-24 — 115 -> 58.5 ms/tok, ~65% of practical ceiling

Baseline 115 ms/tok (2026-11-22) -> **58.49 ms/tok median** (3 runs:
58.49/59.28/58.44, BEST env + graphs, text md5 5cbe8e06... identical all
runs). Wins since baseline: REXL3 weights, CUDA-graph key fix, per-layer
LFU expert caches, sgemv MoE (86->72) + sgemv dense (71->58).

### Decode-token breakdown (GGML_OP_PROF, graphs off; step 66.15 ms/tok)

| bucket | ms/tok | % |
|---|---|---|
| MoE expert GEMV (gemv_moe_s, ~323us x 48 visits, kbench-derived) | ~15.5 | 23 |
| MoE non-GEMV (staging wait ~10 + had/situ/wsum/resolve ~4) | ~13.8 | 21 |
| dense EXL3 GEMV excl unembed (exl3_mz 5.85 + wr_du 2.67) | 8.5 | 13 |
| ctrl/norms/other (MUL_MAT 5.6, mix 3.5, elementwise ~11.9, conv/clip 2.5) | 23.5 | 36 |
| attention (attn_cb x48) | 1.3 | 2 |
| unembed (inside exl3_cb) | ~1.9 | 3 |
| router_cb x48 | 1.0 | 2 |
| host/graph gap (step - GPU span) | ~0.6 | 1 |

OPPROF fn medians (ms/tok): moe_cb_exl3 29.33, exl3_mz 5.85, exl3_cb 4.55,
mix 3.47, conv 1.46, attn 1.34, router 1.04, clip 1.03. op-level:
MUL_MAT 5.63, CONCAT 2.00, ADD 1.99, RMS_NORM 1.75, MUL 1.54, UNARY 1.38,
SUB 1.05, CONT 0.99, SET_ROWS 0.69, GET_ROWS 0.51.

### Expert cache (PFSTAT, k=20): 18.6% miss, ~83 MiB PCIe/tok

L12-16 measured 22.3 misses/tok (14-24%/layer); L17-23 extrapolated -> ~36
misses/tok, ~83 MiB staged @ ~8 GB/s ~10 ms — matches the ~13.8 ms MoE
non-gemv residual. Prefetch bank hits ~0 (predicts wrong pass) — dead
weight. Ceiling refreshed in THEORY.md: practical ~30-38 ms/tok.

### Ranked levers (gap x likelihood)

1. **Elementwise/glue swarm ~12 ms** — ~2900 tiny kernels (~4us avg).
   Fuse rms_norm+scale, kill redundant CONCAT/VIEW/CONT, batch SET_ROWS.
   Likelihood HIGH (pure kernel-count work), gap ~10 ms.
2. **Cut staged bytes ~10 ms** — INKWELL_DEV_MARGIN_MB ~1100 puts layers
   12-17 on_dev (frees ~80 MiB cache/layer too); then k=20 -> 30+ on the
   rest. Bank prefetch ineffective as-is (rework or drop). Gap ~8-10 ms.
3. **MUL_MAT ctrl path ~5.6 ms** — 352 generic mul_mat calls on ~1.1 MiB
   F32 weights; route ctrl+gate_inp through gemv_s. Gap ~3 ms.
4. **MoE gemv ~15.5 ms @ ~60 GB/s** — mid-payload BW cap; occupancy/tile
   work (nky_s sweep untested e2e). Gap ~5 ms, harder.
5. **wr_du singles (2.7 ms, 48 calls)** — fold into the mz fused launch.
   Gap ~1.5 ms, easy.

Rejected already (do NOT repeat — see ledger): spec decode/MTP
(structurally bounded ~5-6 t/s < serial 17 t/s; union ~28/visit, 4.4 GB
expert bytes/tok), global expert pool (<8% coverage), exl3v2 serial gemv
(sgemv beat it), nky>2 split-K, dp4a int8-x (2e-2 err), dp4a-bcast x8,
RESOLVE_DIRECT (neutral), __ldg-no-smem + warp-per-tile + TB batching.

Next: lever 1 (swarm fusion) — biggest gap, pure kernel-count reduction;
then lever 2 (one-line env first: INKWELL_DEV_MARGIN_MB sweep).

### Review (2026-11-24) — lever 2 corrected, new lever 0
- Lever 2 as written does not reconcile: VRAM is full (margin 10), and moving a
  layer from cached to on_dev costs ~(1 - 20/128) x 200-360 MiB, it frees nothing.
- Real finding: on_dev placement is greedy by layer index (runner.rs:915-939), so
  layers 0-11 sit in VRAM. The loop is 6-17 x 3 passes (loop_config), so loop
  layers 12-17 (3 visits/tok each, 1882 MiB) are pinned+cached while prelude 0-5
  (1 visit/tok each, 1336 MiB) is resident. Expert MiB/layer (safetensors):
  0:246 1:246 2:197 3:206 4:210 5:230 | 6:224 7:252 8:273 9:290 10:292 11:310
  12:246 13:279 14:310 15:325 16:359 17:363 | 18:216 19:214 20:246 21:246 22:248 23:247.
- New lever 0 (config-only, highest gap x likelihood): place on_dev by visits/MiB
  (loop layers first) within the same VRAM budget -> 30 instead of 24 visits/tok
  resident, cached visits 24 -> 18. Also emulates the user's ~2.2bpw quant
  (all loop layers resident).
- Lever 1 caveat: OPPROF is graphs-off; graphs already remove ~7.7 ms of launch
  cost, and fusion under graphs was ~neutral before (fused mix, ledger). Measure
  glue under graphs before investing.
