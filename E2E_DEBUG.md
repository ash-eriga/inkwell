# E2E debug state (runner.rs) — 2026-11-22 (checkpoint)
Run: cd ~/inkwell && LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin GGML_PASCAL_GRAPHS=1 ./target/release/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run

## Round 4 (2026-11-22 night): i-quants all dp4a

- iq2_xs i8 (+30-40%), iq2_s i8 (+50%) — same smask[256]+vsubss4 sign trick as iq2xxs_i8; per-32 x-quant. Defaults on; opt-outs INKWELL_IQ2XS_FP / INKWELL_IQ2S_FP / INKWELL_IQ2XXS_FP.
- 2-warps-per-row on i-quant i8 kernels REGRESSED (iq2s 0.336->0.382) — 1 warp/row kept.
- iq1_s i8 remains rejected (-23%, delta-shift doesn't amortize).
- kernels.cu got 8x self-concatenated during a failed write — restored from HEAD; watch for it.
- mmv_f (~5.4ms/tok, ~600 launches) = manual attention mul_mats (K*Q, V*attn, rel_bias table) + mix() states F32 GEMVs — next structural target: fused decode-attention kernel (per visit) or route via ggml flash_attn_ext with mask=rel_bias + softcap mb/lsc params (ops.rs already wraps it).
- decode ~95-100ms/tok contended; MoE i8 ~35ms, dense i8 ~13ms, mmv_f ~5.4, swarm ~6, rest attn/conv/misc.

## Round 3 (2026-11-22 night): real per-kernel split (nsys, graphs off, ~64 tok)

| kernel | ms/tok | note |
|---|---|---|
| gather | ~11 | warmup only — misses_total=0 after (verified MOESTATS) |
| q8_0_i8 dense | 9.4 | 172 launches |
| q2_0_i8 dn | 8.3 | 236us avg |
| iq2_xs moe | 7.5 | worst gu (439us) |
| iq2xxs_i8 | 7.2 | |
| iq4_xs | 5.1 | |
| nvfp4_i8 | 4.5 | |
| q6_k_i8 | 4.0 | NEW — real Q6_K interleave is per-128-half |
| iq1_s+iq2_s | 7.2 | |
| mmv_f | 5.4 | F32 — no dp4a |
| q5_k_i8 | 1.5 | NEW |
| swarm (cpy/bcast/concat) | ~6 | |

- bench GB/s numbers were ~4x inflated for 256-val formats (row_b formula bug in bench_moe) — trust nsys per-launch times.
- dense i8 GEMVs at 1024 threads; q2_0_moe_i8 stays 512 (regressed at 1024).
- Commits: 0e1c939 q6k, 191278e q5k, af0997b block bump, f35ef8b iq2xxs i8 default.

## Kernel rewrite round 2 (2026-11-22 eve)

- iq2_xxs paired-smem float2: REVERTED — layout collided (NU=290 < 2*ngroups=576 → values overwrote each other). The "+17%" bench win was a broken kernel touching less memory. Caught by new sanity check: fp-moe vs fp-dense (unchanged reference kernel) → was 2.19e4, now 0. e2e golden restored exactly.
- LESSON: always diff a modified kernel against an untouched reference kernel on the same inputs; junk-weight bench only checks speed, not correctness.
- iq2xxs_i8 now DEFAULT-ON (per-32 x quant: maxdiff 1.186e3 junk vs fp; e2e golden). Opt out: INKWELL_IQ2XXS_FP=1. ~96 GB/s vs fp 75.4.
- Flat float4 x-reads kept (no collision possible): iq1_s 76.6, nvfp4 62.3, iq2_xs 111.5, iq2_s 110.8, iq4_xs 217 GB/s.
- Latest decode wall ~6.09-6.33s/8tok (contended).

## Current: ~100ms/tok (~10 t/s contended) vs upstream 6.15 t/s. Target 15 t/s (66ms).
- CUDA graphs REPLAY: persistent ctx + n_kv padded to 64 + -inf mask; embd_row on GPU → 1 split.
- Device MoE: resolve/alloc/gather in-stream; misses repoint to arena (1 PCIe read). budget=vfree-1.5G, k=65-78.
- RAM fix: madvise(DONTNEED) after expert pin (~4GB freed).
- Custom dense GEMV: mmq() routes Q8_0 (type8) mul_mats to inkwell_mmvq_q8_0_i8 inside the graph (CUSTOM=210/tok).
- CAVEAT: kern_sum under graphs only counts capture-phase launches — use wall time.
- MEASURED (post-graph): compute ~85ms/tok, build ~1ms. MoE GEMVs ~54ms/tok real
  (48 visits x ~3 matrices; iq2_xxs ~500us, others 250-460us). misses_total=0 after
  warmup — cache hit ~100% (loop x3 revisits reuse experts; gather is front-loaded
  in early tokens). THE WALL IS NOW PURE GEMV BANDWIDTH (~22-60 GB/s vs 200 target).
  iq2xxs_i8 exists behind INKWELL_IQ2XXS_I8=1 (1024thr; ~100GB/s vs fp ~72;
  text coherent but ~1.6% logit noise on that tensor — opt-in). Block-size
  sweep: i8 kernels prefer 512, iq4_xs prefers 1024. mmv_f ~9.5us avg x1500, type13/14 ~6.7ms, swarm ~10ms.

## Current blocker
sched allocs 6915 MiB CUDA0 graph buffer (all expert leafs copied to GPU) → OOM.
MUL_MAT_ID ops land on CUDA0. set_tensor_backend(cpu) called on expert leafs at load
AND on gate_up/hid/down/ffn_out outputs in layer() via self.set_cpu.
Hypothesis: ggml-cpu lacks mul_mat_id support for some expert types
(Q2_0=42, NVFP4=40 new fork types, maybe IQ quants) → backend_from_buffer→-1
→ expand-gpu assigns CUDA0 → split input copies the whole weight.
Check: ggml-cpu type_traits_cpu[t].vec_dot_type for types 16,17,19,22,23,40,42;
  repack.cpp:5213 supports_op; ggml-cpu.c:2404.

## Already fixed today
- attn path: view4 op added (ops.rs+shim+lib.rs); attn out reshape2 = hd*nh=3072
- mask2 sliced via view2 to n_kv cols (onehot 1024 wide)
- k4/v4 need t::cont before reshape4 (arena views non-contig)
- exp bias name: blk.N.exp_probs_b.bias (not ffn_exp_probs)
- MoE weights: probs reshape3 [1,128,1] before get_rows → [1,8,1]; then
  reshape2 [8,1], sum_rows→clamp→div→reshape3 [1,8,1]→scale 8.0
- expert combine: 8× view2 [n_embd,1] + adds (NOT sum_rows)
- ssm_conv needs 3D input: reshape3(x, n_conv=4, ch, 1)
- ggml ctx pool 4MB→64MB (shim mem_size); cgraph 32768 nodes; sched 32768
- 512MB thread stack for build_forward DFS (main.rs)
- Runner.sched_cpu: Option<(Ptr,Ptr)> set in load(); set_cpu() helper at runner.rs:255

## Suspect pending cleanups
- q4 reshape may need cont (q from clip→mul is contig already — OK)
- SWA ring addressing wrong for pos>=1024 (works <1024)
- `let out = ... permute(0,1,3,2)` before reshape2 — verify head order vs upstream
- std::mem::forget(buf) per-step leak
- ~60 clippy warnings in runner.rs
- DBG prints: "DBG out ne=" in layer(), remove when golden

## Round 2 fixes applied (2026-11-22 ~06:00)
- USAGE_WEIGHTS on dev/cpu weight bufs → MUL_MAT_IDs land on CPU (was OOM 6915MB CUDA0). shim: ink_buf_set_usage.
- tensor_get on view-of-view fails (one-level view_src only) → read logits direct from softcap output tensor (contiguous prefix).
- sched needs ink_sched_reset between steps (is_alloc stays true → no realloc).
- ggml ctx is append-only → per-step throwaway ctx (ink_ctx_new/free in step()), weights stay in persistent ctx; in_buf (16MB dev) reused via self.in_buf.
- e2e RUNS: 8 tokens in ~2.1s (~4 t/s) but GIBBERISH.
- ggml_permute semantics: dst.ne[ax_i]=a.ne[i] (swaps coincide, rotations differ).
- Attention rewritten to upstream non-FA MHA: kq=mul_mat(k_view[hd,nkv,hkv], permute(q,0,2,1,3))→[nkv,1,nh]; +kq_b from get_rows(table,rel_pos) where table=cont(permute(mul_mat(pad(rel_bias,0,1),r)-reshape3(sinks,1,nh,1),1,0,2,3)); kq_b=cont(permute(reshape4(rows,nh,nkv,1,1),2,0,1,3)); sink0 concat dim0 + softmax + view4 slice; kqv=mul_mat(cont(transpose(v_view)),attn)→[hd,1,nh]→reshape2(hd*nh).
- rel_pos i32 [n_kv,1] per class (full/swa), distance = n_kv-1-j clamped to rel_extent(1024) [fixed from 1025].
- still GIBBERISH → next suspects:
  * build_lora_mm is NOT plain mul_mat: build_hadamard_input(w,cur) for hdmd->rot-registered weights (rot+perm_rep+signs + llama_mul_mat_hadamard) + w_s=get_w_scale(w) scale mul after. Check which weights have rot/scales (llama-model.cpp:2076 w_scales; hdmd->rot population).
  * NVFP4 needs prec BF16 acc.

## KEY RESOURCES for resuming
- Upstream reference: ~/lfm-gtx1060-bench/src-v2/src/models/inkling2.cpp (attention ~296-380, ffn ~380-413, helpers 165-272), inkling2-loop.cpp (loop driver 76-115), llama-graph.cpp (build_lora_mm 1809-1890, build_hadamard_input 1747+, build_attn_mha 3106+, non-FA MHA else-branch ~3170-3210, MoE build_moe_ffn ~2298-2500).
- llama-model.cpp: load_hparams_hadamard (2259+), load_tensors_hadamard (2254), get_w_scale (2621), w_scales map (llama-model.h:827).
- The runner mirrors upstream structure already (control/mix/convolve/dynamic taps/loop driver w/ expanded slots/rel-bias gather). STILL gibberish.
- llama.cpp golden: "The capital of France is" ids=[371,151451,149879,151194,149884] → " Paris". Server: llama-server -m <gguf> -ngl 99 -ncmoe 99 -t 4 -ub 256 -c 4096 (needs GGML_CUDA_ENABLE_UNIFIED_MEMORY=1), /tokenize, /completion.
- Likey remaining bug sources, in order: (1) hadamard-folded weights need build_hadamard_input pre-transform — check GGUF metadata keys prism.hadamard.* and which weight names get rot registered (load_tensors_hadamard); (2) w_scale mul per weight (which tensors have .scale); (3) NVFP4 BF16 acc hint; (4) KV write pos for SWA ring when pos>=1024; (5) conv state/store ordering.
- Build: cd ~/inkwell && cargo build -p inkwell-cli; Run: LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin ./target/debug/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run
- Pending cleanups: DBG prints (runner.rs TDBG call + "DBG out ne"), ~60 clippy warnings, mem::forget leaks.

## Round 3 (madrugada) — bisseção por-tensor
- Dump upstream por-token: `llama-eval-callback` patchado p/ 1 token/process → /tmp/upstream-pertok.txt (+full em /tmp/upstream-full.txt)
- **Bug 1**: BOS — upstream NÃO adiciona BOS p/ esse modelo; removido do runner
- **Bug 2**: token_embd/output precisam F32 (upstream converte no load) — add `ink_dequant` shim, converte no upload ✓ (row371 = -25.553419 bit-exact)
- **Bug 3**: graph input tensors precisam GGML_TENSOR_FLAG_INPUT (`ink_set_input`) senão gallocr clobbers — fix
- **Bug 4**: dump tensors precisam `t::output()` senão memória é reusada
- **Bug 5**: KV write via cpy/store SEM dep-edge com leitura → trocado por `set_rows` in-place (b precisa [hd,1,hkv]: permute 0,2,1,3)
- Estado: embd/Q/K/V/R **bit-exatos** token0. kqv_out-0: meu -32.46 vs upstream 3.63 (atenção ainda diverge). l_out-0: -26.02 vs -25.78 (quase). result_norm tok0: 317.4 vs 110.66.
- tau formula confirmada: log(clamp((pos+1)/8192,1,inf))*0.1+1 ✓
- Próximo: diff kq_b/kq_ext vs upstream node_93/kq_mask_rel_pos (upstream KV width=256 padded)

## Round 4 — E2E funciona + kernels próprios
- **SCALE BUG RESOLVIDO**: scale 1/sqrt(128) aplica só em k·q; bias rel pos soma DEPOIS (upstream FA semantics: softmax(k·q·scale + bias)). kqv_out-0 = 3.6307 vs upstream 3.6319 ✓
- **Decode end-to-end**: " Paris.\n\nA. Proper" — gen0 id=152717 golden ✓. Release: encode 4tok/0.93s, decode ~5.4 tok/s (vs upstream ~6). compute=180ms/tok, build=1ms.
- Kernel próprio 1: dequant IQ2_XXS — bit-exact (5.3M vals, 0 diff)
- Kernel próprio 2: GEMV IQ2_XXS fused — rel diff 1.3e-5 (fp32 order)
- Kernel próprio 3: GEMV Q2_0 fused — rel diff 1.0e-4
- **Zero-copy GEMV: 8.6 GB/s via PCIe** (cudaHostRegister mapped) — experts streamam de RAM sem upload. Estimativa: ~20 t/s só com experts zero-copy.
- LUTs em crates/inkwell-cuda/src/iq_luts.h (extraídas de ggml-common.h, marcadas __device__)
- Próximo: IQ1_S/NVFP4 GEMV (outros tipos de expert), pager real (LRU + prefetch), spec decode.

## Round 5 — overnight final
- 64-token greedy decode: fully coherent text, sustained ~5.8 t/s
- Batched expert GEMV (ids[slot] indirection over registered host or device ptr)
  + fused swiglu_pair — primitives for the real pager MoE path
- Coverage: ALL expert formats have own GEMVs now (IQ2_XXS/IQ1_S/Q2_0/NVFP4)
- TODO next: wire zero-copy expert GEMVs into runner MoE path (replace mul_mat_id),
  pinned host pool for expert streaming, spec decode, remaining type coverage
  (IQ2_XS/IQ3_S/IQ3_XXS/IQ4_XS/Q6_K for non-expert tensors)

## Análise p/ próxima sessão
- Experts no CPU (mul_mat_id) = ~120ms dos ~180ms/tok. Split do grafo por visita p/ interpor kernels próprios custaria ~24 syncs — não vale sozinho.
- Plano real p/ TPS: (1) pager com cache LRU de experts NA VRAM (pool ~3-4GB livre), H2D assíncrono no draft/prefill — sim do pager projeta ~40-60% hit. (2) spec decode ×~2.5. (3) batched MoE GEMV já pronto em kernels.cu.
- Alternativa: host-mapped ggml buffer pros experts + mul_mat_id CUDA — ggml pode ler tensor de host-pinned em ALGUNS ops; verificar ggml_backend_cuda_host_buffer.

## 2026-11-22 — beat upstream

- decode ~6.6-6.7 tok/s sustained (148-152ms/token) vs upstream 6.15 — WON
- pieces: VRAM LRU expert cache (Cache in moe.rs; k=75/layer ~3.2GiB, ~89.5% hit;
  promote on 2nd use; misses read zero-copy from pinned host tables)
- embd quantized host-side (row dequant per step into embd_row CPU tensor);
  output.weight stays quantized (mul_mat mmvq path on CUDA) — freed ~3.3 GiB
- kernels take `const void* const* Wp` per-slot pointer arrays; x_stride=0 gu / N_FF dn
- headroom: keep >=1 GiB for ggml compute buffers (700MB OOMs)
- debug envs: INKWELL_MOE_T (callback times + hits), INKWELL_MOE_DBG, INKWELL_MOE_DUMP, INKWELL_DUMP
- NEXT: spec decode — basalt ~/basalt/src/models/inkling2/dspark.cpp (747 LOC) +
  kernels/dspark.cu (722 LOC) is the reference impl of Inkling2's DSpark drafter
  (Markov head + screen + recurrence, NVFP4 products/Q8_0 context proj, own cuda graphs).
  inkwell-spec crate already has verify/rollback loop + ngram reference drafter.

## 2026-11-22 — micro-gargalos round 2 (commits df6380c..59216b6)

- **Concat projections** (df6380c): per layer, same-type {attn_q 3072, attn_gate 3072, attn_rel 384} byte-concatenated at load into one weight → 1 mul_mat + view1 slices in step(). INKWELL_NOCAT=1 disables (control). dev_bytes headroom +240MB for cat+pad tensors.
- **GOTCHA: `ggml_view_1d` offset = BYTES in this fork** (not elements!) — slices use row*4 for F32. Proven by `examples/cat_test.rs` (standalone mul_mat+view1 repro).
- **GOTCHA: no u32 loads at odd block offsets** — qs at byte 2 of 18B q2_0 block is misaligned → sticky CUDA fault surfacing later in moe_cb DtoH (panic lib.rs:268). Two u16 reads are the aligned way.
- **op histogram** `INKWELL_OPCNT=1` (shim ink_graph_op_counts): per token — RESHAPE 1574, VIEW 1195, CONT 1103, MUL_MAT 1014, MUL 714, CONCAT 596, ADD 536, RMS_NORM 514, PERMUTE 496, SCALE 435, CPY 193, TRANSPOSE 168.
- nsys shares (~8 tok, contended desktop): MoE kernels ~55%, dense GEMV ~22%, misc swarm ~23% (concat_cont ~14ms/tok, cpy ~16ms, bin_bcast ~19ms — batch-dim ops split ~8 launches/node).
- Attention already batched (mul_mat over ne2/ne3 heads) — the 4170 mmvf launches/tok = batch-split nodes, not per-head calls.
- FAILED experiment kept for reference: KV write `cont(permute)` → reshape elision caused a sticky crash even after revert (was actually the misaligned q2_0 u32 — untangle when retrying).
- Next targets: concat/cpy swarm via fewer graph ops (mix states rebuild per call, ~600 CONCAT), clip 9→fewer ops, dn-expert low GB/s (q2_0 27, nvfp4 45).

## 2026-11-22 — DP4A int8 family (commits 41e428e, ce952d4)

- `mmvq_nvfp4_moe_i8` / `mmvq_q2_0_moe_i8`: x quantized to int8 per 64-val block
  (per-block symmetric sx in smem), weights decode to int8 lanes → `__dp4a`.
  nvfp4 +15-25%, q2_0 +43% @2304. ~0.05-0.1% logit noise (same class as q8_1).
- GOTCHA: nvfp4 nibble packs are interleaved {qlo.lo, qhi.lo, qlo.hi, qhi.hi}
  → vals {0-3,4-7,8-11,12-15}; wrong pack order = 4% logit error (caught by
  bench golden vs fp kernel — ALWAYS bench-compare vs fp version).
- iq1_s next: per-8-val group = 2 dp4a + delta*sum(x_i8) term (needs per-8 x
  sums — precompute during quantize pass).
- Remaining: iq1_s/iq2_xxs/iq2_xs/iq2_s/iq4_xs dp4a variants; swarm
  cpy/concat (~30ms/tok); mix() states rebuild per call (~600 CONCAT/tok).

## DP4A sweep — verdicts per format (2026-11-22)

- q2_0_i8, nvfp4_i8: WIRED (43%/17% gains).
- iq1_s_i8: rejected — 8-val groups don't amortize x-quant pass.
- iq2_xxs/xs/s: sign-masks separate from magnitudes; per-lane sign application
  in dp4a needs byte-select tricks (__vnegss4/LOP3) — group of 8 doesn't
  amortize. Path forward if attempted: sign-apply via vset/vneg SIMD video
  ops or restructure to 16-val groups sharing sign work. Low priority vs
  swarm (~30ms/tok in cpy/concat/mix rebuild).

## Structural push for 15 t/s (contended GPU, apps running)

Decomposition (nsys): MoE experts ~74ms/tok was 2/3 of ~149ms. Fixes:

- `862a35d` miss = 1 PCIe pass (ptr repointed at VRAM slot after promote; was 2x)
- `a80c828` dn promote + ptr upload on side stream, overlapped with gu GEMV
- `2bd1d29` eager promote uses>=1 (loop visits x3 => subsequent hits in VRAM)
- `225dde6` PERSISTENT ggml ctx + reset/step: fresh ctx per step made tensor
  addresses differ => CUDA-graph key never repeated => ~7100 raw
  launches/token (~33ms host API). Now graphs capture+replay.
- After: MoE ~31ms/tok (was 74). decode runs ~6.1-10.4s/64tok (noise: desktop
  contention) ~8.5-10.5 t/s vs upstream recipe 6.15.

Remaining buckets: dense mmvq_q t8 ~9.4ms, mmv_f ~4.4, swarm ~7, attn/conv ~8,
lm_head ~4.3, ~96 backend splits (custom ops are CPU-only in ggml => D2H of
x/ids + sync readback of output per visit). Device-side resolve would kill
the split round-trips.
