# E2E debug state (runner.rs) — 2026-11-22
## Status: graph builds all 48 visits; sched compute fails.
Run: cd ~/inkwell && LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin ./target/debug/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run 2>&1 | tail -20
Debug sched: GGML_SCHED_DEBUG=2

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
