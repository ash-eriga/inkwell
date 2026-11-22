
## R10 — dp4a affine → fp16-exact per-col kernel (2026-11-22)
- kernel `inkwell_exl3_gemv_dp4a`: 16 thr/block, 1 col/thread, inverte tile_perm (i=f(r,c) bijectivo).
- Bug 1: afim `s*k_inv + k_bias*sx` omitia `1024*k_inv*sx` (dec=(1024+bs)*k_inv+k_bias) → maxdiff 627.
- Bug 2 (mais grave): afim fecha o fp16-hfma por elemento → erro ~0.01/out → argmax flip → `" a 1."`. Fix: decode `exl3_mul1(win)` fp16 por elemento no loop interno (ainda sem atomics). Gold: **maxdiff 0.0000, cos 1.00000** em dados reais. e2e 9.37s/8tok, texto coerente.
- s/sx validados via `dbg_dp4a` example (GPU s = CPU s exato por tile).
- Staging: `warm_staging()` no load — cudaMalloc dentro de cb durante graph capture envenena o ctx (CUDA 900 → "function failed to launch" em cublasSgemm depois). Trap documentada.
- Restante: MoE ~86ms/call mesmo staged; não-MoE 143ms/tok vs baseline 103.

## R11 — ms/tok unit fix
decode print covers n_gen tokens (INKWELL_GENS). ms/tok = elapsed/n_gen. All prior ms_tok fields in this ledger used /8 — correct values: 146, 52.8→... recompute as elapsed/gens. True state ~216-280ms/tok.

## R12 — wall-time decomposition (nsys full run, GENS=4)
- GPU busy ~82ms/tok; wall ~222ms → ~140ms/tok em gaps CPU.
- cuda_api_sum: 631 cudaStreamSynchronize (1.03s!) + 20784 cudaLaunchKernel; só 120 cudaGraphLaunch — graphs parciais.
- Kernel sum: gemv_dp4a(dense) 258µs×578 = dominante; gemv_moe 360µs×288; stage 790µs×96.
- UNIDADE CORRIGIDA: "decode 8 tok in X" cobre n_gen tokens → ms/tok = X/n_gen (GENS=8 → ~222ms/tok, ~4.5 t/s).
- Próximos levers: (a) matar os ~158 syncs/token (provável DtoH/top-k/router por camada); (b) dense gemv tile width; (c) expert predictor prefetch mata stage PCIe.
