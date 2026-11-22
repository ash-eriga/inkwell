
## R10 — dp4a affine → fp16-exact per-col kernel (2026-11-22)
- kernel `inkwell_exl3_gemv_dp4a`: 16 thr/block, 1 col/thread, inverte tile_perm (i=f(r,c) bijectivo).
- Bug 1: afim `s*k_inv + k_bias*sx` omitia `1024*k_inv*sx` (dec=(1024+bs)*k_inv+k_bias) → maxdiff 627.
- Bug 2 (mais grave): afim fecha o fp16-hfma por elemento → erro ~0.01/out → argmax flip → `" a 1."`. Fix: decode `exl3_mul1(win)` fp16 por elemento no loop interno (ainda sem atomics). Gold: **maxdiff 0.0000, cos 1.00000** em dados reais. e2e 9.37s/8tok, texto coerente.
- s/sx validados via `dbg_dp4a` example (GPU s = CPU s exato por tile).
- Staging: `warm_staging()` no load — cudaMalloc dentro de cb durante graph capture envenena o ctx (CUDA 900 → "function failed to launch" em cublasSgemm depois). Trap documentada.
- Restante: MoE ~86ms/call mesmo staged; não-MoE 143ms/tok vs baseline 103.
