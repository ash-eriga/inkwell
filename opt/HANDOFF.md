# HANDOFF — Inkwell REXL3 port (overnight session)

## Missão
Meta overnight: **30 TPS sem speculative decoding** (~33 ms/tok). Se chegar lá e sobrar
tempo, pode seguir para speculative decoding (o REXL3 tem drafter MTP embutido —
`mtp.safetensors`, markov_rank 256, block_size 8 — e o usuário aprovou a "magiquinha"
depois do alvo). Mantenha texto coerente (`generated text` deve parecer português/inglês
real, ex: " the capital of France.\n\nThe ").

## Estado atual (HEAD = 2f1b0d6)
- **decode ~1.5–1.6 s / 8 tok ≈ ~190 ms/tok** (INKWELL_GENS=8). Coerente.
- Progressão desta sessão: 1171 → 222 → ~190 ms/tok.
- Baseline GGUF (IQ2_XXS): 103 ms/tok. Meta: 33 ms/tok.
- GPU-bound agora: eval ~2 ms + drain ~135–150 ms/tok. ~35 ms de gaps de replay.
- Kernels dominantes (último profile eager): `gemv_moe` ~80 ms/tok,
  `gemv_dp4a` ~23–28 ms/tok, `had_in/out_moe` ~5 ms, ggml mul_mat_vec ~4 ms.
- ~13/24 camadas de experts VRAM-residentes (~2.6 GB); 11 pinned fazem stage PCIe
  (~1.5–2 ms/visita × ~22–26 visitas ≈ 25–40 ms/tok).
- Decode determinístico (dp4a sem atomicAdd — slices + soma em ordem fixa no had_out).

## Comandos
```bash
cd ~/inkwell
cargo build -p inkwell-cli --release   # ou -p inkwell-cuda -p inkwell-cli
LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin \
  GGML_PASCAL_GRAPHS=1 INKWELL_GENS=8 timeout 300 \
  ./target/release/inkwell ~/models/inkling --run
```
- "decode 8 tok in Xs" cobre n_gen tokens: ms/tok = X/N.
- Rebuild ggml-cuda: `cd ~/lfm-gtx1060-bench/build-inkwell-dp4a && cmake --build . --target ggml-cuda -j8` (ggml-base para ggml-backend.cpp).
- Debug env: `INKWELL_CG_PROF=1` (eval/drain por compute), `GGML_CUDA_GRAPH_DEBUG=1`,
  `INKWELL_MEMCPY_DEBUG=1`, `INKWELL_CPY_DEBUG=1`.
- nsys NÃO está no PATH neste host; gold test: `cargo run --release --example gold_exl3 -p inkwell-cuda` (GPU vs ref CPU, deve passar ~5e-4).
- Zumbis: `pgrep -f target/release/inkwell | xargs -r kill -9` — um processo preso causa CUDA OOM no próximo run.

## Arquivos-chave
- `crates/inkwell-cuda/src/kernels.cu` — kernels EXL3 + launcher `ink_launch_exl3_moe`.
- `crates/inkwell-cuda/src/lib.rs` — FFI (`exl3_moe_async` tem `on_dev`).
- `crates/inkwell-run/src/moe.rs` — `MoeArgsExl3{on_dev}`; buffers: xh/gu/hid/dn
  N_SLOTS×N_EMBD f32, exl3_stage N_SLOTS×3×1004544 B, sdescs_{g,u,d} 128×32 B.
- `crates/inkwell-run/src/runner.rs` — decisão DevBuf-vs-pinned por camada
  (`f_free > total + 1050MiB`); scratch agora `k + 8*n` floats.
- `~/lfm-gtx1060-bench/src-v2/ggml/src/ggml-cuda/ggml-cuda.cu` — FNV graph key
  (ops+type+src->data+node->data, n_nodes<<32), prop-memcmp com exclusões
  (src ptrs, view_src, name, CUSTOM userdata), CGP timing. Prints debug env-gated
  (GKEY/CGP/INCPY/GRAPHDIFF) — limpar antes de finalizar.
- `opt/PLAN.md`, `opt/ledger.jsonl`, `opt/E2E_DEBUG.md`, `opt/THEORY.md`,
  `opt/REXL3.md` — metodologia e histórico (atualize ledger + PLAN por experimento).

## Kernels EXL3 (sm_61)
- `exl3_window(rg, i, ka, halfk)` — extrai janela de r-bits; `exl3_mul1` decodifica.
- `inkwell_exl3_gemv_dp4a` — dense GEMV: 64 thr, 4 tiles/bloco, rings+xhs shared,
  split-K via gridDim.y (nky≤8), partials em slices de scratch, had_out soma
  `nky` fatias em ordem fixa (determinístico). memsetAsync das fatias antes.
- `inkwell_exl3_gemv_moe` — por-slot expert GEMV: grid (tn/4, nslots), 64 thr,
  rings[4*80]+xhs[16] shared, 2 syncs/iter tk. Store direto `t[slot*n+...]`.
- `inkwell_exl3_moe_stage` — copia trellis pinned→VRAM staging (só camadas pinned).
- `inkwell_exl3_had_in/out`, `_moe` variants, `inkwell_situ_pair`,
  `inkwell_expert_wsum`, `inkwell_conv_step` (conv fundido).
- `ink_launch_exl3_moe` — stage (se !on_dev) → had_in → gemv → had_out (×G,U) →
  situ → had_in → gemv → had_out (D) → wsum. G e U compartilham `x` mas suh/svh
  diferentes.

## Armadilhas (aprendidas com sangue)
- **atomicAdd fp32 = decode não-determinístico** — drift de logits muda argmax entre
  runs. Use slices + redução em ordem fixa.
- **In-block split-K do moe foi revertido** — a versão com part[4][64] corrompia a
  soma (dupla contagem) e era ~9% mais lenta mesmo corrigida. Layout 64-thr/4-tile
  é o campeão local.
- **`__ldg`/leitura global direta no inner loop = catástrofe** (~20× mais lento).
  Mantenha rings/xhs em shared.
- `xh`, `dn`, `hid`, `gu` são buffers por-slot — índices `slot*dim`, não
  `slot*dim/2` (bug na tentativa de fusão G+U).
- `had_in_moe` usa o `suh` do desc do próprio peso — G e U precisam de xh separados
  (fusão abortada; se retomar, aloque xh2 = nslots*KG floats).
- `ggml_view_1d` offsets são em BYTES. Kernels custom devem lançar em
  `cuda_ctx->stream()`. Custom-op cb não pode cudaMalloc/cudaDeviceSynchronize
  durante capture. Reordenar fp drifta logits.
- Compute buf ggml ~876 MB — reserve ≤ ~950 MiB causa OOM de arena; 1050 é o ponto.
- Medir com apps desktop ligados (metodologia da skill). Um commit por mudança validada.

## Próximas alavancas (ordem de payoff estimado)
1. **INT8 xh + DP4A no gemv_moe** (a decomposição que o usuário pediu):
   `s = k_inv·Σ xh·bytesum + k_bias·Σxh` — bytesum é soma de 4 u8 ⇒
   `Σ_i xh_i·bs_i = Σ_b dp4a(xh_i8, byte_plane_b)`. Quantizar xh por bloco-128
   com escala (modo-1 error-feedback ≈16-bit efetivo, ref: exllamav3
   `exl3_gemv_int8.cu`). Inner loop vira IMAD+dp4a puro — maior ganho estrutural.
2. **Esconder staging PCIe**: stage kernel das camadas pinned num segundo stream,
   ou prefetch da camada L+1 durante compute de L (sem predictor ainda).
3. **Fusão control chains**: had_out+wsum, situ+had_in.
4. **Predictor de experts** (Lanilen/Inkling-2-Mini-Expert-Predictor): prefetch →
   PCIe some do caminho crítico. Precisa pós-treino na nuvem — coordenar com o
   usuário antes de gastar nele.
5. Spec decode (MTP drafter) — só depois dos 30 TPS.
6. Limpar prints debug (GKEY/CGP/INCPY/GRAPHDIFF em ggml-cuda.cu e ggml-backend.cpp).

## 2026-11-23 progresso (sessão 2)
- LUT dec(bytesum): REJEITADA (~216 vs ~200 base, A/B intercalado 4×).
- FIX: extern Rust `ink_launch_exl3_moe` não declarava `on_dev` (lixo na stack) — corrigido.
- FIX LATENTE: EXL3_STAGE_PROJ 1004544 assumia w≤24 (ka≤3); experts reais têm w até 32
  (trel_b até 1327104) → stage overflow para slot vizinho. PROJ=1334016 agora.
- Cache de experts VRAM por camada pinned: kernels.cu tem inkwell_exl3_cache_resolve
  (serial, victim pula slots reivindicados na visita — atomicAdd%k colidia com hits),
  inkwell_exl3_moe_ccopy (slot_proj param = max_slot por camada = max_e(trel_b)+6912),
  ink_launch_exl3_moe_cached. moe.rs: Exl3Cache{buf,tag,rr,k,slot}, Staging +exl3_map/miss.
  runner.rs: cache alocado p/ pinned se f_free > cbytes + 1100MiB; env INKWELL_EXL3_CACHE_K
  (default 8, ≥8 obrigatório, 0 desliga).
- Estado: bench_cache4.log rodando — verificar coerência + ms/tok.
- OBS: só ~12 camadas pinned (il≥12); cache 32MB/camada; free pós-load ~983-1214MiB;
  compute buf 835MiB; graph instantiate precisa ~150-200MiB livres → OOM se exagerar.

## 2026-11-23 progresso (sessão 2, cont.)
- KV arenas F32→F16 (só load_st): kv 835→419MiB. Kernel attn_decode virou template<KV16>,
  strides k/v em BYTES; set_rows F32→F16 nativo do ggml. Golden mantido.
- Cache lazy pós-load (depois de kv_buf+warm_staging): margem 260→190MiB; ~10-11/12 pinned cached.
- dev_buf headroom 300→160MiB; INKWELL_MAX_CTX env (default 8192, 4096 libera ~134MiB).
- DevBuf::try_new_bytes (Option) — OOM degrada p/ staged.
- gemv_dp4a + gemv_moe: acúmulo AFIM em bytesum (t=KINV*Σxh*bs+KOFF*Σxh, KOFF=1024*KINV+KBIAS)
  — remove decode fp16 por elemento. ~160-170ms/tok.
- NSYS PROFILE (nsys em ~/lfm-gtx1060-bench/cuda-12.6/bin): gemv_moe 39% (369µs×144),
  gemv_dp4a 21% (99µs×289), ccopy 21% (1.34ms×21 misses). Resto <2%/kernel.
- Piso não-MoE (INKWELL_MOE_SKIP): 84ms/tok. MoE atual ~85ms.
- Ledger: exl3-cache-kv16 med ~171; exl3-affine med ~168 (variância desktop ±15).

## estado pós-afffine (auto-save)
- gemv_moe reescrito: 4 tiles/blk × 64thr (256/blk), r split 4-way + shfl_xor reduce,
  tk batch ×4/sync, rings[4*4*32]. Launches <<<gg/gd,256>>>.
- nsys: ~/lfm-gtx1060-bench/cuda-12.6/bin/nsys profile --trace=cuda + stats --report cuda_gpu_kern_sum
- Ledger tags: exl3-cache-kv16(~171), exl3-affine(~168), exl3-moe-tkb(~169), exl3-cache-k12(running)
- K=12 test: INKWELL_EXL3_CACHE_K=12 env (coverage vs hit-rate tradeoff)

## decode kernel profile (inkprof4, graph-trace=node)
- Per token: gemv_moe ~75ms (346us x216), gemv_dp4a ~40ms (93us x433), ccopy ~23ms, stage ~2ms. GPU ~100% busy.
- Decode = ~14 chunked graph launches/token (GGML_CUDA_GRAPH_CHUNK_SIZE default 512; chunk0 test = no gain), ~12 streamSyncs, ~10 small memcpyAsync. ZERO eager launches.
- r8-split e element-parallel gemv_moe PIORARAM; r4+tkb mantido (~346us/call, leve melhoria sobre 369).
- Margin 190->100MiB: 11 layers cached. K16 > K32/K48/K64.
- nsys decode-only trick: --delay/--duration p/ pular load+encode; kernels em graph precisam --cuda-graph-trace=node.

## SESSÃO 3 estado (commit e6914bf + margin 100)
- RUN: cd ~/inkwell && cargo build -p inkwell-cli --release && LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin GGML_PASCAL_GRAPHS=1 INKWELL_GENS=8 ./target/release/inkwell ~/models/inkling --run
- Bench: bash opt/rexl3_bench.sh <cfg> <N> → opt/ledger.jsonl
- Profile: ~/lfm-gtx1060-bench/cuda-12.6/bin/nsys profile --trace=cuda --cuda-graph-trace=node -o /tmp/p ./target/release/inkwell ... ; stats --report cuda_gpu_kern_sum|cuda_api_sum. GGML lib = src-v2 (build-inkwell-dp4a).
- CG_PROF: INKWELL_CG_PROF=1 → eval/drain por step (eval=CPU dispatch, drain=GPU exec).
- ATUAL: ~150 ms/tok mediana (variância desktop ±20). Ledger: exl3-cache-k16~149.5, m100~164(noise), chunk0~159, moe-r8/elem=PIOR, ctx4k~153.
- GPU-bound confirmado: per-tok = gemv_moe 75ms + dp4a 40ms + ccopy 23ms + stage 2ms + resto ~25ms.
- gemv_moe: kernel atual = r4-split (64thr/tile, 4 tiles/blk, 256 thr, grid tn/4 × nslots), tk batch ×4 por sync, acumulador afim KINV*bs+KOFF. Launches <<<gg/gd,256>>> com dim3(NG/64,nslots) e (ND/64,nslots) — NÃO mudar o /64 (kernel usa blockIdx.x*4+g). Staged path NÃO foi convertido (ainda roda o mesmo kernel r4 — consistente).
- dp4a (dense exl3 gemv): ainda layout velho 4 tiles×16thr×64thr, r-loop serial 16 — mesmo padrão do moe ANTES do r4. Aplicar o mesmo r4/tkb ali é experimento pendente.
- PRÓXIMA ALAVANCA: markov prefetch — hist[layer][8] ids do token anterior; na visita da camada L, prefetch na side-stream p/ cache da camada L+1. Cuidado: custom ops SÃO capturados no CUDA graph → side-stream requer event fork/join capturável; ccopy ~0.7ms/miss, ~30 misses/tok.
- Cobertura cache: 11/12 pinned (moe=12 sobra por ~30MB). Margin=100MiB em runner.rs ~linha 847.
- Env knobs: INKWELL_EXL3_CACHE_K(default 16 agora), INKWELL_MAX_CTX, INKWELL_MOE_SKIP(piso não-MoE 84ms), INKWELL_CG_PROF, INKWELL_MOE_STATS(only non-exl3 path), GGML_CUDA_GRAPH_CHUNK_SIZE, GGML_SCHED_DEBUG.
- git stash #0 = versão "element-parallel" REJEITADA (~190ms).
- Meta: 15 TPS=66.7ms (avisa e continua), depois 30 TPS=33ms, depois spec decode (mtp.safetensors markov_rank=256 block=8).
