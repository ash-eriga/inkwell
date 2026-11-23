# HANDOFF — Inkwell REXL3 port (overnight session)

## Missão
**60 ms/tok (~16.7 TPS) no caminho NÃO-spec primeiro**, depois speculative.
Spec já implementado/verificado mas não paga (verify ~500ms, accept ~14%).
REXL3 obrigatório (sem quants menores). Avisar o usuário em PT ao bater 15 TPS estáveis —
NUNCA parar. Mantenha texto coerente (`generated text` deve parecer português/inglês
real, ex: " the capital of France.\n\nThe ").

## Estado atual (HEAD = c9613d6+oracle-dumps)
- **NON-SPEC PLATEAU: ~97-100 ms/tok (~10.4 TPS)**, golden coerente " Paris. The capital of
  the United States is Washington, D.C. ...". Todos os levers de numérica exata esgotados:
  ~800MB de pesos/tok a ~10GB/s efetivo (latência de VRAM, não PCIe nem BW).
- **Spec decode (DSpark/mtp.safetensors) IMPLEMENTADO e VERIFICADO**: driver em
  `inkwell-cli` (INKWELL_SPEC=mtp), drafter em `crates/inkwell-run/src/draft.rs`,
  feats taps em runner.rs (FEAT_IDX=(5,23,41)+final-norm), ctx-K/V de 3 camadas,
  markov head (state 256 + rnn + hidden_out + unembed + argmax cap30).
- **Aceitação medida ~14% (1.98 tok/round, 101 rounds)** — flat com ctx cheio (~200 pos).
  Verificado por ORÁCULO NUMPY (/tmp/mtp_oracle.py + /tmp/mtp_bisect.py): dequant de todas
  as 29 matrizes EXL3 por unit-vectors (mesma matemática do kernel), forward do draft
  replicado — TODOS os estágios dentro de ruído fp32 (GPU é determinístico a 3e-4; diffs
  GPU-vs-oracle são propagação de ~2e-4 base). Draft tokens são plausíveis-mas-difíceis;
  vLLM/B200 reporta accept_len 3.5-5.2 em prompts naturais — nosso texto é repetitivo mas
  semanticamente difícil (nomes de capitais). ngram k=4..6: 1.19-1.21 tok/round < MTP.
- **Spec economics**: verify ~470-570ms + draft ~46ms por round → ~3.5 tok/s vs serial
  10.3 tok/s = 3x PIOR. Verify dominado por `inkwell_exl3_gemv_moe_ntok` (~2.6ms/call,
  ~60 calls/round): a união de experts deduplica pouco (8 tok × 8 experts de 128 → ~60/64).
- Bug fix real: bf16_to_f32 em draft.rs (shift precedence) — commit d6d903e; down-proj MoE
  lendo expert-0 p/ todos os slots — fix anterior ("golden" antigo era saída bugada).
- **LEVER REAL p/ 60ms**: gemv_moe serial lê ~7MB/expert a ~12GB/s efetivo — VRAM real
  ~150GB/s → kernel é instr/latency-bound, headroom ~4x. Se gemv_moe subir pra ~40GB/s:
  MoE 48ms→~15ms → total ~65ms. ANTES de mais spec, otimizar esse kernel.
- Próximos: (1) otimizar inkwell_exl3_gemv_moe (smem/prefetch/occupancy — o ntok compartilha
  o mesmo decode interno); (2) kd-adaptive draft len (theta=0.4 cumprod, NÃO implementado);
  (3) variable-length verify (step_block é fixo-8).

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

## SESSÃO 3b — PREFETCH WIP (não commitado)
- dp4a-r4 (256thr, r split 4-way, tk batch x4, stride 80): ~150-157, NEUTRO, golden ok. Commitável.
- ccopy split gridY=4: neutro.
- PREFETCH design final (in-flight): resolve-aware bank. Events in-cb ilegais em capture (901/905). Novo design: host-side prefetch pós-compute na side stream; bank dedicado/layer + ptag+pready seqlock; resolve kernel ganhou args (ptag,pready) e map pode codificar EXL3_PF_BASE+p; ccopy ganhou arg bank.
- kernels.cu: ink_launch_exl3_prefetch2 + pf_resolve/pf_flag/pf_tag + moe_pcopy ADICIONADOS; resolve/ccopy JÁ assinados novos; launch moe_cached JÁ passa args novos.
- lib.rs: decl ink_launch_exl3_moe_cached extern JÁ atualizada c/ ptag/pready/bank + prefetch2 decl adicionada; falta wrapper exl3_moe_cached_async (args novos) e apagar exl3_prefetch_async/wait órfãos.
- moe.rs: Exl3Prefetch struct existe {hist,map_p,miss_p,sd_g,u,d,ev_done} — REESCREVER p/ {hist,bank,ptag,pready,pf_map,pf_miss}; cb ainda chama exl3_prefetch_wait/exl3_prefetch_async/moe_idlog — REMOVER chamadas órfãs, MANTER memcpy_ids8_async(pf.hist).
- runner.rs: cache ctor tem prefetch: Option<Exl3Prefetch>; precisa alocar bank=8*3*slot bytes; host call site: após ink_sched_compute em step(), loop moe_args_exl3 com cache → cu::exl3_prefetch2_async(descs, hist, tag, bank, ptag, pready, pf_map, pf_miss, 8, k, slot).
- baseline bom ~150ms/tok mediana.

## SESSÃO 3c — regpref pipeline (~135 ms/tok)
- DESCOBERTA: fill global→smem serializado por __syncthreads era o custo real do gemv (remover math todo = neutro).
- gemv_moe + gemv_dp4a: prefetch do próximo tk-batch em REGISTRADORES durante o math, store p/ smem depois do sync → 155→~135 ms/tok mediana. Golden.
- Prefetch de experts host-side (side stream, bank/ptag/pready) implementado e correto mas ~neutro: PFSTAT hits ~6/224 visitas — predição fraca. Mantido (machinery pronta).
- Crashes esporádicos no load = OOM por zumbi ou VRAM apertada; i1 do bench morre às vezes.
- Próximos: (a) mesmo regpref no had_in/had_out stage loops; (b) fusionar had_out+situ; (c) ccopy serial ainda ~20ms.

## SESSÃO 3d (~131 ms/tok, HEAD 683b4f6)
- regpref pipeline COMMITADO: ~155→131 mediana.
- hadamard warp-shuffle (had128_warp): neutro-positive, commitado. g/u had fused z=2 (had_in/had_out), gemv sequencial (fused z=2 quebrava — misterioso).
- DIRECT misses (sdesc→host zero-copy): ~151ms, PIOR — PCIe streaming no gemv perde pra cópia.
- tk8 batch: regs demais, 142ms — rejeitado.
- PFSTAT: ~1-3 misses/visita; bank prefetch ~2% hits — neutro.
- gemv_moe ~instruction-bound (window+dp4a+fma por elem); DRAM floor é ~10x menor.

## SESSÃO 3e — PROFILE REAL (nsys) — muda o mapa
HEAD: 93f7a3d (~128-133 ms/tok). DESCOBERTA (eager + graph trace):
Por token ~122ms: **~96ms dentro de 14 graph execs + ~13ms host (build graph ggml em step()) + ~13ms launch/tail**.
Kernels em eager (por token, ÷8 runs): gemv_moe 47.6ms (216×220µs), dp4a 33.6ms (433×77µs),
ccopy 19ms (25×~750µs), stage 8.2ms (6×1.37ms), pcopy ~2ms, mul_mat_vec 6ms, had ~4ms, attn 1.3ms.
→ as cópias PCIe totais ~29ms/token são o 2º maior custo; gemv_moe instr-bound ~10 inst/elem.
- Kernel-table NÃO cobre kernels de graph-replay (ficam invisíveis no ncu/nsys kernel sum).
- Rejeitados: tk8 (142ms, regs), window i-map hoist (144ms, regs), sx tail-compute (neutro ~133).
- Aceito: `(i0+1)%nw` → select (~128ms). Commitado.
- BUG RESOLVIDO do dia: replace() sem escopo zerou sx do kernel dp4a → "SUL CONTROL". Cuidado com replaces amplos.
- Próximos alvos por payoff: (1) cópias PCIe 29ms — menos misses (pin quente) OU cópia mais rápida (>2.6GB/s);
  (2) gemv_moe 47.6ms instr-bound; (3) 13ms host build/token (ggml sched — difícil); (4) ~4ms graphLaunch (293µs×14).
- Rejeitado: ccopy nparts 16→32 / 256thr (~155ms — sem ganho; PCIe já satura).

## PRÓXIMAS ALAVANCAS (ordem de payoff)
1. **Unir os 14 graph chunks em 1-2** (~13ms host gaps + ~4ms launch): o callback moe_cb_exl3 já emite
   tudo device-side (resolve+ccopy+gemv+had+wsum); a quebra é do lado ggml (nó host/custom-op por visita
   + readback logits). Estudar ink_sched em crates/inkwell-ggml para fazer a op MoE "stream-capturable"
   ou fundir visitas num único nó custom por camada.
2. **gemv_moe 47.6ms** instr-bound: variantes fracassaram por regs. Tentar: 2 tiles/blockDim=512 com
   smem xhb compartilhado (metade dos blocos, mesma lógica); ou descarte antecipado de pfb words nulos.
3. **dp4a 33.6ms** (433 launches): split-K já divide; testar maxrregcount local p/ subir occupancy.
4. **ccopy 19ms**: já pinned; mais paralelismo não ajudou (testado 32×256 ~155ms). Caminho: menos misses
   (warm cache com experts mais quentes por camada no load — hist já existe em moe.rs) ou K>16 nas
   camadas de maior miss (VRAM margin ~100MiB → talvez +1-2 slots em 2-3 camadas).
5. **13ms build/token**: ggml graph build em step() — custo fixo; reduzir via ctx reuse mais agressivo
   seria projeto grande; deixar por último.
Mediana atual ~128ms. Meta 15 TPS = 66.7. Alvo 33.
- Rejeitado: GGML_CUDA_GRAPH_CHUNK_SIZE 0 (~145ms — instantiate serializa; chunks pipelinem) e 2048 (~131, neutro).
  Os 14 chunks são por contagem de nós (default 512) e o pipeline instantiate|exec já é eficaz.

## SESSÃO 4 — CORREÇÃO DE MEDIÇÃO (crítico!)
`decode 8 tok in Xs` incluía `Tokenizer::from_gguf` (~280ms) → **todas as medidas tinham +~35ms/token**.
Real agora: ~105.6ms/tok (~9.5 TPS). Baseline real era ~165 não ~200.
Timers: PROF build~3ms compute~97ms logits~0.3ms; stepwall~105ms; loop main ~1ms.
Fixes commitados: top() linear + timer antes do tokenizer + bench aceita unidade ms.
Ledger histórico: subtrair ~35ms de tudo.

## SESSÃO 4b — pós correção de medição (~96ms = 10.4 TPS)
- **LFU eviction (cnt[128] por expert em device): commitado** — ~96 vs ~99 controle.
- Cobertura real: 24 camadas MoE; on_dev 0-12, cached 13-19 (7×k16), staged 20-23.
  Free na hora dos caches ≈708MiB; downstream precisa ~250MiB → orçamento ~458MiB:
  k16→7, k12→9 (sem ganho ~97.5), k10→10 (~123 com margens erradas — instável).
- Rejeitados: __launch_bounds__(256,6) (113, spill), i_r hoist (neutro), xhb→regs xr[16] (neutro),
  ccopy 32×256 (155), chunk 0/2048.
- **PARADOXO RESOLVIDO parcialmente**: fused gemv z=2 quebrou de novo (texto em branco).
  O write `t[slot*n]` com n=NG gera layout p-major; had_out lê s-major — análise dizia
  impossível ser golden, mas é. Provável subtileza nos offsets — NÃO mexer sem debug real.
- pkill gotcha: usar `pkill -9 -x inkwell` (o -f casa com a própria shell).
- nohup: rodar bg com `nohup env ... < /dev/null &` — subshell (cmd &) morre com a call.

## INCIDENT (2026-11-23) — recovered
- `git checkout` wiped uncommitted deltas in src-v2/ggml-cuda.cu → runtime broken (illegal memory access).
- Lost delta reconstructed + COMMITTED (src-v2 `5a99456`): (1) `GGML_OP_CUSTOM` in `ggml_backend_cuda_device_supports_op` → tensors land on device; (2) `case GGML_OP_CUSTOM` in `ggml_cuda_compute_forward` → calls `p.fun(dst,0,1,userdata)`; (3) op-seq `get_key` (hash of op+type+src-count, NOT ne/nb); (4) `ggml_backend_cuda_stream` shim; (5) PASCAL_GRAPHS gate on graph-eval path; (6) CUSTOM exclusions in `graph_update_required`.
- Rule: ALWAYS commit src-v2 changes immediately.
- State after recovery: ~100 ms/tok golden at HEAD c292e3f.

## nsys breakdown @~100ms/tok (per token, GPU busy ~115ms)
- exl3_gemv_moe: 43.6ms (med 323µs, ~144 launches)
- exl3_gemv_dp4a (dense): 33.7ms (med 86µs, ~433 launches!)
- exl3_moe_ccopy: 14.8ms; exl3_moe_stage: 8ms (uncached L21-23)
- had_in/out_moe+had_in/out: ~8ms; attn_decode 1.3ms; misc ~5ms
- Cache hits REAL ~80% (PFSTAT hits= só prefetch-bank; misses reais ~1.5-2/visita). margin=50 → cached moe13-20, neutro.
- REJEITADOS A/B: nky-split moe gemv (108/113ms), direct-loads sem smem (131ms), prefetch-depth-8 (neutro, mantido).
- gemv_moe não é starved de blocos (~288) nem math-bound (~5Gops); suspeita: latência das chains smem/syncs.
- Não testado: fundir had_in/out no gemv (economiza 2 launches+roundtrip), batchar os 433 dp4a (proj pequenas agrupáveis?), exl3_moe_stage p/ L21-23 via ccopy k=8.

## Cache tuning results (2026-11-23)
- WIN: k=25 + margin=50 defaults (commit) → ~96ms/tok. hits>coverage.
- Rejected: k=32 (fewer layers), k=8 (misses), k=40+noprefetch (103ms), k=32+noprefetch (98.8).
- Prefetch bank ~0 hits (verified via PFSTAT) — dead weight ~32MB/layer, could be removed to free VRAM.
- Uncached layers (19-23 at k=25) still use moe_stage serial (~8ms/tok): try routing them through ccopy with tiny cache, or shrink slot to per-proj granularity.
- Dense gemvs ARE device-resident (exl3_w=145, dev=308MiB) — the 34ms is VRAM latency too, NOT PCIe. PCIe cost = only host expert misses (~15ms).
- ncu blocked: no sudo/CTRPERM on this box.

## Estado 23-11 ~fim do dia
- ~96ms/tok (~10.4 TPS) commitado. Breakdown: moe gemv 44ms VRAM-latency, dense gemv 34ms VRAM-latency, copies ~15ms, misc ~8ms.
- Variance ±30ms entre runs (desktop apps ON) — todo A/B precisa ≥3 samples ou fica ambíguo.
- IDEIA não testada (maior EV): os 433 launches densos/tok incluem várias projeções pequenas por camada (n=384-3072). Agrupar projeções independentes num único custom op/desc-set (padrão do fused gu+up) corta chains serializadas. Ver runner.rs exl3_w usage e os custom_4d calls ~894/1390.
- gemv_moe latency ideas esgotadas no formato atual (smem pipeline + prefetch já otimizados; k-split/double-buffer/direct-loads rejeitados).
- Próximo grande passo teórico: int8 xh → dp4a int8×int8 no inner loop (muda numérica, medir EAR).

## Nota de continuidade
- Dense exl3 sites: proj() @runner.rs:889 (chamado por camada p/ q/gate/rel via proj0_cat concat já feito), head @1390. proj0_cat já concatena q+gate+rel — fusão adicional: o-proj+rel_logits+attn_sink independentes poderiam concat no load (mesmo k) → menos launches.
- Caminho p/ batching: em exl3_of() aceitar lista de bases e concat trel na dim n (igual proj0_cat faz para 3 já).
- Zumbis: sempre `pkill -9 -x inkwell; nvidia-smi --query-compute-apps` antes de run.

## 2025-11-23 manhã — estado + MTP
- Estável ~96-102ms/tok (~10 TPS), golden. HEAD inkwell 8bf80ba+, src-v2 5a99456.
- Teto sem spec estimado ~60-70ms (450MB pesos/tok × latência VRAM+PCIe). 15 TPS sem spec improvável.
- mtp.safetensors baixado (236MB) p/ ~/models/inkling/. Drafter = 3 camadas REXL3 (mesmo formato trellis/suh/svh/mul1 → reusar exl3_of + gemv_dp4a) + markov_head (rnn_proj 5120→768, hidden_out 256→2304, markov_w1 [201024,256] logits) + mask_embed, fuse_weight [4,4], norm.
- mtp_config: block_size 8, markov_rank 256, markov_head_type rnn_hidden, target_layer_ids [5,23,41,48] (visit-space: última visita de cada attn_res block), local_layer_ids [0,2] (SWA), sliding_window 1024.
- Plano: (1) parsear mtp no runner (exl3_of path), (2) drafter fwd 3L + markov head, (3) loop draft-8 → verify-prefill no target, (4) medir acceptance.

## MTP/DSpark — design (ref: vllm vllm/models/inkling2/nvidia/dspark.py)
- Draft: 1 passada paralela no bloco[8]: x[0]=norm(embd(anchor)), x[1:8]=mask_embed; 3 camadas (RMSNorm→attn RoPE com ctx-KV de features do target via kv_proj_ctx→MLP); contexto = fuse_weight softmax das hidden states do target nos visits {5,23,41,48} via dflare_norm; Markov head: logits = fold_weight[201024,256] @ latent + fold_bias + base, latent via rnn_proj[5120→768] + markov_gate seq/step, prev-token via markov_w1[vocab,256].
- fold_weight = lm_head @ hidden_out (foldar no load — head é exl3! desquant ou aplicar em 2 estágios).
- Verify = step_block(ids[8], pos): mesmas ops do step() mas ntok=8:
  * dense exl3 gemv BATCHADO (ntok=8 numa leitura de peso) — ÚNICA amortização que importa (~34ms/token → ~34ms/bloco).
  * attn por token (8 máscaras/rel_pos separadas), conv sequencial, MoE ids[8] por token (~×2 tráfego de expert), KV set_rows idx[8].
  * KV garbage pós-rejeição é auto-mascarado (mask limita a pos+1) → ok; conv_state snapshot+restore se aceitação parcial (~5.3MB memcpy).
  * CUDA graphs: verify graph = seq de ops diferente → cache key separado ok.
- Estimativa: verify-8 ~130-160ms/bloco; aceitação a~4 → ~15-25 TPS.

## 23-11 noite — corrida p/ 60ms non-spec
- HEAD 65480cf+ ; ~97ms/tok estável golden. Breakdown real (nsys run inteiro): gemv_moe ~45ms, gemv denso ~34ms, ccopy ~13ms, moe_stage ~10ms (2-3 camadas sem cache), misc ~10ms, build host ~2-14ms.
- DESCOBERTA: moe lê ~430MB/tok (8 experts × ~2.2MB × 24 camadas) a ~9.5GB/s — mesmo throughput do denso (~10GB/s). O problema NÃO é o kernel ser ruim: ~800MB/tok a ~10GB/s efetivo. Teto teórico a 15GB/s efetivo ~55ms — quase impossível chegar a 60 sem spec.
- Feito: split hit/miss nos gemv/had_in moe (hit-pass ANTES do ev_join → overlap com bulk copy) — correto, ~neutro a k=25 (misses raros). Commitado.
- Rejeitados: wk||wv pair 2-stream (SM contention + event overhead → pior), NKY sweep, batch4→8 denso (quebra xhb fill), DEV_MARGIN 800/500, MAX_CTX 512/1024 (neutro).
- bs range = 0..1005 → int8×int8 (xh,bs) inviável sem perder 3 bits de bs — int8-xh puro morto.
- Restam só: (a) fusão de projeções independentes no load (concat trel na dim n — padrão proj0_cat), ~2-4ms; (b) int8-xh com bs 2-termos (numérica muda); (c) spec decode (o caminho real).

## 23-11 noite pt2 — step_block verify CORRETO (21620b9)
- VTEST baseline was stale (compared pos-1 logits): now snapshots conv, runs
  step() at same pos, restores, then step_block — real comparison.
- Bug: tau8 faltava o bias +1.0 de scale_bias nas layers full (il%6==5) —
  q/r escalados por alpha*ln em vez de alpha*ln+1 → divergência nas layers
  full. Fixado: row0 max|d|=0.00036, argmax match.
- snapshot_conv/restore_conv (host) adicionados p/ spec rollback.
- in_buf 64MB→8MB. b_*/s_lout* dumps per layer p/ bisect futuro.
- verify-8 ~0.9s wall (sem cuda-graph + ops por-token serializadas) —
  precisa acelerar p/ spec pagar; próximos: medir breakdown, cuda-graph o
  bloco, batch MoE/router por token.
- WALL non-spec confirmado ~97ms: ~800MB/token a ~9-10GB/s efetivo uniforme
  (VRAM-latency), todas fusões menores rejeitadas em A/B — caminho real = spec.
- verify-8 compute=807ms ≈ 101ms/tok: per-token ops dominam (moe ~45ms/tok
  ×8 = ~360ms). Próximo lever p/ spec pagar: MoE batched por union de
  experts (rotear 8 tokens juntos, gemv ntok por expert único) + graph do
  bloco. Serial per-token moe reaproveita pesos só por token.
- DESIGN moe-ntok (verify): por camada, rotear 8 tokens (router mmq por
  token já existe), montar união de experts no host, e por expert u fazer
  had_in(xh dos tokens seletores) + gemv_ntok(w_u) sobre as linhas
  selecionadas + had_out×weight, acumular no out por linha. Reusa
  exl3_ntok_cb com x = linhas selecionadas compactadas [k,#sel]. Precisa:
  readback dos ids por token (argsort já roda em dev — copiar pra host ou
  kernel de gather), e mapear expert→desc (descs_* pinned já indexa por
  expert id). ~30-60 experts únicos vs 64 leituras → ~2x no gemv_moe.
  DEPOIS: cuda-graph do step_block (mesma maquinaria chunked do step()).
- MATH spec p/ MoE-Inkling (revisto): união de experts ~40-60/128 por 8
  tokens → moe reads ~300MB/token (vs 422 serial) = só ~12-18ms/token de
  economia. verify-8 807ms ≈ 8×97: per-token ops são o custo, não há o
  3×-por-leitura que o LFM denso teve (todo peso compartilhado lá).
  Spec ainda paga SE verify < ~500ms E drafter acertar ~4+/8.
  Levers restantes p/ verify: cuda-graph do bloco (overhead ~13ms×?),
  batch conv/attn/router por token (small ops ×8 serializam).

## verify union-MoE economics (measured 2026-11-23)
- Expert overlap across 8 verify tokens: ~20.3 unique experts/visit (64 selections) → 3.2x dedup possible. b_ids{il}t{j} dumps added behind INKWELL_DUMP.
- Verify kernel budget (nsys): gemv_moe ~370ms (of ~840ms block), ccopy ~100, stage ~90, dp4a_ntok ~50, had ~40, misc ~190.
- Union-batched plan: expand (expert,consumer) pairs on GPU; new gemv_moe_ntok reads each unique expert once for ≤8 consumer xh vectors; wsum groups by token. Est verify ~450ms.
- CUDA graphs confirmed DEAD on Pascal: is_enabled() false (arch gate) even with GGML_PASCAL_GRAPHS; patch attempt → cudaGraphInstantiate OOM. Reverted.
- Non-spec serial remains ~97ms floor.

## CORRECTNESS BUG FIXED (2026-11-23)
- `inkwell_exl3_had_in_moe` read `x[e]` shared across slots; for the DOWN chain x=hid ([8][1152]) → slots 1-7 used expert 0's gate*up output. Bug since original commit 72a02e1.
- Fix: `x[(descs2 ? 0 : slot*k) + e]` (down chain only). Old golden " the capital of France." was the BUGGED output; new golden = " Paris. The capital of the United States is Washington, D.C. ...".

## 2026-11-23 — UNION MoE COMMITTED (79a567e), spec driver WIP
- Non-spec plateau CONFIRMED ~97ms/tok (~10.3 TPS). User: "60ms primeiro, depois speculative" — non-spec esgotado; seguindo spec path (user avisado).
- verify-8 (step_block): CORRECT (max|d|=0.00014, argmax match) and now ~463-492ms (was ~840ms) via union-batched MoE:
  - kernels.cu: `inkwell_exl3_union_expand` (dedups 8x8→~20 unique experts, emits eids/euni/etok/estart/epos), `inkwell_exl3_gemv_moe_ntok` (gemv per expert applied to all consumer x-vecs, grouped output), `inkwell_exl3_expert_wsum8`, launcher `ink_launch_exl3_moe_ntok`.
  - had_in_moe gained optional `euni`/`etok` params (miss by union idx; x offset by token).
  - resolve fix: nexp>k → map=EXL3_DIRECT (was map=-1 → wild sdescs → illegal access). NEEDED upstream too since serial nslots can exceed k if k<8.
  - moe.rs: VScratch (64-slot scratch) + `moe_cb_exl3_ntok`; runner.rs layer_blk: batched router (logits8 [128,8] → sigmoid → +pb → argsort_top_k → cont → ids3[8,8,1] get_rows → w8 [8,8] norm+scale) + ONE custom_4d [n_embd,8] + convolve per token.
  - ids8 MUST be `t::cont` (argsort_top_k returns a view).
- Layout: x8/out8 [n_embd,8] col-major → row t at offset t*2304. desc tables indexed by expert id [128][4] u64.
- INKWELL_MNTOK_DBG=n env left in launcher: 1..5 + 99 = stage markers with cudaGetLastError prints. Keep for now (dbg=0 inert).
- Kernel cost after union: gemv_moe_ntok ~1.5-1.9ms avg/launch (~120 launches/verify), ccopy still big (PCIe misses), dp4a_ntok ~50ms/verify.
- Spec driver exists: crates/inkwell-spec (SpecLoop/NgramDraft/accept_len/commit). NEXT:
  1. conv rollback WIP (IN PROGRESS, uncommitted): conv_step gained `float* snap` param (writes post-shift state to snap[3*region_off+3c+r]); ink_launch_conv_step + conv_step_async gain trailing `snap` arg — lib.rs edit applied to ffi decl only (see diff); conv_step_async body still needs `, snap, stream()` + caller `a.snap`.
  2. ConvStepArgs needs `snap: *mut f32`; convolve() takes snap ptr (null in serial, conv_snap[slot]+j*3*n_sconv*4 in block). Allocate conv_snap[slot][8][3*n_sconv] lazily on first step_block (DevBuf per conv_state entry).
  3. rollback_conv(emitted): if emitted<8 memcpy snap[slot][emitted-1]→conv_state[slot] (D2D) for every slot. KV arenas need NO rollback (stale slots overwritten next round).
  4. main.rs spec loop (env INKWELL_SPEC): committed `id`@pos; draft = ids[1..7] via NgramDraft/MTP; step_block([id,d1..d7]) → rows j predict pos+j+1; verify[j]=argmax(row j); accept d_{j+1} iff verify[j]==d_{j+1}; emitted=n+1 (n accepted + bonus v_n); rollback_conv(emitted); pos+=emitted; id=emitted last token.
  5. Verify uses K=10 cache during tests (VRAM tight: K=15 OOM'd on the batched graph ~110MB extra vs serial). Desktop apps on.
- Zombie cleanup: `pkill -9 -x inkwell` before every run.

## DSPARK DRAFTER — algoritmo completo (ref: ~/megakernel-zoo/engines/inkling-2-mini_b200/offline-c/engine/draft.py — implementação funcional do MESMO mtp.safetensors)
Config (config.json mtp_config): markov_head_type=rnn_hidden, block_size=8, 3 layers, H=2304, nh=? kv=?, hd=128, markov_rank r=256, window=swa, target_layer_ids=[5,23,41,48] = visits {5,23,41} (input do layer_blk) + 48 = final-norm output. mtp shapes: q_proj 2304→3072, k/v_proj 2304→1024 (nh*hd=3072→24 heads, nkv*hd=1024→8 kv), o 3072→2304, gate/up 2304→9216, down 9216→2304, k/v_proj_ctx 2304→1024 cada.
PESOS: tudo EXL3 (trellis/suh/svh/mul1) menos: norms bf16, fuse_weight[4,4], mask_embed[2304], markov_w1[201024,256] bf16, rnn_proj.bias[768], hidden_out.bias[2304]. Normas/embed/rope/softmax em f32; GEMMs bf16 (nós fazemos f32 — ok, drafter só propõe).
FOLD: hidden_out é EXL3 [256→2304]+bias; fold_w[V,r]=unembed@hidden_out.weight (f64→bf16), fold_b[V]=unembed@hidden_out.bias — foldar 1x no load (~200058×256).
CICLO (após verify de 8 rows com tgt[8]=argmax por row):
- accept: a = #drafts líderes iguais (vtok[1..]==tgt[..]); c_new=c+a+1; anchor=tgt[a]; frow=a; emit=tgt[0..a+1).
- ctx_append: p/ CADA row j do verify, feats[j,4,H] (visits 5,23,41 inputs + finalnorm out) → p/ cada layer l: u=_mix(feats[j],alpha[l])→dflare_norm→kv_ctx proj→k_norm_ctx+RoPE(pos=c+j)→kc[l][pos], vc[l][pos]. Rows>a reescritas no próximo ciclo (draft só atende pos<c_new).
- propose: x=[rms(embd(anchor))*embed_norm; mask_embed×7]; pos=c..c+7; 3 layers: in_ln→qkv→q_norm+rope,k_norm+rope (rope halves: x[:64]cos - x[64:]sin, GPTJ?) → attn: q[8,24,128] vs keys=kc[l] (ctx rows pos<c + block rows 8) mask (global: pos<c; local layers: pos>=c-window) + softmax → o_proj→+res→post_ln→MLP(silu(gate)*up→down); residual separado por _add_rms(x,residual,w): normed value → próxima op, raw soma → residual.
- hid = add_rms(x,residual,norm)[1:] [7,2304]; f=_mix(feats[frow],alpha[3],m_norm) [2304].
- markov: inp=cat(hid[i],f)[4608]→context[i]=inp@w_ctx[2H,3r]+rnn_b [7×768]; base[i]=hid[i]@unembed [7,V]; state[256]=0; prev=anchor; loop i=0..6: raw=context[i]+cat(state,w1[prev])@w_rec[512→768]; gt=sig(raw[:256]); cand=tanh(raw[256:512]); state=gt*state+(1-gt)*cand; latent=tanh(raw[512:768]); logits=cap*tanh((base[i]+fold_b+latent@fold_w^T)/30); prev=argmax→draft[i+1].
- kd adaptativo: pconf=top1 prob/step; kd=maior k com cumprod>=theta(0.4) → próximo verify usa kd+1 rows.
CUSTO estimado Pascal: base GEMM lê unembed 922MB bf16 (~90-120ms); fold_w 7×102MB (~70ms); 3-layer tr ExL3 (~uns ms); total draft ~150-200ms. Ciclo ~550-600ms → precisa ~5-6 tok/round p/ ganhar de 97ms serial. Verify ~400ms hoje (~3.5GB/s efetivo vs 1.4GB de pesos → chão ~140ms): OTIMIZAR verify é o lever crítico.
- feats capture: feats[j] = x_in (residual) no INÍCIO de layer_blk p/ vi ∈ {5,23,41} + normed output final — por row j. step_block deve expor feats[8][4][2304].

## 2026-11-23 late — spec verify is GPU-bound; moe_ntok is the target
State: serial ~97ms/tok golden `" Paris. The capital of the United States..."`.
Spec works mechanically (accept 25-47% noisy, 2.2-4.3 tok/round) but ~4.4 tok/s
< serial — verify ~570ms, ~93% GPU-busy, ~70% = inkwell_exl3_gemv_moe_ntok
(latency-bound ~10GB/s effective; union ~60/64 experts ≈ serial bytes).
CUDA graphs confirmed working on block path (GGML_PASCAL_GRAPHS + pnexp patch).
NEXT: (a) moe_ntok inner loop efficiency (vectorize tile_base reads — pfb[]
loads are 4B/thread; try uint4); (b) per-row swarm ~50ms minor; (c) draft att
residual 0.28 vs oracle — likely bf16-reference rounding; verify by comparing
draft logits vs reference api.py if needed; (d) kd-adaptive draft len.

## Spec verify = PCIe-bound on expert reads (root cause found)
- `descs_g/u/d` are host-PINNED descriptor tables (zero-copy); a miss expert's
  tile_base points into the host weight blob → `gemv_moe_ntok` reads weights
  over PCIe inline at ~8-10GB/s.
- Verify union ≈ 60/128 experts PER LAYER × 48 visits ≈ ~4GB over PCIe ≈
  ~460ms of the ~570ms verify (moe_ntok ~70% of kernel time). Cache_k=4
  hits almost nothing; K≥8 OOMs (draft+graphs+caches).
- Structural: serial reads 8 experts/layer/token → same ~4GB per 8 tokens.
  Spec can only win via (a) expert dedup across rounds in VRAM, or
  (b) emit ≥ ~6.
- NEXT LEVER (chosen): GLOBAL expert pool in VRAM (~256-384 slots × ~2MB
  ≈ 512-768MB) keyed by (layer,expert), shared across all layers+rounds —
  replaces per-layer k=4 caches during spec. Expected hit ~40-60% →
  verify ~300-400ms → spec ≈ breakeven at emit 5, wins at 6.
- Serial path stays on per-layer caches (K=25 optimal there).

## Spec verdict: PCIe-bound, cannot beat serial on this model+GPU
Verify ≈ union ~55 experts × ~2.25MB × 48 visits ≈ 5.9GB over ~10GB/s PCIe
≈ ~570ms — the byte count is the wall, not the kernel. A VRAM pool can't
cover the working set (~218MiB free). Spec floor ≈ (430+draft)/emit; even
perfect accept (emit 8) only reaches ~76ms/tok; realistic emit ~3-4.3 loses
to serial 97. Options forward: (a) serial-side expert-byte cuts (~48ms of
the 97 is the same PCIe wall — hot-static residency ~200MB ≈ ~5%), (b) lazy
verify (~140ms/tok floor — still loses), (c) call spec done mechanically.


## 2026-11-23 late — spec-decode deep dive (definitive bottleneck)
- Union size nu ≈ 8-18 (median ~12), NOT ~60 → expert bytes/verify ≈ 1.2-1.3GB
- PCIe measured ~11GB/s plain loads, ~12.4GB/s ccopy-gather → bytes are NOT the wall
- `inkwell_exl3_gemv_moe_ntok` is INSTRUCTION/LATENCY-bound (~5GB/s effective):
  pool=40 (all experts VRAM-resident) changed nothing; ~4-5 instr per useful MAC
  in the consumer-FMA inner loop (smem ld + 2 FMA + predicate); ~2.2G instr/launch
  ≈ 1.6ms compute floor; ~1.5ms of it is dead blocks (grid.y=64, nu~12)
- FIXED real bug: union_expand could emit nc>8 → xhb[8][8][16] smem OOB →
  split experts into ≤8-consumer chunks (commit). Fix correctness + crashes.
- Rejected: host readback of nu (illegal in graph capture), slot-loop over
  gridDim.y (serializes, ~2x worse), #pragma unroll (worse), NKY>1 (worse),
  window-16 (err=900 crashes — cause unknown, NOT the nc OOB), sx-hoist
  (fragmentation bug, marginal anyway)
- verify ~550-700ms (gemv ~260ms, dp4a_ntok ~63, had ~140, rest + host gaps)
- spec ~3.4-5.1 tok/s, accept 21-50%; serial ~97ms (~10.3 TPS)
- TO WIN: need verify ~300ms AND emit ~4.3 → gemv inner-loop rewrite is THE
  lever (cut instr/MAC ~2x or pre-decode trellis→fp16 then plain FMA gemv)
