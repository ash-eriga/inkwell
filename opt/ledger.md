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

## 2026-11-23 — global pool experiment + PCIe floor verdict (7fbe1c4)
- Implemented global expert pool (INKWELL_MOE_POOL=N): resolve+claim keyed
  by (arch*128+uid). Fixed pass2 tag missing key_base + cnt[-1] OOB.
- RESULT: pool k=24-32 → verify ~576ms UNCHANGED. Why: union needs ~55
  experts/layer-visit × 48 visits ≈ ~2600 distinct keys/round (~5.9GB) —
  a 24-192-slot pool covers <8% → ~0 effective hits. Pool OOMs >32 anyway.
- VRAM during spec: only ~218MiB free — no room for meaningful residency.
- FLOOR: verify ≈ 5.9GB expert bytes / ~10GB/s PCIe ≈ ~570ms, irreducible
  while experts live in host memory. Spec needs emit≥6.3 to beat serial;
  drafter tops ~4.3-5.3 → spec is structurally bounded ~5-6 tok/s.
- Kernel-level: miss-gemv ran before ev_join (phase-order race fixed) —
  kept for correctness, no perf change.

## 2026-11-23 late2 — moe_ntok inner restructure (kept: ~5-7% verify)
- Transposed xhb to [c][frag][b] + float4 consumer FMAs (decode bs4 once) +
  uint4 (16B/lane) ring fills + xhb tail-row zeroing for correctness.
- verify ~534-613ms (was ~580-650). Small but real. gemv still ~3.4-3.7ms/visit
  → bound shifts further from instr to dependent-chain latency (18 windows ×
  stage→sync→compute→sync).
- Rejected same session: y=32 fixed (nu>32 incorrect), sx-hoist (bug).

## 2026-11-23 (sessão 4 — retomada por agente novo, HEAD ec940d9)
Baseline própria (apps de desktop ligados, `opt/spec_bench.sh`, medianas):
- serial K=25, 48 tok: **88.1 / 89.1 ms/tok (≈11,3 TPS)** — mais rápido que o "~97" do
  handoff; a régua do spec é (verify+draft)/emit < 88 ms.
- spec K=4 (handoff cmd), ANTES do fix abaixo: 4.5 / 4.5 / 5.7 TPS, emit 2.8-3.7,
  verify ~580 ms, draft ~48 ms. 1ª run morreu com OOM no graph instantiate (cache
  come VRAM até sobrar 126 MiB; margin 50 é pouco) — flake conhecido.

### BUG FIX: anchor do spec processado em pos+1 (commit seguinte)
- `INKWELL_SPEC=mtp` chamava `step_block(&vtok, c)` com `c = pos + 1`, mas o anchor
  fica em `pos` (frame do VTEST, do path n-gram e do `Mtp::cycle`). Resultado: o token
  anchor entrava DUAS vezes ("capital of of the Netherlands") e o texto do spec
  divergia do serial no 1º token (" the Netherlands..." vs " Japan is Tokyo..."),
  terminando em loop "Ireland is Dublin ×3".
- Fix: `c = pos`. Depois: 38/39 linhas de verify == serial-96; texto do spec == serial
  (" Japan is Tokyo. The capital of China is Beijing...").
- Aceitação REAL do drafter neste prompt (pós-fix): **21% / 2,45 tok/round** (antes
  "26-38%" medido no contexto corrompido — número inflado).

### VTEST8 (harness novo, `INKWELL_VTEST8=1`): bloco de 8 tokens vs serial
A cada passo serial roda `step_block` sobre os 8 tokens recém-processados (conv
snapshotted 8 passos antes) e compara as 8 linhas com os logits seriais:
- 96 passos → **712 linhas, 0 divergências de argmax**, max|d| típico 1e-4…1e-3.
- MAO: max|d| salta p/ **1,2-1,4** nas linhas logo após tokens raros ("C" de
  "Washington, D.C.") e decai em ~6 tokens. Não é bug de bloco (argmax igual): é
  fp-chaos — um near-tie discreto interno (provável router top-8) vira com ruído de
  1e-4. Consequência: o spec pode divergir do serial quando o top-1 do serial tem
  margem < ~0,2 (visto 1×: " Canada"→" Mexico", marginais a 0,15). Aceitável.

### PERFIL REAL DO VERIFY (nsys, `--cuda-graph-trace=node`, 12 blocos)
Janela ≈ 498 ms, **GPU 98% busy** (não é gap de host). Por bloco:
| kernel | ms | % | chamadas | média |
|---|---|---|---|---|
| gemv_moe_ntok | 358 | 73,5% | 96 | 3,73 ms |
| gemv_dp4a_ntok (denso) | 47 | 9,6% | 283 | 166 µs |
| had_in_moe + had_out_moe | 16,3 | 3,3% | 192 | 104 / 65 µs |
| had_in/out densos | 12,1 | 2,5% | 4528 | 2-3 µs |
| swarm (sgemm, concat, bcast, cpy, conv, attn…) | ~54 | 11% | ~13000 | 2-13 µs |
Entre blocos: ~85 ms (45 ms de kernel: 7× unembed do draft a 1,7 ms + ctx; ~40 ms host).

### CORREÇÃO DO HANDOFF: união real é ~28, não ~12
`INKWELL_DUMP_IDS` (novo) gravou os ids de roteamento reais de 15 blocos: união por
visita **min 13 / mediana 28 / máx 44** (média 28,4). O "nu ≈ 8-18 (mediana ~12)" do
handoff estava errado. Consequências:
- gemv_moe_ntok lê ~2,8 GB por bloco (não 1,15 GB): VRAM 3,2 GB/s efetivos no geral;
  por camada: VRAM-residentes 4,3 ms/visita (~54 MiB → 12,6 GB/s = 8% da VRAM),
  pinned 10-15 ms/visita (~4,5 GB/s de PCIe).
- **O verify faz MAIS trabalho de MoE por token emitido que o serial**: 10,7 G pesos
  / 2,45 tok = 4,4 G por token vs 3,05 G do serial (744 MB). A dedup da união só
  poupa 2,3× (28 de 64 seleções), e o emit é 2,45.
- Pisos: decode ≈ 7 instr/peso → ~10 µs por matriz a 100% de issue; PCIe das 21
  visitas pinned ≈ 1,14 GB ≈ 104 ms/bloco. Logo gemv_moe_ntok tem piso ~130-150 ms
  (não ~55): headroom ~2,4×, não 6×.
- Especulação só vence o serial com kernel no piso E emit ≥ ~2,6 E resto do bloco
  (denso+swarm+host+draft, hoje ~185 ms) cortado para ≤ ~70 ms.

### Amostra pós-fix + aceitação em outros prompts (apps ligados)
- spec fixpos K=4, 4 runs: **4.3 / 3.5 / 3.5 / 3.3 TPS (mediana 3,5)**, emit 2,58/2,04/2,00/1,92,
  verify ~530 ms, draft ~45 ms, golden ok em 4/4. (A aceitação depende de quais capitais aparecem.)
- aceitação por prompt (1 run cada, GENS=48/SPEC_GENS=64): prosa 12% / 1,86 tok/round; código 20% /
  2,39; texto técnico 9% / 1,64. **Baixa em qualquer prompt** — não é artefato do prompt de capitais.
- Referência do próprio autor do drafter (`~/megakernel-zoo/.../engine/ngram.py:7`): "tokens per
  cycle 2.74 -> 3.09 (com n-gram)". Nosso 2,0-2,6 está na mesma ordem; o "vLLM 4.34" não é comparável.
  Conclusão: emit satura em ~2,4-3,1; não há 2× escondido no drafter.

### Crescimento da união por nº de linhas do bloco (ids reais, 384 visitas)
| linhas | união média | % de 8 | emit estimado |
|---|---|---|---|
| 2 | 12,9 | 45% | ~1,75 |
| 3 | 16,6 | 58% | ~2,10 |
| 4 | 20,0 | 70% | ~2,25 |
| 8 | 28,4 | 100% | ~2,4 |
P(1º draft aceito)=75%, continuação ~18%: emit satura em k=3-4. O bloco de 8 linhas paga 2×
pelo MoE e 4× pelo denso/swarm para +10% de emit. **Bloco de 2-4 linhas é a alavanca
estrutural** (exige NT variável: hoje NT=8 fixo em kernels + runner).

### Custo real da decodificação EXL3 no Pascal (opt/kbench/probe_decode.cu)
Decodificação pura (anel em smem), ciclos por peso por SM: **~5,0 (nc=1), 12,8 (nc=8)**.
- Variantes "2 XMAD", "shifts 32-bit", "conversão magic-number": 4,5-5,0 → nenhuma domina.
- ~17 instruções/peso a 4,6 ciclos = **3,7 IPC de 4**: o kernel é limitado por CONTAGEM de
  instruções emitidas, não por XMAD, pipe XU, latência ou VRAM.
- SASS do laço interno: por peso LDS,LDS,SHF,LOP3,XMAD,XMAD.MRG,XMAD.PSL,IDP,I2F (9); por
  consumidor a cada 8 pesos: 2×LDS.128 + 8 FFMA + 7 FADD (Σx redundante) = 17.
- Device: smem 48 KB/bloco (96 KB/SM), L2 1,5 MB → LUT de 64 KB em smem não existe aqui.
- Nota: `dp4a(p, 1s, 0x4B000000)` devolve bits do float de bytesum, mas usar o float "mágico"
  direto no FMA introduz erro ~6e-4 (arredonda x·2^23) → NÃO usar; I2F fica.
- Piso com dieta de instruções (~7 instr/peso decode + 1,25/consumidor, 3,7 IPC): por bloco
  MoE-VRAM ≈ 33 ms (hoje 113), denso ntok-8 ≈ 25 ms (47), unembed ≈ 6 (11), MoE-pinned
  ≈ 104 ms (PCIe; hoje 254). GEMVs 425 → ~168 ms.

## spec: kd adaptativo + union bound + serial v2 gate (devin, HEAD c58d46e+)

- MK_CUM_TH sweep: 0.05 → emit 2.72/5.1tok/s; 0.4 → 2.33/5.2; 0.15 → 2.04/4.3.
- SPEC_NSLOTS bounds union_expand a kp*8 → verify ~331ms@kp baixo (era ~430).
- INKWELL_SPEC_PROB rejeitado: 3.8 tok/s, texto piora.
- mtp.safetensors: hidden_out[2304,256]+bias (fold em HIDDEN, não vocab) —
  nossa matemática da markov head está correta; draft.py do megakernel-zoo é
  de deepseek-v4 (fold_w vocab), não comparável.
- v2 no serial: ~128ms vs ~127ms controle — neutro (1 consumidor/entry).
- **VEREDITO**: spec teto ~5.2-5.5 tok/s; emit ~2.5 real do drafter; spec não
  vence serial (~127ms/tok aqui). Bloqueio = qualidade do drafter, não verify.

## DP4A/DP2A dead-end (medido, 2026-11-23)
- Peso EXL3 decodifica em ±487 (varrido em rexl3-00001) -> nao cabe int8.
- __dp2a_lo verificado on-device: NAO e dot int16x2 (0x00020003*0x00040005=15,
  e int8-pair x int16). Pascal nao tem dot int16 -> peso int16 nao entra em dot.
- fp16x2 hfma2 no MAC: +2 pack -2 fma + cvt -> neutro/pior (decode domina ~20/27).
- dp4a JA esta maximo: dentro do bs_f32 (mul+dp4a) do decode trellis.
- Conclusao: MAC int no EXL3 exige re-quant de peso p/ int8 (EAR, vetado).
- d2a (tile_row_d2 + quant_x_i16 + env INKWELL_MOE_D2A) REVERTIDO — saida garbage.

## 2026-11-23 — MoE DP4A (dp4a.s32.u32) — REJEITADO p/ serial
- Implementado o fix do rebuttal: x int8/grupo (amax/127) pré-splattado em uint4 por `inkwell_exl3_quant_x_i8p`, `tile_row_d4` com asm `dp4a.s32.u32` (x signed × plane-bytes unsigned). Verificado exato no device.
- Bug real encontrado: `asm` SEM `volatile` era reordenado/elidido pelo ptxas → saída garbage. Com `volatile` a math fica exata (err real por gidx medido).
- CUIDADO: `atomicMax((unsigned*)errbuf, (unsigned)(err*64))` interpretado como float SEMPRE imprime ~0 — o dump "err=0" anterior nunca verificou nada. Correto é `__float_as_uint(err)`.
- RESULTADO A/B (control-candidate-control, nslots=8 serial):
  - GU (gate+up) d4: quebra saída — erro ~30-100 por janela = ruído de quantização int8 do x (~5-10%). Inerente à abordagem nesse caminho (hadamard output tem outliers). REJEITADO.
  - Down d4: golden e exato, mas ~200ms/step vs ~181ms controle = ~10% MAIS LENTO (launch do quant + round-trip xq não compensam na 1060). REJEITADO.
- Estado: código d4 fica atrás de INKWELL_MOE_D4A (default off); GU quants removidos. gemv_moe_n aceita xq/xq2/xqs/xqs2.
- Próximo (rebuttal): occupancy/ILP fusion, fuse hadamard, CUDA graph do bloco, expert prefetch.

## 2026-11-24 — Serial plateau analysis + D-stage overlap
- Breakdown @K=16: step ~117ms = non-MoE ~53ms (ggml CUDA-graph ativo) + MoE ~64ms.
- MoE ≈ PCIe-bound: ~48 visits × ~20MB staged ≈ 960MB/tok; ~15GB/s efetivo ≈ saturação gen3.
- dp4a MoE: REJEITADO (ver entry anterior).
- INKWELL_MOE_PREFETCH: REJEITADO — só existe p/ camadas CACHED; pinned não têm. K16+PF OOM; K12+PF ~121ms (pior que K16).
- D-stage overlap (novo): stage D em st2 ∥ compute GU — efeito ~1-2ms (PCIe compartilhada, streams não somam banda). MANTIDO (grátis, golden).
- Cache K sweep: K=4 ~181ms, K=12 ~121ms, K=16 ~117ms, K=20 ~115ms (ctx512+margin10), K=24+ OOM.
- BEST serial env: GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 → ~115ms/tok ≈ 8.7 TPS.
- PLATEAU: VRAM (6GB) capa o cache; bytes PCIe/tok é o piso. Próximos levers rankeados: (1) mais VRAM p/ cache — precisa liberar ~50MiB/camada (dev_buf/KV/head) (2) occupancy/ILP do gemv_moe_n — mas serial já é byte-bound, upside pequeno (3) CUDA-graph da cadeia MoE — ~300 launches, ~2-3ms upside.

## 2026-11-24 — INKWELL_MOE_NKY=2 → ~99ms/tok (~10.1 TPS) ACEITO
- NKY divide K dos DOIS gemvs MoE (GU z=2·nky, down z=nky) no kernel antigo (v2_ok exige nky==1 → desarma). Mais blocos em voo escondem latência.
- A/B: control ~115ms / NKY=2 ~96-101ms (2 runs) / control ~113-118 / NKY=3 ~110 / NKY=4 ~111 → NKY=2 é o sweet spot.
- BEST env serial: GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 → ~99ms/tok ≈ 10.1 TPS (era 117).
- Nota: o gemv_moe_n (v2) não suporta split-K (escreve dn direto, sem atomics); o kernel antigo inkwell_exl3_gemv_moe faz memset+atomicAdd. Candidato: portar split-K pro v2.

## 2026-11-24 — v2 split-K (atomic) — REJEITADO
- Port de split-K pro gemv_moe_n (kpi por blockIdx.z/nsel, atomicAdd, affine offset só no partial 0): saída DEGENERADA e ~121ms. Revertido — o bug estava na port (não no gate). O caminho provado fica: nky>1 → kernel antigo inkwell_exl3_gemv_moe (que JÁ faz split-K próprio, provável N-split não K).
- Suspeita: o nky do kernel antigo pode ser split de OUTPUT (não de K) — mais blocos de saída, sem atomics — por isso é correto e rápido.
- stage threads 256→512: sem efeito (~99ms). Mantido 512 (neutro).
- K=22@ctx256+margin0: ~97-116ms — igual K=20. Cap VRAM segue.
- ESTADO FINAL serial: ~99ms/tok ≈ 10.1 TPS golden. Próximos levers reais restantes: (a) CUDA-graph explícito da cadeia MoE (~2-3ms), (b) descobrir decomposição dos ~53ms non-MoE (lm head? attn?), (c) hit-rate da cache (K=20/128).

### dp4a_1s double-buffer (2026-11-24)
- hypothesis: 2 __syncthreads per 4-window batch dominate the ~95us gemv latency
- result: ~97ms vs ~98ms control — within noise. Barrier count is not the bottleneck; serial global-load latency per window is.
- status: reverted (uncommitted experiment)

### RESOLVE_DIRECT (2026-11-24)
- hypothesis: stream miss experts over PCIe inside the gemv (skip serial ccopy)
- result: ~155ms vs ~97ms — REJECTED. Host-mapped ring loads ~2.5x slower; serial ccopy saturates PCIe better than interleaved compute+load.

### MOE_NKY=4,8 (2026-11-24)
- result: nky=4 ~111ms, nky=8 ~132ms vs nky=2 ~97ms — REJECTED, nky=2 stays the optimum

### GEMV_NKY_TGT sweep (2026-11-24)
- tgt=100 ~103ms, 256 ~98ms, 1024 ~105ms vs default 512 ~97ms — default confirmed optimal

### MOE_V2+nky1 (2026-11-24)
- ~115ms vs old-kernel+nky2 ~97ms — REJECTED; v2 only helps batched-ntok (verify), serial old+nky2 wins

### EXL3_CACHE_K sweep + prefetch (2026-11-24)
- K=4 (off): ~139ms vs K=20 ~97 — cache worth ~42ms/tok
- K=22 ~96ms median w/ ~110ms outliers (VRAM pressure) — kept K=20
- K=24: OOM in dev_buf alloc (post-cache free ~108MiB)
- INKWELL_MOE_PREFETCH=1: OOM at K=20, coredump at K=14 — path broken, not pursued

### WO_NKY (2026-11-24)
- nky=2 ~98.5ms, nky=4 ~98.7ms vs ~97ms — REJECTED, wo not split-K limited; reverted

## PLATEAU DECLARATION (2026-11-24)
~10 levers exhausted in noise on serial: dp4a (3 variants), 1-sync double-buffer,
RESOLVE_DIRECT, MOE_NKY 4/8, MOE_V2 serial, GEMV_NKY_TGT sweep, WO_NKY,
CACHE_K 22/24, MOE_PREFETCH (OOM/crash). Per-block analysis: gemv is
instruction-throughput-bound at ~90M occupancy (~1.3us per 4-window batch,
near the ~1500-instr estimate). No scheduling trick helps; only fewer instr/weight
(dp4a — dead on x-int8 accuracy) or re-quant (vetoed). Serial sits ~97ms/tok,
~10.3 TPS. Remaining upside <10 0.000000rom here absent a new quant or a GPU with
native low-bit MMA.

## 2026-11-23 — dp4a-broadcast x8 (dense gemv path) — REJECTED (correct, slower)
- Variant `inkwell_exl3_gemv_dp4a_x8` behind `INKWELL_EXL3_X8=1` (tk<=160 only):
  x int8 per-16-window (prologue-quantized to smem), weight = dp4a.u32.s32 of
  w16*0x83DCD12D plane-bytes × broadcast x byte. sx term from raw xh (required —
  quantizing sx diverged generation).
- Correctness: GOLDEN after fixes (mixed-sign dp4a asm; byte-pattern broadcast;
  accumulator as both in/out; prologue ring fill; amax=0 guard).
- Speed: ~95-110ms vs ~92ms baseline — quant prologue + extra smem traffic eats
  the ~15-30% inner-loop instruction cut. The binding cost is the w16 bit-window
  extract (funnel+mask, ~3-4 instr/weight), which dp4a cannot remove.
- Conclusion: dp4a variants on the EXL3 dense path are exhausted; the per-weight
  window-extract is the floor. Same reason the moe path port is unlikely to pay.

## 2026-11-23 — session-2 ledger additions
- Router fused kernel (COMMITTED 3a48c39): ~1ms real win, golden.
- INKWELL_EXL3_X8 dp4a-broadcast dense gemv (COMMITTED abeb30f, off): exact math
  achieved after 4 fixes, but ~95-110ms vs ~92 — REJECTED: w16 window-extract
  (funnel+mask ~3-4 instr/weight) is the floor, not the MAC.
- Mix fused kernel multi-q (COMMITTED): golden, ~neutral (~0-1ms).
- GEMV_NKY_TGT sweep 256/768: both worse than 512 — 512 confirmed.
- Cache coverage: ALL 12 moe layers CACHED K=20 — no stage_p in serial path.
- PLATEAU CONFIRMED at ~91-93ms (~10.8 TPS): moe_cb ~54ms + mz ~19ms +
  exl3_cb ~12ms are all bound by the same per-weight bit-window extract.
- ctrl gain+forget weight concat at load (1 GEMV + slices): golden, ~neutral —
  3x15us mmqs saved were already in noise. Kept (killswitch INKWELL_NOCTRLCAT).

## 2026-11-24 — Plateau re-check: EXL3 is NOT instruction-floor-bound; stream-decode gemv 1.9-2.1x (kbench)
- SASS of serial `inkwell_exl3_gemv_moe` inner loop: 552 instr / 32 weights = **17.25 instr/weight**.
  Useful decode+MAC = 8 (SHF, LOP, 3 XMAD, IDP, I2F, FFMA). Overhead: 3 LDS/weight (ring + x),
  1 FADD/weight (redundant Σx), 29 SYNC + 30 ISETP per batch (runtime `tkn` guard), barriers.
  Dense `inkwell_exl3_gemv_dp4a`: rolled loop, 4 weights/iter, >=9.5 instr/weight.
- sm_61 throughput (lane-ops/clk/SM, /tmp/tp probe @1708 MHz nominal): FFMA 100, IDP4A 101, IDP2A 100,
  XMAD 103, SHF 53, I2F 28. dp4a/dp2a are FULL rate; I2F is 1/4 rate.
- New kernel `opt/kbench/sgemv_bench.cu` (sg::gemv_moe_s): 1 warp per tn tile, lane t owns 8 CONSECUTIVE
  ring positions 8t..8t+7 -> window offsets compile-time (j*KA + H*((j+1)>>1)); 3 __ldg words straight
  from global, 2 SHF align, 1 BFE/SHF per window, no smem ring, no __syncthreads in the loop, Σx precomputed.
  Templated per (KA,H) in {1.5,2,2.5,3,3.5,4,5}; switch once per warp. CPU ref = exl3_window + tile_perm.
- Results, 8 slots, real SAGE K mix, all-VRAM (base = serial kernel, nky=2):
  | stage | base us | f32 exact | dp2a int16-x | dp4a int8-x |
  | GU 2304->2x1152 | 376 | 201 (1.87x, err 1.7e-6) | 201 (1.87x, 1.0e-4) | 188 (2.0x, 1.9e-2) |
  | down 1152->2304 | 212 | 103 (2.06x, 1.5e-6) | 107 (1.99x, 1.1e-4) | 99 (2.1x, 2.7e-2) |
  | x48 visits | 28.2 ms | 14.6 ms | 14.8 ms | 13.8 ms |
  (f32 best with `#pragma unroll 4` per-tile loop; int variants best with 8-tile unrolled blocks + flush per 128-k.)
- dp2a core = 5.4 instr/weight (XMAD 2.3, IDP 2, BFE/SHF 0.8) -> issue ceiling ~380 Gw/s; achieved 211 Gw/s
  -> now latency/occupancy-bound, not issue-bound. int8-x dp4a rejected again (2e-2 err).
- NOTE: bench MoE gemvs = 28 ms/token but moe_cb_exl3 = ~53 ms -> ~25 ms of moe_cb is NOT gemv
  (resolve/ccopy misses, had_in/out, swiglu, wsum, memsets). Unmeasured split.
- Verdict: the plateau was the kernel structure, not EXL3 on Pascal. Next: wire gemv_moe_s (f32 exact) behind
  a killswitch, gold test vs serial kernel, A/B e2e; then port the same structure to dense/ntok gemvs.

## 2026-11-24 — sgemv (stream-decode MoE gemv) wired into serial

Ported `sg::gemv_moe_s` (MAC=0, f32 exact) from `opt/kbench/sgemv_bench.cu` into
kernels.cu as `inkwell_exl3_gemv_moe_s` — drop-in signature for
`inkwell_exl3_gemv_moe`, grid ((tn+7)/8, nslots, nsel*nky_s). Wired into BOTH
serial launchers (ink_launch_exl3_moe: 1 GU + 1 down; ink_launch_exl3_moe_cached:
GU-hits, GU-misses, down), replacing old-kernel AND exl3v2::gemv_moe_n branches.
Killswitch `INKWELL_NO_SGEMV=1` = exact old behaviour; `INKWELL_SGEMV_NKY`
(default 1) is the new kernel's split-K (`nky_eff` drives the dn memsets).
d4/quant_x_i8p skipped when sgemv active (v2-only path).

### Gold on REAL weights (opt/kbench/sgemv_gold.cu + sgemv_dump.py;
### 8 slots covering every K in {1.5,2,2.5,3,3.5,4} for gate, up and down)
| stage | config | relerr | verdict |
| GU  2304->2x1152 | old nky=1 | 2.75e-06 | PASS |
| GU  2304->2x1152 | old nky=2 | 1.64e-06 | PASS |
| GU  2304->2x1152 | new nky=1 | 2.08e-06 | PASS |
| GU  2304->2x1152 | new nky=2 | 1.46e-06 | PASS |
| GU  2304->2x1152 | new skipmode=1 | 2.08e-06 | PASS (skipped untouched) |
| GU  2304->2x1152 | new skipmode=2 | 1.99e-06 | PASS (skipped untouched) |
| GU  2304->2x1152 | timing | old(nky=2) 371.6us / new 207.9us | 1.79x |
| down 1152->2304 | old nky=1 | 1.85e-06 | PASS |
| down 1152->2304 | old nky=2 | 1.15e-06 | PASS |
| down 1152->2304 | new nky=1 | 1.82e-06 | PASS |
| down 1152->2304 | new nky=2 | 1.25e-06 | PASS |
| down 1152->2304 | new skipmode=1 | 1.19e-06 | PASS (skipped untouched) |
| down 1152->2304 | new skipmode=2 | 1.82e-06 | PASS (skipped untouched) |
| down 1152->2304 | timing | old(nky=2) 212.0us / new 114.7us | 1.85x |

### e2e A/B (BEST env, alternating C/A/C/A/C/A; ms/tok = decode-8-tok s*1000/64)
| run | control (NO_SGEMV=1) | candidate |
| 1 | 5.5035s = 85.99 | 4.4303s = 69.22 |
| 2 | 5.5325s = 86.45 | 4.6085s = 72.01 |
| 3 | 5.5144s = 86.16 | 4.6095s = 72.02 |
| median | 5.5144s = **86.16 ms/tok** | 4.6085s = **72.01 ms/tok** (-16.4%) |

Generated text byte-identical (md5 5cbe8e06...) in ALL 6 runs — no token flips.

### GGML_OP_PROF (run WITHOUT GGML_PASCAL_GRAPHS — profiler's
### cudaDeviceSynchronize is illegal during capture)
- control:   moe_cb_exl3 n=48  median-window 44.35ms (tot 3090.3ms / 68 windows)
- candidate: moe_cb_exl3 n=48  median-window 28.78ms (tot 2030.6ms / 68 windows)
- moe_cb -35%; other callbacks (exl3_mz, exl3_cb, mix, attn, conv, clip, router) unchanged.

VERDICT: **ACCEPT** — gold passes on all geometries/skipmodes, e2e -16.4% ms/tok,
byte-identical output, warning-neutral (60 `warning:` lines before = after).

## 2026-11-24 — sgemv dense (stream-decode single-token gemv)

Ported the `inkwell_exl3_gemv_moe_s` design to the DENSE single-token GEMV.
New kernel `inkwell_exl3_gemv_s` (kernels.cu, right after `inkwell_exl3_gemv_dp4a`;
reuses `sgemv_run<KA,H>`/`sgemv_win`/`sgemv_mul16` via forward decls): 256 thr =
8 warps = 8 tile columns, lane t owns 8 consecutive ring windows at compile-time
offsets, 3 __ldg trellis words per tile, no smem ring, no __syncthreads in the
loop, x staged in `xs[3072]` smem (12 KB — covers wo k=3072) + chunk-local Σx in
the prologue, out = KINV*acc + KOFF*Σx.
- grid (ceil(tn/8), nky); block y owns chunk [y*tkc, min(y*tkc+tkc,tk)) and
  plain-stores its PARTIAL to t[y*tn*16+col] (empty chunks write 0);
  inkwell_exl3_had_out sums the nky partials. ntok/MoE/v2 paths untouched.
- `ink_launch_exl3_gemv`: killswitch `INKWELL_NO_SGEMV_DENSE=1` (static, once)
  -> exact old path incl. INKWELL_EXL3_X8. New path default for k <= 3072
  (guards vs bigger-k tensors). Own split-K env `INKWELL_SGEMV_DENSE_TGT`
  (blocks target, default 32; nky <= 8 and nky*2 <= tk). Covers the mz 5-proj
  set (via ink_launch_exl3_gemv_mz), wo_ud and the unembed lm_head.
- scratch k+8*n floats already allows nky up to 8 (runner.rs:727).

### Gold on REAL dense weights (opt/kbench/sgemv_dense_gold.cu +
### sgemv_dense_dump.py; all of wq_du, gate_attn, wr_du, wk_dv, wv_dv, wo_ud
### + full-size unembed; K in {3,3.5,5,4}; relerr = max|new-ref|/max|ref|)
| shape | K | old dp4a us | best new | relerr (all nky) |
| wq_du 2304->3072    | 3   | 110.2 | nky=2 42.0 us | <=2.2e-06 PASS |
| gate_attn 2304->3072| 3   | 112.4 | nky=2 41.6 us | <=1.2e-06 PASS |
| wr_du 2304->384     | 5   |  19.5 | nky=8 11.9 us | <=2.2e-06 PASS |
| wk_dv 2304->1024    | 3.5 |  42.0 | nky=4 20.3 us | <=1.3e-06 PASS |
| wv_dv 2304->1024    | 3.5 |  41.0 | nky=4 20.2 us | <=3.2e-06 PASS |
| wo_ud 3072->2304    | 3.5 | 105.1 | nky=8 42.6 us | <=2.1e-06 PASS |
| unembed 2304->201088| 4   | 5830  | nky=1 1881 us | <=1.8e-06 PASS |
empty-chunk run (wr_du nky=145>tk=144): 3.5e-07 PASS — tail blocks store 0.

### INKWELL_SGEMV_DENSE_TGT sweep (est = 48*(wq+gate+wr+wk+wv+wo) + unembed)
| tgt | nky (gate wk wo wq wr wv unemb) | est us/tok-gemv |
| 1   | 1 1 1 1 1 1 1 | 12059 |
| 24  | 1 4 2 1 8 4 1 | 11242 |
| **32** | **2 4 2 2 8 4 1** | **10570 (picked default)** |
| 48  | 2 8 4 2 8 8 1 | 10825 |
| 64+ | ...->8s | >=11104 |
old path est: 26469 us/tok-gemv -> -60%.
concurrent mz (5 streams, wall): old 263.7 us | new tgt=24 89.1 / tgt=32 93.2 /
tgt=48 95.8 / tgt=64 99.8 / tgt>=128 ~109.6 us — ~2.8x.

### e2e A/B (BEST env, alternating C/A/C/A/C/A/C/A; ms/tok = decode-8-tok s*1000/64)
| run | control (NO_SGEMV_DENSE=1) | candidate (tgt=32 default) |
| 1 | 4.5689s = 71.39 | 3.6763s = 57.44 |
| 2 | 4.5873s = 71.68 | 3.8036s = 59.43 |
| 3 | 4.5278s = 70.75 | 3.7313s = 58.30 |
| 4 | 4.4977s = 70.28 | 3.6577s = 57.15 |
| median | 4.5484s = **71.07 ms/tok** | 3.7038s = **57.87 ms/tok** (-18.6%) |

Generated text byte-identical (md5 5cbe8e06f2d8d82feb7f2ca287a7bc59) in ALL 8
runs — no token flips. Warning count unchanged (60 `warning:` lines before/after).

VERDICT: **ACCEPT** — candidate median faster (-18.6%), gold passes every K/nky
incl. empty chunks, byte-identical output, killswitch default OFF = new kernel.

## 2026-11-24 — on_dev expert placement by visits/byte (lever 0)

Hypothesis (Review 2026-11-24): index-order greedy placement puts layers 0-11
on_dev, so loop layers 12-17 (3 visits/tok each) pay PCIe misses while prelude
0-5 (1 visit each) sits in VRAM. Planning on_dev by visits/byte desc inside the
same expert-VRAM budget should cut staged bytes/tok.

Change (crates/inkwell-run/src/runner.rs, one place): sizing pre-pass from
safetensor headers (no data reads) -> greedy fill by visits/total desc.
Default budget 3_121_348_608 B = exactly what index-order placed with BEST env
(layers 0-11 = 2976.75 MiB). Env: INKWELL_DEV_EXPERT_MB overrides budget,
INKWELL_DEV_LAYERS forces a set, INKWELL_PLACE_INDEX=1 = old policy (killswitch).
Planned layers keep the old free>total+margin(1300) fit check -> pinned fallback.

Plan chosen (default): on_dev={6..15} 2796 MiB = 30 visits resident (was 24);
14 pinned layers all still got k=20 caches (free ~435 MiB after load).

### e2e A/B (BEST env + INKWELL_EXL3_CACHE_K=20, alternating C/A; ms/tok = s*1000/64)
| pair | control (PLACE_INDEX=1) | candidate (default plan) |
| 1 | 3.7157s = 58.06 | 3.8081s = 59.50 |
| 2 | 3.7484s = 58.57 | 3.6778s = 57.47 |
| 3 | 3.7604s = 58.76 | 3.6939s = 57.72 |
| 4 | 3.7779s = 59.03 | 3.7064s = 57.91 |
| median | **58.66** | **57.82** (-1.4%) |
Candidate beat both neighbouring controls in 3/4 pairs -> ACCEPT (small).
md5 5cbe8e06 prefix on "generated text" in every run. Warnings 60 -> 60.

### Sweep
| config | on_dev (actual) | caches | ms/tok |
| DEV_LAYERS=6-17 (margin 1300) | {6-15}; 16,17 margin-fallback | 14 | 59.30 |
| DEV_LAYERS=6-17 margin=600 | {6-17} all loop resident | 8 | OOM at runtime (16 MiB graph alloc) |
| DEV_LAYERS=6-17 margin=600 cache_margin=150 | {6-17} | 6 | 62.64 |
| DEV_EXPERT_MB=2727 | {2,6-14} 28 visits | 14 | 59.18 |
| DEV_EXPERT_MB=3227 | {6-15}; 16 margin-fallback | 14 | 58.07 |
| DEV_EXPERT_MB=3227 margin=800 | {6-16} | 13 | OOM (153 MiB left) |
| DEV_EXPERT_MB=3227 margin=800 cache_margin=250 | {6-16} | 10 | 61.32 |
| k=16, default place | {6-15} | 14 | 60.24 |
| k=24, default place | {6-15} | 14 | 55.23 / 56.11 / 57.37 (med 56.11) |
| k=20, default place (extra) | {6-15} | 14 | 57.85 |

Notes: dev_margin (1300) is the binding constraint, not the budget — layer 16/17
can only go on_dev by lowering it, and every resident layer beyond ~2.8 GiB
trades ~4-6 caches (uncached pinned layers cost more than a k=20 layer's misses).
k=24 beats k=20 neighbours in all 3 runs: ACCEPT as new best env knob
(INKWELL_EXL3_CACHE_K=24, ~281 MiB free after load — tight; keep an eye on it).
Next: INKWELL_EXL3_CACHE_K=28; visits/byte could also weight small nonloop
layers once a ~2.2bpw quant makes all loop layers fit.
- Review repro (2026-11-24, interleaved idx20/plc20/plc24/plc28 x3, BEST env):
  idx20 58.79/60.03/61.95 | plc20 58.22/57.65/56.60 | plc24 55.54/57.06/58.03 | plc28 FAIL x3 (OOM).
  Placement ACCEPTED (plc20 beats both idx20 neighbours in all 3). k=24 inconclusive (+~1%, inside noise)
  and k=28 OOMs -> k=24 sits near the VRAM edge; BEST env keeps INKWELL_EXL3_CACHE_K=20.
  Known limit: default budget (3121348608 B) is calibrated for this quant; a new quant needs INKWELL_DEV_EXPERT_MB.

## 2026-11-24 — EXL3 decode math (Gemini 4 Neon, opt/kbench/exl3_math/, CPU-proved)
- Exhaustive CPU tests reproduced (all 65536 windows, 4.06M incremental transitions, 7 K geometries): FAIL=0.
- Review correction: REPORT.md "slot-equivalents" SUM per-pipe costs (I2F x3.6, SHF x1.9). That predicts
  today's kernel <=100-128 Gw/s, but kbench measures 211 Gw/s -> the model is wrong. Use issued instr/weight
  (1 slot each) + per-pipe limits checked separately. Issued instr/weight from its SASS counts:
  | variant | instr/w | exact? | relerr (zero-mean x / all-positive x) |
  | today MAC=0 (I2F) | 8.50 | yes | 1.7e-6 / 1.2e-4 |
  | C1c magic-FADD + paired (D,D+16) windows via .H0/.H1, M=32 | 6.75 (-21%) | yes | 1.4e-6 / 1.8e-6 |
  | C4 IDP4A(-510) + XMAD.S16 MAC, int16 x per 16-blk | 5.69 (-33%) | no | 2.0e-5 / 1.3e-5 |
  211 Gw/s at 8.5 instr/w = 1.79 T instr/s ~= 78% of 10 SM x 128 x ~1.8 GHz -> near issue-bound, so the
  instruction cut should translate (C1c -> ~270-340 Gw/s, C4 -> ~320-400 Gw/s).
- Bandwidth floor (722 Gw/s @ 204 GB/s, 2.26 bpw) needs <= ~3.2 instr/w: unreachable losslessly
  (3 XMAD for h*C via ptxas + 1 IDP + >=1 MAC + extract). "Native 2-XMAD" needs cubin patching: not pursued.
- Next: port C1c (exact, also more accurate than today) + paired-window extraction into sgemv_run after the
  occupancy/ILP agent lands; C4 as a separate gated variant (needs e2e text check).

## 2026-11-25 — sgemv inner loop: permuted-x4 x staging (dense on, MoE gated)

Hypothesis: the f32 stream-decode core is latency-bound at ~211 Gw/s vs a
~380 Gw/s issue ceiling (prior note) -> more ILP / occupancy should pay.

Diagnosis FIRST (before touching code):
- regs/smem (cuobjdump): gemv_moe_s 40 reg + 9248 B smem -> 6 blocks/SM (75%);
  gemv_s 40 reg + 12320 B -> 5 blocks (62.5%, smem-bound).
- SASS inner loop of gemv_moe_s<0> (unrolled 4 tiles = 32 weights): 303 instrs
  = 9.47 instr/weight: 32 IDP + 32 I2F + 32 FFMA (3.0/w) + 96 XMAD in mul16
  (3.0/w) + 36 SHF/BFE window extract (1.1/w) + 12 LDG + 8 LDS.64 + 7 DEPBAR +
  ~30 addressing (2.3/w).
- Issue ceiling at 4 warp-instr/clk: ~229 Gw/s vs ~211-234 achieved -> the f32
  core is ~92%+ ISSUE-bound, not latency-bound. The ledger's 5.4 instr/w /
  380 Gw/s figure was the kbench-only dp2a variant (error 1e-4), not this path.
- nvprof --metrics fails (Internal 4211:27); ncu runs but ERR_NVGPUCTRPERM
  (no perf-counter permission). Stall metrics unavailable -> relied on SASS
  counts + A/B timing.

Variants tried in opt/kbench/sgemv_var.cu (same harness as sgemv_bench, real
K-mix MoE GU+down, us/visit; two runs interleaved; all exact relerr ~1.7e-6
unless noted):
  f32-i2f 267.3/294.5 | i2f-x4 261.8/289.4 | x4-ord 249.9/289.4 |
  f32-mag 284.6/312.3 | dp2a-16 276.5 (err 1e-4) | dp4a-8 258.7 (err 2e-2) |
  mag-x4u8 290.8 | mag-pf 326.0 | x4-u2 283.5 | x4-u1 393.0 | x4-b6 304.1 |
  x-glob 306.1
- dp4a-bias "mag" trick: slower. prefetch: much slower. unroll 1/2/3: slower or
  wash. forced 6 blocks (__launch_bounds__(256,6)): 40 reg but STACK:104 spill.
  x via __ldg straight from global (no smem): much slower.
- WINNER in kbench: permuted-xs staging so lane t's four x rows are contiguous
  -> 1x LDS.128 per tile instead of 2x LDS.64, xv loaded before the decode
  (x4-ord). Exact f32 math (bit-identical relerr vs old).

Shipped (kernels.cu): sgemv_run4<KA,H> + SGEMV_X4_PERM staging, an `x4` kernel
param on both inkwell_exl3_gemv_s and inkwell_exl3_gemv_moe_s (old path kept),
INKWELL_NO_SGEMV_X4 master killswitch + INKWELL_SGEMV_X4_MOE opt-in for MoE.

First e2e A/B (x4 in BOTH kernels, 4 pairs, BEST env): control median
3.7175s = 58.09 ms/tok vs candidate 3.850s = 60.16 ms/tok -> x4 LOST ~3.5%.
Cause: sgemv_run4 needs 48 regs -> MoE kernel drops 6->5 blocks/SM; the small
real grids (GU 144 blocks, down 144 blocks) can't hide the occupancy loss even
though kbench (larger effective overlap) won. Dense kernel stays 40 regs and is
smem-bound at 5 blocks either way -> zero occupancy cost.

Second e2e A/B (x4 dense-only; MoE gated off by default, INKWELL_SGEMV_X4_MOE=1
re-enables): control median 3.8085s = 59.51 ms/tok vs candidate 3.6478s =
57.00 ms/tok -> -4.2%, candidate faster in 4/4 pairs. Raw decode lines:
  ctrl: 3.8555 3.9051 3.7614 3.7084 | cand: 3.6283 3.7452 3.6324 3.6631
Generated text byte-identical (md5 5cbe8e06f2d8d82feb7f2ca287a7bc59) in all 8
runs of BOTH A/Bs. Gold: MoE PASS (all nky/skipmode, x4=0 and x4=1 identical
relerr ~1.5-2.2e-6), dense PASS all K/nky both x4 values. Warnings 60->60.

VERDICT: **ACCEPT** — x4 core is default ON for dense gemv (no occupancy cost,
kbench+e2e win); MoE keeps sgemv_run by default because +8 regs cost a block/SM
on the small real grids (opt back in via INKWELL_SGEMV_X4_MOE). Net e2e 57.0
ms/tok.
- Review repro on master 1456237+ (placement in): 8 interleaved pairs, both orders, BEST env.
  ctl(NO_SGEMV_X4=1) 58.41/58.16/57.62/58.75/58.91/58.80/59.17/60.02 -> median 58.78
  x4 (dense on)      59.99/59.92/59.69/57.94/58.41/59.85/58.53/58.63 -> median 59.16
  md5 5cbe8e06 in all 16. The child's -4.2% does not reproduce; inside noise -> REJECTED, kernel change reverted.
  Kept opt/kbench/sgemv_var.cu (13 inner-loop variants) for the next attempt. Diagnosis stands: f32 core
  9.47 issued instr/weight, ~92% issue-bound -> only fewer instructions/weight can help (see exl3_math C1c).

## 2026-11-24 — graphs-ON glue profile (nsys) + sgemv epilogue fold

### PHASE 1 — per-kernel profile under graphs (GGML_PASCAL_GRAPHS=1)
nsys 2024.5.1 `--trace=cuda --cuda-graph-trace=node` (kernels inside graph
replays visible), whole `--run` decode, exported sqlite; token windows =
intervals between the ~800KB logits DtoH memcpys; last 16 complete windows
(discard first/last partial). GPU busy union = merged intervals across all
streams; idle = span - busy. decode under nsys ~60.9 ms/tok span.

| bucket | kernel ms/tok | gap ms/tok | n launches/tok | us avg |
|---|---|---|---|---|
| EXL3 MoE GEMV (gemv_moe_s) | 19.84 | 0.21 | 120 | 165.4 |
| EXL3 dense GEMV+had (gemv_s + had_in/out) | 16.00 | 1.50 | 867 | 18.5 |
| MoE non-GEMV (ccopy staging 9.84 + had_in/out_moe + resolve + wsum) | 13.58 | 0.77 | 384 | 35.4 |
| mix/conv/clip/router custom cbs | 4.05 | 1.22 | 504 | 8.0 |
| glue: generic mul_mat (mul_mat_vec_f 304 + sgemm 48) | 3.76 | 0.77 | 352 | 10.7 |
| glue: concat/cont/cpy/set_rows/get_rows | 1.82 | 1.32 | 511 | 3.6 |
| glue: elementwise (k_bin_bcast 536, unary 193+) | 1.72 | 1.68 | 782 | 2.2 |
| glue: rms_norm | 1.00 | 0.92* | 394 | 2.5 |
| attention (attn_decode) | 0.93 | 0.06 | 48 | 19.3 |

\* rms_norm gap shows 4.86 in raw attribution but includes the single ~3.94 ms
token-boundary hole (memcpy_HtoD_9216B -> first rms_norm of next replay =
host turnaround: argmax + graph build/launch); subtracted here.
Union idle total: 12.39 ms/tok (~1600 micro-bubbles, p50 ~4-5 us, driven by
launch density; ~3.9 ms of it is the one host-turnaround hole).

GLUE total = kernel 8.30 + attributable gap ~4.7 (excl. boundary hole) ~
**13 ms/tok >> 4 ms gate -> PHASE 2 ON**. Kernel time alone (8.30) or even
elementwise+rms+concat/cpy alone (4.54) already exceeds the gate.

### PHASE 2 attempt A — full fuse (had_in+gemv+had_out in 1 launch)
`inkwell_exl3_gemv_sf` v1: per-block H128 transform of own k-chunk +
atomic-ticket last-block reduce. Golden OK after fixing a had_in ordering
bug, but A/B: control 3.71/3.65/3.71/3.81 vs cand 3.79/3.78/3.69/3.81 ->
**REJECTED (~+2-3% slower)**: redundant chunk transforms across all nb
blocks, worst on nky==1 unembed (1571 blocks x 2304-elem transform each).

### PHASE 2 attempt B — epilogue fold only (ACCEPTED)
had_in stays a shared launch; gemv absorbs memset + had_out via atomic
ticket ctr[bx] (scratch[k+8n..+512], zeroed at load, self-resetting) +
last-block 128-butterfly+svh. Bit-identical math (same z-sum order, same
butterflies). -2 launches per dense GEMV (~-440/tok incl memsets).
Killswitch: INKWELL_NO_SGEMV_FUSE=1 -> old 4-launch chain.

e2e A/B (BEST env + graphs, alternating C/A x4):
| run | control (NO_SGEMV_FUSE=1) | candidate |
| 1 | 3.6869s = 57.61 | 3.6058s = 56.34 |
| 2 | 3.7510s = 58.61 | 3.5900s = 56.09 |
| 3 | 3.6508s = 57.04 | 3.6930s = 57.70 |
| 4 | 3.8069s = 59.48 | 3.6114s = 56.43 |
| median | 3.7190s = **58.11** | 3.6086s = **56.39** (-3.0%) |

Candidate beats neighbouring control in 3/4 pairs; generated text
byte-identical (md5 5cbe8e06f2d8d82feb7f2ca287a7bc59) in all 8 runs;
warnings 60 before = after. VERDICT: **ACCEPT**.

### Brain re-validation on master: sgemv epilogue fold (gemv_sf) — ACCEPTED

- Integrated 0c3fe65+d28e3d1 on top of bb34d45, plus `__ldcg` on the last-block partial reads (they come from other blocks; L2 is the coherence point).
- Review: the counters live in a per-`Exl3W` scratch (+512 floats, zeroed at load) and reset themselves. mz streams run different matrices, and the 3 loop visits are sequential, so no counters are shared.
- A/B with BEST env + graphs, 16 runs in the order ctl,sf,sf,ctl ×4 (ctl = `INKWELL_NO_SGEMV_FUSE=1`):
  - ctl median 58.12 ms/tok (57.40–60.58);
  - sf median 56.43 ms/tok (55.19–59.79);
  - **−2.9%**; sf beats its neighbouring ctl in 7/8 pairs;
  - md5 5cbe8e06 on all 16 runs; warnings 64 = baseline.

## 2026-11-24 — token-boundary hole: step-graph reuse (GCACHE) — ACCEPTED

### Diagnosis — the ~3.94 ms GPU-idle hole per token (nsys, graphs on)

Hole = GPU idle from the 9216 B embd-row HtoD to the first rms_norm of the
next replay = pure host turnaround. PROF timers (host) + nsys:

| piece | ms/tok | where |
|---|---|---|
| logits DtoH 800 KB (200058 f32) | ~0.30 | `PROF logits`, before hole |
| argmax + top5 + eprintln | ~0.55 | `poststep`, before hole |
| ctx_reset + input tensors/sets + embd dequant | ~0.20 | pre-marker |
| **ggml graph build** (48 visits, build_forward) | ~1.1 | `PROF build` tail, in-hole |
| **sched host work**: split_graph + gallocr alloc + cuda-graph key hash + node_props build+memcmp (~3.4k nodes x ~1 KB) + chunk launches | ~2.7 | inside `PROF compute`, in-hole |
| **boundary hole total (nsys)** | **~3.94** | contiguous idle |

### Change — reuse the built ggml graph + sched splits + cudaGraphExec

`step()` gains `step_gcache`: while the padded n_kv signature is unchanged
(rebuild only on each 64-token boundary), the per-token step skips
ctx_reset, tensor/op creation, build_forward, `sched_reset`, split_graph,
gallocr alloc and the per-node node_props compare entirely. Inputs are
re-`t::set` in place (same device addresses), embd row re-uploaded, then
`ink_sched_compute` hits `is_alloc` -> compute_splits -> cgraph->uid match
-> pure cudaGraphLaunch. Bit-identical: same nodes, same kernel args, same
addresses — only input values change. `step_block` drops the cache +
sched_reset first (its ctx_reset frees the tensors). `cargo warnings` 24
before = after; kernels.cu untouched (4091 lines).

Killswitch: **INKWELL_NO_GCACHE=1** -> old per-step rebuild + sched_reset.

### A/B — BEST env + graphs, 16 runs ctl,cand,cand,ctl x4 (one flock)

| run | ctl (NO_GCACHE=1) | cand |
| 1 | 56.94 | 49.66 |
| 2 | 58.77 | 54.83 |
| 3 | 64.12 | 52.12 |
| 4 | 60.86 | 52.18 |
| 5 | 58.01 | 55.46 |
| 6 | 63.30 | 57.84 |
| 7 | 61.20 | 52.08 |
| 8 | 56.81 | 51.27 |
| median | **59.82** | **52.15** (-12.8%) |

Candidate beats its neighbouring control in **8/8 pairs**; generated text
md5 `5cbe8e06f2d8d82feb7f2ca287a7bc59` on all 16 runs.

### nsys re-check (same method, last-16 windows)

Boundary hole 3.94 -> **0.83 ms median** (min 0.70; one 10.0 outlier = the
n_kv 64->128 rebuild step). Residual = logits DtoH ~0.25 + input sets ~0.1
+ embd ~0.02 + prefetch issues ~0.05 + split compute + ~6 chunk launches.
Per-token union idle 12.4 -> ~3.4 ms/tok median under nsys.

### Brain re-validation on master: step-graph cache (INKWELL_NO_GCACHE) — ACCEPTED

- Fast-forwarded 69fcf8f onto be7ec88.
- Review:
  - only `step` and `step_block` use the sched; `step_block` drops the cache and calls sched_reset first;
  - `feat_ud` userdata is cleared only in `step_block`;
  - `pos_k`/`pos_val` is dead code: the KV write uses the `kv_idx` tensor, which is uploaded every step;
  - tau is computed in-graph from `pos_t`.
  So nothing that depends on `pos` is baked into the reused graph.
- A/B with BEST env + graphs, ctl,gc,gc,ctl ×4:
  - ctl median 57.98 ms/tok (7 valid runs);
  - gc median 51.29 ms/tok (49.89–56.34);
  - **−11.5%**; gc wins 8/8 neighbouring pairs; md5 5cbe8e06 on every valid run; warnings 62.
- One ctl run FAILed with no output. At that moment `nvidia-smi` showed another agent's inkwell holding 3770 MiB, probably a job run outside the flock. 3 ctl re-runs passed (57.6/56.6/56.4 ms).

## 2026-11-24 — sgemv paired-window core (M=32/lane, A1/A2/A3+B1): verified, slower, reverted

GOAL: port the REPORT.md paired-window extraction into sgemv_run: lane owns 32
consecutive ring windows, pairs (D,D+16)/(D-1)+IADD share one funnel shift,
mul16 reads .H1/.H0 halves directly (3 XMAD, zero BFE/LOP), exact f32 MAC kept
bit-identical via magic FADD(-2^23) (same (float)bs FMA input as I2F). 8 accs
per lane reproduce the OLD per-lane FMA order exactly, so the column tree is
bit-identical (relerr 9.95e-07 == baseline in bench; e2e would be text-identical).

NO load-time transform needed: pairing is translation-invariant. For integer K
the per-lane chunk offset o = (KA+16) mod 32 is identical for all 8 lanes ->
folded into compile-time extraction offsets E_j = D_j + o on raw words
(w0 = g*KA-1 mod NW). For half-K o alternates 16 by lane parity -> per-tile
realign W[m]=shf(R[m+1],R[m],o) (2-4 SHF/32w). All in-kernel, no re-layout.

CPU verification (opt/kbench/exl3_math/plan_pw.cpp, KEPT): greedy staged
planner (aligned -> (d,d+16)+bonus -> d%32==1 IADD -> (d,d+15) -> singles)
generates the emission-ordered op list per K and simulates every lane's
extraction vs ref_window: PASS all 7 K, 512000 checks each, 0 fails. Extract
instr/w: K1.5=0.69, K2=0.44, K2.5=1.03, K3=0.94, K3.5=1.06 (no pair structure
exists for 3.5: 2 aligned+30 singles), K4=0.38, K5=0.94. SASS check on the
compiled K=2 loop: ~7.2 issued instr/w vs 9.47 today (-24%). Mul16 folded to
4-XMAD+BFE until CM was sourced from shared memory (opaque) -> clean 3-XMAD.

MEASURED (sgemv_var pw32 vs f32-i2f baseline, interleaved, real K-mix):
  launch_bounds(256,6): 40 regs + STACK:32 spills -> GU+down 443-495us vs 311us
  launch_bounds(256,5): 48 regs, no stack    -> 370-375us vs 254-263us  (-42%)
  unroll 2:                                 -> 495us (stack 72, worse)
  pw32 relerr 9.95e-07 (GU) 1.27e-06 both stages -> math correct, perf wrong.
  Regs: MoE pw 48 (5 blk/SM, or 40+spills at 6) vs 40 baseline; dense pw 48 vs 40.

WHY IT LOST: ~25% fewer issued instr/weight but ~40% SLOWER -> the 92%
issue-bound model did not transfer. Structural suspects, ranked:
  1. warp=4 tile columns -> ceil(tn/32) grids: down stage 80 blocks, GU 96
     (with 25-10% idle tail groups) vs 144/288 baseline -> fewer resident warps
     on the small real grids; per-tile critical path is 4x longer with only
     ~2-4 LDGs in flight per lane (unroll-1 needed for regs -> no overlap).
  2. 8 accs + up-to-7 stash regs + NR word-index regs -> 48 regs (5 blk/SM);
     forcing 40 spills the inner loop.
  3. giant per-K bodies (K=3.5 all-singles) -> icache pressure across mixed-K blocks.
Lesson: on sm_61 this inner loop is closer to latency/scheduling-bound than the
issued-instr model suggests; M=32's 4x-less-parallelism tradeoff needs a
different shape (e.g., M=16 with 2 lanes/col + xor-reduce, or keeping M=8 with
cross-lane pair sharing via shfl) to keep warp count up.

VERDICT: **REJECTED** — code reverted (kernels.cu, sgemv_var.cu). Kept
plan_pw.cpp (the verified extraction planner: emits + validates the per-K op
lists; reuse for any future paired-window attempt).

## 2026-11-24 — glue fusions round 2 (ctrl cluster + rel-bias gather)

### Per-token ggml op map (INKWELL_GRAPHDUMP, pos=0, 4599 nodes)

OPCNT (node counts/token; views/reshape/permute are free, no kernel):
ADD=184 SUB=144 MUL=410 CONCAT=260 RMS_NORM=394 MUL_MAT=352 SCALE=5
CPY=1 CONT=96 RESHAPE=606 VIEW=823 PERMUTE=240 GET_ROWS=48 SET_ROWS=96
CLAMP=1 UNARY=241 CUSTOM=697 LOG=1 -> ~2900 kernel-launching nodes/token.

Per-visit node skeleton (REXL3 path, all attn projs EXL3):
- norm() = RMS_NORM+MUL -> backend already fuses (rms_norm_mul_f32). The 394
  rms_norm launches already include the weight muls; ~166 MUL nodes are
  likewise already elided, as are the 48 UNARY(sigmoid)+MUL gate muls
  (unary_mul fusion). Remaining real launches below.
- control() x4/msg-visit (24 msg-visits): RMS_NORM, RMS_NORM, CONCAT[4608],
  MUL_MAT down [4608->64], MUL_MAT gf_cat [64->4608], 2x VIEW, 2x
  UNARY(tanh), MUL, SUB, MUL, ADD = 11 launches per site, 44/visit, 1056/tok.
- per visit: sinks MUL_MAT [2304->24], rel_bias MUL_MAT [16,1025]x[16,24]
  (sgemm), SUB+CONT+GET_ROWS+CONT rel-gather (4), set_rows x2, mix CONCATs,
  residual ADD x~1.75, misc.
- generic mul_mat (352): 96 ctrl_down + 96 gf_cat + 48 sinks + 48 gate_inp
  [2304->128] + 48 rel_bias sgemm + 16 conv_proj. F32 weights ~2.4MB/ctrl
  pair; ~300MB weight reads/tok total.
- CUSTOM 697: mz(48)+conv k/v/post/ffn(4x48=192... ~11-12/visit)+clip x3+
  attn+router+moe+mix x2-3+exl3 out.

### Attempt C1 — ctrl-cluster fusion + rel-bias gather (exact-math)

runner.rs `control()`: two custom ops replace 9 launches/site:
- `rms2cat` (inkwell_rms2cat): 2 blocks x 1024 thr, replicates ggml
  rms_norm_f32 vec path (vpt=1, float4, same block_reduce<SUM> tree,
  __fdiv_rn) -> [rms(cur); rms(msg)] in 1 launch (-2/site).
- `ctrl_apply` (inkwell_ctrl_apply): y = (cur - tanh(g3[ng+i])*cur) +
  tanh(g3[i])*msg with __f*_rn intrinsics to block fmad contraction ->
  same op order as tanh+mul+sub+mul+add (-5/site).
- `relbias_gather` (inkwell_relbias_gather): kq_b[j+h*nkv] =
  mm[rel_pos[j]+h*ext] - sinks[h]; replaces SUB+permute/CONT+GET_ROWS+
  permute/CONT (-3/visit).
Launch delta: ctrl -7x96=-672; relbias -3x48=-144; total ~-816 launches.
Killswitches: INKWELL_NO_CTRLFUSE=1, INKWELL_NO_RELGATHER=1 (independent).

### A/B — on master 41d0798 (+gcache), BEST env, 8 pairs ctl,cand,cand,ctl (one flock)

ctl = INKWELL_NO_CTRLFUSE=1 INKWELL_NO_RELGATHER=1; cand = both fused.

| pair | ctl a | cand a | cand b | ctl b |
| 1 | 52.06 | 54.39 | 49.09 | 52.08 |
| 2 | 56.93 | 61.51 | 48.97 | 52.21 |
| 3 | 50.60 | 48.32 | 50.51 | 50.86 |
| 4 | 51.47 | 47.11 | 47.11 | 50.25 |
| median | 52.06 | | | |
cand median 49.09 (-5.7%); adjacent-pair wins 6/8
(W: c2b, c3a, c3b, c4a, c4b — L: c1a 54.39>52.06, c2a 61.51 outlier).
md5 5cbe8e06f2d8d82feb7f2ca287a7bc59 all runs. **ACCEPTED** (commit
1f8c66a). Note: measured on post-gcache master where per-launch bubble
is smaller (~3.4ms union idle) — gain is mostly launch-count-driven.
OPCNT after (fusions on): ADD 88 MUL 218 CONCAT 164 RMS_NORM 202
UNARY 49 CUSTOM 937 SUB 0 CONT 0 GET_ROWS 0 -> kernel-launching
nodes/token 2830 -> 2114 (-716, -25%). CONT/GET_ROWS/SUB fully
eliminated from the per-token graph.

### Brain re-validation on master: glue fusions (ctrl-cluster + rel-bias gather) — ACCEPTED

- Cherry-picked 1f8c66a, bdbf4be and f1dc491 onto e8ed4bb.
- Review fixes, made by the brain:
  - The fused `control()` path assumed gain = g3[0..n) and forget = g3[fo..fo+n) without checking it. It now runs only if `go == 0 && fo == n && ne(cat,1) - fo == n` and falls back to the unfused chain otherwise.
  - `Box::leak` of the forget offset on every graph build was replaced by carrying `ng` in the userdata pointer value.
  - The fusion stayed active after the guard (gain unchanged, below).
- A/B with BEST env + graphs, ctl,gf,gf,ctl ×4 (ctl = `INKWELL_NO_CTRLFUSE=1 INKWELL_NO_RELGATHER=1`):
  - ctl median 51.28 ms/tok (49.76–52.52);
  - gf median 47.74 ms/tok (47.05–52.16);
  - **−6.9%**; gf wins 8/8 neighbouring pairs; md5 5cbe8e06 on all 16 runs; warnings 62.
- Master decode is now ~47.7 ms/tok (~21 tok/s).

## 2026-11-24 — EXL3 decode math Round 2 (real weights, SASS + silicon microbench, `opt/kbench/exl3_math2/`)

Explored 4 directions (`opt/kbench/exl3_math2/{test_cpu.cpp,sass_kernels.cu,count_sass.py,bench.cu,REPORT.md}`), keeping exact shipped 8-tiles/block grid parallelism (144/144 blocks), `<= 40` regs (0 spills), and 6 blocks/SM (fixing `41d0798`'s `-42%` regression from 48 regs + 4× grid shrink):

- **Lossless ~3 instr/w at batch-1 is PROVABLY UNREACHABLE on `sm_61`:** Real weights across all 7 $K$ (65,664 tiles = 16.81M weights) have $H_1(B)=7.999$ b/byte (`zratio=1.0004`, $H(\text{bs})=9.04\text{–}9.25$ b/w) -> incompressible; only 0%-bloat bijections exist. Exact FP32 batch-1 lower bound is `>= 5.63–6.19` instr/w (hand-assembled 2-XMAD SASS) / `>= 6.63–7.19` (stock `ptxas 12.6`, 3-XMAD impossible to lower to 2 via PTX). Achieved: `6.875` (`M=32-Strided C0c`) and `7.219` (`M=16-Paired C0c`).
- **Key silicon discoveries (interleaved min-of-31 vs shipped `inkwell_exl3_gemv_moe_s` under `flock`):**
  - **2-way (`M=16`) / 4-way (`M=32`) in-block `tk`-split** preserves 8 tiles/blk and `<= 40` regs (`0` spills), flipping `41d0798`'s `-42%` into real gains on `K=2.0` (`40.7%` of SAGE MoE):
    - **`M=16-Paired C0c` (exact I2F, L1-rot, 39 regs):** `321.1` vs `280.8 Gw/s` (**`1.14–1.16×`**), `7.6e-7` err.
    - **`M=32-Strided C4` (NEW 0%-bloat Strided-Crumb bijection for `K=2.0` + replicated 16-entry `SHFL` + `ISCADD` = 0 XMADs in loop + `XMAD.S16`, 40 regs):** **`355.9` vs `280.3 Gw/s` (`1.27×`)**, `3.9e-5` err (`180×` below 1 EXL3 quant step; exact `M=32-Strided C0c` ties `M=16` at `319.3 Gw/s`, `1.14×`). Verified 0/0 fails across all 10,368 real `K=2.0` tiles (2.65M weights).
    - **`I2F` (`C0c`) beats Magic `FADD32I` (`C1c`) by +12% on silicon** (`321.1` vs `286.9 Gw/s` despite identical 7.2 instr/w) because `I2F` offloads 1.0 op/w from the saturated ALU pipe onto Pascal's idle SFU/CVT pipe. `C3b` folded per-tile loses (`0.98×`); 64KB/128KB `LDG` LUTs collapse to `17.0–17.4 Gw/s` (`0.06×`).
  - **Full SAGE All-7-K Mix (interleaved vs shipped `~260 Gw/s`):** gains shrink to **`1.03×` exact** (`M=16-Paired C0c` `266.1 Gw/s` / `Hybrid C0c` `267.6 Gw/s`, `5.4e-7`) and **`1.05×` sub-step** (`M=16-Paired C4 blk=16` `274.7 Gw/s` / `Hybrid C4` `274.0 Gw/s`, `2.0e-5`, 0 spills) because heterogeneous-K MoE launches are bounded by non-`K=2.0` wave tail blocks (where 0 `(D, D+16)` pairs exist for `K ∈ {2.5, 3.0, 3.5, 5.0}`). Expected serial MoE e2e upside is only `~0.6–1.0 ms/tok` (`~1–2%`).
  - **Approx Pareto cliff vs EXL3 quant step (`KINV=0.006767`, `rms_rel≈7.0e-3`):** `C0c` (`0.0001×` step) and `C4` (`0.002–0.007×` step) are safe; all cheaper approximations (`C6 dp4a-8` `0.6–1.8×`, `A-BS8` `1.1–23×`, `A-2BYTE` `148×`) jump to/past 1 full quant step; `C3a` fails Dist B (`1.09×` step).
  - **Multi-token batching `B=1..4` (`M=16-Batch C0c`, exact FP32, 40 regs, 0 spills, `2–8.2 KB` smem):** becomes MORE compute-bound per loaded weight (`Rd_Gw/s`: `288 -> 244 -> 203 -> 184`), while effective throughput crosses the `722 G MAC/s` (`<= 3.0` instr/w/tok) barrier at **`B=4`**: `B=2` **`487.9 G MAC/s` (`1.68×`)**, `B=3` **`608.2` (`2.11×`)**, **`B=4` `734.0 G MAC/s` (`2.53×`, `2.828` instr/w/tok, `6.3e-7`)** (`B=4 C4`: `734–741 G MAC/s`, `2.453` instr/w/tok).

## 2026-11-24 — kslot step 0: exposed staging + size census (worktree t3-c88bd85d)

Instrumentation (this commit, all env-gated, off by default):
- `INKWELL_SKIP_CCOPY=1`: ccopy kernels write slot metadata but move no data
  (mode 3). Timing-only: output garbage, graph shape identical.
- `hm` i32[2] per-cache selection hit/miss counters in `inkwell_exl3_cache_resolve`;
  read via `INKWELL_PF_STAT` (now works without prefetch, cumulative per layer).
- `INKWELL_KSIZES=1`: per-layer expert-blob size census at cache alloc.

### Exposed staging (BEST env, k=20; ms/tok = s*1000/64)
| ctl | skip_ccopy |
| 50.53 | 41.71 |
| 51.92 | 41.25 |
| 49.87 | 42.38 |
| median **50.53** | **41.71** |
Exposed staging = **~8.8 ms/tok** — nearly all of the 9.84 ms ccopy kernel time
is on the critical path (side-stream bulk copy does NOT hide it). md5
5cbe8e06 on every ctl run.

### Hit rate at k=20 (62 tok, cumulative PFSTAT)
- prelude L0-5 (1 visit/tok, 496 sel each): miss ~47-50% (!)
- loop L16-17 (3 visits/tok, 1488 sel each): miss 25.3% / 32.5%
- L18-23 (1 visit/tok): miss 23-30%
- TOTAL 3011/8928 = **33.7% miss ≈ 48.6 misses/tok** (higher than THEORY's
  18.6% — that measured the old on_dev={0-11} placement; prelude layers
  have poor k=20 locality).

### Size census (per-proj blob = 10368*w*4+6912 B; class = max over 3 projs)
- classes are few per layer (2-6); distinct sizes {492,654,816,978,1140,1302} KiB.
- sum over 14 cached layers: sum(max)=1707.4 MiB vs sum(esz)=1170.3 MiB
  (ratio 1.46 — matches review's 42.5/28.3 = 1.50).
- k_eff = ck*max*128/sum(esz) per layer: 24.5-34.6, mean 27.9 (vs k=20).
- L16/L17 (biggest experts, 3 visits/tok): k_eff only ~24.6 — least gain where
  traffic is highest. L2/L22: k_eff ~34.
- Notable: L0/L20/L21 are nearly single-class (123-127 experts share one size
  class + 1-5 outliers) — per-class pools behave almost exactly like today's
  LFU there, just with right-sized slots.

## kslot: variable-size expert slots via device free-list (commit b8c0995)

Design (final): each cached layer's buffer = one arena of 6912-B granules
(= suh+svh; every proj blob is exactly 6*w+1 granules). New resolve kernel
`inkwell_exl3_kslot_resolve` keeps today's GLOBAL LFU but slots are variable:
per miss it first-fits 3 extents from a sorted free list (sorted insert +
adjacent merge on free), evicting the least-used unclaimed resident until the
expert fits; nothing evictable -> EXL3_DIRECT. soff[v*3+p] byte offsets are
resolve-maintained and read by ccopy (prefetch bank keeps fixed stride).
Killswitch: `INKWELL_NO_KSLOT=1` restores uniform ck x max_slot.

Rejected first attempt (measured, not committed): per-K-class pools with
equal-resident-fraction waterfill. md5 correct but L17 misses 484 -> 886 —
visits skew toward the large classes, so per-class partitioning starved the
hot pool. Global-LFU + allocator keeps the working-set choice free.

### Results (BEST env, k=20)
- md5 5cbe8e06 on all 26 validation + A/B runs; free VRAM margin 511 MiB.
- Selection misses (62 tok, PFSTAT): 3011 -> 2329 (-22.6%); hit 66.3% -> 73.9%.
  Per layer: L16 377->359, L17 484->465, L22 114->50, L18 118->61, etc.
  (misses/tok ~48.6 -> ~37.6; staged MiB/tok roughly proportional ~83 -> ~64).
- E2E A/B, 16 interleaved ctl,cand,cand,ctl pairs (ctl = INKWELL_NO_KSLOT=1):
  ctl median 50.87 ms/tok, cand median 50.18 (-0.69 ms/tok, -1.4%);
  pair wins 11/16 (69%). Median faster; pair-win rate just under the 75% bar —
  the remaining misses are the bigger blobs and part of the copy is hidden by
  the side stream, so the byte savings translate at ~1/3 of the naive estimate.

### Brain re-validation on master: kslot variable-size expert slots — ACCEPTED

- Cherry-picked 0046981 and 640db2b onto e4d6044.
- Review:
  - the free list holds at most 3·64+1 extents, inside KSLOT_FLMAX = 256;
  - the eviction loop terminates, since each pass evicts one entry or breaks;
  - experts claimed by this token (hits or new allocations) are never evicted;
  - experts that do not fit fall back to EXL3_DIRECT;
  - the prefetch bank keeps the uniform stride.
- A/B with BEST env + graphs on top of the glue fusions, ctl,ks,ks,ctl ×4 (ctl = `INKWELL_NO_KSLOT=1`):
  - ctl median 47.87 ms/tok (46.82–50.84);
  - ks median 46.78 ms/tok (45.77–47.18);
  - **−2.3%**; ks wins 6/8 neighbouring pairs; md5 5cbe8e06 on all 16 runs; warnings 62.
  The child's run on the older base gave −1.4% with 11/16 pair wins.
- Exposed staging measured by the child (INKWELL_SKIP_CCOPY, debug only): ~8.8 ms/tok. It is still the largest single item; the next lever is prefetch.

## 2026-11-24 — M=16-Paired C0c port into shipped sgemv cores (K=2/2.5/3/4): verified, shipped (`INKWELL_NO_PW16`)

PORT of r2's exact FP32 `M=16-Paired C0c` into `inkwell_exl3_gemv_s`, `inkwell_exl3_gemv_sf`,
`inkwell_exl3_gemv_moe_s` (`crates/inkwell-cuda/src/kernels.cu`). The L1 ring rotation is folded
into the lane bit-offset in-kernel (rot bit q == raw bit q+KA-16 mod R) — no load-time data
transform, so ntok/dp4a/v2/dbg readers stay valid. CM reaches the MAC asm from the opaque
`__device__` var `ink_pw16_cm` (the r2 CM trick — a literal refolds to 4-XMAD and costs ~26%).
src[sub-1] for the rotation is fetched via `__shfl` from lane sub-1, not a 3rd LDG (LSU-bound).
New kernel arg `pw16` = per-K enable bitmask; `ink_pw16_mask()` = `INKWELL_NO_PW16 ? 0 : PW16_KMASK`.
2-way in-block tk-split (part=tid>>7), 16 lanes/tile, xor-merge + smem cross-part reduce, epilogue
`fmaf(KINV, acc, (KOFF+510*KINV)*sx)`. Dispatch: `pw = (pw16>>kk)&1` per block (kk=ka*2+hk).

Per-K kbench (DRAM-bound moe_s, distinct tensors, nky=1, interleaved min-of-31, GU 144x72 / DN 72x144):
  K1.5 -8.5/-5.6 | K2.0 +6.0/+8.0 | K2.5 -0.2/+5.0 | K3.0 +3.6/+7.7 | K3.5 -11.8/-4.0 | K4.0 +6.8/+8.5 | K5.0 -19.0/-13.2
Enabled: PW16_KMASK = bits{4,5,6,8} = K2.0, K2.5, K3.0, K4.0. In the r2 harness the K=2 port measured
265 vs shipped 247 Gw/s (+7%); r2's physically-rotated variant reaches ~295 — the ~10% gap is the
in-kernel rotation overhead, accepted to keep all other readers valid.

GOLD (sgemv_gold MoE, sgemv_dense_gold, real dumps): PASS on all shapes/skipmodes; pw16 errs
~0.9-2.0e-6 <= shipped 2.75e-6 (e.g. GU nky=1: 1.99e-6 vs 2.75e-6). Regs: moe_s/gemv_s/gemv_sf all
40 regs, moe_s 0 spills, gemv_sf 4B stack spill; 6 blk/SM maintained. kernels.cu 4164 -> 4494 lines.

E2E A/B (--run, GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10
INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2, flock, 8 pairs ctl,cand,cand,ctl):
  ctl  s: 3.169 3.128 3.053 3.035 3.278 3.169 3.095 3.298  -> median 3.148 (49.2 ms/tok)
  cand s: 3.079 3.044 3.044 3.139 3.072 3.008 3.106 3.069  -> median 3.070 (48.0 ms/tok)
  cand median -2.5%, pair wins 6/8 -> ACCEPT.
  generated text md5 identical ctl==cand (no divergent token).

## 2026-11-24 — Speculative drafter cost model, sm_61 kernels & real acceptance proxy (`opt/kbench/drafter/`)

Full study (`opt/kbench/drafter/REPORT.md`) answering cost model, head/body design, `sm_61` CUDA-graph kernels, reduced-vocab coverage, and real DSpark quantization proxy against `46.8 ms/tok` (`21.37 tok/s`) serial on GTX 1060 6GB:

- **Cost model & binding constraint (`cost_model.py`, anchored to 46.8 ms serial + 384-visit `ids_block.bin` + `exl3_math2` `M=16-Batch C0c`):**
  - Mean union `U(B)` grows `8.00 -> 12.89 (B=2) -> 16.59 (B=3) -> 19.96 (B=4) -> 22.91 (B=5)`, with `61–76%` of union experts hit by only 1 token. Even with an ideal ragged batched MoE verify kernel, `verify(k)` is modeled at `69.9 ms` (`k=1, 1.49×`), `90.4 ms` (`k=2, 1.93×`), `108.9 ms` (`k=3, 2.33×`), `126.1 ms` (`k=4, 2.69×`).
  - Under strict prompt acceptance (`p1=0.75, q=0.18`), **only `k=1 (B=2)` beats serial** (`emit=1.75`, budget `+12.0 ms` break-even / `+4.5 ms` for +10%); `k=2` has negative break-even budget (`−2.2 ms`). Under ledger empirical (`emit(2)=2.10`), `k=2` budget is `+7.7 ms` (0%) / `−1.3 ms` (+10%). Reaching +20% requires ragged `B=2..3` verify AND `p1 >= 0.83` (`k=1`) or `emit(2) >= 2.36` (`k=2`).
  - VRAM theft from expert caches costs `~+0.0148 ms/MiB × (U(B)/8)` per verify round (`+3.1 ms/100 MiB` at `B=3`).
- **Measured `sm_61` kernels (`drafter_kernels.cuh` + `bench_drafter.cu`, verified bit-exact `relerr ~6e-8`, single CUDA Graph, zero host sync, device-side argmax):**
  - `head_argmax_i8_dp4a<3>` (`40` regs, `0` spills, `6` blk/SM): **173.7–176.2 GB/s (85–86% of 204 GB/s DRAM peak)**; `V'=32K H=2304` = `436 us`, `H=1024` = `202 us`.
  - `head_argmax_i4g128_dp4a<1..3>` (`38–40` regs, `0` spills, `6` blk/SM, `+8` unsigned nibble pack + fp16 g128 scales): `V'=32K H=1024` (`16.6 MiB`) = **`108.5 / 115.5 / 146.4 us` (160.6 GB/s)**; `H=2304` (`37.2 MiB`) = `238.6 us` (`163.7 GB/s`).
  - `gemv_i8_dp2a<3>` (int16 act × int8 W, `40` regs, `0` spills, `6` blk/SM): `150.7 GB/s` (+13% time vs `dp4a`). `fused_rmsnorm_quant<i8/i16>`: `3.1–4.1 us`.
  - Resident EXL3 K=4.0 unembed (`unembed.bin`, `0 MiB` extra VRAM): `B=1..4` = **`1.76 / 2.11 / 2.61 / 3.00 ms`** (`273 -> 666 GMAC/s`, `2.44×`).
- **Reduced-vocab coverage (`vocab_coverage.py`, 640 real target greedy tokens across 5 prompts vs independent 4.12M-token corpus ranking `vocab_rank_corpus.bin`):** `4K: 71.4%`, `8K: 77.8%`, `16K: 86.7%`, **`32K: 94.84%`** (Python/tech 100%, prose/Rust 93.8%), `64K: 95.94%`. Naive `id < V'` collapses to `33.1%` at `32K` because Inkling's leading-space English tokens sit in `149860..199999`.
- **Real DSpark acceptance proxy (`acceptance_proxy.cu`, 12 prompts = 84 real draft steps, 84/84 oracle verified):**
  - Full 3L + Markov + `32K` head: **`Int8 + Int16 X (dp2a)` = `84/84 (100.0%)`**, **`Int8 + Int8 X (dp4a)` = `36/36 (100.0%)` on d1..3 / `80/84 (95.2%)` on d1..7**, zero loss in real target `emit`.
  - Head-only `Int4-g128 + Int8 X (32K)` = **`12/12 (100%)` d1 / `34/36 (94.4%)` d1..3** (zero loss in target `emit`), whereas **per-row `Int4` collapses to `67%` head-only / `58%` full on d1** (disqualified).
- **Recommended design:** **`EAGLE-1-Med`** (`1L, D=1024, FF=2816, 16q/4kv hd=64, SWA-128`, 4-tap fuse, symmetric per-row `int8` `dp4a` body) + **tied `V'=32,768` `int4-g128` nibble-unpack head** (`29.9 MiB` VRAM; measured CUDA-graph round **`0.24 / 0.46 / 0.68 ms`** for `k=1/2/3`; modeled **`24.14 tok/s` (+13.0%)** at `k=1` under current acceptance and **`25.08 tok/s` (+17.4%)** with trained EAGLE acceptance). Requires compact ragged `B=2..3` verifier in Inkwell.

### Brain re-validation on master: pw16 (M=16-Paired C0c core) — NOT ACCEPTED, kept opt-in

- Cherry-picked 95b1b50 onto 952e1ff. Review: the reduce scratch reuses `xs` only after a `__syncthreads`; every thread reaches `pw16_reduce`; nky=2 atomics add two terms onto zero, so the result is order-independent; per-block K dispatch keeps mixed-K launches correct.
- nsys, one run each (68 steps, graphs on): total kernel 3791.7 → 3718.9 ms; `gemv_moe_s` 1267.5 → 1230.3 (−0.55 ms/tok), `gemv_sf` 922.5 → 890.1 (−0.48 ms/tok). Kernel gain ≈ **−1.07 ms/tok (−1.9%)**.
- E2E A/B, BEST env, ctl = `INKWELL_NO_PW16=1`, ctl,pw,pw,ctl:
  - batch 1 (8 pairs): ctl 46.45, pw 46.39 ms/tok, 5/8;
  - batch 2 (8 pairs, pre-registered: accept if 16-pair total ≥12/16 and median faster): ctl 49.64, pw 49.37, 6/8;
  - total **11/16**, median 47.97 → 47.04. md5 5cbe8e06 on all 32 runs. Other agents were building concurrently (batch 2 is ~3 ms slower overall).
- Verdict: below the bar. Code stays on master **default off**, opt-in with `INKWELL_PW16=1`. Re-test on an idle machine; the expected effect (~1 ms) is close to the pair noise.
## 2026-11-24 — BF16 ctrl/router weights + ctrl-cluster GEMV fusions (worktree t3-f8755cfb)

### Finding: ggml-cuda mmvf already handles BF16 src0 on sm_61
`mul_mat_vec_f<nv_bfloat16, float>` runs the identical strided col2 loop
and F32 `acc += v*u` (bf16->f32 is an exact 16-bit shift; the file's BF16
was widened to F32 at load). Block reduce is type-independent. So a
loader-only change is bit-exact. supports_op: prec_src1=UNDEFINED ->
passes; should_use_mmvf: sm_61 !mmf -> ne11<=8 -> mmvf. The NT=8 block
path also shifts cublas->mmvf per-column, making block-vs-serial MORE
consistent (same per-column order).

### Change 1 — BF16 storage (commit 2178e4b; killswitch INKWELL_NO_BF16CTRL)
densew! keeps file BF16 for: controllers down/gain/forget (+gf_cat),
mlp.gate.weight, attn_sink_proj, k/v_sconv.weight_proj. dev_bytes sized
to match. All are mul_mat src0 only; rel_bias pad + non-matmul misc stay
F32. VRAM peak 5010 -> 4938 MiB (-72). md5 5cbe8e06 on all runs.
A/B 4x (ctl,cand,cand,ctl): ctl med 48.96, cand med 48.86, wins 3/8 --
inside noise (run sigma ~2.5ms vs ~0.3ms expected kernel saving).
Misses/tok UNCHANGED (13046 cum): kslot budget is pinned by
INKWELL_EXL3_CACHE_K=20, so freed VRAM does NOT auto-grow arenas; it is
headroom for a bigger K or other work.

### Change 2 — gf_cat GEMV + ctrl_apply fused (inkwell_gf_apply;
killswitch INKWELL_NO_GFAPPLY). One warp per output i: 2x 32-pair dots
(gain row i, forget row fo+i) + ctrl_apply tail; replicates mmvf's
warp-per-row 64-col dot + shfl_xor 16..1 -> bit-identical (md5 holds).
Removes the g3 intermediate + 96 mmvf + 96 ctrl_apply -> 96 launches.

### Change 3 — rms2cat + ctrl_down GEMV fused (inkwell_rms2down;
killswitch INKWELL_NO_RMSDOWN). Per output row: 256-thr block emulates
the 1024-thr rms2cat reduce exactly (real thread t owns virtual tids
{t,t+256,t+512}; real warp w covers virtual warps {w,w+8,w+16}), stages
gate[4608] in smem, then the mmvf-256 dot (same pair map, same reduces).
-96 rms2cat + -96 mmvf -> +96 fused launches.

### Combined fusions A/B (ctl = INKWELL_NO_RMSDOWN=1 + INKWELL_NO_GFAPPLY=1,
cand = both on), 4x (ctl,cand,cand,ctl):
ctl: 47.47 45.38 48.79 49.67 47.23 45.71 47.80 47.75 (med 47.61)
cand: 45.42 47.02 46.41 47.25 46.68 46.16 47.32 48.85 (med 46.85)
-> -0.76 ms/tok (-1.6%), wins 5/8 -- under the 6/8 gate, trend positive.
md5 5cbe8e06 on all 16 runs; warnings 62; VRAM peak 4934 MiB (<5120);
misses 13046 cum unchanged. Launches/tok: -192 vs unfused ctrl chain
(2 fused ops/site replacing 4). Serial sanity path: 48.30 base ->
47.62 (bf16) -> 46.38 (+gf) -> 46.22 (+rms2down).
Commits: 2178e4b (BF16 storage), a25ff40 (gf_apply), 282ef33 (rms2down).

### Brain re-validation on master: BF16 ctrl/router + gf_apply + rms2down — ACCEPTED

- Rebased 2178e4b..4c791b8 onto 7e047e6 (pw16 opt-in included). Review: BF16 src0 goes through ggml-cuda mmvf with the same F32 accumulation order; both fusions copy the mmvf reduce tree. Killswitches INKWELL_NO_BF16CTRL / INKWELL_NO_GFAPPLY / INKWELL_NO_RMSDOWN.
- A/B with the pw16 env (MAX_CTX=512, CACHE_K=20, MOE_V2, NKY=2, graphs), ctl,bf,bf,ctl ×8, ctl = all three killswitches:
  - ctl median 48.09 ms/tok, bf median 46.58 ms/tok, **−1.51 ms (−3.1%)**;
  - bf wins **14/16** neighbouring pairs; md5 5cbe8e06 on all 32 runs; warnings 62.
- VRAM peak 5010 → 4934 MiB (child measurement). The 76 MiB freed are not yet spent on caches (CACHE_K pins the arena).

### Pre-gate expert prefetch (branch t3/pregate-expert-prefetch, f9715d4) — REJECTED, not merged

- d=1 pre-gate on fcur claims the top-M experts of the next layer in its kslot arena and copies them on the side stream. Killswitch INKWELL_NO_PREGATE; md5 identical.
- Misses fall ~3.6× (37.6 → ~15/tok at M=12), but the token gets slower: M=12 59.2–62.5 ms, M=8 54.0 ms, vs 46.78 baseline.
- Cause: PCIe 3.0 bytes are the cost. Claims move ~125 MB/tok against ~86 MB/tok of demand; the 1.24× prediction waste is wire time, and the ~0.7 ms window between issue and join per visit cannot hide 0.5–0.9 ms of copies. A ≥2-visit predictor would be needed, and step 0 found none.
- Lesson: on this box, fewer bytes over PCIe wins (cache sizing, smaller streamed experts), not earlier bytes.
## 2026-11-24 — Per-layer expert-cache sizing (worktree t3-cb1d9264)

### Method
- `INKWELL_IDSEQ=<path>` (moe.rs): stream-ordered `cudaMemcpyAsync` + sync on
  the compute stream appends `[layer i32][8 i32]` per MoE visit; eager only
  (cb runs once per capture under CUDA graphs, so trace with graphs off).
  A default-stream copy raced the router kernel — first trace was garbage.
  Traces: 68 tok default prompt, 74 tok robot-paint prompt.
- `opt/tools/kslot_replay.py`: gexp granules from safetensors headers
  (`w=8*(k16/16)+4*(k16%16!=0)`, blob=6912*(6w+1)); replays the exact
  `inkwell_exl3_kslot_resolve` policy (first-fit extents, global least-cnt
  eviction, u64 claimed, kmax<=64). Validated vs `INKWELL_PF_STAT`: per-layer
  misses within ~2% at ck=20.
- `opt/tools/kslot_alloc.py`: per-layer miss-byte curve (4 MiB steps) then
  greedy marginal missMiB/MiB; hand-checked ~10 variants on both traces.
  Greedy948 (t1 52.7) lost to F964 (51.7); F964 peaked 5076 MiB (4 MiB under
  cap) so the default table is F-mid trimmed to 940 MiB (V1).

### Table (streamed layers; sim on 68-tok trace)
| L  | visits | MiB before->after | hit% before->after | missMiB/tok before->after |
|----|--------|-------------------|--------------------|---------------------------|
| 0  | 68     | 47 -> 56          | 59.4 -> 66.7       | 6.24 -> 5.11 |
| 1  | 68     | 47 -> 56          | 65.4 -> 72.2       | 5.31 -> 4.27 |
| 2  | 68     | 57 -> 56          | 73.9 -> 74.1       | 3.30 -> 3.28 |
| 3  | 68     | 47 -> 48          | 59.9 -> 59.9       | 5.34 -> 5.34 |
| 4  | 68     | 57 -> 56          | 68.9 -> 68.6       | 4.05 -> 4.13 |
| 5  | 68     | 57 -> 64          | 64.3 -> 68.2       | 5.52 -> 4.86 |
| 16 | 204    | 76 -> 156         | 76.0 -> 92.8       | 17.42 -> 5.02 |
| 17 | 204    | 76 -> 156         | 68.9 -> 91.7       | 22.81 -> 6.04 |
| 18 | 68     | 57 -> 48          | 88.4 -> 85.3       | 1.62 -> 2.03 |
| 19 | 68     | 57 -> 44          | 86.2 -> 83.5       | 1.86 -> 2.25 |
| 20 | 68     | 47 -> 48          | 82.2 -> 81.8       | 2.75 -> 2.81 |
| 21 | 68     | 57 -> 52          | 84.9 -> 83.5       | 2.32 -> 2.54 |
| 22 | 68     | 66 -> 48          | 90.6 -> 84.0       | 1.45 -> 2.47 |
| 23 | 68     | 57 -> 52          | 83.6 -> 80.3       | 2.52 -> 3.02 |
Total 812 -> 940 MiB; sim miss bytes 82.6 -> 53.2 MiB/tok (trace1),
94.1 -> 71.5 (trace2). Maxing L16/L17 at cap while starving small layers
was WORSE on both traces (61.1/74.6) — the LFU curves flatten.

### Result (real, INKWELL_PF_STAT cumulative ~63 tok)
misses ~2329 -> ~1745 (-25%); L16 359->112, L17 465->127. ccopy bytes
scale with miss bytes (~-35 MiB/tok predicted).

### A/B (8 interleaved pairs ctl,cand,cand,ctl; ctl = INKWELL_NO_CACHEL=1)
ctl:  50.00 46.12 46.36 48.07 49.60 51.82 46.34 51.46 (med 48.84)
cand: 44.06 41.68 45.38 42.88 41.49 47.19 44.23 44.47 (med 44.15)
-> **-4.69 ms/tok (-9.6%), wins 8/8.** md5 5cbe8e06 on all 16 runs.
VRAM peak: ctl 4934-4962, cand 5052 (cap 5080). warnings 62.

### Data only: L15 streamed (not shipped, not A/B'd)
Streaming L15 frees ~324 MiB of resident experts; re-spent as cache
(L15=120, L16/17 at ~210 cap) the sim predicts 53.2 -> 50.6 MiB/tok (-5%).

Commits: see HEAD. Killswitch `INKWELL_NO_CACHEL=1` (uniform ck sizing);
`INKWELL_CACHE_MB_L="0:56,16:156,..."` per-layer override.

### Brain re-validation on master: per-layer expert-cache sizing (INKWELL_NO_CACHEL) — ACCEPTED

- Rebased 8c51c01 onto 98d30c8 (BF16 ctrl + fusions included). Review: the table is MiB per layer; resident layers skip it, so the coming quant with L16/L17 resident drops those entries by itself. The other entries need a re-run of opt/tools/kslot_replay.py on the new quant. A layer that does not fit is still skipped whole (same as uniform sizing).
- A/B, the usual env, ctl,cl,cl,ctl ×8 (ctl = `INKWELL_NO_CACHEL=1`):
  - ctl median 45.60 ms/tok, cl median **42.98 ms/tok**, **−2.62 ms (−5.7%)**;
  - cl wins **16/16** neighbouring pairs; md5 5cbe8e06 on all 32 runs; warnings 62.
- VRAM peak 5,052 MiB (child measurement), under the 5,120 cap.

### Brain re-validation: register-preloaded inkwell_mix + had_in merge (branch t3/mix-kernel-multiblock-bitexact, 69e97b1) — PENDING, noisy A/B

- The child's multi-block mix profiled without `--cuda-graph-trace=node` and saw 240 launches instead of 8,160; with node trace it was slower (2.66 → 3.11 ms/tok).
- Rewrite: one warp per (column, query) with the lane's 72 values preloaded into registers (156 regs, 0 spills) and post loads batched 8 rows at a time; same op order. nsys (node trace, 68 steps): mix 2.75 → 1.45 ms/tok, had_in 1.18 → 0.60 ms/tok. md5 5cbe8e06.
- 16-pair A/B: ctl 48.38, mx 47.98 (−0.40), 11/16, with runs spread 43–67 ms (4 agents active, zram swapping). Not accepted yet; re-run on a quieter box.
- Re-run on master 3ad8488 (cache sizing in): ctl 45.37, mx 44.07 (**−1.31 ms**), 11/16 again; one 59.7 ms outlier. Pooled 22/32, under the 24/32 bar. Kept on the branch; re-run when fewer agents share the box.

### Dense W4/W5/W6 group-wise GEMV kernel study (`opt/kbench/w4dense/REPORT.md`, sm_61) — READY FOR QUANT KLD CHECK

- Tuned `N32`-tiled F32-activation W4/W5/W6 kernels (`LDG.128`, opaque-AND `I2F.F32.U8` 0-shift unpack, broadcast smem `x` with 0 bank conflicts, in-block split-K, fused `unembed+argmax`) hit **2.65–2.70 SASS ins/w** for W4 (`3.28–3.33` for W5/W6, 40 regs, 0 spills, 6 blks/SM) and **172.3 GB/s** on `unembed` (96.4% of raw `LDG.128` STREAM 178.8 GB/s; zero-compute 4.125 bpw floor = 5.80 ms/tok).
- Modeled dense time (`48×attn6 + unembed`, 31 interleaved reps vs shipped EXL3, CPU FP64 verified `fails=0`):
  - **`W4_g128_sym` (4.125 bpw):** `6.98 ms` vs `8.85 ms` bare `gemv_s` (**1.27×, −1.87 ms**) / `~10.24 ms` shipped `had_in+gemv_sf` (**1.47×, −3.26 ms**; up to −3.3..−5.2 ms at warm SM clocks), **+43.43 MiB** VRAM (+38.07 attn, +5.36 unembed).
  - **`W4_g128_asym` / `W4_g64_sym` (4.250 bpw):** `7.09 / 7.24 ms` (**1.44–1.45×, −3.18 ms** shipped), **+59.89 MiB** VRAM.
  - **`W4_g128_dp2a` (int16 x):** `6.71 ms` (−0.27 ms vs F32, 100× worse rel err `4.2e-5` vs `3.8e-7`) → keep exact F32.
  - **`W5` (`8.12 ms`, +175 MiB) & `W6` (`9.28–9.52 ms`, +307 MiB):** rejected globally (too much VRAM on 6 GB; `W6` slower than bare EXL3).
- Verdict: not ~2× (capped by 192 GB/s DRAM), but **−3.2 ms/tok for +43–60 MiB** fits inside existing headroom; adopt `W4_g128_sym` or `W4_g128_asym`/`W4_g64_sym` if quantizer meets KLD ≤ 0.0475 and top-1 ≥ 94.66%. Format contract in `opt/kbench/w4dense/REPORT.md` §6.

## 2026-11-24 — exl3_gemv_moe_s tail balance study (worktree t3-1f0e2b0b, branch t3/exl3-moe-tail-balance)

Task: shrink `inkwell_exl3_gemv_moe_s` (~18.6 ms/tok, ~114 launches/tok ~163us)
with BIT-IDENTICAL output. GPU shared under flock; all e2e runs use the A/B env
(GGML_PASCAL_GRAPHS=1 MAX_CTX=512 CACHE_MARGIN_MB=10 EXL3_CACHE_K=20 MOE_V2=1 MOE_NKY=2).

### Tail model and what was measured
- Launch structure: GU (9,nslots=8,2)=144 blocks fused gate+up; down (18,8,1)=144
  blocks; cached path splits GU into hit/miss launches. 256 thr/block, 40 regs,
  9.2KB smem -> 6 blocks/SM = 60 resident on 10 SMs -> ~2.4 waves/launch.
- Per-block cost hypothesis was bytes ∝ r_bits_words (K): real dump K mix gives
  words 12..32 per (slot,zsel) item. LPT (heavy-first) sim vs identity order
  predicted 0-6% makespan gain.

### Change 1 — heavy-first ordering (commit fbedae1, opt-in INKWELL_MOEBAL)
Each warp ranks all P<=32 items by r_bits_words via shfl (lane<P holds key,
rank = #{j: kj>ki || eq && j<i}, ballot selects rank==p). Only block->work map
changes -> bit-identical by construction.
- GOLD: sgemv_gold bal/bal2 bitexact PASS (nky=1/2), skipmode bal0/bal1 PASS,
  all relerr <=2.1e-6. e2e md5 5cbe8e06 all runs.
- KERNEL TIMING (moebal_bench, real dumps, nky=1): GU 173.9->173.5us (+0.2%),
  down 94.2->95.8us (-1.7%). nky=2 both worse. VERDICT: NEUTRAL/NEGATIVE.
  Block cost does NOT scale with K strongly enough (kernel is issue/latency-
  bound, not BW-bound) -> ordering pays ~nothing, just adds prologue cost.
  Kept opt-in default OFF. E2E A/B skipped on kernel-level evidence.

### Change 2 — INKWELL_SGEMV_NKY=2: kernel-timing REJECTED
GU 174.9 (nky=2) vs 172.0 (nky=1); down 99.3 vs 93.1us. Doubling blocks doubles
prologue+epilogue cost and loses ~4-6%. Not e2e-tested further. (md5 did hold
on a single sanity run, but speed already loses.)

### Change 3 — MISSPAR (miss-chain on side stream ss): e2e A/B REJECTED, reverted
Moved GU had_in/gemv miss-skipmode=2 work onto the bulk-copy stream ss right
after the trel copy it depends on (concurrent with GU-hit tail instead of
serialized behind ev_join). Same kernels/outputs -> bit-identical.
- A/B 8x(ctl,cand,cand,ctl), ctl=INKWELL_NO_MISSPAR=1:
  ctl med 47.47 ms, cand med 47.75 ms (-0.28 ms), pair wins 8/16 -> FAILS bar.
  md5 5cbe8e06 all 32 runs. Reverted (miss gemv was never the critical path:
  the bulk copy dominates the ev_join wait).

### nsys tail data (master config, 64 tok, graphs on, cuda-graph-trace=node)
inkwell_exl3_gemv_moe_s: 7752 launches (~121/tok), total 1250.4 ms (~19.5 ms/tok),
avg 161.3us, med 134us, min 2us (empty miss grids), max 1.25ms, sd 107us.
Duration spread is dominated by WHICH slots are live (hit vs miss grids), not
just K order; GU-miss launches with few live blocks are latency-floor bound.

### pw16 re-validation on this tree — NOT ACCEPTED (11/16), stays opt-in
A/B 8x(ctl,cand,cand,ctl), cand=INKWELL_PW16=1 (default off), all md5 5cbe8e06:
  ctl med 46.19 ms, cand med 45.75 ms (+0.44 ms), pair wins 11/16.
Same verdict as the earlier master run (11/16): median faster but under the
12/16 bar. My balance work did not land (ordering opt-in, misspar reverted), so
this single A/B covers both "before" and "after" states. PW16_KMASK unchanged.

### Status
- kernels.cu 4797 lines, build warnings 58 emitted (62 incl. summary lines) <= 64.
- Commit fbedae1: moebal ordering, OPT-IN only (INKWELL_MOEBAL=1 / NO_MOEBAL=1).
- Everything else reverted or env-only. Net e2e change on master: none.
- Remaining tail idea (untested): miss-gemv is copy-latency bound, not
  scheduling bound; the real gap is the ~160us avg launch cost itself being
  issue-bound (sgemv inner loop instr/weight), addressed by the pw16 core --
  which needs a cleaner A/B on an idle machine to clear 12/16.
## host-latency pass (step_id, commit 700aecd)

nsys (`--trace=cuda,osrt,nvtx --cuda-graph-trace=node`, baseline best env, /tmp/inkwell_host_base):
per-token GPU idle on the critical path, steady state:

| gap | us/tok | cause |
|-----|--------|-------|
| logits DtoH (800KB) -> next rms_norm | ~895-1159 | host argmax + top5 + 3x unbuffered eprintln + Vec allocs + 10x synced t::set input uploads + embd dequant + upload + graph-key hash + ~6 chunk launches (runner.rs step_cached, main.rs gen loop) |
| last graph node -> logits DtoH | ~53-266 | sched_synchronize wake + Vec alloc + t::get issue |
| inside-token small HtoD chains | ~10-14 x several | per-tensor cudaMemcpyAsync+sync on cudaStreamPerThread |

Fixes shipped in step_id fast path:
- `inkwell_argmax` kernel (kernels.cu): IEEE totalOrder key map + last-max
  tiebreak = host `max_by(total_cmp)` exactly; reads the `out` tensor in
  place, writes {idx,key}, 4B DtoH. Killswitch INKWELL_NO_GARGMAX.
- `ink_sched_compute_async` shim -> `ggml_backend_sched_graph_compute_async`:
  launches splits/chunked cudaGraphExecs without the final all-backend sync;
  host blocks on a cudaEvent recorded after the 4B argmax readback.
  Killswitch INKWELL_NO_ASYNCSTEP (per-tensor synced uploads + blocking
  sched compute, still uses device argmax).
- inputs packed into a pinned mirror of in_buf -> ONE async HtoD covering
  ids/pos/kv_idx/rel_pos/masks/sink_zero (was ~9 synced t::set); embd row
  dequant into pinned stage -> one async HtoD (was dequant + synced set).
  in_pin is double-buffered by pos parity so the next step's pos-dep
  marshal overlaps the running graph (premarshal after launch, pre-sync).
- per-token eprintln!/top5/argmax-scan behind INKWELL_STEPDBG (off by
  default); stepdbg mode keeps the exact old per-token output.
- argmax + readback queued on the ggml stream right after the replay, so
  the GPU-side tail is ~35us instead of ~50-270us.

A/B: see below (ctl = NO_GARGMAX+NO_ASYNCSTEP+STEPDBG, cand = default).

A/B results (16 neighbour pairs, ctl = NO_GARGMAX+NO_ASYNCSTEP+STEPDBG,
cand = default step_id path; batch 2 ran under host load from sibling
agents, so absolute times drift up but pair ordering holds):

ctl  : 46.09 44.49 45.44 45.58 45.52 45.74 45.51 45.95
       47.02 49.35 45.95 48.84 46.54 50.48 51.06 52.15  ms/tok
cand : 44.95 45.26 44.65 44.50 44.87 44.90 44.18 44.85
       46.54 49.00 44.38 49.69 44.64 49.38 47.76 50.87  ms/tok
median ctl 46.02, cand 44.93 (-1.09 ms/tok, ~-2.4%)
pair wins: 14/16 (>=12 required)  md5: 5cbe8e06 on all 32 runs
verdict: ACCEPTED (commit 700aecd)

Remaining boundary budget after this fix (est): event wake + ids write +
embd dequant (~20us) + 2x HtoD issue + prefetch + graph-key hash + ~6
chunk launches + argmax tail ~35us = ~0.15-0.25ms/tok, down from ~1.0-1.4.
