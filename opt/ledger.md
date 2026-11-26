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

## 2026-11-24 — host-latency pass (step_id) accepted on master

Brain re-A/B on master 3f60c55 (16 pairs, ctl = `INKWELL_NO_GARGMAX=1 INKWELL_NO_ASYNCSTEP=1 INKWELL_STEPDBG=1`):
ctl median 45.22, cand 43.16 ms/tok, **−2.06 ms, 12/16** pair wins, md5 5cbe8e06 on all 32 runs.
Fast-forwarded master to `t3/host-latency-token-critical-path` (61ab1dc + ledger). Killswitches
`INKWELL_NO_GARGMAX`, `INKWELL_NO_ASYNCSTEP`; per-token prints now behind `INKWELL_STEPDBG`.
## 2026-11-24 — Multi-block inkwell_mix + merged had_in (worktree t3-3ed86fc4)

### Target 1: multi-block inkwell_mix — NOT ACCEPTED, opt-in INKWELL_MIXMB=1 (3171b17)

- `inkwell_mix<<<1,256>>>` split into `mix_pre` (one warp per column over
  many blocks; per-column lane-strided sums + xor-shuffle trees verbatim)
  and `mix_post` (per-block redundant sequential softmax, output split
  over SMs by element). 4-slot rotated `__device__` scratch (9 KiB).
- nsys, BEST env, one run each: `inkwell_mix` 240 x 27.8us avg (19.2 med)
  = 6.67 ms -> `mix_pre` 240 x 17.7us + `mix_post` 240 x 4.6us = 5.37 ms.
  Kernel saving ~20us/tok. NOTE: measured 3.75 mix launches/tok here, not
  the 120/tok in the task brief — the real total is ~0.10 ms/tok, not 2.64.
- E2E A/B (ctl,cand,cand,ctl) x8, both arms INKWELL_NO_HADMERGE=1,
  ctl = INKWELL_NO_MIXMB=1:
  - ctl: 2.913 3.135 3.000 2.976 2.911 2.936 3.304 3.233 3.217 3.109
         3.021 3.005 3.149 3.014 2.946 2.913 (med 3.009s = 47.02 ms/tok)
  - cand: 3.036 2.962 2.993 2.931 2.937 2.953 3.239 3.318 2.988 3.352
          3.083 3.411 3.445 3.126 2.887 3.001 (med 3.018s = 47.16 ms/tok)
  - wins 8/16. md5 5cbe8e06 on all 32 runs.
- Verdict: real -19% kernel time but ~25x below the E2E noise floor on this
  shared machine; below the bar. Ships default-off, opt-in INKWELL_MIXMB=1
  (INKWELL_NO_MIXMB vetoes).

### Target 2: merged had_in — NOT ACCEPTED, opt-in INKWELL_HADMERGE=1 (7743082)

- `inkwell_exl3_had_in_n` (HadNArgs by value, up to 8 members) does the
  identical 128-point butterfly per member. mz groups projections by
  identical (x,k): leader's stream runs one merged had_in before member
  gemv tails (`exl3_gemv_tail` factored out of ink_launch_exl3_gemv).
  ntok merges its 8 sequential had_ins into one launch.
- nsys, same env: `had_in` 578 x 3.3us = 1.89 ms -> `had_in_n` 208 x 4.2us
  + `had_in` 98 = 1.12 ms per run (-41%, ~12us/tok; launches -272).
- Verdict: same noise-floor problem; no separate A/B run since the kernel
  saving (~0.77 ms/run) is under the measured run-to-run sigma. md5
  5cbe8e06 on all opt-in runs. Ships default-off (INKWELL_NO_HADMERGE
  vetoes).

warnings 59 (47 run + 11 cuda + 1 cli); kernels.cu 4905 lines; VRAM
+9 KiB static scratch only (no new allocations); commits 3171b17, 7743082.

## 2026-11-24 — fused inkwell_mix + had_in merge accepted on master

- `inkwell_mix_fused`: one 256-thread block per (2304/256 column block, query column). Each warp preloads its 72-value
  slices into registers, computes rms and the dot product per row in the old op order, thread 0 runs the softmax
  sequentially, then each thread accumulates its column in row order. md5-identical. Killswitches `INKWELL_NO_MIXMB`,
  `INKWELL_NO_HADMERGE`.
- nsys (node trace, 68 steps): inkwell_mix 2.66 → 1.17 ms/tok; had_in 1.18 → 0.56 ms/tok (−2.1 ms/tok kernel time).
- A/B history (ctl = `INKWELL_NO_MIXMB=1 INKWELL_NO_HADMERGE=1`, 16 pairs each):
  - on 3f60c55 (before the host pass): −1.26 ms, 11/16 — under the bar;
  - on d8c64ee (host pass in): ctl 43.12, mx 42.11, **−1.02 ms, 15/16**, md5 5cbe8e06 on all 32 runs — ACCEPTED.
- Control spread was 42.1–48.8 ms here vs 42.2–51.1 in the previous run; whether the host pass lowered the noise is not shown.
### `mul_mat_vec_f` & MoE Hadamard kernel study (`opt/kbench/mmvf_had/REPORT.md`, sm_61) — READY FOR IMPLEMENTATION

- Attribution (`/tmp/nsys_mix_on.sqlite` vs `/tmp/nsys_pw16_ctl.sqlite`, 50 steady-state steps): BF16 controller fusions already cut `mul_mat_vec_f` from `3.357` → `0.668 ms/tok` (112 launches: 48× `mlp.gate` `0.396`, 48× `attn_sink_proj` `0.209`, 16× `k/v_sconv` `0.063`), paired with `inkwell_router` (`0.837`), `rms2down` (`1.030`), `gf_apply` (`0.701`), `sgemm` relbias (`0.464`), and MoE `had_in`/`had_out` (`1.845 ms/tok`, 69% on `on_dev` layers).
- **Headline finding (`H1`, BIT-EXACT, −3.5 to −4.20 ms/tok, +120 KiB VRAM):** `runner.rs:1129–1139` allocates `descs_g/u/d` in `host_alloc_mapped` (PCIe zero-copy) even when `on_dev == true` (10 loop layers = 30 visits/tok), stalling 480 CTAs/visit on uncached PCIe reads. Real EXL3 microbench (`H0` PINNED `422.73 µs` → `H1` VRAM `282.61 µs`, `−140.1 µs/visit` = **−4.20 ms/tok**: −1.31 ms Hadamard + −1.62 ms GU + −1.43 ms DN); corroborated within 4% by `nsys` (`ON_DEV` DN `131.03` vs `CACHED` `85.35 µs` = −45.68 µs vs bench −47.74 µs).
- **Additional BIT-EXACT wins (`bit_diffs=0`):**
  - `M1b-4` fused `mlp.gate + router_par_128` (`gate_router_fused_exact<4>`): `20.70 → 11.12 µs` (**−0.46 ms/tok** bench / **~−0.55 ms** nsys; matches reordered LDG.128 `11.14 µs`).
  - `M2b-4` + `M2a-4` 0-smem `rms2down_nosmem_mr<4>` (18.7 KiB → 640 B smem) + 3× batched attn controller triads: **−0.33 ms/tok** (up to ~−0.40 ms with by-value struct args).
  - `H3e` clean 6-kernel MoE chain (shipped 40-reg `gemv_moe_s` untouched + `had_in_moe_vec` + `had_out_moe_vec` + `situ_had_in_moe` + `had_out_wsum_moe`): `282.61 → 277.71 µs` (**−0.235 ms/tok**). Fusing `had_in/out` *into* `gemv_moe_s` (`H2`/`H3a`) regressed **+18–20 µs/visit** (16 B stack spill + 2 extra barriers) — rejected.
  - `M4-1` batched `k/v_sconv`: **−0.014 ms/tok**.
- **Optional FLAGGED REORDERED win:** `M3a` `sink_relbias_fused` (BF16 `rel_bias_pad`, `max_abs=4.77e-7` vs cuBLAS K=16): `9.15 → 3.10 µs` (**−0.29 ms/tok** bench / **~−0.55 ms** nsys, **−512 KiB VRAM**).
- **Total strict BIT-EXACT savings: ~−4.6 to −5.25 ms/tok** (projected decode ~37.8–38.4 ms/tok); with `M3a`: ~−5.1 to −5.8 ms/tok (~37.2–37.8 ms/tok).

## 2026-11-24 — Review of master spec v3.0 (`opt/REVIEW_SPEC_v3.md`)

- Format and contracts match QUANT_BRIEF; most runtime numbers check out. Wrong: "smaller non-loop experts" (v3 makes them +436 MiB, est. misses 53.6 → ~59–100 MiB/tok), Option B "13.7 MiB smaller" (+44 MiB vs today), the −1.07 ms pw16 gain (not accepted), Drafter B "fits 128K".
- 32K is not budgeted. Decode attention fails above 12,096 tokens (all scores in shared memory, 48 KiB/block on sm_61; probe `opt/kbench/attn_ctx`: n_kv 12,160 → `invalid argument`). Below that it costs 0.18 / 0.72 / 2.75 ms per visit at 1K / 4K / 8K: ~13 ms/tok at 4K and ~29 ms/tok at 8K, vs 0.93 ms in the short-prompt bench.
- F16 KV at 32K adds +1,008 MiB, so caches go 940 → 0 MiB under the 5,120 cap (replay: 336.8 MiB/tok of misses, +25 to +45 ms); the loader does not enforce the cap. Trims (lazy `Exl3Blk`, `dev_buf` slack, `gf_cat` copy) give back ~282 MiB of caches → ~168 MiB/tok.
- Seven lossy steps lack a no-regression gate vs 0.0475 / 94.66%; drafter labels are dumped before the shipped quant is final; H200 Window 1 estimate 6–17 h vs the spec's 6.2–9.75 h; Canary 2C cannot run the spec's ~80 GB recipe on an L4.

## 2026-11-24 — device-resident EXL3 MoE descriptor tables accepted

- H1 implementation: each layer's immutable `descs_g/u/d` tables (128 × 32 B)
  now default to `DevBuf` device memory instead of pinned host zero-copy.
  `INKWELL_NO_DEVDESC=1` restores the old host-mapped allocation. All MoE
  single-token, cached, ntok, staging/ccopy and prefetch call sites receive the
  device pointer. The exl3-v2 host eligibility scan was moved to load time
  (`v2_descs_ok`) because the launchers can no longer dereference descriptor
  memory on the host; resolve never consumed descriptors. No dense,
  router/controller, expert-blob load, or resolve-kernel changes.
- Build: `cargo build --release` warnings 63 (master budget ≤64).
- Correctness: generated-text md5 `5cbe8e06` on smoke, both nsys runs, and all
  32 A/B runs. Peak process VRAM 5052–5064 MiB, below the 5120 MiB cap
  (+288 KiB nominal descriptor VRAM, below 1 MiB measurement granularity).
- nsys 2024.5.1, `--trace=cuda --cuda-graph-trace=node`, BEST env, 64 decode
  tokens, totals/64:

| kernel | ctl pinned ms/tok | cand device ms/tok | diff |
|---|---:|---:|---:|
| `inkwell_exl3_gemv_moe_s` | 18.314 | 15.893 | −2.421 |
| `inkwell_exl3_had_in_moe` | 1.146 | 0.512 | −0.634 |
| `inkwell_exl3_had_out_moe` | 0.787 | 0.341 | −0.446 |
| target total | 20.247 | 16.746 | **−3.501** |

  Launches unchanged: gemv/had_in 7752 (121.12/tok), had_out 6528 (102/tok).
  Profiled decode spans: ctl 3.103184214s, cand 2.59693755s.

- A/B: 16 interleaved neighbour pairs in run order ctl,cand,cand,ctl ×8,
  one flock, ctl = `INKWELL_NO_DEVDESC=1`:

| rep | ctl1 | cand1 | cand2 | ctl2 |
|---:|---:|---:|---:|---:|
| 1 | 40.92 | 37.81 | 39.66 | 41.42 |
| 2 | 42.62 | 38.01 | 38.38 | 43.75 |
| 3 | 42.32 | 41.46 | 38.29 | 42.42 |
| 4 | 41.50 | 39.76 | 38.42 | 44.45 |
| 5 | 42.53 | 37.99 | 37.75 | 41.63 |
| 6 | 40.34 | 38.43 | 37.28 | 42.94 |
| 7 | 41.98 | 36.87 | 37.56 | 41.64 |
| 8 | 41.03 | 37.95 | 38.98 | 41.25 |

  ctl median **41.81** ms/tok; cand median **38.15** ms/tok;
  candidate **−3.66 ms/tok (−8.8%)**. Pair wins **16/16** (bar ≥12/16).
  md5 `5cbe8e06` on every run. Verdict: **ACCEPTED**.

  **Brain re-validation on master (3fa931c base, 2026-11-24):** 16 pairs,
  ctl median 41.54, dd median 38.28, **−3.25 ms/tok, 15/16** pairs (one dd run
  exited with no output; 6 follow-up runs with full logs all passed with md5
  `5cbe8e06`; the same empty-exit happened once on the startup branch, which
  lacks this change, so it is not attributed to devdesc). ACCEPTED.

## 2026-11-24 — startup-to-first-decode branch (t3/startup-to-first-decode, 0d48987): A/B on master base — HELD for owner

- ctl = master-equivalent binary (devdesc worktree, same crates as 1218f58); cand = branch rebased on 1218f58.
  BEST env, 3 rounds of ctl,st,st,ctl, one flock. md5 `5cbe8e06` on all 12 runs.
- Peak RSS (VmHWM): ctl median **5,148 MiB**, cand **4,590 MiB** → **−558 MiB**, lower in **6/6** pairs.
- Launch → first decode: ctl median 21.24 s, cand 23.21 s → **+1.97 s**, slower in **5/6** pairs.
  Wall: 24.92 → 27.00 s.
- Decode: ctl 38.00, cand 37.77 ms/tok (no regression).
- Verdict: the branch's stated goal (startup time) is not met; the RSS cut is real. Not merged;
  the RAM-vs-startup trade-off goes to the owner.

## 2026-11-24 — production SASS per K (nvcc 12.6, sm_61), for MATH_CHALLENGE_STEP2

- `opt/tools/sass_loops.py` on `cuobjdump -sass` of kernels.cu at 60c73af. Main loop = 4 tiles × 8 windows = 32 w/lane.
- `gemv_moe_s` (40 regs): K1.5 8.41, K2 8.44, K2.5 9.22, K3 9.34, K3.5 9.47, K4 9.44, K5 9.38 instr/w;
  expert-weighted **8.81**. `gemv_sf` (48 regs) within 5 instr per loop; `gemv_s` 40 regs.
- Every loop: XMAD = 96 (3/w), IDP = I2F = FFMA = 32. All LOP32I are 0xffff window masks; SHR is the D=0 `>>16`.
- PTX forms tried for a 2-XMAD mul1 (opaque c × (v>>16), cvt.u32.u16 + mul.lo, mad.lo with 0x83DC0000):
  all ≥3 instructions. 2-XMAD needs a SASS rewrite → brief `opt/MATH_CHALLENGE_STEP2.md`
  (count model: experts 8.81 → 7.58 instr/w).
- K=6 trellis tensors are vision-encoder only (self_attn.*, mlp.c_fc/c_proj, visual.*); not on the text decode path.

## 2026-11-24 — host decode step: CUDA graph chunk shrink — REJECTED, reverted

Scope: host step path only (`runner.rs`); no kernel, VRAM, KV, cache-sizing, or
attention-launch changes. Prior accepted host-latency pass was read first; it
already removed the large graph-build, logits-readback, per-tensor-copy and
stderr/token-boundary costs.

### Pre-change nsys profile

nsys 2024.5.1, `--trace=cuda,osrt --cuda-graph-trace=node`, BEST env,
`/tmp/graphchunk-nsys-ctl.sqlite` (control binary used `INKWELL_NO_GRAPHCHUNK128=1`,
which is the unmodified ggml default path). Token boundary = the 4 B device-argmax
DtoH copy. Last 40 steady tokens:

| measurement | ctl |
|---|---:|
| 4 B DtoH end -> first compute kernel, median | **819.6 us/tok** |
| same, minimum | 696.9 us/tok |
| GPU span median | 37.265 ms/tok |
| GPU union-busy median | 34.852 ms/tok |
| GPU union-idle median | 2.147 ms/tok |
| `cudaGraphLaunch` API duration median | 598.2 us |

Typical ordered critical path inside that boundary hole: `cudaEventSynchronize`
wake (~5-13 us after the 4 B copy), two small pinned HtoD copies, ggml CUDA graph
key/scheduler host work, then the first chunked `cudaGraphLaunch`. The first
graph-launch API call is the dominant host-attributable piece under nsys. Note
that nsys inflates CUDA API durations; these are attribution numbers, not a
prediction of the same magnitude in an uninstrumented run.

### Experiment — `GGML_CUDA_GRAPH_CHUNK_SIZE=128`

Commit `a7e2c2d` made the runner set `GGML_CUDA_GRAPH_CHUNK_SIZE=128` when the
user had not set it. Killswitch: `INKWELL_NO_GRAPHCHUNK128=1` restored the ggml
default (512). Candidate nsys (`/tmp/graphchunk-nsys-cand.sqlite`):

| measurement | ctl 512 | cand 128 |
|---|---:|---:|
| 4 B DtoH end -> first compute kernel, median | 819.6 us | **372.5 us** |
| `cudaGraphLaunch` API duration median | 598.2 us | **135.8 us** |
| GPU union-idle median | 2.147 ms | **1.814 ms** |

This does reduce the instrumented host/idle boundary, but the uninstrumented
end-to-end effect was not a win.

A/B: 16 interleaved neighbour pairs in run order ctl,cand,cand,ctl ×8, one
flock, ctl = `INKWELL_NO_GRAPHCHUNK128=1`; md5 `5cbe8e06` on all 32 runs;
peak process VRAM 5052-5060 MiB (cap 5080):

| rep | ctl1 | cand1 | cand2 | ctl2 |
|---:|---:|---:|---:|---:|
| 1 | 38.36 | 38.06 | 38.49 | 37.96 |
| 2 | 38.61 | 38.64 | 38.30 | 38.25 |
| 3 | 38.42 | 38.60 | 38.72 | 36.50 |
| 4 | 39.02 | 38.26 | 38.25 | 38.38 |
| 5 | 37.33 | 38.75 | 40.13 | 37.80 |
| 6 | 38.22 | 37.53 | 38.61 | 38.43 |
| 7 | 38.43 | 38.71 | 38.51 | 38.74 |
| 8 | 39.88 | 38.25 | 38.63 | 38.70 |

Sorted ctl: 36.50 37.33 37.80 37.96 38.22 38.25 38.36 38.38
38.42 38.43 38.43 38.61 38.70 38.74 39.02 39.88.
Sorted cand: 37.53 38.06 38.25 38.25 38.26 38.30 38.49 38.51
38.60 38.61 38.63 38.64 38.71 38.72 38.75 40.13.

ctl median **38.40** ms/tok; cand median **38.56** ms/tok (**+0.16**, worse).
Pair wins **7/16** (bar >=12/16). Verdict: **REJECTED**; commit `a7e2c2d` was
reverted by `4840361`, so the default numeric/launch path is unchanged.

Likely interpretation: the profiler-visible launch-cost reduction is mostly not
on the uninstrumented critical path, and/or finer chunks repay it with more
chunk-boundary work. No second host-path change was shipped: the remaining
default serial step already uses cached `OnceLock` switches, packed pinned input
upload, async sched compute, device argmax with a 4 B readback, double-buffered
premarshal, and no default per-token logits `Vec` readback. Remaining `env` or
small-copy costs are in the GPU shadow or far below the A/B noise floor.
## 2026-11-24 — mmvf_had production port: router accepted, ctrl/H3e opt-in

Implementation branch `t3/port-mmvf-had-kernels`, rebased on bb8d77e. Commits: e06023c router fusion, cff25d3 controller batching, 426b888 H3e, 7087a15 KV batching, 840c7a9 warning cleanup, 8154204 ctrl opt-in, 7011b1d H3e opt-in.

Kernel-level exactness against the shipped kernels on real inputs (`/tmp/portverify.log`): all `PORT` checks `bit_diffs=0` — `inkwell_router_par_128` ids 8/8/w 0/8, `inkwell_gate_router<4>` ids 8/8/w 0/8, `inkwell_rms2down_mr<4,1>`, all three `inkwell_ctrl_triad` branches, both `inkwell_kv_sconv<1>` outputs, and the full H3e runtime chain end-to-end. md5 `5cbe8e06` in every completed production A/B run.

### M1b-4 fused gate+router — ACCEPTED, default-on (`INKWELL_NO_ROUTERFUSE` vetoes)

16 interleaved pairs ordered ctl,cand,cand,ctl. ctl 46.997 ms/tok median (43.88–51.38), cand 44.6425 (43.11–47.75), **−2.355 ms/tok, 14/16 pair wins**. md5 `5cbe8e06` on all 32 runs. The non-BF16 fallback uses the bit-exact parallel router.

### M2 controller batching/fast rms2down — REJECTED, opt-in `INKWELL_CTRLBATCH=1` (`INKWELL_NO_CTRLBATCH` vetoes)

Stopped after five positional pairs because the candidate lost all five and could no longer reach 12/16. ctl median 40.703 ms/tok (39.10–43.15, n=5), cand 45.6235 (43.18–47.79, n=6), **0/5 pair wins**; md5 `5cbe8e06` throughout. The kernels are bit-exact and faster in the isolated study, but the production graph regressed.

### H3e MoE Hadamard chain — REJECTED, opt-in `INKWELL_HADVEC=1` (`INKWELL_NO_HADVEC` vetoes)

16 interleaved pairs. ctl 40.673 ms/tok median, cand 40.491, **−0.182 ms/tok but only 9/16 pair wins**, below the 12/16 bar. md5 `5cbe8e06` on all 32 runs. The late runs in both arms shifted upward together (external machine noise), but the pair-win count is decisive.

### M4 KV projection batching — kept, md5-checked

`inkwell_kv_sconv_batched<1>` is bit-exact in the port verifier and was enabled throughout the production runs above (all md5 `5cbe8e06`). `INKWELL_NO_KVBATCH` vetoes it.

### nsys before/after (`--cuda-graph-trace=node`, 64 decode steps)

Before (all four vetoes): relevant kernel time — `mul_mat_vec_f` 0.780, `rms2down` 1.166, `gf_apply` 0.792, `had_in_moe` 1.252, `had_out_moe` 0.841, `situ_pair` 0.148, `expert_wsum` 0.144 ms/tok. After (router+KV+H3e enabled; ctrl off): `gate_router_fused<4>` 0.815, remaining `mul_mat_vec_f` 0.252, `kv_sconv_batched` 0.054, `had_in_moe_vec` 0.843, `had_out_moe_vec` 0.315, `situ_had_in_moe` 0.357 ms/tok; `had_out_wsum` was present but below the simple regex table print cutoff. Total kernel time in these two profiles was 56.239 → 53.660 ms/tok (nsys-run attribution, not an A/B acceptance measurement).

Build: `cargo build --release` rc=0, warnings **62** (master budget ≤64), `kernels.cu` **5504** lines.

### Brain re-validation on master (1218f58 crates + docs, 2026-11-24): mmvf_had port — ACCEPTED

- ctl = master-equivalent binary; cand = branch default (router fusion on, kv sconv batching on;
  ctrl batching and H3e stay opt-in). BEST env, 8 × (ctl,cand,cand,ctl), one flock.
- ctl median **37.87**, cand **37.54** ms/tok → **−0.33 ms/tok**, **12/16** pairs. md5 `5cbe8e06` on all 32 runs.
- The child's −2.36 ms (router fusion) was measured before devdesc landed; most of it does not
  survive on the current master. Meets the bar (median faster, ≥12/16, md5). ACCEPTED.
- Killswitches: INKWELL_NO_ROUTERFUSE, INKWELL_NO_KVBATCH.

### sm_61 instruction rates on the GTX 1060 (2026-11-24) — MEASUREMENT

- Tool: `opt/kbench/isa_rates/run.sh`. It runs 1 block of 1024 threads per SM, with 8 independent chains per thread. Counts are taken from the loop SASS, and cycles from clock64.
- Rates, in warp-instructions per cycle per SM:

  | Class | Rate |
  |---|---|
  | FFMA, FADD, LOP3, IDP.4A, XMAD | 3.7–3.9 (issue-bound) |
  | SHF, SHR, BFE, BFI, PRMT, ISCADD | 1.93–2.00 |
  | IADD3 | 1.87 |
  | I2F, POPC, SHFL, LDS.32 (own bank or broadcast) | 1.00 |
  | LDS.32 at random addresses | 0.45 |

- Mixes:
  - 3 FFMA + 1 I2F reached 3.98 instructions per cycle, so the slow pipe runs beside issue.
  - 1 FFMA + 1 I2F, LDS or SHFL was held to 1 pair per cycle.
- Shipped expert loop under the max(N, 2H, 4Q)/Wp model: issue-bound at every K, e.g. K=2 has N=270, H=36, Q=40 per 32 weights.
- Use: this is the cost model of `opt/MATH_CHALLENGE_STEP2.md` §2.5. Pipe sharing between slow classes is still open, and the mix run is queued.

## 2026-11-24 — process VRAM cap, runtime trims, and context-aware expert caches accepted

Implementation commits: `e3934a2` lazy `Exl3Blk` scratch, `fdec037` direct controller gain/forget concat, `628a427` measured `dev_buf` sizing, `9b49dd4` process VRAM cap + budgeted cache tables. Validated after rebasing on device-resident MoE descriptors (`1218f58`); the later `60c73af` master delta was documentation-only. Build warnings **63** (budget ≤64).

- **Runtime trims.** `INKWELL_NO_LAZY_BLKSCRATCH=1` restores eager 8-row verify scratch allocation; `INKWELL_NO_GFCAT_TRIM=1` restores separate device gain/forget weights plus their concatenated duplicate; `INKWELL_NO_DEVBUF_TRIM=1` restores the 160 MiB `dev_buf` reserve. The control arm measured the review's claims exactly: `dev_buf use=105.4 MiB alloc=237.0 MiB slack=131.6 MiB`; the trimmed arm measured `use=78.4 MiB alloc=85.0 MiB slack=6.6 MiB`. Pre-cache process VRAM fell from 4,050 MiB (ctl) to 3,772 MiB (candidate), freeing **278 MiB** for caches on the paths tested.
- **Cap accounting.** `nvmlDeviceGetComputeRunningProcesses_v2` reports the process's own device memory; a `cudaMemGetInfo` baseline fallback remains for systems without NVML. Default target `INKWELL_VRAM_CAP_MB=5080`; default runtime reserve `INKWELL_RUNTIME_RESERVE_MB=96` (measured post-cache 4,972 → external peak 5,034 MiB, +62 MiB). `INKWELL_NO_VRAMCAP=1` restores physical-free-memory sizing; `INKWELL_NO_CACHE_BUDGET=1` restores the fixed legacy 940 MiB table.
- **Budget logs.** ctx 512: `target=5080 reserve=96 used_pre_cache=3772 budget=1212 table=1200`; `cache_allocated=1200 used_post_cache=4972`. ctx 32768: `target=5080 reserve=96 used_pre_cache=4780 budget=204 table=192`; `cache_allocated=192 used_post_cache=4972`.
- **Cache tables** (24 layers; layers 6–15 are device-resident and therefore zero). ctx 512 selected total 1,200 MiB: `[100,80,68,80,84,96, 0×10, 156,164,52,48,64,68,64,76]`. ctx 32768 selected total 192 MiB: `[0,4,0,4,4,0, 0×10, 56,12,24,24,24,20,12,8]`. The legacy ctl table is 940 MiB: `[56,56,56,48,56,64, 0×10, 156,156,48,44,48,52,48,52]`. Replay on the 68-token routing trace estimates 53.4 → 34.8 MiB/tok of streamed-layer misses at ctx 512 (940-table greedy vs 1,200) and 336.8 → 189.6 MiB/tok at 32K (zero vs 192). Tables were produced by the existing kslot greedy marginal-value replay in 4 MiB steps.
- **Correctness/peak probe**, one flock, 100 ms `nvidia-smi --query-compute-apps` polling: ctx 512 rc 0, md5 `5cbe8e06`, peak **5,034 MiB**; ctx 32768 load plus short decode rc 0, peak **5,034 MiB**. Both are below the 5,080 MiB target.
- **A/B at ctx 512**, one flock, 16 interleaved neighbour pairs in order ctl,cand,cand,ctl ×8. ctl = all new killswitches (`INKWELL_NO_LAZY_BLKSCRATCH`, `INKWELL_NO_GFCAT_TRIM`, `INKWELL_NO_DEVBUF_TRIM`, `INKWELL_NO_VRAMCAP`, `INKWELL_NO_CACHE_BUDGET`); cand = defaults. All 32 runs rc 0 and md5 `5cbe8e06`. ctl peak 5,052 MiB on every run; cand peak 5,034 MiB on every run.

| rep | ctl1 | cand1 | cand2 | ctl2 |
|---:|---:|---:|---:|---:|
| 1 | 36.31 | 35.31 | 35.21 | 37.08 |
| 2 | 37.59 | 35.69 | 35.19 | 38.34 |
| 3 | 39.46 | 36.39 | 35.82 | 36.75 |
| 4 | 37.67 | 35.60 | 35.96 | 37.61 |
| 5 | 37.80 | 36.13 | 34.32 | 38.28 |
| 6 | 37.72 | 36.21 | 35.88 | 37.74 |
| 7 | 38.28 | 37.57 | 36.42 | 38.19 |
| 8 | 38.79 | 35.92 | 36.05 | 39.13 |

  ctl median **37.77** ms/tok; cand median **35.90** ms/tok; candidate **−1.87 ms/tok (−4.9%)**. Pair wins **16/16** (bar ≥12/16). Verdict: **ACCEPTED**.

### Brain re-validation on master (54909d9 + rebase, 2026-11-24): VRAM cap + trims + budgeted caches — ACCEPTED

- ctl = master `54909d9`; cand = branch rebased onto it (`da6cfaf`). BEST env, 8 × (ctl,cand,cand,ctl), one flock.
- ctl median **37.62**, cand **35.55** ms/tok → **−2.07 ms/tok**, **15/16** pairs. md5 `5cbe8e06` on all 32 runs.
- Process VRAM peak (nvidia-smi, 200 ms): ctl 5,052 MiB, cand **5,034 MiB**. Both are under the 5,120 cap; cand also meets the ≤5,080 target.
- Killswitches: INKWELL_NO_LAZY_BLKSCRATCH, INKWELL_NO_GFCAT_TRIM, INKWELL_NO_DEVBUF_TRIM, INKWELL_NO_VRAMCAP, INKWELL_NO_CACHE_BUDGET.

### sm_61 slow-pipe sharing (2026-11-24) — MEASUREMENT

Measured with `opt/kbench/isa_rates`. Results are pairs per cycle per SM for 1:1 mixes.

| Mix | Pairs/cycle | Reading |
|---|---|---|
| SHF + BFE | 0.98 | shared half-rate pipe |
| SHF + ISCADD | 0.98 | shared half-rate pipe |
| SHF + PRMT | 1.84 | PRMT is separate |
| SHF + I2F | 1.00 | separate (I2F binds) |
| I2F + POPC | 0.50 | shared quarter-rate pipe |
| SHFL + LDS | 0.50 | shared quarter-rate pipe |
| I2F + LDS | 0.99 | separate |
| I2F + SHFL | 0.67 | partial contention |

IADD3 and LOP32I ran alone: IADD3 at 1.94 per cycle (half rate), LOP32I at full rate.

## 2026-11-24 — warp-parallel kslot resolve, re-validation on master: NOT ACCEPTED (11/16)

- Control: master 74eeeb6. Candidate: `t3/warp-parallel-kslot-resolve-1` b28547b, rebased on 74eeeb6.
- Interleaved A/B, 16 pairs (ctl,cand,cand,ctl), one flock. Env: `GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=512 INKWELL_CACHE_MARGIN_MB=10 INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2`.
- Median ctl 35.617 ms, cand 34.929 ms (−0.69 ms). Candidate wins 11/16; the gate is 12/16. md5 5cbe8e06 in all 32 runs. Peak 5,034 MiB both.
- Per-pair Δ (cand − ctl, ms): −4.57 0.98 −0.64 −0.66 −0.77 −1.65 0.31 −0.69 −1.79 1.70 −5.70 −1.22 −0.88 1.74 −3.76 0.62.
- Noisy host: the machine had rebooted 6 min before (run 1 cold), and another agent's CPU jobs held load average ~4.5. The child had measured −0.56 ms.
- Decision: not merged. Re-run on a quiet host before deciding again.

## 2026-11-24 — dense W4 opt-in branch, default-path check on master: INCONCLUSIVE

- Candidate: `t3/dense-w4-runtime-optin` ec2625b, rebased on 74eeeb6 (conflicts were append-only: kernels.cu, runner.rs log line, ledger). Build: 62 warnings.
- Default path only (no W4 file, `w4_w=0`); load logs identical to master apart from the `w4_w` field (dev_buf 78.4/85.0 MiB both).
- Interleaved A/B, 16 pairs, same env as the resolve-warp check. Median ctl 34.550 ms, cand 35.219 ms (+0.67 ms); candidate wins 6/16. md5 5cbe8e06 in all 32 runs; peak 5,034 MiB both.
- Per-pair Δ (cand − ctl, ms): 0.36 2.46 0.02 0.07 1.11 −0.09 0.05 −0.19 0.67 0.73 −1.90 −0.35 −1.62 0.15 −3.87 1.01.
- Reading: no clear regression (6/16 is a coin flip), but no clean pass either. Not merged; re-run on a quiet host together with resolve-warp.
## 2026-11-24 — split-K decode attention for long context: correctness gate failed

Branch `t3/attn-splitk-flash-decode`, rebased on master `32a522a`. Pieces: `c56a355` warp-split decode-attention kernel and host dispatch, `912e609` attention/logit dump hooks, `1215027` attention-context probe. `INKWELL_NO_ATTNSPLIT=1` restores the old kernel; default dispatch keeps the old kernel below n_kv 320 and uses chunks 64/128/256 above that.

Synthetic probe (FP16 K/V, FP64 CPU reference): old/new max abs about **1-2e-8**, normalized rel **4e-7..3.2e-6**; split vs FP64 max abs about **1-3e-9**, rel **1.4e-7..3.9e-7**. The old kernel still reports `invalid argument` at n_kv >= 16384; split runs through 131072.

Indicative per-visit times (dynamic chunks, one global visit): 320 **0.040 ms**, 512 **0.053**, 1K **0.065**, 2K **0.121**, 4K **0.222-0.231**, 8K **0.427-0.454**, 16K **0.85**, 32K **1.70**, 64K **3.2-3.35**, 128K **6.29-6.37 ms**. That is only **80-85 GB/s** at long context, so the 1.5x KV-floor target (~120 GB/s) was **missed**, although it is about **6x** faster than the old kernel at 8K and removes the >12K launch failure.

End-to-end exactness gate **failed**. Under the same bounded expert-cache configuration (cacheK8 with 24 MiB per-layer arenas), forcing split at all lengths on an exactly 8000-token prompt completed in 419.116 s encode and 4.475680874 s for the decode line; the old kernel control completed in 486.033 s and 4.751162691 s. Logits over 16 generated rows differed by **max_abs 2.33707** with top-1 agreement **16/16**. The old-kernel ck20 vs ck24 control was bit-identical (`max_abs=0`), so the difference is not an expert-cache artefact. A scalar-strided/bit-compatible dot-order experiment (`bitdot2`) did not improve synthetic old/new error and reduced bandwidth (32K about 1.96 ms vs 1.70 ms committed), so it was discarded.

32K end-to-end did **not** reach prefill in the cache24 configuration. KV allocation was 1187 MiB; after bounded cache arenas only 44 MiB remained free after moe21, and the run failed before prefill in a ggml `cudaMalloc` of 16 MiB (`out of memory`). This is a memory-capacity failure, not a split-kernel launch failure. Real-tensor FP64 validation was attempted but the first diagnostic used an exactly 4096-token prompt, so the first decode position was 4097 and no power-of-two dump fired; a corrected 4095-token run was queued but not completed before foreign GPU benchmarking took the lock.

Standard post-rebase benchmark (ctx512, default split inactive): md5 **5cbe8e06**, decode line **2.581207358 s / 8 tokens = 40.331 ms/token**. Build after rebase: rc0, **62 warnings** (limit 64).

Verdict: **do not enable as a production exactness-preserving path yet**. It is a useful correctness-capability/performance prototype, but the owner's end-to-end exactness gate and the bandwidth target both remain unmet.

## 2026-11-24 — split-K merged with default threshold at the old kernel's limit

- Orchestrator change on top of the child's work: `INKWELL_ATTNSPLIT_MIN` now defaults to **12097** (was 320). The old one-block kernel fails at n_kv ≥ 12,160 (shared memory); 12,096 is the last validated length. Below the threshold the output is bit-identical to master; above it the old kernel could not run at all.
- Reason: the end-to-end exactness gate at 8K failed (max_abs 2.337 on logits, top-1 16/16), so split-K is not yet shown to be numeric noise where the old kernel works. Lowering the default needs that demonstration.
- Check on master 9a70b74 + this branch: ctx512 md5 5cbe8e06 (2 runs; decode 2.653 s and 2.227 s for 8 tokens). Build: 62 warnings.

## 2026-11-24 — challenge 2 deliverable: verified locally (Lean + CPU); port delegated

- Lean gate on this machine: Lean v4.37.0-rc1 with Mathlib aad247f. `lake build` succeeds (18,311 jobs). No `sorry`; the only match is a comment. `#print axioms` on the 14 main theorems shows only propext, Classical.choice and Quot.sound, or no axioms. The extraction and program chain does not use native_decide or bv_decide.
- Independent CPU check (orchestrator script, outside the package): all of these pass.
  - The 2-XMAD identity, exhaustive over 2^16.
  - F6 (IDP with accumulator 0x4B000000, then FADD −2^23) for all 913 values of s.
  - Lane-map bijection; (k,n) and start positions; hi/lo extraction on 200 random rings per K; ≤2 four-word chunks per lane.
- Gap in their cost model: lane-uniform register selection. With per-lane base words, the ring-word index varies across lanes on 6/8 slots for K=1.5 and 6–7/8 for odd K. K=2 and K=4 are uniform. The fix is a lane-aligned stream, about +2 SHF and +1 LDG per tile. Their x-layout assumption A-X is unproven.
- Corrected estimate: E ≈ 6.5 with A-X, ≈ 7.05 without. They claim 6.28; today's E is 8.81.

## 2026-11-24 — challenge 2 port prototype (05f28e0, branch t3/ch2-expert-gemv-core): not merged

- ptxas never emits the exact 2-XMAD product; it always emits 3 XMADs. The prototype rewrites the cubin at build time (`opt/kbench/ch2/patch_sass.py` via build.rs; killswitch `INKWELL_NO_CH2CORE=1`): 463 PSL sites, 40 regs, 0 spills.
- The retired XMAD becomes a NOP, so the issue count does not drop: 8.72–8.84 instr/weight with NOPs, 7.72–7.84 if the NOPs could be removed.
- The magic-FADD variant (F6) is bitwise exact for K1.5–4 but slower than I2F on the dominant Ks. Per-K speed (new/old): K1.5 0.887×, K2 0.90×, K2.5 ~0.99×, K3 1.01×, K3.5 1.02×, K4 1.00×. K5 is wrong (4608/9216 bad) and is gated back to the shipped core.
- The lane maps (paired views) were not ported. The final I2F build has no GPU validation (no md5, no A/B).
- Next: a feasibility probe on t3/sass-reasm-probe — sm_61 SASS reassembly that deletes the NOPs, with a byte-identical round trip, per-K bitwise checks and timing. Go/no-go for a SASS-level challenge-2 loop.

## 2026-11-24 — split-K attention noise gate: PASSED; default lowered to 320

Branch `t3/splitk-attn-noise-gate`. Harness commit `6d100f4` (plus bytecode cleanup `df514a6`) added filtered power-of-two attention dumps, an fp64 real-tensor probe path, exact-length prompts, full-vocabulary fp64 KLD comparison, and state-backed A/B benchmarking. The graph-cache bypass is required: power-of-two positions normally reuse the graph captured at the start of their 64-token shape bucket, so the original layer-5 dump hook never fired without it.

### Real-tensor fp64 accuracy

Real old-kernel Q/K/V/bias/mask dumps from an 8K run, two independent global layers at each of n_kv 1024/4096/8192. `old_ref` is the one-block kernel against a CPU fp64 reference; `split_ref` is split-K chunk 32 against the same reference. Errors are max absolute and output-scale-relative.

| n_kv | layer/pass | old_ref abs | old_ref rel | split_ref abs | split_ref rel | closer |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | l5p0 | 1.26889e-5 | 1.47259e-6 | 1.82527e-6 | 2.11828e-7 | 7.0x |
| 4096 | l5p0 | 4.52656e-6 | 5.50399e-7 | 1.42224e-6 | 1.72935e-7 | 3.2x |
| 8192 | l5p0 | 1.51894e-5 | 1.73143e-6 | 1.8783e-6 | 2.14105e-7 | 8.1x |
| 1024 | l11p0 | 3.85549e-6 | 9.74415e-7 | 1.22606e-6 | 3.09867e-7 | 3.1x |
| 4096 | l17p0 | 4.72457e-6 | 1.05643e-6 | 9.33353e-7 | 2.08701e-7 | 5.1x |
| 8192 | l23p0 | 3.7129e-6 | 1.06185e-6 | 6.51375e-7 | 1.86287e-7 | 5.7x |

Chunk 64 was similarly close (0.5-1.9e-6 absolute in these cases). Split is closer to fp64 than old in every real case.

### End-to-end divergence and noise floor

Exact 4096-token prompt, `INKWELL_MAX_CTX=4352`, 256 generated positions, full-vocabulary fp64 KLD. Arms: old (`INKWELL_NO_ATTNSPLIT=1`), production split (`INKWELL_ATTNSPLIT_MIN=320`, chunk 32), and the exact-in-real-arithmetic noise-floor perturbation (`INKWELL_ATTNSPLIT_CHUNK=64`).

- old vs split32: generated text identical (`b08cc28f`), top-1 **256/256**, first mismatch none, max logit abs diff **2.29755**; KLD mean **0.000728/0.000711**, p95 **0.00250/0.00240**, max **0.00547/0.00532**.
- split32 vs split64 noise floor: top-1 **255/256**, first mismatch row 3; full-trajectory KLD mean **0.00347/0.00323**, p95 **0.0103/0.0119**, max **0.399/0.277**. The first flip had small margins (0.150 and 0.084), after which the two numerically valid trajectories diverged autoregressively.
- old vs split64 showed the same amplification pattern: first mismatch row 3, full KLD mean **0.00350/0.00309**, max **0.397/0.271**.
- Thus old-vs-split divergence is inside the noise floor's full-trajectory range, with better top-1 agreement than the floor itself. The row-3 margins show the mechanism directly: old/split remained tied despite margins of 0.026/0.150, while the chunk reorder's 0.150/0.084 margins flipped.

The 8K end-to-end split-arm quality run was not repeated; 4K is the required gate and 8K was optional under the brief. The fp64 table above does cover 8K real attention inputs.

### Decode speed

State-backed decode A/B. The prompt state was encoded once with the control kernel, all KV/conv persistent state tensors were saved, and later processes reloaded that state. A 1024-token validation produced byte-identical generated text from serial and loaded state. Later runs start with cold expert caches in both arms; the first control is warm from its encode, so if anything pair 1 is biased against the candidate. Four interleaved pairs, order ctl,cand,cand,ctl, one flock, 64 generated tokens/run.

4K (`INKWELL_VRAM_CAP_MB` default):

| pair | ctl 12097 ms/tok | cand 320 ms/tok | cand−ctl |
|---:|---:|---:|---:|
| 1 | 56.513 | 52.445 | −4.068 |
| 2 | 51.649 | 42.648 | −9.001 |
| 3 | 50.772 | 43.601 | −7.171 |
| 4 | 48.435 | 39.011 | −9.424 |

Median 4K: ctl **51.210**, cand **43.124**, **−8.086 ms/tok**, **4/4 pair wins**; peaks 4894/4902 MiB.

8K (`INKWELL_VRAM_CAP_MB=4900` after the default target OOMed once while foreign desktop memory was resident):

| pair | ctl 12097 ms/tok | cand 320 ms/tok | cand−ctl |
|---:|---:|---:|---:|
| 1 | 72.857 | 46.851 | −26.006 |
| 2 | 66.344 | 44.608 | −21.736 |
| 3 | 70.378 | 44.182 | −26.196 |
| 4 | 71.929 | 44.199 | −27.730 |

Median 8K: ctl **71.154**, cand **44.403**, **−26.750 ms/tok**, **4/4 pair wins**; peaks 4682/4690 MiB. Generated-text md5 was `44dbfe6b` in every speed run. These are controlled relative measurements, not warm-cache absolute production numbers; the reduced 8K cache budget slows both arms.

### Dispatch note

The original dynamic chunk selection was dead: an unset `INKWELL_ATTNSPLIT_CHUNK` became zero and was immediately clamped to 32. The validated production configuration at 1K-12K is therefore chunk 32. The dispatcher now makes that explicit up to n_kv 16384 and uses 256 above that point so `INKWELL_ATTN_MAX_SPLIT=512` does not disable split at long context. Explicit `INKWELL_ATTNSPLIT_CHUNK` remains an override.

### Verdict

Noise gate **passed**: split is uniformly closer to fp64, old-vs-split end-to-end divergence is within the exact-reorder noise floor, and top-1 is unchanged for 256/256 positions at 4K. Decode is faster in every interleaved pair at both 4K and 8K. ctx512 generated-text md5 remains **`5cbe8e06`**. `INKWELL_ATTNSPLIT_MIN` is therefore lowered to **320**. Build rc0, **61 warnings** (limit 64). `cargo test --workspace` is not a valid gate here because the pre-existing `crates/inkwell-gguf/examples/embdcheck.rs` example does not link `inkwell_ggml` (unrelated compile error).

## 2026-11-25 — split-K threshold 320 merged (407c049), re-validated on master

- Merged `t3/splitk-attn-noise-gate` (132fe6e): `INKWELL_ATTNSPLIT_MIN` defaults to 320; chunk 32 up to n_kv 16384, 256 above. The gate evidence is in the child's entry above: split-K is closer to fp64 than the old kernel in all 6 real cases (3.1–8.1×), and at 4K old-vs-split KLD is below the split32-vs-split64 noise floor (top-1 256/256, same text).
- Master check after merge: ctx512 md5 5cbe8e06 (2 runs; decode 2.449 s and 2.606 s for 8 tokens on a busy host). Build: 61 warnings.
- Also merged: KV state save/load (`INKWELL_STATE_SAVE` / `INKWELL_STATE_LOAD`) for long-context tests that skip the encode.

## 2026-11-24 — Challenge-2 exact expert-GEMV core, emission and patching evidence: IN PROGRESS

Branch `t3/ch2-expert-gemv-core`. Solver artifacts archived in `opt/kbench/ch2/` (`DESIGN.md`, `lane-maps.json`).

**ptxas emission.** Focused probes in `opt/kbench/ch2/emit.cu` retried algebraic forms, split/merged `mul.wide.u16` asm, immediate constants, volatile boundaries, direct `0x83dc0000` multiplication, PRMT/zero-packed halves, and ptxas `-O1/-O2/-O3`. Stock ptxas never emitted the required two-instruction product. The best direct forms still emitted either (a) three XMADs (`lo*0xd12d`, `lo*0x83dc`, `XMAD.PSL q,1,t`) or (b) a redundant low-half XMAD before the useful `XMAD.PSL`. Explicit zero-packing was worse (an extra `XMAD.PSL.CLO`). This independently confirms the earlier Round-2 emission failure.

**Binary rewriting prototype.** `opt/kbench/ch2/patch_sass.py` rewrites the compiler-emitted triple

```text
LP    XMAD   t, U.H?, 0xd12d, RZ
LOW   XMAD   p, U,    0x83dc, t
HIGH  XMAD.PSL p, U.H1, 0x83dc, p
```

by changing LOW in place to `XMAD p, U.H1, 0xd12d, RZ` (immediate bits 20..35, r8 `.H1` bit 53, r39 addend RZ=0xff) and replacing LP with Pascal NOP `0x50b0000000070f00`. Patched disassembly shows exactly the useful pair `XMAD U.H1,0xd12d` + `XMAD.PSL U.H1,0x83dc` before IDP/FADD. In `inkwell_exl3_gemv_moe_s`: 463 product sites, 467 retired low-product instructions, 2,258 machine words rewritten (duplicate unrolled encodings), zero non-PSL `0x83dc` XMADs remain. The rewritten production kernel remains at 40 registers/thread and 0 spills.

**Important limitation.** The retired LP instructions remain as NOPs because sm_61 control-word/instruction packing makes deletion a full SASS reassembly problem. Thus the abstract non-NOP cost falls, but the issued-instruction stream still contains one patched NOP per weight. Current patched-loop counts for the first six K cases are 8.719–8.844 issued instructions/weight including NOPs and 7.719–7.844 excluding NOPs, versus the shipped-model baseline 8.41–9.47. Any real gain must therefore come from pipe-pressure relief and/or extraction pairing, not from the nominal 2-XMAD count alone.

**CPU exactness model.** `opt/kbench/ch2/test_cpu.py`: 1,120,000 randomized view extraction cases over every K and 20,000 product/bytesum cases pass. It verifies `ch2_view` against the shipped low-aligned window extraction and verifies that the patched high-half product is exactly `(0x83DCD12D*w) mod 2^32` with the IDP bias producing exact integer `s`.

GPU validation and performance measurements are pending behind the shared `/tmp/inkwell-gpu.lock`.
**Standalone per-K measurement, magic-FADD patched variant.** `opt/kbench/ch2/ch2_perk` compares the shipped stream core (`pw16=0`) with the patched CH2 branch on distinct MoE slots, min-of-31 interleaved timing. Bitwise output was exact for K=1.5 through K=4 (`badbits=0/9216`, maxabs 0), but K=5 failed (`badbits=4608/9216`, maxabs 153); the failure is isolated to the unpaired K5 path and is not yet root-caused. Measured MoE gate/up and down speedups (old/new, larger is better):

| K | gu | dn | reading |
|---|---:|---:|---|
| 1.5 | 0.887x | 0.889x | about 11% slower |
| 2.0 | 0.902x | 0.899x | about 10% slower |
| 2.5 | 0.994x | 0.990x | neutral |
| 3.0 | 1.011x | 1.005x | neutral |
| 3.5 | 1.023x | 1.012x | slight gain |
| 4.0 | 1.000x | 1.000x | neutral |

This confirms the Round-2 warning: magic FADD exactness does not translate into speed on sm_61 because it uses the saturated main ALU pipe, while the patched NOP still occupies the instruction stream. Since K1.5/K2 dominate expert work, this variant is not acceptable.

**Final branch state.** The conversion defaults to exact I2F rather than magic FADD (`CH2_USE_MAGIC_FADD` preserves the solver form for experiments). This is still exact integer `s`; it is not an approximation. The broken K5 CH2 path is gated out (`kk != 10`) so the runtime cannot silently produce wrong K5 expert output; K5 uses the shipped core. Thus the branch is a correctness-safe prototype, not an accepted implementation of the all-K challenge design. The solver lane maps/pairing have not been ported; the current prototype keeps the shipped consecutive-lane mapping and changes only the exact scalar core. No end-to-end A/B was run because the isolated dominant-K measurements already fail the performance gate and the shared GPU lock remained occupied by foreign long-running validation jobs. Build after the final state: rc0, 62 warnings (limit 64), patched production object has zero remaining non-PSL `0x83dc` XMADs.

## 2026-11-24 — Challenge-2 sm_61 SASS reassembly/NOP deletion probe: GPU-BLOCKED, NO-GO FOR PRODUCTION YET

Branch `t3/sass-reasm-probe`, based on 05f28e0. Probe artifacts/scripts: `opt/kbench/ch2/reasm_*.py`, `reasm_drv_perk.cc`, `reasm_pipeline.sh`, `cuasm-96a9f72-local.patch`.

**Toolchain evaluated.** CuAssembler (cloudcores) pinned to upstream commit `96a9f72baf00f40b9b299653fcef8d3e2b4a3d49`, Python venv `sympy==1.14.0`, `pyelftools==0.32`, `mpmath==1.3.0`, CUDA 12.6 `nvdisasm/cuobjdump`. The upstream sm_61 repository is present but sparse; it could not assemble the production cubin until it was augmented from the cubin's own `cuobjdump -sass` stream (102,660 instruction samples; four new encoding facts, about 36 s). One upstream bug had to be patched locally: `hasattr(self, '__mCuInsAsmRepos')` does not name-mangle, so a custom repository was always discarded when `.__elf_flags` was parsed. maxas-family tools were not a viable alternative for this task because maxas targets Maxwell/sm_50, not Pascal/sm_61.

**Round trip, full cubin: not byte-identical.** The extracted production `kernels_unpatched.1.sm_61.cubin` is 1,319,664 bytes, sha256 `45855f8a144da25920f1c007ccf488e4b31c57848d70e9fced8b19c55cb2ae71`. CuAssembler reassembly is 1,322,768 bytes, sha256 `22403c4719d2cfe39155ed9d84e82684ba377dd44ba39dbd7b4d773291105088`. It has the same 490 section names; 368 sections are byte-identical, 49 text sections differ in size, and 73 same-sized sections differ in content. Main causes: CuAssembler counts tail padding in many `.text` section sizes (usually +32/+64 bytes), changing downstream ELF offsets (`e_shoff` 0x13a790 -> 0x13b3b0, `e_phoff` 0x142210 -> 0x142e30); `.nv.info` indirect-branch target entries are rewritten and some branch-instruction addresses move backwards by 8 bytes to control-word boundaries; one non-target text section differs in two control bytes because two predicated ISCADD instructions carry reuse bits not displayed by nvdisasm. Importantly, the target `.text._Z23inkwell_exl3_gemv_moe_s...` section round-trips byte-identically (137,728/137,728 bytes, zero differing bytes), although its `.nv.info` indirect-branch metadata does not.

**NOP deletion and reassembly.** Existing `patch_sass.py` was applied to the extracted cubin (`fake=463 lp=466 words=2167`). The six patched main loops were disassembled, 32 retired-XMAD NOPs per K were deleted for K=1.5/2/2.5/3/3.5/4 (192 total), and their stall/yield/wait-bar information was carried forward conservatively (stall distributed forward if one control word could not hold it; producer scoreboard fields were absent and caused a hard fail if encountered). Reassembled `cuobjdump` loop counts are exactly: K1.5-K3.5 247 instructions/32 weights = 7.719 instr/weight with 0 NOPs; K4 248/32 = 7.750 with 0 NOPs. This matches the prior non-NOP abstract count and removes one issued instruction per weight versus the patched-with-NOP kernel.

**GPU validation/timing.** Not completed. A driver-API harness (`reasm_drv_perk.cc`) was built and queued exactly as required under `nice -n 19 flock /tmp/inkwell-gpu.lock`, but `/tmp/inkwell-gpu.lock` remained continuously held by foreign long-running jobs, with two more foreign jobs already queued ahead/around the probe. The queued probe was cancelled rather than adding more pressure. Therefore there is no bitwise correctness table and no measured per-K timing table in this run. Static instruction-count reduction is exactly 1.000 issued instruction/weight for every K tested: 8.719 -> 7.719 for K1.5/K2/K2.5/K3/K3.5 and 8.750 -> 7.750 for K4 (an 11.4-11.5% issue-count upper bound, not a measured speedup).

**Verdict.** **NO-GO for writing the full challenge-2 loop in SASS or integrating this into the build yet.** Reasons: (1) full-cubin round trip is not byte-identical; (2) CuAssembler needs a local correctness patch plus augmentation from the input cubin before it can assemble the production kernel; (3) ELF padding/layout and `.nv.info` indirect-branch metadata are not reproduced faithfully; (4) the NOP-deleted cubin has not been loaded or bitwise-validated on the GPU. The positive result is narrower but real: the target kernel's text section round-trips byte-identically, CuAssembler can represent labels/branch fixups well enough to physically delete 192 NOPs, and the resulting disassembly has the intended 7.719-7.750 instructions/weight with zero NOPs. A future conditional go would require fixing/vendoring the assembler defects, a fail-closed target-section/metadata comparison, and the missing GPU bitwise/timing gate.

## 2026-11-24 — Challenge-2 SASS NOP-deletion GPU gate: BITWISE FAIL, NOT ACCEPTED

Branch `t3/sass-reasm-gpu-gate`, based on `8bf3991`. Regenerated artifacts with the now-executable `opt/kbench/ch2/reasm_pipeline.sh` under `~/.cache/inkwell-sass-reasm/out/`; the pipeline was independently rerun end-to-end and reproduced the same hashes. Unpatched `45855f8a…`, patched-with-NOPs `25dc4bcf…`, patched-NOP-deleted `fb190b1e…` (full hashes in `opt/kbench/ch2/reasm_gpu_gate_results.txt`). The first queued harness attempt failed before CUDA initialization because `libcudart.so.12` was not on the runtime loader path; the harness and pipeline now link `libcudart_static`, and the measurement below used that binary.

Method: all three cubins were loaded with `cuModuleLoadData` by `opt/kbench/ch2/reasm_drv_perk.cc`; K=1.5/2/2.5/3/3.5/4 only (K5 remains on the shipped core). Inputs and MoE shapes match `ch2_perk.cu` (8 slots, gate/up tk=144/tn=72 and down tk=72/tn=144, x quantized in 1/512 steps) and the default exact I2F conversion. Correctness used nky=1. Timing used nky=2, memset+kernel, rotating three-way interleaving, min-of-31.

**Bitwise.** Patched-with-NOPs was bit-identical to the unpatched shipped core at every K and shape (`badbits=0`, `maxabs=0`). NOP-deleted was **not** correct:

| K | gate/up badbits/N, maxabs | down badbits/N, maxabs |
|---|---|---|
| 1.5 | 7480/9216, 56.3 | 13224/18432, 44.3 |
| 2.0 | 9216/9216, 85.2 | 18432/18432, 59.5 |
| 2.5 | 7928/9216, 56.3 | 13632/18432, 39.6 |
| 3.0 | 7976/9216, 60 | 14584/18432, 46.6 |
| 3.5 | 8280/9216, 64.5 | 15400/18432, 49.9 |
| 4.0 | 9216/9216, 361 | 18432/18432, 271 |

**Timing.** Times are ms; ratios greater than 1 are speedups. `u/n` is unpatched/patched-with-NOPs, `u/d` unpatched/NOP-deleted, and `n/d` patched-with-NOPs/NOP-deleted.

| K | shape | unpatched | with NOPs | NOP-deleted | u/n | u/d | n/d |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.5 | gu | 0.0797 | 0.0829 | 0.0734 | 0.961 | 1.086 | 1.129 |
| 1.5 | dn | 0.0819 | 0.0859 | 0.0778 | 0.954 | 1.053 | 1.104 |
| 2.0 | gu | 0.0799 | 0.0836 | 0.0764 | 0.955 | 1.045 | 1.094 |
| 2.0 | dn | 0.0819 | 0.0847 | 0.0786 | 0.967 | 1.043 | 1.078 |
| 2.5 | gu | 0.0900 | 0.0849 | 0.0737 | 1.059 | 1.220 | 1.152 |
| 2.5 | dn | 0.0916 | 0.0867 | 0.0788 | 1.057 | 1.162 | 1.099 |
| 3.0 | gu | 0.0907 | 0.0850 | 0.0746 | 1.067 | 1.216 | 1.139 |
| 3.0 | dn | 0.0932 | 0.0870 | 0.0795 | 1.071 | 1.172 | 1.095 |
| 3.5 | gu | 0.0928 | 0.0850 | 0.0768 | 1.092 | 1.209 | 1.107 |
| 3.5 | dn | 0.0952 | 0.0876 | 0.0808 | 1.086 | 1.179 | 1.085 |
| 4.0 | gu | 0.0891 | 0.0847 | 0.0796 | 1.052 | 1.119 | 1.064 |
| 4.0 | dn | 0.0911 | 0.0866 | 0.0824 | 1.053 | 1.106 | 1.050 |

`cuobjdump -sass` on the exact timed NOP-deleted cubin (`fb190b1e…`) confirms zero NOPs and 247 instructions/32 weights for K1.5–K3.5 = **7.719 instr/weight**, and 248/32 for K4 = **7.750 instr/weight**. The corresponding patched-with-NOPs loops are 8.719/8.750 instr/weight. After filtering NOPs and normalizing branch targets, the visible non-NOP instruction streams of the patched-with-NOPs and NOP-deleted loops are identical for all six K cases.

**Reading.** Deleting one instruction per weight does produce a real measured per-K gain in this harness: at K1.5 the NOP-deleted kernel is 12.9%/10.4% faster than patched-with-NOPs (gu/dn) and 8.6%/5.3% faster than the shipped unpatched core; at K2 the gains are 9.4%/7.8% over patched-with-NOPs and 4.5%/4.3% over unpatched. Those are the dominant expert-work cases. However, the measured NOP-deleted kernel is bitwise wrong everywhere, so this is **not an accepted optimization** and no runtime integration is justified. Since the visible arithmetic instruction sequence is unchanged, the failure is isolated to the reassembled control/scheduling/metadata or encoding details rather than the intended scalar math. The positive result is narrower: patching with NOPs is bitwise correct, physical deletion is mechanically possible and fast, but the current conservative control-carry reassembly does not preserve correctness.

## 2026-11-25 — Challenge-2 SASS NOP-deletion root cause fixed: bitwise PASS, conditional build-step GO

Branch `t3/sass-reasm-bisect`, based on `50d2ecd`. Fixed artifacts under `~/.cache/inkwell-sass-reasm/out/`: unpatched `45855f8a…`, patched-with-NOPs `25dc4bcf…`, fixed NOP-deleted `07ef9b14db171dc248af865064effbb9b0760033e2a242af0c58d4559907b49f` (1,320,720 bytes). Full harness output is in `opt/kbench/ch2/reasm_gpu_gate_results.txt`.

**No-deletion bisect.** A no-deletion CuAssembler round trip of the patched-with-NOPs cubin (`bab26f66…`) was bitwise correct on the GPU for all six K values and both shapes. Its target `.text` is not byte-identical: 170 bytes / 161 qwords differ, but every differing qword is a control qword, zero instruction payload words differ, and every differing bit is a hidden low reuse bit (bit 17/38/59) attached to a patched NOP. Those bits are invisible because `patch_sass.py` rewrote the instructions to NOP in place and nvdisasm does not print `.reuse` on NOPs. The target `.nv.info` differences are confined to `EIATTR_INDIRECT_BRANCH_TARGETS` label reconstruction. Thus the leading hypothesis—that CuAssembler encoded the rewritten `XMAD .H1, immediate, RZ` instruction word incorrectly—was rejected: the instruction words are exact, and the no-delete round trip executes exactly.

**Root cause.** The NOP-deletion transformer carried stall/yield forward onto the first surviving instruction. On Pascal the 4-bit stall field belongs to the instruction that has just issued and controls the gap before the next instruction. Moving it forward therefore puts the delay after the dependent consumer instead of before it. A clean K=2 site is patched `.text` `0x177b8`: `XMAD R24,R37.H1.reuse,0xd12d,RZ` produces R24; one deleted NOP carries S05; `0x177d8` `XMAD.PSL R24,R37.H1,0x83dc,R24` consumes it. A linear issue+stall spacing model gives 9 cycles in the patched kernel, 8 cycles with backward transfer, and only 3 cycles with the original forward transfer; XMAD fixed latency is about 6 cycles. In the final layout the exact field is the first control slot's stall bits `[3:0]` in control qword `0x17660`: forward `0x081fd400fec00711` has stall 1, fixed backward `0x081fd400fe200716` has stall 6.

A nearest-producer scan found six forward-critical XMAD windows with spacing original/back/forward: K2 `0x177b8` 9/8/3, K2 `0x17bd0` 10/9/5, K4 `0x20e28` 8/7/2, K4 `0x20eb8` 8/7/2, K4 `0x212f0` 10/8/4, K4 `0x215b0` 11/8/4. This explains why K2 and K4 were 100% bad in the original NOP-deleted gate. For contrast, the correct patched loops already contain IDP→I2F dependences at spacing 3, so the smaller IDP spacings produced by deletion are not inherently unsafe.

**Controlled experiment.** Two NOP-deleted cubins were built with identical target `.nv.info` and identical instruction payload words. The forward-stall control (`630370e4…`) and backward-stall fix (`07ef9b14…`) are the same size and differ in exactly 280 bytes, all in target `.text` control qwords (148 control qwords, zero instruction qwords). Backward was bitwise exact for every K/shape (rc 0). Forward remained massively wrong (rc 2): K2 and K4 were 100% bad in both shapes, with the same kind of damage as the original failure. This isolates the stall-direction field as the executable root cause.

**Metadata fix.** CuAssembler's `CuNVInfo.updateNVInfoFromDict` also discarded regenerated values: when an offset attribute was present in the update dictionary it appended the old value rather than `d[attr]`, and non-autogen `*OFFSETS` attributes were kept before the update branch could run. After NOP deletion this left `EIATTR_EXIT_INSTR_OFFSETS` and `EIATTR_COOP_GROUP_INSTR_OFFSETS` at stale pre-deletion addresses. `cuasm-96a9f72-local.patch` now contains the corrected update logic (verified with `git apply --check` against pristine `96a9f72`) in addition to the pre-existing repository-selection patch. In the fixed cubin the final three EXIT offsets are `135160/135536/135608`, matching the actual `0x20ff8/0x21170/0x211b8` EXIT instructions; S2R offsets are unchanged because all precede the deletion ranges. All indirect-branch-target entries also precede the deletion ranges, so their CuAssembler control-word-boundary reconstruction artefact is not deletion-dependent. The metadata bug is a real reassembly defect and is fixed, but the controlled forward/back experiment shows it was not the measured arithmetic root cause.

**Transformer fix.** `reasm_remove_loop_nops.py` now pre-marks all 192 deleted NOPs, moves wait masks forward to the next surviving instruction, and moves stall/yield backward to the previous surviving instruction. It deliberately does not re-add the deleted issue slots; that is the intended speedup. Six wait masks are preserved. Two K4 backward sums saturate at the 4-bit maximum (`wanted 17/16`, stored 15), still far above the XMAD latency. No branch labels intervene in any of the 157 deleted NOP runs.

**Bitwise after fix.** All six K values and both MoE shapes were exact for both patched-with-NOPs and fixed NOP-deleted kernels:

| K | gate/up badbits, maxabs | down badbits, maxabs |
|---|---|---|
| 1.5 | 0/9216, 0 | 0/18432, 0 |
| 2.0 | 0/9216, 0 | 0/18432, 0 |
| 2.5 | 0/9216, 0 | 0/18432, 0 |
| 3.0 | 0/9216, 0 | 0/18432, 0 |
| 3.5 | 0/9216, 0 | 0/18432, 0 |
| 4.0 | 0/9216, 0 | 0/18432, 0 |

**Timing after fix.** Same harness as before: nky=2, memset+kernel, rotating three-way interleaving, min-of-31. Times are ms; ratios greater than 1 are speedups.

| K | shape | unpatched | with NOPs | NOP-deleted | u/n | u/d | n/d |
|---|---|---:|---:|---:|---:|---:|---:|
| 1.5 | gu | 0.0816 | 0.0850 | 0.0775 | 0.960 | 1.053 | 1.097 |
| 1.5 | dn | 0.0829 | 0.0870 | 0.0788 | 0.953 | 1.052 | 1.104 |
| 2.0 | gu | 0.0814 | 0.0850 | 0.0768 | 0.958 | 1.060 | 1.107 |
| 2.0 | dn | 0.0835 | 0.0868 | 0.0797 | 0.962 | 1.047 | 1.089 |
| 2.5 | gu | 0.0910 | 0.0850 | 0.0784 | 1.071 | 1.161 | 1.084 |
| 2.5 | dn | 0.0932 | 0.0879 | 0.0809 | 1.060 | 1.152 | 1.087 |
| 3.0 | gu | 0.0908 | 0.0850 | 0.0778 | 1.069 | 1.167 | 1.092 |
| 3.0 | dn | 0.0939 | 0.0870 | 0.0809 | 1.079 | 1.161 | 1.076 |
| 3.5 | gu | 0.0930 | 0.0850 | 0.0788 | 1.094 | 1.179 | 1.078 |
| 3.5 | dn | 0.0952 | 0.0881 | 0.0819 | 1.081 | 1.163 | 1.075 |
| 4.0 | gu | 0.0901 | 0.0848 | 0.0806 | 1.063 | 1.118 | 1.052 |
| 4.0 | dn | 0.0917 | 0.0870 | 0.0819 | 1.053 | 1.119 | 1.063 |

**Reading and verdict.** The fixed NOP-deleted kernel is bitwise exact and 5.2–10.7% faster than patched-with-NOPs in this harness, or 4.7–17.9% faster than the unpatched shipped core. For the dominant K1.5/K2 cases the net gain over unpatched is 4.7–6.0%. `cuobjdump` confirms 7.719 instructions/weight for K1.5–K3.5 and 7.750 for K4, zero NOPs. **Conditional GO** for a fail-closed offline build step: it may emit this artifact only if it verifies the patched-with-NOPs baseline byte-for-byte, the zero-NOP/instruction-count shape, regenerated EXIT/COOP offsets, and the full GPU bitwise gate. **NO-GO** for unconditional runtime integration or for trusting CuAssembler without that gate: CuAssembler remains an external sparsely maintained tool, the full-cubin round trip is not byte-identical, hidden reuse state is not representable on NOPs, and indirect-branch-target label reconstruction still has control-word-boundary artefacts.

### CH2 NOP-less production integration — ACCEPTED, default ON

Branch `t3/ch2-nopless-integrate`, based on master `8d3e840`, integrated the validated sm_61 SASS-reassembled CH2 expert GEMV as an optional driver-API kernel (`40de564`) and then enabled it by default after the acceptance gate (`3c02652`).

**Build design.** `crates/inkwell-cuda/build.rs` first compiles the ordinary ptxas kernel object untouched. It then runs `opt/kbench/ch2/build_nopless.py` on a copy of the cubin extracted from that exact object: exact-I2F byte patch, CuAssembler disassembly, deletion of the 192 loop NOPs with the backward stall transfer, reassembly with the production repository, and NVInfo EXIT/COOP correction. The generator verifies the pinned CuAssembler commit `96a9f72baf00f40b9b299653fcef8d3e2b4a3d49` plus both local patches, the patched-with-NOPs pattern counts, zero NOPs in all six target loops, 7.719 instructions/weight for K1.5–K3.5 and 7.750 for K4, exact EXIT/SHFL metadata targets, `REG:40`, `SHARED:9248`, and zero stack/local spills. The generated cubin is embedded as a byte range in the static kernel library; it is written only after every check passes. If CuAssembler or its venv is absent the build prints one note and embeds an empty range; any generation or validation failure likewise emits no artifact. Either way the build succeeds and uses the normal ptxas kernel. Discovery is `INKWELL_CUASM_ROOT`/`CUASMROOT` or `~/.cache/inkwell-sass-reasm/CuAssembler`, and `INKWELL_CUASM_PYTHON`/`VENVPY` or `~/.cache/inkwell-sass-reasm/cuasm-venv/bin/python`.

**Runtime.** At MoE staging warmup the runtime loads the embedded cubin with `cuModuleLoadData`, resolves `_Z23inkwell_exl3_gemv_moe_sPKmS0_PKiPKfS4_iPfiiiiiiiS2_ii`, and runs a one-time bitwise comparison against the runtime-linked shipped core on one small deterministic input for K1.5, K2, K2.5, K3, K3.5, and K4. K5 is unchanged because the optimized core excludes `kk=10`. Failure to load, launch, or match exactly disables the feature with one log line and falls back to the runtime-linked kernel. `INKWELL_NO_CH2NOPLESS=1` is the killswitch. Before acceptance `INKWELL_CH2NOPLESS=1` was the opt-in; after acceptance the feature is on unless the killswitch is present. Production launches use `cuLaunchKernel` on the caller's stream through a shared dispatcher and fall back synchronously if the driver launch itself errors.

**Artifact and startup.** Generated cubin `sha256=e84eaf7fa18990bd395c5a6080876b0f6628d49cea7971140d50edae6cdb9bce`, `1,320,720` bytes. Observed startup self-test cost: 5.04 ms in the ctx512 run and 5.17 ms in the logits-dump run. Log: `ch2nopless: enabled cubin=1320720 bytes selftest=5.04 ms` / `5.17 ms`.

**Correctness and graph/VRAM gate.** Under `GGML_PASCAL_GRAPHS=1`, `INKWELL_MAX_CTX=512`, cache K20, MoE v2/nky2, and the shared GPU lock, the feature-on run exited 0 and generated-text md5 was `5cbe8e06`. Eight-step full-logit dumps with the feature on and off compared byte-identical (`cmp` rc 0). The graph-enabled run completed normally, so the driver-API launch remained capture-compatible. Peak process VRAM in every A/B run was 5,034 MiB, below the 5,080 MiB ctx512 requirement. Release build rc 0; warning count 62 (limit 64; master baseline 61).

**End-to-end A/B.** `~/.cache/inkwell-ab/ab_rw.sh`, CTL `/home/ash/inkwell/target/release/inkwell`, CAND this branch with the feature enabled, 16 pairs in order ctl,cand,cand,ctl inside one `nice -n 19 flock /tmp/inkwell-gpu.lock`. CTL median **38.658 ms/tok**; CAND median **37.063 ms/tok**; candidate won **15/16** pairs; every run md5 was `5cbe8e06`; peak was 5,034 MiB; script rc 0. Per-pair candidate-minus-control deltas (ms/tok): `-1.838 -1.882 -0.835 -1.690 -1.058 -1.365 -1.599 -1.378 -1.720 -0.455 -1.827 -1.802 -1.727 -1.791 -2.048 +1.030`. Acceptance criteria were met (median faster, ≥12 wins, exact md5), so the default was flipped on in the separate commit `3c02652`.

## 2026-11-25 — ch2 NOP-less expert kernel merged (887150b), re-validated on master

- Merged `t3/ch2-nopless-integrate` (40de564, 3c02652, c829bf8). Child A/B: 16 pairs, CTL median 38.658 vs CAND 37.063 ms/tok, 15/16 wins, md5 5cbe8e06 on every run, peak 5,034 MiB, busy host.
- Master check after merge: ctx512 md5 5cbe8e06 in 2 runs. Decode 2.161 s and 2.167 s for 64 tokens (the log line says "8 tok", but `INKWELL_GENS` defaults to 64) = **33.8 ms/tok**. Log: `ch2nopless: enabled cubin=1320720 bytes`, self-test 4.8–8.4 ms. Build: 62 warnings.

## 2026-11-25 — TOP32 K1.5/K2 stream (branch t3/ch2-top32-integrate, 84def41): NOT accepted, 11/16

- Source: the round-1 hill-climb (069a05c + c2a8df8, cherry-picked onto 887150b) plus a build-step fix. The fix labels loops by their SHF immediate set (ptxas lays them out in a different order from the source), passes the detected loop ranges to the NOP remover, and sets the expected shape to 233/32w for K1.5/K2 and 248/249 for K2.5–K4, with patch fingerprint (463, 465, 1646, 470). The generated target-kernel SASS is identical to the GPU-gated cubin.
- Isolated GPU gate (orchestrator, `reasm_drv_perk`, ref = 45855f8a): bitwise exact for all six K values, gate/up and down. Against the current NOP-less kernel: K1.5 +6.7–6.9%, K2 +6.0–6.2%, K2.5/K3 +0.6–1.3%, K3.5 −0.3–0.5%, K4 −0.7–1.8%.
- End-to-end A/B (16 pairs, ctl,cand,cand,ctl, one flock): CTL median 33.795 vs CAND 33.453 ms/tok (−0.34), **11/16 wins**, md5 5cbe8e06 on every run, peak 5,034 MiB. Below the ≥12/16 gate, so not merged.
- Kept on its branch to be stacked with the round-2 hill-climb result and A/B'd again as a new candidate.

## 2026-11-25 — TOP32 + neighbour-lane shuffle expert kernel merged (79cd1e1)

- Candidate: branch `t3/ch2-top32-integrate` (0e6a474) = round-1 TOP32 (069a05c + c2a8df8) + round-2 donor-lane `SHFL.IDX` for W1 (51b3780 + 7e02786), plus the fail-closed build step changes: K-labelled loops, detected ranges, K1.5/K2 shape 225/32w with LDG 4, fingerprint (463, 465, 1806, 470). The target-kernel SASS is identical to the GPU-gated cubin a9d56a52, which is bitwise exact for all six K values, gate/up and down.
- K1.5/K2 loop: 247 → 225 instr per 32 weights (7.719 → 7.031/w). K2.5–K4: 248/249 (+1 vs before).
- A/B (16 pairs, ctl,cand,cand,ctl, one flock; CTL = master 60f1cfe): CTL median 32.368 vs CAND **31.890 ms/tok** (−0.478), **15/16 wins**, md5 5cbe8e06 on every run, peak 5,034 MiB. Accepted.
- Master check after merge: ctx512 md5 5cbe8e06 in 2 runs; generated cubin 270a4be4 (1,319,312 bytes); build 62 warnings.

## 2026-11-25 — donor-lane K2.5–K4 expert loops merged (round-3 hill-climb)

- Candidate: branch `t3/ch2-donor-integrate` (376e449) = 6de5276 + 9919252 + 21f8374 from `t3/sass-hillclimb-3` (donor-only, MOV removal off), plus the build-step shape (233/32w, SHF 36, LDG 4 for K2.5–K4) and fingerprint (463, 467, 1710, 468). The target-kernel SASS is identical to the GPU-gated cubin f77d3133, which is bitwise exact for all six K values, gate/up and down.
- Loops (instr per 32 weights): K1.5/K2 225, K2.5–K4 233 (was 248/249).
- A/B (16 pairs, one flock; CTL = master 438eb48): CTL median 32.133 vs CAND **31.782 ms/tok** (−0.350), **13/16 wins**, md5 5cbe8e06 on every run, peak 5,034 MiB. Accepted.

## 7afd673 — ccopy bulk grid 2 parts per (slot, proj) — ACCEPTED, merged (32dd91d)
- A/B 16 pairs vs pre-ccopy master: CTL 32.311 vs CAND 31.696 ms/tok median (−0.615), 15/16 wins, md5 5cbe8e06 all runs.
- Post-merge master (donor + ccopy) 3 runs: 32.139 / 31.662 / 31.502 ms/tok, md5 5cbe8e06 ×3. Killswitch-equivalent: INKWELL_CCOPY_PARTS=1.

## 2026-11-24 — warp-parallel kslot resolve (worktree t3-dd0e9b83)

- `inkwell_exl3_kslot_resolve_w`: one warp stages tags, slot offsets and the live free-list in shared memory. Hit lookup, first-fit, free-list shifts and victim min-count scans use warp ballots/reductions; miss processing remains in serial selection order. `INKWELL_NO_RESOLVE_WARP=1` restores the serial kernel; `kmax > 64` also takes the serial path.
- Correctness: `opt/kbench/kslot_gold` compared every mutable state word after each of **22,004 calls** (4 edge, 11,000 random-valid, 11,000 real-trace): bit-identical. Mix: pf 2,137, force_direct 1,220, hm=null 1,111, pnexp 2,583, kmax>64 fallback 1,512. L16 microbench in the harness: serial 13.45 us, warp 6.05 us per launch.
- nsys (`--cuda-graph-trace=node`, 1,224 launches): serial avg **67.10 us**, median 52.15 us, total 82.134 ms; warp avg **19.64 us**, median 17.15 us, total 24.042 ms. Kernel saving ≈58.1 ms/run ≈0.91 ms/token.
- Final A/B on master 1218f58 (ctl = `INKWELL_NO_RESOLVE_WARP=1`, ctl,cand,cand,ctl ×8):
  - ctl: 38.12 39.18 37.81 40.47 40.69 38.54 38.85 37.78 37.81 37.67 37.79 37.77 38.20 38.03 37.73 37.32 ms/tok
  - cand: 37.18 37.42 37.24 39.80 39.47 39.34 41.08 37.54 38.02 36.92 36.45 37.52 36.95 36.64 37.30 37.07 ms/tok
  - median ctl **37.92**, cand **37.36**, diff **−0.56 ms/tok**; pair wins **13/16**; md5 **5cbe8e06** on all 32 runs.
- Verdict: **ACCEPTED** by the E2E bar. The median kernel time (17.15 us) is still above the stretch target of ~10–15 us. A shared-memory count-table variant was tested and reverted: it slowed the production kernel to 19.98 us avg/19.02 us median.
- warnings 63; kernels.cu 5,237 lines. Implementation commits 74c8176 through 2e53abb on `t3/warp-parallel-kslot-resolve-1`.

## t3/kslot-warp-retry — warp-parallel kslot resolve — ACCEPTED, merged
- Retry of b28547b (earlier 11/16 on noisy host). A/B 16 pairs vs master 4e7e6d5: CTL 30.851 vs CAND 30.316 ms/tok median (−0.534), 15/16 wins, md5 5cbe8e06 all runs, peak 5,034 MiB.
- Killswitch: INKWELL_NO_RESOLVE_WARP=1.
- Post-merge master dd6232a, 3 runs: 30.126 / 30.393 / 29.981 ms/tok, md5 5cbe8e06 x3, peak 5,034 MiB.

## t3/ctrlbatch-sel3 — controller triad batching, spill fixed, default-on — ACCEPTED, merged
- Root cause (study 4a7e58c): runtime blockIdx.y index into by-value param pointer arrays spilled them to local memory on sm_61 (rms2down_nosmem_mr<4,3> STACK 96, gf_apply_batch<3> STACK 120, ~849 MB/tok local traffic) — the +4.9 ms CTRLBATCH regression. ink_sel3 constant-offset select -> STACK 0 on all three kernels.
- A/B 16 pairs vs master 16adb17: CTL 31.673 vs CAND 31.226 ms/tok median (−0.447), 13/16 wins, md5 5cbe8e06 all runs, peak 5,034 MiB. Killswitch INKWELL_NO_CTRLBATCH=1.

## 2026-11-25 — Decode host-latency pipeline: ACCEPTED (16/16, -0.714 ms median)

Branch `t3/host-latency`, based on master `79cd1e1` (same code as ledger-only `438eb48`). The acceptance control was a frozen binary built from `79cd1e1`; this matters because the shared master checkout advanced during the run. Decode ms/token is `decode 8 tok in Xs` multiplied by `1000/64` (64 tokens are generated despite the label).

**Measurement.** The fresh profile study found three host-side boundary costs: `embed.weight` mmap cold faults at 0.955 ms/tok uninstrumented (6 x 9.2-13.4 ms stalls in one 64-token run; 15 faults and 1.544 ms/tok under nsys), the pos=64 KV-bucket graph rebuild at 40.7 ms host + 12.4 ms GPU once, and an HtoD-to-first-kernel gap that is mostly nsys overhead (about 50-70 us exposed uninstrumented, not the 1.247 ms/tok node-traced value). Earlier token-boundary profiling showed a median gap of about 0.873 ms from the 4-byte readback end to the first kernel of the next token, inflated by nsys.

**Accepted fix.** The safetensors BF16 embedding table is copied once at load into pinned mapped host memory. In the normal cached decode path, after token N's argmax and 4-byte readback are queued, the host marshals token N+1 into the opposite parity pinned input buffer and queues token N+1 before synchronizing token N. A tiny stream-ordered device kernel reads the winning id from the argmax buffer, writes it into the next graph input, and converts that BF16 embedding row directly into the device input buffer. The small inputs and embedding row share one packed input HtoD span; when the device resolver is active the copy is truncated before the embedding row because the kernel writes that row on device. Killswitches: `INKWELL_NO_PIPESTEP` disables the pinned table/pipeline, `INKWELL_NO_ONEH2D` disables input packing, and `INKWELL_NO_EMBED_WARM` disables the unpinned fallback rewarm. `INKWELL_STEPPROF` reports host-phase timings.

**Acceptance.** Interleaved `ctl,cand,cand,ctl` x 8 against frozen `79cd1e1`, paired positionally as (1,2) and (4,3) in each group. Every run generated md5 `5cbe8e06`; peak process VRAM was 5034 MiB on every run (limit 5080). Build warnings were 62 (limit 64).

| metric | ctl `79cd1e1` | cand |
|---|---:|---:|
| median ms/token | 32.278 | **31.564** |
| mean ms/token | 32.5733 | **31.5961** |
| positional pair wins | — | **16/16** |
| peak MiB | 5034 | 5034 |

**Host-memory cost.** The full BF16 table copy is 883.4 MiB of pinned host memory. Process RSS does not account for it like ordinary anonymous memory: per-run peak RssAnon medians were 549,732 KiB candidate vs 544,676 KiB control (about +5 MiB), with RssFile unchanged. Median minimum `MemAvailable` while processes were alive was 5,524,946 KiB candidate vs 6,466,928 KiB control, about 918 MiB lower, consistent with the pinned table. This is a real host-memory tradeoff even though `/proc/pid/status` mostly hides it.

**Negative results.** An earlier packed-upload + `cudaLaunchHostFunc` host-callback pipeline on master `887150b` was correct but not accepted: median 33.608 ctl vs 33.5525 cand, only 8/16 positional pair wins. The callback enqueue cost roughly hundreds of microseconds and was removed; the final design uses the device resolver instead. Increasing KV buckets from 64 to 128 to avoid the pos=64 rebuild was also rejected: in a repeated probe pin+pipeline averaged about 31.552 ms/tok with 64 padding vs 31.999 with 128 padding because the extra padded attention work cost more than the amortized rebuild. The unpinned mmap rewarm path was correct and reduced cold-fault exposure, but was not better than the final pinned/device-resolver design.

**Residual.** The pos=64 KV-bucket graph rebuild is still present. Simply padding to 128 is worse, and the ggml CUDA graph key currently hashes the op sequence rather than shapes, so a one-shot shape-128 prewarm is overwritten by subsequent shape-64 use. Fixing this likely needs shape-aware graph keys or a deliberate multi-shape graph cache, not just host scheduling.

## t3/host-latency (aac66a8) — pipelined decode input resolution, pinned embed — ACCEPTED, merged
- Re-validated on master 16adb17 (merge 561d038): CTL 31.885 vs CAND 30.041 ms/tok median (−1.843), 16/16 wins, md5 5cbe8e06 all runs, peak 5,034 MiB.
- Host RAM cost: 883 MiB pinned BF16 embed table (MemAvailable −918 MiB). Killswitches INKWELL_NO_PIPESTEP / INKWELL_NO_ONEH2D / INKWELL_NO_EMBED_WARM.
- Post-merge master fab1ac7 (ctrlbatch + host), 3 runs: 29.711 / 29.899 / 29.619 ms/tok, md5 5cbe8e06 x3, peak 5,032 MiB, 62 warnings.

## Queue (orchestrator, 2026-11-25 night) — studies done, awaiting SWE slots
- Graph rebuild at KV buckets: study 8f82e2b (t3/kv-bucket-graph-rebuild-study). +12.4 ms per 64-tok boundary (0.19 ms/tok amortised), multi-graph cache ruled out (26 MiB/graph set). Top design: device-scalar n_kv, ~0.32–0.40 ms/tok incl. padded-KV waste. Killswitch INKWELL_NO_DYN_NKV.
- Expert GEMV floor: study 5f5efbe (t3/expert-gemv-instruction-floor). ~1.0–1.4 ms/tok exact headroom: uint4 4-tile interleave (−0.61), M=16 2-col/warp K2 (−0.58), R=2 blocking (−0.44).

## 2026-11-25 — exact GQA-3 shared-KV short-context attention (branch `t3/attn-gqa3`, `f2de4de`): NOT accepted, 11/16

- Implementation: `inkwell_attn_decode_gqa3_exact`, grid `(hkv,1)x256`, one CTA per KV head. K/V rows are read once with `uint2`/`LDG.64`, Q with `float4` when aligned, K rows are tiled through shared memory, and each V row is tiled once then reused for all three query heads. The q·k lane order (`lane+32*t`), shuffle trees, softmax max/sum trees, `__expf`, and ascending-j per-output V accumulation reproduce `inkwell_attn_decode<true>` exactly. Dispatch requires `kv16 && nh == 3*hkv && hd == 128 && n_kv < INKWELL_ATTNSPLIT_MIN` plus 8-byte K/V alignment; `INKWELL_NO_ATTN_GQA_EXACT=1` is the killswitch. Commit `f2de4de`; release build rc 0, warnings **62** (limit 64).
- Standalone exactness/performance harness: `opt/kbench/attn_gqa3/probe.cu`, random F16 K/V, random F32 Q/bias, trailing `-inf` masks for n>=32. Bitwise output diff count was **0/3072 outputs** for every case `{1,7,32,63,64,65,128,200,319}`.

| n_kv | bit diffs | old us | new us | speedup |
|---:|---:|---:|---:|---:|
| 1 | 0 | 5.12 | 5.60 | 0.91x |
| 7 | 0 | 5.82 | 5.73 | 1.02x |
| 32 | 0 | 9.22 | 7.17 | 1.29x |
| 63 | 0 | 14.34 | 9.22 | 1.56x |
| 64 | 0 | 14.08 | 9.22 | 1.53x |
| 65 | 0 | 14.34 | 9.70 | 1.48x |
| 128 | 0 | 23.55 | 13.31 | 1.77x |
| 200 | 0 | 34.37 | 19.33 | 1.78x |
| 319 | 0 | 52.19 | 27.65 | 1.89x |

- Variant note: row-staged K + split-chain V, tiled-K + split-chain V, tiled-K + all-three-head V, and streaming-K + all-three-head V were all bit-exact in the same harness. The integrated tiled-K/all-three-head V table above was the best normalized result at the time it was frozen; absolute probe times shifted with GPU clock/queue state, so cross-run variant comparisons were not used for acceptance.
- ctx512 validation (3 runs, frozen candidate `91411f7e`): generated-text md5 **`5cbe8e06` every run**; decode 31.885 / 29.999 / 32.148 ms/tok; peak **5,034 MiB** every run (cap 5,080).
- End-to-end A/B: `ab_rw.sh`, 16 neighbouring pairs in one `nice -n 19 flock /tmp/inkwell-gpu.lock`; CTL was frozen master `16adb17` (`ffd3eb28`), CAND the frozen candidate. `absum`: `runs ctl=16 cand=16 ctl_med=31.846 cand_med=31.684 delta=-0.162 wins=11/16 md5=['5cbe8e06'] peak=5034`. Per-pair candidate-minus-control ms/tok: `-0.181 +0.104 -0.334 -0.253 -0.147 +1.169 +0.463 -3.658 -0.617 -0.014 -0.057 -0.198 +1.133 -1.493 -1.541 +1.151`.
- Verdict: **NOT accepted**. The median was 0.162 ms/tok faster and every md5/VRAM gate passed, but the candidate won only **11/16** pairs, below the required 12/16. Part 2 (`splitw` cleanup) was not attempted because the Part 1 acceptance gate failed.

## t3/attn-gqa3 (f2de4de) — exact GQA-3 shared-KV short-ctx attention decode — ACCEPTED on 2nd attempt, merged
- kbench: 0 bit diffs at n_kv {1,7,32,63,64,65,128,200,319}; 14.08 -> 9.22 µs at n_kv=64, 1.5–1.9x for n_kv ≥ 63.
- Attempt 1 (child, vs master 16adb17-era): −0.162 ms, 11/16 — failed pair gate.
- Attempt 2 (orchestrator, merged onto fab1ac7, CTL frozen fab1ac7): CTL 29.735 vs CAND 29.418 ms/tok (−0.317), 14/16, md5 5cbe8e06 all runs, peak 5,032 MiB. Combined 25/32.
- Killswitch INKWELL_NO_ATTN_GQA_EXACT=1.
- Post-merge master 54dc27f, 3 runs: 29.589 / 29.312 / 29.654 ms/tok, md5 5cbe8e06 x3, peak 5,032 MiB, 62 warnings.

## 2026-11-25 — dense fused CH2 gemv_sf (branch t3/ch2-dense-sf): ACCEPTED, default ON

Branch `t3/ch2-dense-sf`, based on `376e449`, ports the validated sm_61 CH2 pipeline to the dense fused EXL3 GEMV `inkwell_exl3_gemv_sf` through the driver-loaded clone `inkwell_exl3_gemv_sf_ch2`. Main implementation/audit commits: `a8d97a5`, `6c9c540`, `fa317da`; correctness repair `511ab4b`; post-acceptance default flip `9aa2927`.

**Audit.** The inkling dense projection set has 24 layers and two recurrent passes per token. Distinct dense matrices by K: K3 = 48 (`wq_du` + `gate_attn`), K3.5 = 72 (`wk_dv` + `wv_dv` + `wo_ud`), K4 = 1 (unembed), K5 = 24 (`wr_du`). Per-token invocations: K3 96, K3.5 144, K4 1, K5 48. Shapes: `wq_du`/`gate_attn` K2304xN3072; `wk_dv`/`wv_dv` K2304xN1024; `wo_ud` K3072xN2304; unembed K2304xN201088; `wr_du` K2304xN384. Shipped scalar loop audit, using the same instruction classification as the expert audit:

| loop | instr/32w | instr/weight | core | extract | LDG | LDS | addr/loop | ctl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K2.5 (unused here) | 293 | 9.156 | 192 | 36 | 12 | 8 | 28 | 6 |
| K1.5 (unused here) | 269 | 8.406 | 192 | 32 | 8 | 8 | 20 | 3 |
| K2 (unused here) | 269 | 8.406 | 192 | 32 | 8 | 8 | 20 | 3 |
| K3.5 | 304 | 9.500 | 192 | 36 | 12 | 8 | 28 | 7 |
| K3 | 302 | 9.438 | 192 | 36 | 12 | 8 | 28 | 7 |
| K4 | 305 | 9.531 | 192 | 32 | 12 | 8 | 28 | 6 |
| K5 | 305 | 9.531 | 192 | 36 | 12 | 8 | 28 | 4 |

The dense K3/K3.5/K4 NOP-less CH2 loops are 232 instructions per 32 weights (**7.250/w**), with SHF 36, LDG 4, LDS 8, SHFL 8, XMAD 64, IDP 32, FFMA 32, and zero NOPs. K5 deliberately remains the shipped core because the unpaired K5 CH2 path is not exact. The pw16 layout remains unused because the bitwise requirement preserves the shipped scalar reduction/layout.

**Transforms and exactness.** The dense clone uses the exact 2-XMAD high-half product (`w*0xd12d + ((w*0x83dc)<<16)`), the exact integer byte sum and I2F form, unchanged FP accumulation order, CH2 ring windows, donor-lane geometry, and SHFL where proven. CPU proofs passed: view cases 1,120,000; high-half product cases 20,000; TOP32 rotate views 320,000; tile-level GEMV exactness 280; TOP32 donor cases 64; generic donor lanes K2.5/K3/K3.5/K4 (256 donor cases, 24,576 stream cases). The runtime startup self-test compares the driver kernel bitwise against the shipped kernel for K3, K3.5, K4, and the unchanged K5 fallback before enabling.

**Build design.** `build_nopless.py` now validates the dense kernel in the same fail-closed cubin as the expert: pinned CuAssembler `96a9f72baf00f40b9b299653fcef8d3e2b4a3d49` plus the two local patches; patched-source fingerprint `(fake=583, lp=587, words=2198, hi_d12d=588)`; three dense loops with shape `(232 instructions, 36 SHF, 4 LDG)`; zero NOPs; corrected EXIT/COOP metadata; dense `REG:40`, `SHARED:12848`, zero stack/local spills. Generated cubin sha256 `d0c95bd0d5afc97c82b6baf216995d8aee9a28d181c81d8d26da8e827adc719e`, **1,362,896 bytes**. Release build rc 0, **62 warnings** (limit 64). Final release binary sha256 `d4613b22d54185223cc6b80c281a5e3feb2b1f99f1af36cf9c48c1f1c06e791f`. Killswitch: `INKWELL_NO_CH2DENSE=1`; `INKWELL_CH2DENSE=0` also opts out after the default flip.

**Dead end and root cause.** The first dense artifact was bitwise-correct for K3.5/K4 but wrong for K3 (`badbits=3072/3072`, maxabs `1.22e3`) and absurdly fast (`0.0060 ms` versus `0.0338 ms`), so it was invalid. The first mitigation commit `3431c43` excluded K3, but further inspection showed the stream was not the cause. CuAssembler represents `.nv.constant2` jump tables as opaque bytes; deleting NOPs shifted case labels, while the BRX table entries stayed at their old absolute addresses. kk=6/K3 is table-index 0 and its target follows the K3.5 loop, so dispatch entered `0x158` bytes late. CuAssembler's `.nv.info` indirect-target list is label-based and correct after reassembly, so `511ab4b` uses it as the authoritative old→new mapping, rewrites only those constant-bank entries, verifies the full constant bank byte-for-byte, and fails closed on any mismatch. K3 was restored. This repair is also required for correctness independent of performance.

**Isolated GPU gate.** `dense_sf_drv_perk`, production shapes and nky, min-of-31, all bitwise exact:

| case | bad bits | ref ms | cand ms | ref/cand |
|---|---:|---:|---:|---:|
| K3 / gate | 0/3072 | 0.0328 | 0.0284 | **1.1558x** |
| K3.5 / wo | 0/2304 | 0.0379 | 0.0317 | **1.1935x** |
| K4 / unembed | 0/201088 | 1.7375 | 1.4374 | **1.2087x** |
| K5 / wr fallback | 0/384 | 0.0119 | 0.0120 | 0.9893x |

K5 is the unchanged fallback; the small ratio difference is noise. Before the indirect-table repair the runtime correctly failed closed with `ch2dense: disabled: bitwise self-test failed`; after the repair the ctx512 smoke logged `ch2dense: enabled cubin=1362896 bytes selftest=0.34 ms`. Startup self-test cost: **0.34 ms**.

**End-to-end exactness gates.** Feature-on ctx512 run exited 0, generated-text md5 **`5cbe8e06`**, decode `2.034917125 s` for 64 tokens (31.796 ms/tok despite the misleading log label). Eight-step full-logit dumps with the feature on and off both exited 0, were 6,401,888 bytes each, and compared byte-identical (`cmp` rc 0). The graph-enabled run completed normally.

**End-to-end A/B.** `~/.cache/inkwell-ab/ab_rw.sh`, CTL `/home/ash/.cache/inkwell-ab/ctl-376e449/inkwell`, CAND this branch with `INKWELL_CH2DENSE=1`, 16 interleaved neighbour pairs in order ctl,cand,cand,ctl inside one `nice -n 19 flock /tmp/inkwell-gpu.lock`. CTL median **32.5345 ms/tok**; CAND median **31.290 ms/tok** (**−1.2445 ms/tok**); candidate won **16/16** pairs. Every run md5 was `5cbe8e06`; peak VRAM was **5,034 MiB** (limit 5,080); script rc 0. Per-pair candidate-minus-control deltas (ms/tok): `-1.302 -1.223 -3.282 -0.396 -2.319 -1.346 -1.559 -1.055 -1.154 -1.227 -1.132 -0.942 -1.688 -1.149 -0.177 -0.862`. Acceptance criteria were met, so the default was flipped on in separate commit `9aa2927`.

**Follow-up, not implemented.** `inkwell_exl3_gemv_sf` still uses static `__shared__ float xs[3072]` (12,848 B), capping occupancy at 30 CTAs; the 36-CTA `wo_ud` and similar dense matrices therefore run in two waves. Dynamically sizing shared memory to the actual K is predicted to save another 0.2–0.5 ms/tok bit-exactly and should be investigated separately.

## t3/ch2-dense-sf (6450a8b) — ch2 NOP-less pipeline on dense gemv_sf K3/K3.5/K4 — ACCEPTED, merged
- Child A/B vs 376e449: −1.2445 ms, 16/16. Includes fail-closed repair of .nv.constant2 indirect-branch tables after NOP deletion (511ab4b).
- Re-validated merged onto 54dc27f (74527c7), CTL frozen 54dc27f: CTL 29.480 vs CAND 28.267 ms/tok (−1.212), 16/16, md5 5cbe8e06 all runs, peak 5,032 MiB. Self-test 0.41 ms.
- Killswitch INKWELL_NO_CH2DENSE=1. Follow-up: dynamic smem for gemv_sf xs[] (2-wave tail on 36-CTA wo_ud), predicted −0.2–0.5 ms.
- Post-merge master e03df94, 3 runs: 30.075 / 34.351 / 28.961 ms/tok, md5 5cbe8e06 x3, ch2dense enabled x3, peak 5,032 MiB, 62 warnings. Run 2 outlier: desktop GPU processes (t3code/electron) active; A/B pairs are the reference.

## t3/mix-rework (8f1c9f0) — msg batching (12×qcols=4 → 1×qcols=48 per pass) — PARKED (A/B inconclusive)
- Bit-exact (0 diffs, all shapes + packed vs 12 independent). kbench: 132.37 → 14.57 µs per pass (×2 passes ≈ −0.235 ms/tok deterministic).
- Narrow 1A/1B rejected (qcols=1 slower; A/B 2/16, 6/16).
- Batch-only A/B: −0.269 (11/16), retry −0.348 (9/16) on a noisy host. Not merged; no third solo retry.
- Plan: re-gate bundled with the next small exact change (e.g. gemv_sf dynamic smem) on a quiet host.

## Neon study: path to 20 ms (spec decode / batched verify) — partial, child crashed (provider error); data recovered by orchestrator
Worktree t3/spec-decode-feasibility-study (uncommitted diagnostic hooks `INKWELL_LOOP_NP`, `INKWELL_SKIP_EPI`, `INKWELL_SELFSPEC_EVAL`). Raw: /tmp/spec_study_results.txt, /tmp/spec_study_part2.txt.
- Batched EXL3 GEMV kbench, B tokens sharing the same weights, 0 bit diffs B=1..8. B4/B1 time: MoE GU K2 1.37×, K3 1.20×, MoE DN 1.35×, dense wq 0.97×, wo 0.95×, unembed 1.45×. Caveat: real MoE verify routes different experts per token, so the expert union grows. This bench is a lower bound for MoE cost.
- nsys baseline 9c735bd: ccopy 31.3% of GPU time (72 inst, 449 µs avg), gemv_sf_ch2 25.1%, gemv_moe_s 24.1%, mix 2.3%, attn 1.5%.
- Reduced-pass self-draft, measured decode with graphs, 32 tok: np3 30.06, np2 21.87, np1 16.16, np0 9.19 ms/tok. Skipping the epilogue: np2 18.60, np1 13.09.
- Self-spec acceptance, 5 prompts × 16 samples, greedy, draft = fewer loop passes:

| draft | p1 range | emit@k=4 range |
|---|---|---|
| np2+epi (36 visits) | 0.81–1.00 | 3.44–4.88 |
| np1+epi (24 visits) | 0.75–1.00 | 3.13–4.88 |
| np0+epi (12 visits) | 0.44–0.69 | 1.88–2.88 |
| no-epilogue variants | ≤0.25 | ≤1.25 (useless) |

- Existing paths are BROKEN on master: `INKWELL_VTEST8`, `INKWELL_SPEC=7` (n-gram) and `INKWELL_SPEC=mtp` all abort with `GGML_ASSERT(view_size + view_offs <= ggml_nbytes(view_src))` (ggml.c:1868). Probably a regression from the kslot/arena or NT-batch views.
- Orchestrator estimate: np1 draft at 16.2 ms/tok is too expensive for a naive scheme. Drafting k=3 costs 48 ms and a full 4-token verify ≥40 ms, which is about 27–31 ms/tok and no gain. A viable path must reuse the draft's prelude + pass-1 work in the verify (LayerSkip-style shared prefix: only passes 2–3 + epilogue run batched). It must also amortise ccopy/expert streaming across the verified tokens.

## Neon study: encode/prefill (batched prefill) — partial, child crashed (provider error); data recovered by orchestrator
Raw: /tmp/prefill-study/results/{kbench,default_ctx512,vtest8_ctx512}.log. Only part A (kbench) and part B1 ran. Parts B2–D (512/2K/8K windows, nsys) did not run.
- Current encode is serial and slower than decode: `encode 4 tok in 261 ms` (65 ms/tok) and 551 ms on a second run. Decode is ~29 ms/tok.
- Batched EXL3 GEMM kbench (sm_61), µs per token:

| shape | M=4 | M=16 | M=64 |
|---|---|---|---|
| MoE gu K2.0 | 39.8 | 16.8 | 5.6 |
| MoE gu K3.0 | 36.4 | 16.6 | 6.0 |
| MoE dn K2.5 | 24.5 | 12.5 | 4.35 |

  Dense wq K3 and wo K3.5 sit at ~10–14 µs/tok for M 64–512, at ~1.0–1.4 TFLOP/s. Bit-exactness was checked only at M=16 (0 diffs). The other rows were not checked (-1).
- Flash prefill attention kbench (nh24/hkv8/hd128, F16 KV): 0.19–0.24 TFLOP/s, far below the 1060's ~4.4 TFLOP/s FP32 peak.
  - SWA costs 56–66 µs/tok.
  - FULL attention cost grows linearly with position: 228 µs/tok at 4K, 1.79 ms/tok at 32K, 3.55 ms/tok at 65K.
  - Implied 64K full-attn prefill at this kernel: ~115 s. The attention kernel needs a rewrite before 64K prefill is usable.
- ctx512 default prompt: md5 5cbe8e06.
- VTEST8 works in this probe binary, but the block verify is not bit-exact vs serial: j0–j7 max diff 2.5e-4 to 8.1e-4. It is also slow (8-token block 701 ms, 87.6 ms/tok) and OOMs in cudaGraphInstantiate at gen 8 with CACHE_MARGIN_MB=10.
- Conclusion: batched MoE GEMM gives a 6–8× per-token gain at M=64 vs M=4. A real prefill path needs: batched expert GEMM over the token chunk, a faster flash-prefill attention, and graph memory headroom. Exactness vs serial must be gated (md5 of generated text after prefill, plus a KLD gate if float order changes).

## t3/dyn-nkv — post-merge re-validation on master 9c735bd (scratch merge 5c3dc31) — NOT MERGED (gate miss)
- Frozen CTL e03df94, 16 interleaved pairs: CTL 28.684 vs CAND 28.507 ms/tok (−0.178), wins **11/16** (gate ≥12), md5 5cbe8e06 all runs, peak 5,032 MiB.
- Child result had been −0.331, 14/16 vs 731adf3. The run shared the host with queued delegate GPU jobs.
- Exactness is proven (0 bit diffs, kbench). The feature's main value is long context: shape-invariant graphs, no rebuild except at pos=256. Decision: one retry on a quiet host, or gate it as a long-ctx feature (non-regression at 512 + measured win at 4K/32K).

## Spec-decode study — final (bb1a8be, opt/studies/spec-decode-2026-11-25.md)
- Supersedes the partial entry above. An oracle drafter (0 ms) still has a floor of 20.13 ms/tok at B=2. Real routing grows distinct experts 8.00→12.89/16.59/19.96 (B=2/3/4), so the ragged MoE verify costs 1.67/2.19/2.68×.
- Best training-free drafter: int4-requantised mtp.safetensors (~120 MiB, 1.45 ms) + ragged B=2 verify, modelled at 25.48 ms/tok. Self-spec np1 costs 29.98 ms/tok (no gain).
- ≤20 ms requires a trained ~30 MiB EAGLE-style drafter plus a bit-exact ragged B=2..4 verify (modelled 19.5–22 ms/tok). It also requires fixing the Runner::control() GGML_ASSERT regression.

## Prefill study — final (d2bc6bc, opt/studies/prefill-2026-11-25.md)
- Supersedes the partial prefill entry above.
- Serial step() encode: 32.8–40.8 ms/tok (512–8K).
- Existing step_block (NT=8, after 3 bugfixes left unstaged on t3/prefill-chunked-study) is 1.6–1.9× slower, 52.6–78.1 ms/tok, and peaks at 5,406 MiB (over the cap). Cause: MOE_NKY=2 forces gemv_moe_ntok (40.8–49.9 ms/tok vs 12.4–13.5 for ch2nopless).
- step_block is not exact: atomicAdd nky=2, mix8, control, clip and router all change FP order. At 512 tok the first-gen KLD is 3.53e-5, but rare MoE top-8 flips give K_max_abs 10.83, and greedy text diverges at gen 56/64.
- Architecture: loop state is per-token; sconv is a parallel FIR; only attention is cross-token. PCIe streaming of the 14 host-pinned MoE layers (3.4–4.9 GiB/chunk) needs C=256–512 with double buffering.
- FP32 tiled prefill at C=512 with GQA-3 flash attention is predicted at 81.9 tok/s (2K), 80.2 (8K), 75.2 (32K) and 69.6 (64K; needs K4V2). It borrows 108.8 MiB from the expert cache.
- Plan: FP-order prefill gated by teacher-forced KLD (<1e-3 first token, <5e-4 64-tok mean), killswitch INKWELL_NO_PREFILL. Phases: 0 = step_block fixes + step_id encode; 1 = prefill.cu kernels; 2 = PrefillArena + Runner::prefill.
- Orchestrator note: ~80 tok/s is the FP32 path. Challenge 4 (integer Tier Q numerics) targets ≥150 MAC/SM-cycle for the 500 tok/s goal.
## t3/swa-ring-fix — ring-correct sliding-window KV cache — implemented, with oracle-noise caveat

Base `9c735bd`. Implementation in `crates/inkwell-run/src/runner.rs` plus teacher-forcing/debug harness in `inkwell-cli/src/main.rs` and `opt/swa-ring-fix/`.

**Paths fixed.** SWA KV slot selection, relative-position generation, and visibility masks were centralized and applied in serial input upload, packed one-H2D/pipelined decode, cached graph premarshaling, and the NT-token `step_block` path. Default slot is now `pos % 1024`; `INKWELL_SWA_LEGACY_MIN=1` restores the old `min(pos,1023)` behavior only for comparison. `INKWELL_SWA_ORACLE=1` allocates full-length SWA arenas, writes `kv_idx=pos`, uses `rel_pos=pos-slot`, and masks everything older than the last 1024 positions. Ring rel-positions are `(pos-slot) mod 1024`, clamped to `rel_extent`; unfilled/padded slots remain masked. Normal, split-width, and GQA-3 attention consume the supplied per-slot mask/rel-bias and needed no kernel change. State snapshots are versioned `INKWST02` because ring slot meanings changed; oracle and ring state layouts are not interchangeable.

**NT overwrite hazard.** Once the ring is full, every later write in an 8-token batch can evict a row still visible to an earlier token in that batch. `step_block` now detects this regime and makes each later KV write depend on the previous token's attention output through a mathematically neutral `0*x+1` scalar. A separate pre-existing batch bug was also fixed: the fused controller path is serial-only and returned `[n,1]` for `[n,NT]`, causing the ggml view assert seen on master. Batch control now slices each token, applies the exact serial fused control path, and concatenates the token outputs.

**Teacher-forced oracle table** (natural text, 128 rows, full-vocabulary float64 KLD):

| context | top-1 agreement | mean KLD | median | p95 | max | row0 KLD |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 127/128 = 0.9922 | 0.0002231 | 7.57e-5 | 0.001020 | 0.005919 | 0.000170 |
| 4K | 125/128 = 0.9766 | 0.0049746 | 0.001872 | 0.018443 | 0.069943 | 0.000551 |

The 2K independent-run gate passes. The raw independent 4K run misses the requested gate, but the failure is numerical amplification rather than a demonstrated slot/mask bug. Evidence:

- From mathematically identical 4K states, ring vs oracle teacher-forcing for 16 rows was essentially exact: top-1 16/16, mean KLD 4.41e-9, max KLD 1.94e-8, row0 bitwise identical.
- Ring-vs-oracle KV state comparison at the end of independent 4K encodes: 81,872/81,920 rows (99.94%) were closer to the correct oracle position than to either adjacent position; correct/adjacent difference ratio mean 0.0293, p95 0.1126. This validates the physical ring write-slot mapping despite fp divergence.
- One controlled NT block starting at position 4095 across the ring wrap, again from mathematically identical ring/oracle states: top-1 8/8, row0 KLD 9.08e-5, mean KLD 0.004324, max KLD 0.015089. KLD grows over the 8 rows because scan-order noise recurs through the block, but there is no systematic top-1 divergence.

**ctx512 regression.** Canonical validation passed all three runs: md5 `5cbe8e06` each time; decode 30.063 / 29.952 / 30.549 ms/tok; peak VRAM 5,032 MiB each run (cap 5,080); `ch2nopless: enabled`.

**4K free-run.** Fixed binary rc0 and coherent text. First 200 characters:

```text
phone-mic status\nphone-dict "texto"       # ditar e colar no campo de foco (backend escolhido)\nphone-dict --backend parake "texto"\nphone-dict --backend hex "texto"\n```\n\n`phone-dict` aceita `--ba
```

Single-run decode at 4K: master 43.599 ms/tok vs fixed 48.663 ms/tok. The master baseline used a prompt one token shorter because shell command substitution strips the trailing newline; treat this as indicative, not a rigorous paired benchmark. Master's generated text degenerated to repeated `||||...`, while the fixed binary stayed coherent.

**NT/VTEST status.** Master crashes even at small context in `INKWELL_VTEST8` with `GGML_ASSERT(view_size + view_offs <= ggml_nbytes(view_src))`; this is the pre-existing serial-only controller shape bug, not an SWA regression. The fixed branch no longer crashes. However serial-vs-block logits remain non-exact even at ctx512/base4 where ring and legacy indexing are identical and the wrap hazard is false (8 rows, 2 argmax mismatches, maxdiff up to 9.5). That pre-existing batch numerical discrepancy means VTEST8 cannot isolate SWA overwrite correctness end-to-end. The controlled ring-vs-oracle BTEST above is the meaningful SWA-specific NT evidence.

**Build/state.** Release build rc0. Warning count 55 warning blocks / 59 lines beginning `warning` (limit 64). Validation artifacts are in `opt/swa-ring-fix/artifacts/` and intentionally not committed.

## t3/swa-ring-fix — A/B and merge (master ed6f659)

- 16-pair A/B: CTL frozen e03df94 vs CAND master bcb7f52 + ring fix (scratch merge 05a343e), ctx512, interleaved ctl,cand,cand,ctl.
- Result: ctl_med 29.009, cand_med 29.044 ms/tok (delta +0.035, within noise). Wins 9/16. md5 5cbe8e06 on all 32 runs. Peak 5,032 MiB.
- Verdict: MERGED as a correctness fix. Short context costs nothing; without the fix, output past 1K is garbage ("||||" at 4K). Killswitch `INKWELL_SWA_LEGACY_MIN=1`.
- Open: 4K speed (48.66 vs 43.60 ms/tok single run) is not comparable, because master's 4K output is garbage. It needs a proper 4K A/B against a correct baseline.
- Post-merge check on master ed6f659 (release build, 59 warnings), ctx512: md5 5cbe8e06 ×3; 28.97 / 28.40 / 29.86 ms/tok.

## t3/sf-dynsmem (07208de) — msg batching alone (`INKWELL_NO_SF_DYNSMEM=1`), orchestrator A/B
- Bundle (dynsmem + msg batching), child A/B: +0.083 ms/tok, 7/16 wins. REJECTED. Isolated, dynamic smem is 0.76–0.96× of static. Occupancy is 6 CTAs/SM before and after (register-capped), so the premise did not hold.
- Msg batching only, 16 pairs vs frozen CTL e03df94: ctl_med 28.455, cand_med 28.564 (+0.109). Wins 11/16. md5 5cbe8e06 on all runs. Peak 5,032 MiB.
- Per-pair deltas (cand − ctl, mean of the two runs in each ctl,cand,cand,ctl block, 8 blocks): −0.06, −2.63, +1.85, −1.36, −0.01, +0.28, −0.41, −0.39. Median ≈ −0.2.
- The host was loaded (4 agents, with 30–31 ms outliers on both arms).
- Verdict: near-miss, not merged. Queued for an improvement plus re-A/B when a code slot frees.

## t3/moe-phase1 (a967ddb) — expert GEMV phase-1 exact transforms (T2 + simple T3): superseded by the Challenge 3b port
- 7.141 → 6.843 instr/w. Bit-exact on all K and both shapes. Kernel geomean 1.040×, normalized. md5 5cbe8e06 ×3, peak 5,032 MiB, 62 warnings.
- A/B vs frozen CTL e03df94: −0.104 ms (9/16), then −0.117 ms (11/16). Median faster both times, pair gate missed.
- Rejected inside the branch: T7 (bit-exact, slower on dn); T6 (+0.57 ms with T7, 3/16; 8/16 without); post-SASS MOV removal (illegal address); K4 paired extraction (ptxas reassociated the XMADs).
- Verdict: not merged on its own. It edits the same loop as t3/challenge3b-whole-loop-port, which gets lower (6.775 instr/w in part 1; target 6.442). Its techniques were handed to the 3b part-2 agent, and the near-miss retry is folded into that A/B.

## t3/challenge3b-whole-loop-port — practical whole-loop hybrid — BIT-EXACT, PARTIAL vs requested schedule

**Scope.** Ports the implementable parts of the Challenge 3b whole-loop schedule into the production sm_61 expert GEMV module while preserving the shipped arithmetic core, reduction, epilogue, weight layout, and FFMA order. K5 remains on the scalar path. This is deliberately recorded as a **practical hybrid**, not the full direct-`LDG`/no-donor schedule.

**Implemented.**
- Shared-memory x is permuted at block start so each lane reads `x[r0],x[r0+1],x[r0+8],x[r0+9]` as one conflict-free `LDS.128` (`r0=2*(t&3)`). Dense CH2 K3/K3.5/K4 uses the same layout; scalar/pw16 layouts are unchanged.
- Production expert widths `tn=72` and `tn=144` are compile-time specialized. ptxas folds the per-tile row stride into `LDG` immediate offsets; unusual widths use the exact scalar path with unpermuted x.
- K2 uses the aligned stream and exact byte-permutation windows for j=3/j=7.
- K4 uses paired funnels `(j,j+4)`, one shared `PRMT sel=0x0765` for j=1/j=5, and the free high half for j=3.
- K2.5/K3 no longer fetch the provably unused W2 ring word. K3.5 still needs W2.
- `INKWELL_NO_CH3B=1` disables both runtime-patched CH2 users (`ch2nopless` expert and `ch2dense` dense), making it a true pre-change fallback. The expert startup self-test now covers all six K values in both production MoE shapes (`tk=144,tn=72` and `tk=72,tn=144`) before enabling.

**Not implemented / deviation from the requested construction.** Donor `SHFL` remains for W1 (and for W2 in K3.5). Free low-half XMAD operands are not yet patched, so K4 low halves still need byte-permutation movement; generic K2.5/K3/K3.5 windows still use aligned funnels. An isolated direct cyclic-`LDG` experiment removed every donor shuffle but compiled the expert at **REG:48** and increased fixed-loop instruction counts by roughly 2–6 instructions per 32 weights, because wrap-safe cyclic ring lanes require independent 64-bit bases on sm_61 and `LDG` has no register+register index form. It was rejected rather than silently accepting a slower/nonconforming form.

**Final expert SASS instruction counts.** Per 32 weights after NOP deletion; both tn=72 and tn=144 copies were shape-checked separately. Baseline is the previously shipped production nopless kernel.

| K | baseline instr/32w | new instr/32w | instr/w | target instr/32w | extraction actually present |
|---|---:|---:|---:|---:|---|
| K1.5 | 225 | **215** | **6.7188** | 207 | 32 SHF, 0 PRMT, 4 LDG, 4 LDS, 4 SHFL |
| K2 | 225 | **215** | **6.7188** | 205 | 24 SHF, 8 PRMT, 4 LDG, 4 LDS, 4 SHFL |
| K2.5 | 233 | **219** | **6.8438** | 207 | 36 SHF, 0 PRMT, 4 LDG, 4 LDS, 4 SHFL |
| K3 | 233 | **219** | **6.8438** | 207 | 36 SHF, 0 PRMT, 4 LDG, 4 LDS, 4 SHFL |
| K3.5 | 233 | **223** | **6.9688** | 211 | 36 SHF, 0 PRMT, 4 LDG, 4 LDS, 8 SHFL |
| K4 | 233 | **211** | **6.5938** | 189 | 8 SHF, 20 PRMT, 4 LDG, 4 LDS, 4 SHFL |

Weighted static instruction count: **6.774625 instr/w** vs baseline **7.141** (−5.13%); Challenge 3b target was 6.442125. Both kernels remain `REG:40 STACK:0 LOCAL:0` (expert shared 9,248 B; dense shared 12,848 B).

**Build/pipeline.** `build_nopless.py` remains fail-closed and now validates all 12 expert fixed-stride loop copies plus the 3 dense generic loops. Patch fingerprint `fake=1078 lp=1088 words=4388 hi_d12d=1086`; duplicate loop ranges are accounted per range, not per K. Generated cubin sha256 **`b4f032bdcaecbd32a0bfb71dfa3cdb594f6656d293ed4a5e4e95a987963e1a08`**, **1,478,480 bytes**. Final release build rc **0**, warnings **58** (limit 64). Final binary sha256 **`4fc1d051586a05a9ecc5a5db7bcd16c382713f663817e710e02fc1152bd195bf`**.

**Per-K GPU gate.** `reasm_drv_perk`, both MoE shapes, min-of-31. All bitwise rows were **`badbits=0`** for both the old production nopless module and the new candidate. New/old production min-time speedups:

| K | gu old ms | gu new ms | gu speedup | dn old ms | dn new ms | dn speedup |
|---|---:|---:|---:|---:|---:|---:|
| K1.5 | 0.0696 | 0.0676 | **1.0298x** | 0.0724 | 0.0702 | **1.0305x** |
| K2 | 0.0695 | 0.0666 | **1.0447x** | 0.0716 | 0.0696 | **1.0285x** |
| K2.5 | 0.0717 | 0.0694 | **1.0327x** | 0.0737 | 0.0717 | **1.0286x** |
| K3 | 0.0727 | 0.0696 | **1.0441x** | 0.0753 | 0.0736 | **1.0226x** |
| K3.5 | 0.0744 | 0.0734 | **1.0131x** | 0.0768 | 0.0765 | **1.0042x** |
| K4 | 0.0767 | 0.0745 | **1.0292x** | 0.0788 | 0.0776 | **1.0161x** |

Workload-share aggregate of those isolated timings: gu **+3.61%**, dn **+2.64%**, equal-shape aggregate **+3.12%**. This is only marginally above the 3% honesty threshold and far below the modelled 14% whole-loop target.

**End-to-end validation.** `validate.sh` on the final release binary, three runs each. Default: md5 **`5cbe8e06` ×3**, decode ms/tok **31.093 / 28.480 / 30.855**, peak VRAM **5,032 MiB** each, `ch2nopless enabled` and `ch2dense enabled`. With `INKWELL_NO_CH3B=1`: md5 **`5cbe8e06` ×3**, decode ms/tok **32.341 / 31.975 / 31.810**, peak VRAM **5,032 MiB** each, no CH2 module enable line. Single-run comparison uses run 1: default **31.093 ms/tok**, killswitch **32.341 ms/tok** (−1.248 ms/tok for that pair; noisy single runs, not the 16-pair A/B). Startup self-test cost in the first default run: expert **13.72 ms** (12 production-shape cases), dense **0.33 ms**; validation script rc **0**.

**Interpretation.** Bitwise and end-to-end exactness gates pass, register/spill limits pass, warning and VRAM limits pass, and the isolated expert microbench is marginally above 3% on the equal-shape aggregate. Nevertheless this does **not** implement the requested no-donor/direct-`LDG` schedule and does not approach the modelled 14% decode saving, so it should be treated as a partial hybrid rather than a completed Challenge 3b port.

## t3/challenge3b-whole-loop-port (93f69cc) — orchestrator A/B — MERGED
- 16 pairs vs frozen CTL e03df94: ctl_med 28.467, cand_med 28.200 (−0.267 ms/tok). Wins 13/16. md5 5cbe8e06 on all runs. Peak 5,032 MiB.
- Per-block deltas: −0.13, −0.08, −0.24, +0.08, −0.20, −0.24, −0.11, −0.44.
- Part 2 (direct LDG, per-window SHF, K4 PRMT 20→4, free half operands; target 6.442 instr/w) is in progress on a child branch of 93f69cc.
- Post-merge check on master e7ac0fb (59 warnings), ctx512: md5 5cbe8e06 ×3; 28.08 / 29.17 / 30.34 ms/tok; ch2nopless enabled.
  - Runs 2–3 went through `gpu-run -p high` after the plain-flock attempt starved for more than 1 h.

## ACTQ study — int8 activation KLD calibration (9f4b5fd on t3/actq-kld-epsilon-study-3; doc copied to opt/studies/actq-kld-2026-11-25.md; hooks not merged)
- Teacher-forced, 256 scored rows, one prompt. Sim applied to every EXL3 input, unembed included.

| mode | KLD 512 | KLD 4K | top-1 (512 / 4K) | PPL Δ (512 / 4K) |
|---|---:|---:|---:|---:|
| row8 | 3.1e-3 | 6.6e-3 | 96.5% / 91.8% | +0.66% / +1.48% |
| g128_8 | 2.3e-3 | 4.2e-3 | 96.9% / 94.5% | +0.26% / +0.66% |
| row12 | 1.4e-3 | 3.1e-3 | 96.9% / 96.1% | −0.08% / +0.74% |

- **Noise floor is the key finding.** At 4K, output noise with ε = 1e-6 … 1e-4 all give mean KLD 1.3e-3 … 3.4e-3 and top-1 ~97–98%. Baseline-vs-baseline is exactly 0.
- Cause: chaotic amplification through the 3-pass loop and discrete top-8 routing flips.
- Consequence: an absolute gate of KLD ≤ 1e-3 at 4K cannot be met by any non-bit-exact change, FP32 reorders included (that covers the planned FP-order prefill).
- 256 rows resolve PPL only to about ±0.5%: noise 3e-6 gives +0.46%, noise 1e-5 gives −0.09%.
- The child's power-law ε values (5.5e-4 at 512, 7.5e-5 at 4K for KLD 1e-3) fit a curve that is flat at 4K. Do not use them as budgets.
- Dumps: /tmp/actq-xh/ (15 files; sha256 in the study).
- Next steps: owner decides the gate definition (absolute vs relative to the FP-noise floor); then a larger-sample eval (≥2K scored tokens × several prompts) of the floor (ε 1e-7/1e-6), g128_8 and row12.

## t3/challenge3b-decode-schedule-part2 — direct-window feasibility audit — NEGATIVE / UNRESOLVED, no kernel change

**Scope.** Second implementation attempt starting from `t3/challenge3b-whole-loop-port` (`93f69cc`). This entry records negative evidence gathered before changing the production kernel. The branch intentionally keeps the part-1 bit-exact kernel unless a later commit adds a validated SASS transform.

**Experimental dead ends reproduced locally.**
- A compile-time induction-ring-pointer (`qind`) probe stayed `REG:40 STACK:0 LOCAL:0`, but expert patched-loop counts improved only by 0–2 instructions/32w: K1.5/K2 247->246, K2.5/K3 251->250, K3.5 255->254, K4 243->241/243 depending on copy. It also changed the fail-closed patch fingerprint from `(fake=1078 lp=1088 words=4388 hi_d12d=1086)` to `(1078,1080,6094,1094)`, so it was not a safe drop-in and was reverted.
- A CUDA-level K4 low-half probe (`mul.wide.u16 lo`) reproduced the parallel-attempt failure exactly: ptxas common-subexpression-fused the high and low fake products. K4 loops collapsed to malformed forms with only 16 `XMAD.PSL` sites and 64 total XMADs rather than the patcher's 32/96 structure. Thus the low-half rewrite must be done after ptxas, not by another plain CUDA formulation.

**Structural evidence against the published static target as a literal sm_61 warp schedule.**
1. **Cyclic ring boundary.** With the present zero-bloat linear ring layout, lane `w0` values cover every ring-word index for K1.5/K2 (and nearly all indices for the other Ks). Therefore every cyclic successor edge `(w,w+1 mod NW)` is needed by some lane. If two direct loads use one per-lane base and uniform immediates, their signed offsets must be identical for every lane. A non-wrapping lane requires spacing `+1` word; the wrapping lane requires spacing `-(NW-1)` words. No lane-dependent base register can change that difference. Reordering rows or pipelining across rows changes the constant but leaves the same two different differences. This is the concrete memory-layout hole in L4' as implemented on linear memory; candidate_3b.c models `ring[(w0+n)%nw]` mathematically but does not provide a zero-cost physical realization.
2. **Lane-local free windows are not warp-uniform.** sm_61 XMAD `.H0/.H1` selectors and SHF operations are warp-uniform instructions. For K1.5 the 16 free lane-windows are split 8 lanes at j=2 and 8 different lanes at j=7; K2.5 similarly splits j=4/j=7, K3 splits j=7 by parity, and K3.5 splits j=6/j=7. K2 j=7 is high-half free on even lanes and low-half free on odd lanes. These cannot all be free in one warp-uniform operand selector without a lane/window remapping that candidate_3b does not construct. The often-quoted 30-SHF extraction count is therefore not directly attainable as a warp-level static schedule merely by patching selectors.
3. **K4 low-half register lifetimes are the real blocker.** A conservative cuasm audit of the part-1 K4 loops found 14 movable low-half `PRMT` sites per expert loop, but only 3 per loop are trivially safe. In the other sites the extracted source register is overwritten before the low-half PSL (often by the paired high-half PSL) or the first-product temporary is still live. A complete rewrite requires local SASS register reallocation and downstream use renaming, not just deleting each `PRMT`. This is consistent with the parallel attempt's note that plain H0 forms do not survive ptxas.

**Consequence.** The requested 6.442125 instr/w target should not be treated as established attainable for the production warp-uniform sm_61 kernel without an additional construction that explicitly solves (a) cyclic physical memory adjacency, (b) warp-uniform lane/window selector alignment, and (c) ≤40-register dataflow. The part-1 hybrid remains the best validated implementation at 6.774625 instr/w. Acceptance measurements on this branch are measurements of that unchanged part-1 artifact plus this audit, not of a new direct-window schedule.


## t3/challenge3b-decode-schedule-part2 — qind control reduction — BIT-EXACT, PARTIAL; A/B gate missed

**Implemented.** The fixed-width expert CH2 paths other than K4 now use a compile-time induction ring pointer (`qind`) instead of recomputing `p + a*stride` in every tile. K4 was deliberately left in its part-1 source form because its qind register allocation diverged between the tn=72 and tn=144 copies without a uniform win. Dense runtime-width CH2 source and all extraction/data-layout arithmetic are unchanged. No donor `SHFL` was removed and no direct cyclic `LDG` schedule was added. `INKWELL_NO_CH3B=1` remains the killswitch.

**Rejected experiments.**
- A conservative post-ptxas K4 low-half operand rewrite statically removed 3 `PRMT`s per expert loop, but the final nopless cubin was not bit-exact: K4 gu **`badbits=4560/9216`** and dn **`badbits=8336/18432`**. The patched-with-NOPs variant remained exact, isolating the failure to the operand/lifetime rewrite. It was removed.
- An aggressive post-SASS qind induction-MOV rotation caused an illegal memory access. The entry copy is part of pointer register-pair packing/initialization, and the base register can be reused after the closing induction update; eliminating it requires loop-carried register reallocation rather than local deletion. It was also removed.
- The earlier negative audit remains binding for the rest of the requested schedule: cyclic ring adjacency, lane-dependent free-half selectors, and ≤40-register lifetimes are unresolved.

**Static counts.** Per 32 weights after NOP deletion; both expert copies have the same count.

| K | part 1 | this branch | target |
|---|---:|---:|---:|
| K1.5 | 215 | **214** | 207 |
| K2 | 215 | **214** | 205 |
| K2.5 | 219 | **218** | 207 |
| K3 | 219 | **218** | 207 |
| K3.5 | 223 | **222** | 211 |
| K4 | 211 | **211** | 189 |

Weighted static count: **6.744156 instr/w**, versus part-1 **6.774625** and target **6.442125**. Extraction remains part-1-shaped: SHF+PRMT `32+0` K1.5, `24+8` K2, `36+0` K2.5/K3/K3.5, `8+20` K4; LDG 4/32w and donor SHFLs remain.

**Build/pipeline.** Patch fingerprint `fake=1078 lp=1084 words=5382 hi_d12d=1090`. A function-by-function SASS diff audit showed replacements only in `gemv_moe_s` and `gemv_sf_ch2`; all other kernels were unchanged. Generated cubin sha256 **`8768643f01d8bc58387177c6404c7df2af9aa8949982944b08e6c557800ec282`**, **1,476,816 bytes**. Release binary sha256 **`5ba838158bcd8c6d7ffd6013ff57442d25167cb54d0b1bb1eb4e7a051f1f5dce`**, release rc **0**, warnings **62/64**, expert `REG:40 STACK:0 LOCAL:0`.

**Per-K GPU gate.** `reasm_drv_perk`: every K1.5–K4 row, gu and dn, nops and nopless, had **`badbits=0`**. Min-of-31 timings (`unpatched` is the harness pre-patch compiler baseline; `part1` is the prior nopless artifact):

| K | shape | unpatched ms | part1 ms | qind ms |
|---|---|---:|---:|---:|
| K1.5 | gu | 0.0798 | 0.0676 | **0.0666** |
| K1.5 | dn | 0.0826 | 0.0702 | **0.0686** |
| K2 | gu | 0.0799 | 0.0666 | **0.0666** |
| K2 | dn | 0.0816 | 0.0696 | **0.0696** |
| K2.5 | gu | 0.0891 | 0.0694 | **0.0696** |
| K2.5 | dn | 0.0922 | 0.0717 | **0.0734** |
| K3 | gu | 0.0911 | 0.0696 | **0.0707** |
| K3 | dn | 0.0932 | 0.0736 | **0.0737** |
| K3.5 | gu | 0.0922 | 0.0734 | **0.0727** |
| K3.5 | dn | 0.0948 | 0.0765 | **0.0763** |
| K4 | gu | 0.0887 | 0.0745 | **0.0754** |
| K4 | dn | 0.0908 | 0.0776 | **0.0778** |

**End-to-end validation.** ctx512: md5 **`5cbe8e06` on all 3 runs**, decode **30.090 / 28.758 / 30.614 ms/tok**, peak **5,032 MiB** each, `ch2nopless: enabled`; validate rc **0**.

**16-pair A/B.** `runs ctl=16 cand=16 ctl_med=28.553 cand_med=28.396 delta=-0.157 wins=6/16 md5=['5cbe8e06'] peak=5032`. The median is 0.157 ms/tok faster, but the win count is only **6/16**, so the requested **wins ≥12/16** gate is not met. Exactness and memory gates pass.

**Interpretation.** This is a small validated control reduction over part 1, not the requested decode schedule. The large remaining gap is unchanged: no direct ring `LDG`, no donor elimination, no warp-uniform free-half extraction construction, and no valid K4 low-half PRMT removal. The branch is useful as bit-exact evidence and as a small static-count improvement, but it is not an accepted Challenge 3b implementation under the stated gates.

## t3/challenge3b-decode-schedule-part2 (37d454f) — orchestrator verdict — NOT MERGED
- Rejected: 6/16 wins vs CTL, far from 12/16, and the static gain is only 6.775 → 6.744 instr/w. Part 1 (13/16, −0.267) on master stays the best validated version.
- Keep the audit. It shows candidate_3b.c is not SIMT-faithful.
  - The ring wrap breaks the "uniform immediate offsets" assumption.
  - Free half-word windows are lane-dependent, while XMAD .H0/.H1 selectors are warp-uniform.
  - K4 low-half reuse needs register reallocation.
- So the 6.442 instr/w target is unproven for a real warp. Any follow-up needs a SIMT-faithful re-statement of Challenge 3b, with warp-uniform instruction streams, per-lane data only in registers, and physical ring adjacency.

## t3/dyn-nkv-swa-ring-port — dynamic n_kv on ring-correct master

- Port of `t3/dyn-nkv` onto ring-correct master `ed6f659`/`098b4e8`. Ring semantics preserved: SWA valid count `min(pos+1,1024)`, slot `pos%1024`, ring relative position `(pos-slot) mod 1024`; full layers use `pos+1`. `INKWELL_SWA_ORACLE=1` keeps the explicit rel_pos/mask arrays and full-length arenas.
- Killswitch: `INKWELL_NO_DYN_NKV=1` falls back to padded master shapes/kernels. End-to-end check: default vs killswitch logit dumps for 64 greedy ctx512 tokens were byte-identical (`cmp` rc0).
- Margin work beyond the original port: normal full/ring/legacy relative positions are now computed inside the rel-bias gather from the device pos/n_kv scalars, so the pinned-input marshal no longer writes rel_pos arrays and the pipelined input copy carries only the small scalar block (about 1.5 KiB at ctx512 rather than the fixed-capacity input prefix). Ring modulo uses mask arithmetic for the power-of-two 1024-slot ring. Oracle mode intentionally keeps host arrays because it needs explicit visibility masks.
- Correctness: ctx512 validation 3/3 md5 `5cbe8e06`; ms/tok 31.482 / 32.766 / 29.667; peak 5,032 MiB each. Release build rc0, 59 warning lines (limit 64).
- Ring rel-position probe: 0 bit differences in all full/SWA-ring/legacy cases, including pos=1024/1025/2047/4095 ring-wrap rows; probe rc0.
- Oracle agreement from identical 4K state: 64/64 top-1 agreement over the in-context rows (positions 4031..4094); mean KLD 4.042e-9, median 1.168e-9, p95 1.193e-8, max 1.004e-7, row0 KLD 1.466e-11, row0 max absolute logit difference 1.36e-4. This is the same essentially-exact identical-state standard as the ring-fix entry. Oracle used the full-length state layout produced by `convert_ring_state_to_oracle.py state4032.inkst state4032-oracle.inkst 4096 4031`.
- ctx512 acceptance A/B (`ctl-e03df94`, interleaved ctl,cand,cand,ctl): `runs ctl=16 cand=16 ctl_med=28.335 cand_med=28.038 delta=-0.297 wins=16/16 md5=['5cbe8e06'] peak=5032`. ACCEPTED by the requested gate.
- 4K interleaved decode vs correct master ed6f659 (`INKWELL_MAX_CTX=4096`, prompt length 4032, same master-generated state): ctl ms/tok 42.6553 / 40.0560 / 42.4868 / 57.9384 (median 42.571); cand 38.1354 / 45.9912 / 45.5959 / 37.3387 (median 41.866); median delta -0.705 ms/tok, 2/4 positional wins. All 8 runs rc0 and generated-text md5 `5457bee0`; text stayed coherent Portuguese instructions, not repeated `||||`. Run-to-run variance is high, so treat the 4K speed delta as indicative rather than a gate; the interleaved order is ctl,cand repeated four times.

## t3/dyn-nkv-swa-ring-port (da50a9b) — brain re-validation on master 17c3b3c — ACCEPTED, merged (6aca5eb)
- Scratch merge 6aca5eb (one ledger-only conflict). Build warnings 59.
- A/B 16 interleaved pairs, ctl = master 17c3b3c build, cand = merge build: ctl median 28.399, cand 28.072 ms/tok, **−0.327 ms/tok, 14/16 wins**, md5 5cbe8e06 on all 32 runs, peak 5,032 MiB both arms.
- Killswitch `INKWELL_NO_DYN_NKV=1`. Raw: ~/.cache/inkwell-ab/dynnkv3.out.
## t3/prefill-chunked-encode-phase0-1 — Phase 0 headless encode + Phase 1 prefill kernels

Base `fb9180a`; branch `t3/prefill-chunked-encode-phase0-1`. Implements the serial-encode lm_head skip and the standalone Phase 1 kernels in `crates/inkwell-cuda/src/prefill.cu`; no prefill driver is wired into decode.

**Phase 0.** The remaining step_block hit/miss ordering bug is fixed in `ink_launch_exl3_moe_ntok` (hit expert GEMV now precedes the miss GEMV on the joined stream). Non-final teacher-forced encode rows use `step_prompt_state`: the cached graph ends at the normed hidden state, so lm_head, argmax, and logit readback are omitted while state-update nodes stay unchanged; `clear_step_cache` drops that headless cache before headed generation. Controlled same-session encode A/B, `/tmp/inkwell-phase0-before` vs `/tmp/inkwell-phase0-headless-final`: ctx512 16.554603 s/511 = 32.396 ms/tok before vs 14.947819 s/511 = 29.252 ms/tok after (−3.144 ms/tok, −9.71%), generated text `";"` md5-line hash 21b63086 both; ctx2048 73.761369 s/2047 = 36.034 ms/tok before vs 65.639186 s/2047 = 32.066 ms/tok after (−3.968 ms/tok, −11.01%), generated text `"h"` md5-line hash 55522e02 both. The gain is below the study's ~15% prediction.

**Validation.** Exact final binary canonical ctx512: md5 `5cbe8e06` on 3/3 runs, decode 28.839 / 29.643 / 29.240 ms/tok, peak 5,060 MiB each run (cap 5,080), `ch2nopless: enabled`. Release warning lines 59 (limit 64). INKWELL_VTEST8 runs without the master controller-shape crash but remains numerically non-exact (8 rows, 6 mismatches), consistent with the pre-existing batch FP-order mismatch documented under `t3/swa-ring-fix`. Memory caveat: one earlier exact-final validation attempt aborted in `cudaGraphInstantiate` after a 1,200 MiB cache-table allocation; the quiet-queue rerun passed 3/3 at 5,060 MiB. This margin is thin and Phase 2 should reserve/arena graph memory deliberately.

**Kernel harness.** Re-runnable harness: `cd opt/kbench/prefill && make test` (uses `gpu-run`; binary ignored). Final `prefill_tests all`: OVERALL PASS.

- EXL3 GEMM: 0 bit diffs vs the single-token order for M=1/16/64. M=64 TFLOP/s: MoE gu 0.74, MoE dn 1.05, dense wq 0.89, dense wo 1.17.
- conv_prefill: PASS, 0 bit diffs vs C sequential conv steps.
- flash_prefill_gqa3: PASS, max_abs 1.583e-8/2.282e-8 at pos1100/1500 SWA with ring wrap and 1.770e-8 at pos700 full attention (tolerance 5e-5); 0.91 TFLOP/s. This misses the ≥1.0 target. Best variant was branchless relative-bias; vectorized float K/V lost more occupancy than it saved in conversions, and shared rel_w caching was slower.
- MoE group/sort/grouped GEMM: PASS, 0 bit diffs, 0.68 TFLOP/s on the realistic grouped shape; deterministic fused SiLU/weighted scatter pipeline PASS, 0 bit diffs.

**Phase 2 requirements.** The kernels take caller-owned scratch and already valid device descriptors. The MoE H2D ring contract is documented at the end of `prefill.cu`: a 32 MiB device ring in two 16 MiB buffers, tile t+1 copied on the side stream, buffer reuse only after tile t−2 compute, descriptor rewriting to ring addresses, and events ordered before grouped GEMM. The driver must also preserve prompt-state ordering, use headed logits only for the final prompt row, handle ring wrap/SWA masks exactly, and solve CUDA-graph/activation memory headroom instead of relying on the current marginal 5,060 MiB peak.

## t3/prefill-chunked-encode-phase0-1 (faf17b3) — brain validation — ACCEPTED, merged
- Merged on top of dyn-nkv (scratch f3422c4). Conflicts resolved in runner.rs: the headless prompt-state path now uses dyn-nkv's `step_shape_match`, and `StepCache` carries both `input_copy_len` and `has_head`.
- 3-run canonical validation: md5 5cbe8e06 ×3; decode 28.280 / 27.986 / 28.009 ms/tok; peak **5,060 MiB** (dyn-nkv master 5,032). Warnings 59.
- Decode path unchanged; encode −10% (512: 32.40 → 29.25 ms/tok; 2048: 36.03 → 32.07).
- Open risk: +28 MiB post-cache peak from the headless prompt graph, and one agent-side `cudaGraphInstantiate` abort. The 20 MiB margin to 5,080 is thin. The phase-2 arena must reclaim this.
## t3/prefill-batched-kernel-speedup — batched prefill tile rewrite

Base `f3422c4`; branch `t3/prefill-batched-kernel-speedup`. Design notes: `opt/PREFILL_BATCHED_TILE_DESIGN.md`.

**Implemented.** EXL3 prefill GEMM tiles are derived from the sm_61 shared-memory roofline. Dense M>32 uses a BM=64/BN=64, 128-thread, 4x8 register tile with strided conflict-free scalar shared loads; narrow-N dense shapes use BM=32/BN=64 4x4 tiles to recover output parallelism; ka=3 halfk dense wo keeps the higher-occupancy legacy tile because it measured faster there. The grouped MoE kernel keeps its public launch contract but maps even column CTAs over 128 columns and processes unpadded 32-row pieces, eliminating the half-tile padding at the realistic mean of 32 rows/expert. The reduction order for every output remains k-tile then r ascending; all final numbers below are bit-identical to the scalar references.

**Rejected experiments.** Vectorized float4 weight loads were slower than conflict-free scalar strided loads (grouped 1.54 vs 1.69 TFLOP/s in the comparison run). Narrow-everything dense tiling was much worse (wq 1.04 vs 1.50). Flash head-split CTAs were neutral (0.89 vs 0.92), shared-FP32 K/V conversion was worse (0.63), BQ16 high-occupancy tiling was correct but slower (0.73), key-parallel QK ownership was correct but slower (0.79), and probability-vector/transposed-PV variants were incorrect or no faster, so flash remains the Phase-1 schedule.

**Final harness run** (`/tmp/prefill-final.log`, one run; shared GPU, expect several percent noise): OVERALL PASS, conv 0 bit diffs, MoE grouped/pipeline 0 bit diffs, flash max abs 1.583e-8/2.282e-8 SWA and 1.770e-8 full.

| kernel, realistic M=64/C=512 shape | before TFLOP/s | after TFLOP/s |
|---|---:|---:|
| EXL3 MoE gu 2304x1152 | 0.74 | 0.97 |
| EXL3 MoE dn 1152x2304 | 1.05 | 1.05 |
| dense wq 2304x3072 | 0.89 | 1.37 |
| dense wo 3072x2304 | 1.17 | 1.32 |
| grouped MoE 512x8, ~32 rows/expert | 0.68 | 1.60 |
| flash_prefill_gqa3 | 0.91 | 0.85 in the final noisy sample; kernel unchanged |

Earlier same-binary comparison measurements before final dispatch were grouped 1.69, gu 1.07, wq 1.50, dn 1.14, so the final row should be read as a conservative noisy sample rather than a regression in those paths. No FP32 reordering was accepted; no error-versus-FP64 relaxation is needed.

**Decode validation and build.** Final default release build rc0, 57 warning lines (limit 64). Canonical ctx512 validation of `target/release/inkwell`: generated-text md5 `5cbe8e06` on 3/3 runs; decode 28.202 / 28.188 / 28.253 ms/tok; peak 5,060 MiB each; `ch2nopless: enabled` on all three runs. An earlier normal-ptxas build also passed md5 3/3 but at 33.5–34.8 ms/tok; the final numbers above are the default CuAssembler-enabled binary.

**Interpretation.** Grouped MoE is no longer padding-bound and reaches the intended range. Dense narrow-N shapes are limited by the parallelism/arithmetic-intensity tradeoff: a 4x8 tile gives too few threads at N=1152, while 4x4 restores threads but lowers the shared-memory roof. Flash is now the limiting kernel. Its roofline is not DRAM (~98 FLOP/B); it is Pascal issue/occupancy: 168 registers, ~32 KiB smem, 3 CTAs/SM, redundant half-to-float conversions, and serialization around small K tiles. Tested occupancy/conversion trades either lowered occupancy or added enough register pressure to lose more than they saved.

## t3/prefill-batched-kernel-speedup (e7f0219) — merged (kernels only, not on the decode path)
- Bit-exact retiling: grouped MoE 0.68 → 1.60–1.69 TFLOP/s; dense wq 0.89 → 1.37–1.50; MoE gu 0.74 → 0.97–1.07. Flash is unchanged at ~0.9.
- Agent validation: md5 5cbe8e06 ×3, decode 28.19–28.25 ms/tok, peak 5,060, warnings 57.
- Orchestrator estimate: at ≤8K, flash is ~0.6–0.65 GFLOP/tok (~0.7 ms at 0.9 TFLOP/s). MoE is 6.1 GFLOP/tok (~3.8 ms at 1.6). Flash is not the 140 tok/s limiter below ~16K.

## Challenge 4 (integer Tier-Q numerics, noise-floor gate) — SWE-2.1 max on Devbox — DONE, nothing to adopt
- Q-A = ε 3e-5 (gate A); Q-B = ε 1e-4. Synthetic rows only (no real activation pack on the Devbox). Candidates: cand_qa_resid16 PASS Q-A (Q1 2.01e-5); cand_qa_splitcorr PASS Q-A (2.26e-5); cand_qb_split8 PASS Q-B (4.82e-5).
- m=1 decode: Q-A integer ≈24–26 MAC/SM-cycle vs shipped FP32 25.8, i.e. ≈1.0×. **Integer numerics do not speed up decode.**
- Ragged m=1–4 verify: B=4 costs 2.78× B=1 (Q-A) / 2.71× (Q-B); the distinct-expert floor is 2.495×.
- Prefill m≥32: Q-A is capped by the same 128 MAC/SM-cycle roof as FP32 (112.8 at m=32, 123.8 at m=128). Q-B reaches 143.4 / 162.9 (roof 170.7). Ceiling: Q-A ~434–443 tok/s, Q-B ~535; 64K ~191 / ~209.
- Byte-code bound: the best possible deterministic 1-byte weight code has RMSE 0.56% rms(z), above the ~0.5% the Q-A residual needs. Q-B cannot be repaired into Q-A by a better code.
- Consequence: prefill stays FP32. The 140 tok/s target is ~3× below the FP32 roof, so kernel efficiency is what matters. Solution files stay on the Devbox (work/challenge4/solution/, SOLUTION.md sha256 85fe486f…a825).

## t3/ccopy2-merge (c1e5880 = master f46f37f-line + t3/expert-copy-efficiency d8e8414) — brain re-validation — NOT MERGED (VRAM / gate miss)
- Default (cache32 on), 3 runs: md5 5cbe8e06 ×3, 28.46 / 28.69 / 30.50 ms/tok, **peak 5,096 MiB > 5,080 cap**. Master prefill01 is 5,060; cache32 (reserve 96→64 MiB, 1,232 MiB table) plus the +28 MiB headless prompt graph no longer fit.
- `INKWELL_NO_CACHE32=1`: peak 5,064. A/B 16 pairs vs ctl-master-f46f37f: ctl 28.490 vs cand 28.416 median (−0.075), **8/16**, one cand abort (`cudaGraphInstantiate` out of memory, run 30). The split-D copy + hit pipeline alone carry no measurable win; the −0.328 ms on 17c3b3c came from cache32.
- Action: retry with cache32 after prefill phase 2 reclaims the headless-graph VRAM (phase-2 contract). The OOM in graph instantiate at 5,064 is a margin warning: whatever lands must keep instantiate headroom, not just steady-state peak.

## t3/m3a-sink-relbias-port — production fused sink projection + rel-bias — ACCEPTED

Base `e5dea6c`; branch `t3/m3a-sink-relbias-port`. Ports the M3a `sink_relbias_fused` kernel into decode. The fused custom op is default-on and replaces the BF16 `attn_sink_proj` mmvf, the `[1025,24,16]` rel-bias SGEMM, and the gather/subtract epilogue. `INKWELL_NO_SINKRELBIAS=1` restores the old chain. The padded rel-bias table is stored transposed as BF16 when fused (about 769 KiB less dense weight storage); dynamic full/SWA-ring/legacy positions are computed with the same mode logic as `relbias_gather_dynpos`, and oracle/static modes read the host rel-pos array.

**Kernel-level gate.** Real layer-5 pass-0 inputs at position 64 were dumped from production (`opt/kbench/sinkrelbias_real/`). Against the shipped-path reference (cuBLAS rel SGEMM + mmvf sink + gather), fused V0 had `bit_diffs=645/1536`, `max_abs=9.54e-7`, `max_rel=1.04e-5` (near-zero denominators inflate max_rel; bench-scale max_abs). The production kernel output was bit-identical to harness V0, as expected. Tested alternate K=16 orders V1–V5 were not bit-exact, so the port remains a small FP32 reorder.

**Whole-model quality gate.** Eight distinct real-text prompts, exact 512-token prefixes, teacher-forced 256 continuations each (2,048 scored tokens per arm). KLD is master || candidate, full softcap-corrected softmax rows. The FP32-noise floor is one deterministic `INKWELL_ACTQ_SIM=noise:1e-6` realization over the same rows; its no-noise binary reproduced the master baseline bit-for-bit on prompt 1. Prompt-cluster bootstrap CIs overlap on all three metrics, so the candidate is statistically indistinguishable from the noise floor.

| metric | candidate | candidate 95% CI | noise floor | noise 95% CI |
|---|---:|---:|---:|---:|
| mean KLD | 8.0036e-4 | [5.484e-4, 1.1095e-3] | 1.0631e-3 | [4.944e-4, 1.9385e-3] |
| top-1 disagreement | 1.1230% | [0.6836%, 1.6602%] | 0.7324% | [0.3418%, 1.2207%] |
| forced-target PPL delta | +0.0448% | [-0.1083%, +0.2427%] | -0.0093% | [-0.2171%, +0.1777%] |

**Validation.** Release build rc0; 64 warning diagnostics (limit 64). Canonical validation of the exact candidate binary: md5 `5cbe8e06` on 3/3 runs, decode 27.411 / 27.342 / 27.469 ms/tok, peak 5,048 MiB each, `ch2nopless: enabled`; no cudaGraphInstantiate OOM.

**A/B.** 16 interleaved pairs vs `ctl-master-f46f37f-inkwell`: `runs ctl=16 cand=16 ctl_med=28.079 cand_med=27.445 delta=-0.634 wins=16/16 md5=['5cbe8e06'] peak=5060`. Candidate arm peaked at 5,048 MiB vs control 5,060 MiB. Median is 0.634 ms/tok faster, above the 0.29–0.55 ms/tok prediction; all runs retained the canonical md5.

## t3/m3a-sink-relbias-port (1441a23) — brain validation — ACCEPTED, merged (fast-forward)
- Agent A/B vs ctl-master-f46f37f: 28.079 → 27.445 ms/tok (−0.634), 16/16, md5 5cbe8e06 all runs, peak 5,048 MiB (−12 MiB vs master). Gate A: KLD 8.0e-4 vs noise-floor 1.06e-3, top-1 disagreement 1.12% vs 0.73%, PPL Δ +0.045%; all 95% CIs overlap over 2,048 teacher-forced tokens.
- Brain validate on the same binary (gpu-run -p high), 3 runs: 27.475 / 27.267 / 27.404 ms/tok, md5 5cbe8e06 ×3, peak 5,048 MiB. Killswitch INKWELL_NO_SINKRELBIAS=1. Build warnings 64 (at limit).

## Challenge 9 (Devbox, SWE-2.1 max): parametric INT8 family. Fidelity PASS, no speed win at 3e-5.
- **What it is:** a single candidate (`cand_c9.c`, sha256 b2ba6e21…), parameterised by fidelity target (`C9_TARGET`) and by M. It comes with a selection rule, a T* surface over M ∈ {1..128} × 8 shapes, and a Lean build with no sorry or custom axioms.
- **Real rows, run locally** (harness4 `-f -r`, 5 layers × {attn, moe_gu} and moe_dn):

  | target | Q1 attn+moe_gu | Q1 moe_dn | result |
  |---|---|---|---|
  | robust | 2.3742e-5 | 2.3825e-5 | PASS |
  | Q-A | 2.6739e-5 | 2.6022e-5 | PASS |
  | Q-B | 4.24e-5 | 4.26e-5 | FAIL at 3e-5 (expected) |

- **Speed (cost model, not silicon):** Ch9 found that the integer-ALU and shift work serialises.
  - Under that floor, robust INT8 vs FP32 on moe_gu is 0.995× at m=1, 0.53× at m=8, 0.79× at m=32, 0.85× at m=128. Q-A peaks at 1.03× at m=1 and is 0.80–0.88× at m=32–128.
  - Only Q-B with G64 crosses FP32, at m ≈ 59–66.
  - This withdraws C7's modelled 128.8/135.2 at m=32/128 as an envelope prediction.
- **Verdict:** nothing to port. Challenge 10 is looking for a way past the floor and will deliver sm_61 micro-benchmarks to test the cost model on the real GPU.

## Challenge 10 — NF10-LX INT8 (G128 strip16, base-254 Karatsuba, bank-private 12-bit digit LUT); sm_61 silicon check

- **Fidelity (CPU, real rows):** PASS. Q1 is 2.83e-5 on moe_gu and 2.71e-5 on moe_dn, both under ε 3e-5.
- **Model claim:** ≥1.019× vs FP32 at every m (1.033 at m=1, 1.040 at m=8, 1.059 at m=128). The Lean optimality proof holds only within the NF10 class.
- **GTX 1060 micro-benchmarks** (`gpu-run -p high`; sources and SASS in `~/.cache/c10-mb`):

  | probe | measured | expected | verdict |
  |---|---|---|---|
  | core dp4a | 1.953 warp-instr/clk/SM | — | — |
  | shift SHF | 1.853 warp-instr/clk/SM | — | — |
  | mixed core+shift | 1.877; mixed/core 0.961 | serialised ≈ 1.0; independent pipes ≥ 1.3 | serialisation confirmed, so the Ch9 floor is real |
  | PRMT | 1.870; +shift ratio 1.021 | 1.7–2.1 | PASS, PRMT behaves as an independent pipe |
  | digit LUT, as delivered (`%` index, branchy) | 2.19 lookups/SM-cycle | ≥ 20 (refit below 17) | FAIL |
  | digit LUT, fair variant (add/select index, branchless 2×LDS + SHF.R.W), 256/512/1024 thr | 6.99 / 7.26 / 7.97 lookups/SM-cycle | ≥ 20 | FAIL |
  | lut_unpack | 1.28 M cycles per 2000 iterations, no spills | — | toy probe; no per-weight number |

- **Why the LUT misses:** a 12-bit entry that straddles a 32-bit word needs 2 LDS. One warp-wide LDS per clock then caps throughput at 16 lookups/SM-cycle, already below the refit threshold. With the extraction ALU work the measured rate is about 8.
- **Verdict:** Ch10's speed claim is refuted on silicon. There is nothing to port. INT8 on sm_61 does not beat FP32 at ε 3e-5 with the published constructions. Any next attempt needs a cost model fitted to these measured rates (about 4 warp-instr/clk/SM issue with core and shift serialised; 2 LDS per straddling LUT entry).

## DFlash2 port study: do not port (branch commit d160004, not merged)

- **Setup:** CPU acceptance on the EXL3 target. Non-degenerate corpus, teacher-forced, acceptance measured against the target's own greedy output.
- **Correctness:** feature extraction and the EXL3 unembed are bit-exact against the GPU binary (96/96 argmax).
- **Acceptance:** DFlash2 2.01 vs DSpark 1.86 tokens/block (+8%). It saturates by k=2–3: emit_by_k = 1.59, 1.83, 1.93, 1.97, 1.99, 2.00, 2.01.
- **Cost model:** best DFlash2 config is 30.1 ms/tok (0.91× vs 27.4) against DSpark pruned/int4 at 27.3 ms/tok. With the production/teacher-forced correction of 0.93 the ordering holds (DFlash2 ≈ 28.0). DFlash2 streams ~2× the bytes per round (196 vs 120 MiB) and doubles expert-cache displacement.
- **Verdict:** the 8-token block drafter does not pay on a memory-bound GTX 1060. Any remaining spec-decode leverage is in the verify / expert-union streaming path. Spec decode itself stays cancelled under the release focus.
- **Report:** opt/dflash2/REPORT.md on that branch.

## t3/decode-latency-25ms — HADVEC default-on — ACCEPTED

The fused/vectorized MoE Hadamard chain is now default-on; `INKWELL_NO_HADVEC=1` is the killswitch. Clean 16-pair interleaved A/B against the frozen master control `b38f787ebb05dec43c644d13a7a368c000fb5c4cf58071d4f0630bc277fa3fc7`: `runs ctl=16 cand=16 ctl_med=27.435 cand_med=27.224 delta=-0.211 wins=16/16 md5=['5cbe8e06'] peak=5048`. Candidate binary sha256 `1f268f076bfcf4837e6b247037e48dd3d3e478e83a97a75d72b11f79a0c757e2`. Raw output: `/home/ash/.cache/inkwell-dec25/ab-hadvec-clean/out`; summary: `/home/ash/.cache/inkwell-dec25/ab-hadvec-clean/summary.manual`. PW16 remains rejected under the exactness gate in this tree (`md5 7aa1f58b` rather than `5cbe8e06`).

## t3/decode-latency-25ms — scans and rejected/abandoned directions

- **Dense split-target scan:** frozen control at `INKWELL_SGEMV_DENSE_TGT=16/24/32/48/64` measured 28.184/27.717/27.274/27.895/27.631 ms/tok. Existing default 32 is already best; no change.
- **PW16:** control-binary env scan produced wrong generated-text md5 `7aa1f58b` rather than `5cbe8e06` for both PW16 and HADVEC+PW16. Rejected on exactness before A/B.
- **Dense CH2 dynamic shared memory:** implemented and built, but abandoned before A/B after the occupancy premise was disproved. Master nsys reports `gemv_sf_ch2` REG40/static shared 12,848 B/`sharedMemoryExecuted=78,336`, consistent with six resident CTAs/SM; registers already cap occupancy at six CTAs/SM (`40*256*6=61,440` of 65,536 registers), so reducing shared memory cannot raise the kernel above six CTAs/SM. The built dyn candidate also added 1–2 address instructions to dense loops. No performance result; not committed.
- **Cache-table marginal expansion:** offline exact kslot replay on the 68-token routing trace showed current T1200 miss traffic 34.75 MiB/tok and 17.26 misses/tok. Adding 8 MiB to the best single layer saves only about 0.30–0.34 MiB/tok (and +4 MiB much less), likely below 0.1 ms/tok. Rejected as too small for the VRAM-margin risk.
- **Replacement-policy scan:** same offline replay: global LFU 34.75 MiB/tok; aged LFU 50.05; LRU 40.12; FIFO 51.88; cost-aware 35.32; cost-sqrt 35.27; random 54.26. Existing global LFU is best of these simple policies; no change.

## t3/decode-latency-25ms — ccopy projection split — REJECTED

The cached-MoE bulk miss copy was split into per-projection side-stream launches so the gate+up join could complete before the down copy (`INKWELL_NO_CCOPY_PSPLIT=1` killswitch). The first A/B accidentally used a binary containing the abandoned dense-CH2 dynamic-shared-memory experiment and was discarded. A clean build (binary sha256 `1b21761492ef6972367985a9e3ec430faaf49acd75d80bd1aef7d38e04659de0`, embedded cubin 1,596,432 bytes) validated with md5 `5cbe8e06` in both arms, but the 16-pair A/B was too weak: `runs ctl=16 cand=16 ctl_med=27.346 cand_med=27.298 delta=-0.048 wins=9/16 md5=['5cbe8e06'] peak=5052`. This misses the ≥12/16 positional-win acceptance rule, so the source change was reverted. Raw output: `/home/ash/.cache/inkwell-dec25/ab-cleanpsplit/out`; rejected source snapshot: `/home/ash/.cache/inkwell-dec25/kernels-psplit-rejected.cu`.

## t3/decode-latency-25ms — expert CH2 unroll-8 loop control — REJECTED

Changed the four CH2 expert loops from `#pragma unroll 4` to `#pragma unroll 8` and updated the fail-closed SASS pipeline for 64-weight loop bodies. The generator succeeded structurally: expert loops were reduced from 215/219/223/211 to 423/431/439/415 instructions per 64 weights (a 3.5-instruction saving per 32 weights), with dense CH2 shapes 444/452/427 per 64. However the patched cubin grew from 1,596,304 to 1,670,416 bytes and the whole-model validation was already slower: candidate 27.740 ms/tok vs control 27.210 ms/tok. The interleaved A/B was stopped early after a consistent large regression; complete rows were control 26.980/27.036/27.096 ms/tok vs candidate 27.544/27.613/27.387 ms/tok, all md5 `5cbe8e06`, peak 5,048 MiB. Likely causes are I-cache/branch/scheduling cost from the doubled body and imperfect NOP-control transfer, outweighing the small loop-count saving. Not committed; experiment tree and binary remain under `/home/ash/.cache/inkwell-dec25/t7tree` and `/home/ash/.cache/inkwell-dec25/target-t7u8`.
- Brain validation (gpu-run -p high), 3 runs of the same binary (sha 1f268f07): 27.525 / 27.125 / 27.116 ms/tok, md5 5cbe8e06 in all 3 runs, peak 5,048 MiB, ch2nopless enabled. Merged into master. The new decode control is this binary.

## Prefill phase 2 (t3/chunked-prefill-driver @ e024bb6): brain check, NOT merged

Clean build (sha 3540fe2c), `bench_encode.sh` with chunk 512. Serial is INKWELL_NO_PREFILL=1.

| prompt | serial tok/s | chunked tok/s | speedup | mean KLD | top-1 | peak MiB |
|---|---|---|---|---|---|---|
| 127 | 30.9 | 57.1 | 1.85× | 3.7e-6 | 1.0 | 5,028 |
| 512 | 30.3 | 105.9 | 3.49× | 4.4e-4 | 1.0 | 5,048 |
| 2K | 29.9 | 97.6 | 3.27× | 2.8e-4 | 1.0 | 4,854 |
| 4K | 27.7 | 92.3 | 3.34× | 4.0e-4 | 1.0 | 4,918 |
| 8K | 25.5 | 86.7 | 3.40× | 4.9e-4 | 1.0 | 5,046 |

- **Blocker:** default decode stays bit-exact (md5 5cbe8e06), but it is slower. In an interleaved validate.sh run (2×3 each) against the HADVEC master control, the control measured 27.07–27.55 ms/tok at peak 5,048, and the branch 28.09–28.69 at peak 5,060, slower in 6/6.
  - The branch lacks HADVEC, which accounts for ~0.21 ms.
  - That leaves ~+0.7–0.8 ms of its own regression. Suspect commit: 1f43dbe (arena borrows decode cache memory).
  - The agent must fix this before merge.
- **Tooling bug:** bench_encode.sh never creates `$OUT/$n`; fixed in the validation copy only.
- **Target:** 150 tok/s. The gap is 1.4× at 512 and 1.7× at 8K, and throughput falls with prompt length.
