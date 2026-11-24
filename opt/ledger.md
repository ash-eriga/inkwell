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
