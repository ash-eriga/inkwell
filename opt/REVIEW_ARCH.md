# REVIEW_ARCH — decode-speed architecture review

Date 2026-11-24. Base commit be7ec88 (56.43 ms/tok median, BEST env, graphs on).
Inputs: the master spec v2.2 / runbook v1.4 (`/tmp/inkling2-mini-spec.txt`), `opt/PLAN.md`, `opt/THEORY.md`,
`opt/ledger.md`, `opt/kbench/exl3_math/REPORT.md`, `crates/inkwell-run/src/runner.rs`,
`crates/inkwell-cuda/src/kernels.cu`, and the safetensors headers of the SAGE-REXL3 2.26 bpw quant.
§2.5 also uses the placement-sweep load logs (`/tmp/place_runs/*.log`), nvidia-smi, and two GPU probes run under the shared GPU lock: bare CUDA 12.6 context size, and pinned HtoD copy bandwidth.

## Verdict

1. Keep the core: EXL3/SAGE trellis weights, the Rust host, custom stream-decode GEMVs, CUDA graphs, device-side routing with SM-driven staging, and per-layer expert caches.
2. The spec's speed model counts only PCIe. Its 128K plan keeps no expert cache, so it stages 207 MB/tok (today's caches stage 83 MiB). It budgets nothing for GEMV compute (36 ms/tok of kernel time measured), for launches, glue and the host (~17 ms), or for attention over 128K of KV (5–10 ms est.).
3. Its kernel ceiling for mul1 (~510 Gw/s) is above the proven ALU-only bound (≤492 Gw/s). The measured rate is 211–234 Gw/s.
4. Its "single largest lever" is pinning all loop layers. Pinning by visits/byte measured −1.4%. Forcing all 12 loop layers resident at today's quant measured +8% (62.64 ms).
5. Its per-expert DMA staging measured 12.4–12.6 GB/s here against ccopy's 8.6 GB/s: −3 ms at today's 83 MiB, −7.5 ms at the plan's 207 MB. It needs host-known expert ids, so it costs either the single captured graph (worth 7.7 ms) or a split into 12 graph segments (cost unmeasured).
6. The time goes to three places: launches, glue and the host (~17–20 ms); GEMV issue (9.47 instructions/weight); PCIe staging (≤9.8 ms).
7. Fixing them in that order reaches ~28–34 ms/tok (~30–36 tok/s) on the GTX 1060 at short context with today's caches, with no format change. A requant and a codebook fork are optional extras, each gated on quality.
8. On RTX 30–50 the same weight file stays the best choice. int4, int8 and fp16 transcodes multiply bytes by 2–7× on cards that are bandwidth-bound. Glue fusion is the top lever there too.
9. 128K fits the 6 GB card under the spec's plan, which has no expert caches: 5,095–5,115 MiB against the 5,120 MiB cap (+5 to +25 MiB), and +430 to +468 MiB under the physical ceiling at today's desktop use. It needs a runtime of ≤200 MiB (today's ggml runtime takes 387–518), three requants, and K4V2 kernels. Estimated decode at 128K full context: 56–68 ms/tok after the plan below, 49–61 with DMA staging, 40–53 with a prefetch hiding half the PCIe (§2.5).

## 0. Where the time goes today

Graphs-on nsys profile, 60.9 ms span under the profiler, before the sf epilogue fold (`opt/ledger.md:536-557`):

| bucket | kernel ms/tok | gap ms/tok | launches/tok |
|---|---|---|---|
| MoE GEMV (`gemv_moe_s`) | 19.84 | 0.21 | 120 |
| dense GEMV + Hadamard | 16.00 | 1.50 | 867 |
| MoE non-GEMV (ccopy staging 9.84, had_in/out_moe, resolve, wsum) | 13.58 | 0.77 | 384 |
| mix/conv/clip/router custom ops | 4.05 | 1.22 | 504 |
| generic mul_mat (ctrl/router F32) | 3.76 | 0.77 | 352 |
| concat/cpy/set_rows/get_rows | 1.82 | 1.32 | 511 |
| elementwise | 1.72 | 1.68 | 782 |
| rms_norm | 1.00 | 0.92 | 394 |
| attention | 0.93 | 0.06 | 48 |

- Total idle across all streams: 12.39 ms/tok. That is ~1,600 micro-bubbles (p50 4–5 µs) plus one 3.94 ms host-turnaround hole at the token boundary (argmax, graph build and launch, the 9,216 B embedding-row upload).
- Kernel times sum to 62.7 ms against a 48.5 ms busy span, because the mz streams overlap. Per-bucket kernel time overstates wall time for the dense GEMVs.
- The sf fold later removed ~440 launches and 1.7 ms (58.12 → 56.43, `opt/ledger.md:589-596`).

The physical floor is 1,867 MiB of VRAM traffic per token at 217 GB/s, which is 8.6 ms (`opt/THEORY.md:94`).

## 1. What is right and should stay

**The EXL3 mul1 trellis with SAGE mixed K, for experts and dense layers.**
- Quality per byte decides how many experts fit in 6 GB, and this format has the best published quality per byte.
- The quant card reports top-1 94.66% and mean KLD 0.0475 against the source, measured with float64 logits over the full vocabulary (`opt/rexl3/README.md:26-35`).
- exllamav3's KLD-vs-bpw chart (`doc/qb_kld.png`, its Qwen trace) shows GGUF formats need ~1 bpw more for the same KLD:
  - EXL3 at 3.05 bpw: 0.0177. UD-Q2_K_XL at ~3.12 bpw: 0.0533.
  - EXL3 at 4.05 bpw: 0.0067. UD-IQ4_XS at ~4.05 bpw: 0.0165.
- QTIP Table 1 puts mul1's codebook family (1MAD: multiply, then sum the 4 bytes) at 0.069 MSE on a 2-bit Gaussian. The rate-distortion bound is 0.063. Format work can win at most ~9% MSE.

**Custom stream-decode GEMVs on sm_61.**
- One warp per tile, an smem copy of x, and the dp4a bytesum. After the signed-dp4a bug behind the earlier dead end was fixed, the probe measured 251 Gw/s for the fp32 `tile_row` and 347 Gw/s for a lean dp4a variant (`opt/EXL3_PASCAL_REBUTTAL.md:24-26`).
- Porting the MoE GEMV moved e2e from 86.16 to 72.01 ms/tok (`opt/ledger.md:330`); porting the dense GEMV moved it from 71 to 58 ms/tok (`opt/PLAN.md:45`).
- The kernels are issue-bound, not memory-bound, and the REPORT proves which instructions to remove (§3).

**CUDA graphs.**
- Measured 66.15 → 58.49 ms, −7.7 ms (`opt/THEORY.md:56-58`).
- Every staging or scheduling design has to stay capture-safe.

**Device-side routing and staging.**
- `resolve`, `ccopy` and `moe_ntok` read the device `pnexp`, so the expert choice never goes back to the host (`opt/ledger.md:18`).
- `gemv_moe_s` has a `skipmode` that runs hit slots before miss slots (`kernels.cu:2618-2621`), which lets hit GEMVs overlap the miss copy.
- Interleaving PCIe loads inside the GEMV was measured 1.6× slower (155 vs 97 ms, `opt/ledger.md:215-216`). A separate copy kernel is the right shape.

**Per-layer LFU caches with visits/byte placement.**
- Placement: 58.66 → 57.82 ms (`opt/ledger.md:423`).
- Cache size matters a lot: on the old kernels K=4 ran ~181 ms and K=20 ~115 ms (`opt/ledger.md:192`).

**Host-side embedding gather.**
- The BF16 embedding table is 926 MB (201,024 × 2,304 × 2 B). Gathering one 4.6 KB row on the host keeps it out of VRAM.

**The Rust host with ggml as the graph and capture vehicle.**
- It works and captures. The cost lies in op granularity, not in the language or the scheduler (§2.1).

**Measurement discipline.**
- Interleaved A/B pairs, the md5 of the generated text, a warnings count, and a ledger that records rejections.
- The repro rules caught a −4.2% claim that did not reproduce (`opt/ledger.md:524-529`).

## 2. What is wrong or risky, ranked by speed impact

### 2.1 Launches, glue and the host are unbudgeted and are the largest removable cost (~17–20 ms/tok)

- **Spec:** m=1 throughput is derived from PCIe streaming alone. It assigns 16–17 ms to 207 MB of cold experts and implies 22–29 tok/s (spec line 305). Kernel count, glue and host time do not appear.
- **Measured:**
  - Glue costs 8.30 ms of kernel time plus ~4.7 ms of attributable gaps, ~13 ms in all (`opt/ledger.md:559-561`).
  - The token-boundary hole adds 3.94 ms.
  - There are ~3,960 launches per token, ~2,540 of them glue (mix/conv/router, generic mul_mat, concat/cpy, elementwise, rms_norm).
- **Why it matters:** this is a fixed cost per kernel. GPU speed does not shrink it, so it dominates on RTX cards (§3.4).

### 2.2 The GEMV is issue-bound, and the spec's kernel ceilings are above the hardware bound (~36 ms/tok kernel time at stake)

- **Measured:**
  - `sgemv_run` issues 9.47 instructions per weight and is ~92% issue-bound (`opt/ledger.md:528-529`).
  - kbench reaches 211–234 Gw/s.
  - MoE plus dense GEMV take 35.8 ms/tok of kernel time.
- **Spec table** (spec lines 252-282):

| kernel | claimed rate | spec's described cost |
|---|---|---|
| ported mul1 DP4A | ~510 Gw/s | ext8w + 2 XMAD + 1 dp4a |
| TQIR-INT8 | 380–450 Gw/s | |
| CPTC-INT8 | ~306 Gw/s | |

- **Bound:** the REPORT's ALU-only bound for lossless mul1 is 3.903 issue slots/weight, with native 2-XMAD and the shifter hidden. At 10 SMs × 128 lanes × 1.5 GHz that is 384–492 Gw/s (`REPORT.md:145`). The 510 Gw/s claim would need 3.76 slots/weight, which is below that bound.
- **The spec's own instruction list misses:**
  - the window extraction (BFE/SHF, 1.0–1.35 slots/w);
  - the I2F, which is quarter-rate and costs 3.57 slots/w (`REPORT.md:35`);
  - the FFMA.
- **TQIR and CPTC "4-way dp4a.s32.s32"** need int8 weights. mul1 bytesums span ±510 and decode to ±487 on the real tensors (`opt/ledger.md:168`), which does not fit int8. TQIR therefore means a new quant constrained to int8, with unknown quality.
- **CPTC** decodes through a 64-byte LUT. That is one LDS per weight, costing 3.125 slots on Pascal's quarter-rate LD/ST pipe (`REPORT.md:36`), before its 2 XMAD + 2 SHR + ADD + PRMT. That totals ≥7.5 slots/w, worse than the exact C1c kernel at 6.73. CPTC, XOR-Trellis and NariQ are cited as Sep–Nov 2026 work via `custom_mcp`/`unknown_url` sources. This review could not verify any of their numbers.

### 2.3 The staging design: the spec's plan loses the graph win, and today's slots waste capacity (≤9.8 ms/tok at stake)

**Spec plan.** Double-buffered per-expert `cuMemcpyHtoDAsync` at 12–13 GB/s into 20 MiB of staging (spec line 294). It conflicts with graph capture:
- A copy-engine memcpy needs its source address on the host at launch time, so the host must learn each layer's expert ids. That takes one device→host sync per streamed visit: 12 per token in the spec's plan, up to 48 with caches on every layer. The single captured graph becomes segments.
- Graphs are worth 7.7 ms (§1). Whether 12 segments keep most of that is unmeasured.
- Pinned HtoD by copy engine measured 12.39, 12.50 and 12.56 GB/s for 2, 20 and 256 MiB copies on this machine (probe for this review). ccopy runs at 8.6 GB/s.
- The DMA rate gain at today's 83 MiB: 10.1 → 7.0 ms, −3 ms. At the spec's 207 MB: 24.1 → 16.6 ms, −7.5 ms.

**Spec traffic.** The plan keeps no cache, so 12 non-loop layers × 8 experts stream every token, 764.4M params or 207 MB (spec line 294). Today's caches make the measured traffic lower:
- With k=20 LFU caches the miss rate is 18.6%, which stages ~83 MiB/token (`opt/THEORY.md:80-87`).
- The ccopy kernel measures 9.84 ms/tok.
- How much of that sits on the critical path is not measured. Routing for visit L depends on visit L's attention, so most of the copy waits on the chain router → ccopy → GEMV. Only the `skipmode` hit/miss split hides part of it.

**Today's fixed slots.** Each layer's slot is sized for the largest expert in the layer (max K, `runner.rs:969-971`).
- Over the 14 cached layers, the sum of max K is 42.5 and the sum of mean K is 28.3.
- Slots sized per K class would hold 1.50× as many experts in the same bytes, which is k≈30 instead of k=20.
- The prefetch that exists today hits ~0. It predicts from the previous visit's ids, which belong to the wrong pass (`opt/THEORY.md:83-84`).

### 2.4 "Loop-priority VRAM pinning is the single largest lever" contradicts the ledger

The spec says this at line 306. The ledger says otherwise (`opt/ledger.md:423-450`):
- Placement by visits/byte: −1.4%.
- `DEV_LAYERS=6-17` with all loop layers resident: OOM at margin 600. With cache_margin 150 it ran but kept only 6 caches, at 62.64 ms vs 57.85 ms for the default. That is +8%.

The mechanism:
- Each layer made resident beyond ~2.8 GiB costs 4–6 caches.
- An uncached pinned layer reads zero-copy at 8.6 GB/s, which costs more than a k=20 cache's misses.
- At today's quant the loop experts take 3.441 GiB (2.42 bpw including suh/svh). That is 3,523 MiB against the spec's 3,280.5 MiB at 2.25 bpw.

The lever exists only after a requant (§4 step 7). Even then it is worth 3–5 ms, not the largest item.

### 2.5 128K context fits on paper, with zero expert caches; the runtime line is the unproven one

The spec's 128K plan (Recipe A-Strict, spec lines 294-302) keeps the 12 loop layers resident and streams every expert of the 12 non-loop layers on every token. It has no LFU expert cache. Under that plan, 128K does not displace today's caches; the plan never had them. The arithmetic of every weight and KV line checks out against the headers and the model config, to within 0.6 MiB. The process fits the 5,120 MiB cap with 5–25 MiB to spare if the runtime line (spec line 296) holds at 180–200 MiB. Today's ggml runtime takes 387–518 MiB for that line.

**Budget at 128K, line by line.** MiB unless marked. "Today" means the SAGE-REXL3 2.26 bpw quant and the current loader.

| # | component (spec line) | spec | verified | status |
|---|---|---|---|---|
| 1 | Display reserve (295) | 1,024 (cap 5,120 of 6,144) | nvidia-smi: 6,144 total, 87 driver-reserved, so CUDA sees 6,057 and the cap leaves 937 for the desktop. Desktop measured 494–512 (Hyprland, quickshell, 2 Electron GPU processes) | Wrong: the reserve is 937, not 1,024. It holds 425–443 of slack at today's desktop use |
| 2 | Context + prefill scratch (296) | 180–200 | Bare CUDA 12.6 context: 65 (probe). Today's ggml runtime: 291–309 at init, plus 96–209 that must stay free for compute buffers and graph instantiate (OOM at ~96, runs at ~209; evidence below), so 387–518. 128K prefill scratch: unmeasured | Unverifiable. The spec says "verified via Phase 0", but Phase 0 is pending (spec line 428) and the repo has no such record |
| 3 | Attention, 24 layers (135, 142, 149, 237) | 362.0 | Headers: 644.09M trellis + 1.72M FP16 = 645.81M (spec 645.90M). 201.8 + 97.1 + 80.7 = 379.6 MB = 362.0. Today: 278.7 at 3.63 bpw | Confirmed. Needs a requant to 4–5 bpw (+83 vs today) |
| 4 | lm_head, NariQ 3.5 bpw (156, 237) | 192.4 | 460.93M × 3.5/8 = 201.7 MB = 192.4. Today: 221.3 trellis | Arithmetic confirmed. The NariQ format is unverifiable (§2.2) |
| 5 | Message gates, routers, convs/norms (184, 205, 212) | 73.6 | Headers: 28.31M + 7.08M BF16, 1.60M FP32 → 54.0 + 13.5 + 6.1. Today: stored as F32, 148 | Confirmed. Needs BF16 storage (−74 vs today, §3.5) |
| 6 | GPU scales, LUTs, headers (219) | 38.0 | Today's suh/svh for the same tensors: 31.9 (loop experts 30.38, attention 1.17, head 0.39) | Plausible. The planned formats' metadata is unspecified |
| | **Fixed weights (297)** | **666.0** | 362.0 + 192.4 + 73.6 + 38.0 = 666.0 | Confirmed |
| 7 | Loop core L6–17 at 2.25 bpw (107, 298) | 3,280.5 | 12,230.59M × 2.25/8 = 3,439.9 MB = 3,280.5. Today: 3,523.6 at 2.42 bpw | Confirmed. Needs a requant (−243 vs today) |
| 8 | K4V2 KV + attention state (299) | 932.85 | Global: 8 visits × 131,072 × 864 B = 864.00 (K 512 + V 256 + scales 96 B per head set). SWA: 40 × 1,024 × 864 B = 33.75. Sinks/local: 48 × 64 × 4 KiB = 12.00. rel_bias: 48 × 1,024 × 16 × 4 B = 3.00. Prefill ring: 512 × 8 × 4 KiB = 16.00 (spec 17.00). Conv ring: 48 × 3 × 6,656 × 4 B = 3.66 (spec 3.10) | Total 933.4. Conv ring wrong by +0.56. No K4V2 kernel exists, and its quality is unmeasured |
| 9 | Cold DMA staging (300) | 20.0 | 2 buffers × 4 experts × 2.49 MB (an L20–23 expert at 2.5 bpw) = 19.0 | Confirmed |
| 10 | LFU expert caches | none | 0. The non-loop layers stream 764.4M params, 207.0 MB/token: L0–5 95.6 + L18–19 31.9 + L20–23 79.6 MB | Confirmed. The footprint table's "hot experts cached in VRAM" (lines 117, 124, 131) has no VRAM line, so the plan contradicts itself there |
| 11 | Serial GEMV scratch | missing | 15.7 (`exl3_of` scratch, 145 matrices). Today's loader also allocates 123.6 of 8-row `Exl3Blk` scratch, used only by batched verify | Missing line |
| 12 | Allocation overhead | missing | Per-layer free deltas in the k=20 load log exceed header bytes by 1.8–7.1 per layer, ~100 over 24 layers (likely cudaMalloc granularity) | Missing. One pooled allocation removes it |
| | **Process total (301)** | **5,079.35–5,099.35** | Rows 2–11 with the spec's runtime: 5,094.6–5,114.6 | The spec's own sum is right. Rows 8, 9 and 11 add a net 15.3 |

**OOM evidence on today's path** (ctx 512):
- `DEV_EXPERT_MB=3227 margin=800` died in `cudaGraphInstantiate` with ~96 MiB free after load. The log shows 153 MiB before the last 60 MiB cache (`/tmp/place_runs/sweep_b3227_m800.log:46`). The ledger records that pre-allocation figure as "153 MiB left" (`opt/ledger.md:434`).
- `DEV_LAYERS=6-17 margin=600` failed a 16 MiB allocation with ~44 MiB free (`opt/ledger.md:430`).
- Runs passed with ~209 MiB free (k=24) and ~378 MiB free (k=20 default).
- The 126 MiB OOM (`opt/ledger.md:70-71`) has no log to check.

**Fit.** The physical ceiling is 6,057 − desktop = 5,545–5,563 MiB at today's desktop use.

| runtime assumption | process total | vs 5,120 cap | vs physical ceiling |
|---|---|---|---|
| spec runtime 180–200, pooled allocations | 5,095–5,115 | **+5 to +25** | +430 to +468 |
| spec runtime, per-tensor cudaMalloc (+100) | 5,195–5,215 | −75 to −95 | +330 to +368 |
| today's ggml runtime 387–518, batch scratch dropped, pooled | 5,302–5,433 | −182 to −313 | +112 to +261 |
| today's ggml runtime, batch scratch kept, per-tensor cudaMalloc | 5,525–5,656 | over | −111 to +38 |

128K fits on the GTX 1060 6 GB under the spec's plan, with +5 to +25 MiB under the cap and +430 to +468 MiB under the physical ceiling. It fits only if all of these hold:
1. The whole runtime fits in 200 MiB: context, module load, compute buffers, graph exec and the 128K prefill scratch. The bare context is 65 MiB, so 135 MiB is left for the rest. Today's ggml runtime needs 387–518 MiB.
2. All allocations come from one pool, sized at load. A 5–25 MiB margin leaves no room for lazy allocation; today's path already OOMs with ~96 MiB free.
3. The three requants ship: loop core 2.42 → 2.25 bpw, attention 3.63 → 4–5 bpw, head to a 3.5 bpw format. Ctrl and routers are stored as BF16.
4. K4V2 kernels exist and pass the KLD gate (§4 step 0c). The spec plans cache-aware QAD for this; nothing is measured.
5. The 123.6 MiB batch scratch goes, so speculative verify needs its own budget.
6. MTP stays off, or Recipe A-MTP's −60.75 MiB core applies (spec line 302).
7. The desktop stays under 937 MiB. It uses 494–512 MiB today.

**Decode at 128K full context under the plan.**
- Bytes per token: VRAM weights 1,919 MB (loop experts 645, attention 780, head 202, ctrl and router 85, staged experts 207) plus KV 941 MB (864 MiB global, 33.75 MiB SWA). At 217 GB/s that is a 13.2 ms floor.
- Attention over 128K: 24 heads × 131,072 × 128 × 2 × 8 visits = 6.44 G MAC/token. The FMA floor is 2.9–3.4 ms and the KV-read floor 4.3 ms. Estimate 5–10 ms with K4V2 dequant (spec line 368: 7–10).
- Staging 207.0 MB: 24.1 ms by ccopy (8.6 GB/s), 16.6 ms by copy-engine DMA (12.4–12.6 GB/s measured on this machine). No cache hit is left to overlap with. Per-expert pipelining can hide the staged experts' own GEMVs (~11% of GEMV bytes, 2–3 ms); the rows below do not subtract that.

| state | GEMV | glue, launches, host | attention | staging | ms/tok | tok/s |
|---|---|---|---|---|---|---|
| today's kernels, plan's memory | today's 56.43 ms minus 9.84 ms ccopy, minus 0.93 ms attention (~46–51) | in GEMV column | 5–10 | 24.1 | ~75–85 | 12–13 |
| after §4 steps 1–3 and 6, ccopy | 21–25 | 6–9 | 5–10 | 24.1 | 56–68 | 15–18 |
| same, DMA with 12 graph segments | 21–25 | 6–9 (+ segment cost, unmeasured) | 5–10 | 16.6 | 49–61 | 16–20 |
| plus a pre-gate prefetch hiding half the PCIe | 21–25 | 6–9 | 5–10 | 8.3 | 40–53 | 19–25 |
| all PCIe hidden | 21–25 | 6–9 | 5–10 | 0 | 32–44 | 23–31 |

The spec's 22–29 tok/s (spec line 2, 34–45 ms) needs the last row: every staged byte hidden, plus the full kernel and glue plan. Its own speed model (spec line 305) counts only the 16–17 ms of PCIe.

**Using the slack the reserve does not need.**
- The plan leaves 430–468 MiB unused under the physical ceiling at today's desktop use. After a 100 MiB safety margin, that is 13–15 cache slots per non-loop layer.
- At a 25–45% miss rate, staging falls from 207 MB to ~50–95 MB, about −9 to −12 ms with DMA. The miss rate is an assumption: the measured 18.6% at k=20 is for loop layers 12–16, which repeat 3× per token (`opt/THEORY.md:79-81`). The old-kernel sweep put k=12 within 6 ms of k=20 (`opt/ledger.md:192`).
- This spends the display reserve, so the owner decides (§5 question 3).

**Recipe B.** The 12 loop layers at 2.5 bpw take 3,822.1 MB (3,645.0 MiB), +364.5 MiB over Recipe A. The total reaches 5,459–5,479 MiB: over the cap, but under the physical ceiling by 66–104 MiB if the runtime holds at 200 MiB.

**Savings that are not VRAM.**
- The spec claims that 4-bit embeddings, a 5/6-bit vision tower and a NariQ lm_head together reclaim 1.27–1.39 GB (spec line 92).
- In the spec's own placement, embeddings live in RAM and vision streams only during image prefill. Neither is in VRAM during decode.
- Today's unembed is already a 221.3 MiB trellis, so NariQ's 192.4 MiB head saves 28.9 MiB of VRAM.

**Shorter context.** Each 1K tokens of global KV costs 6.75 MiB at K4V2, about 3 expert slots (2.8–3.6 at 2.0–2.5 bpw). Today's F16 layout costs 32 MiB per 1K: 8 visits × 8 KV heads × 128 × 2 × 2 B. The bench (one prompt, 64 tokens, ctx 512) never exercises this cost.

### 2.6 The speculative-decoding projection does not survive the measured acceptance and expert union

- **Spec:** 31–62 tok/s with Narrow-DSpark at m=5, α = 0.6–0.8 (spec line 368).
- **Measured acceptance:**
  - The stock drafter emits 1.64–2.58 tokens per round across prompts (prose 12%, code 20%, technical 9% acceptance, `opt/ledger.md:121-129`).
  - P(first draft accepted) is 75%; continuation is ~18%.
- **Measured expert union** grows quickly with block rows (`opt/ledger.md:130-138`):

| rows | average union | estimated emit |
|---|---|---|
| 2 | 12.9 | ~1.75 |
| 3 | 16.6 | ~2.10 |
| 4 | 20.0 | ~2.25 |
| 8 | 28.4 | ~2.4 |

- **Implication:** 4 rows read 2.5× the expert bytes and stage 2.5× the misses to emit ~2.25 tokens.
- After the plan below, experts plus staging are ~55% of the token. Verifying 4 rows would then cost ≥1.8× a serial token, even if dense work amortizes perfectly. With a near-free drafter the best case is ~20% faster than serial.
- Today's measured result: spec decode tops out at ~5.2–5.5 tok/s against a faster serial path (`opt/ledger.md:164-165`).

### 2.7 "Symmetric int8 A8 on all linears" is a quality risk the md5 gate cannot evaluate

- The spec freezes symmetric per-block-128 int8 activations for every linear (spec line 441).
- **Measured:**
  - int8 x with the EXL3 affine broke the GU output (D2A reverted, `opt/ledger.md:174`).
  - The runtime comment records "int8-x loses ~10%/window" on GU (`kernels.cu:3085`).
  - The best int8 variant, C6 (per-16 scales, matched sx), reaches 4.5e-3 relative GEMV error. The int16-x variant C4 reaches 2e-5 at ~1 slot/w more (`REPORT.md:66,75`).
- int8 x may well be acceptable for model quality. The md5 identity gate cannot show that, so the project needs a KLD gate before any lossy kernel (§4 step 0).

### 2.8 The benchmark is too narrow to protect cache and placement decisions

- The bench is one prompt, 64 tokens, ctx 512.
- The 81% cache hit rate, the k=24 result (inconclusive, +~1%) and the placement budget were all tuned on it.
- An LFU cache warmed on one prompt can overstate hits on mixed traffic. This review has no measurement of the size of that effect.

## 3. Recommended format and kernel design

### 3.1 One weight file, per-architecture kernels

Keep the EXL3 mul1 trellis for experts and dense layers. Apply lossless layout transforms at load time, per device. Do not transcode to int4, int8 or fp16 on any GPU.

**Issue budget.** Budget per weight = lane-instructions/s ÷ (DRAM bytes/s ÷ bytes/weight), at 2.2 bpw.

| GPU | lanes/SM × SMs × clock | lane-instr/s | DRAM GB/s | Gw/s at 2.2 bpw | instr/w budget | INT32 instr/w budget |
|---|---|---|---|---|---|---|
| GTX 1060 6 GB (sm_61) | 128 × 10 × 1.5–1.7 GHz | 1.92–2.19 T | 204 (217 measured) | 742 | 2.6–2.95 | same (all lanes INT) |
| RTX 3060 12 GB (sm_86) | 128 × 28 × 1.78 | 6.4 T | 360 | 1,309 | 4.9 | 2.4 |
| RTX 3090 (sm_86) | 128 × 82 × 1.70 | 17.8 T | 936 | 3,404 | 5.2 | 2.6 |
| RTX 4090 (sm_89) | 128 × 128 × 2.52 | 41.3 T | 1,008 | 3,665 | 11.3 | 5.6 |
| RTX 5090 (sm_120) | 128 × 170 × 2.41 | 52.4 T | 1,792 | 6,516 | 8.0 | ~8 (unified INT32/FP32) |

- Ampere and Ada SMs run INT32 on 64 of their 128 lanes (GA102/AD102 whitepapers). The trellis's LOP3, SHF, IMAD and IDP4A therefore get half the issue budget there.
- Blackwell runs INT32 on all 128 lanes.

**What this means for the trellis on each GPU class:**
- **GTX 1060.** The trellis cannot reach DRAM bandwidth. The best exact kernel, C1c (6.75 issued/w), peaks at 31–40% of the 722 Gw/s DRAM rate. C4 (5.70) peaks at 36–47%. The ALU-only bound (3.90) is 53–68% (`REPORT.md:135-145`, 1,500–1,920 G issue slots/s).
- **RTX 30.** The trellis sits near the INT32 budget. Estimate: ported without I2F, it needs 2.5–3.5 INT ops/weight (window extract, 16-bit mask, IMAD hash, IDP4A).
  - QTIP Table 17 supports this. On a 3090, Llama-2-7B ran at 127 tok/s at 2 bits and 109 tok/s at 4 bits: halving the bytes gained 17%.
  - At 2 bits that kernel used ~30% of DRAM bandwidth. Something other than DRAM bound it: ALU, launches, or both.
- **RTX 40/50.** The trellis is DRAM-bound, so instruction count stops mattering.

### 3.2 Format comparison per weight

| format | bytes/w | GTX 1060 cost | RTX 3090 cost | text model VRAM | quality vs EXL3 at same bpw (published) |
|---|---|---|---|---|---|
| **EXL3 mul1, 2.15–2.26 bpw (keep)** | 0.27–0.28 | 5.7–6.75 slots/w, issue-bound, 31–47% of DRAM | est. 2.5–3.5 INT/w vs 2.6 budget | ~6.7 GiB | reference: KLD 0.0475, top-1 94.66% at 2.26 bpw |
| int4 g32 (Q4-class), 4.5 bpw | 0.56 | ~2–3 instr/w, DRAM-bound at ≈386 Gw/s (4.5 bpw at 217 GB/s): 1.15–1.45× the C4 trellis rate (263–337 Gw/s) at 2.05× the VRAM | DRAM-bound at 2× the bytes, ~2× slower | ~13–14 GiB (fits no 6/8/12 GB card) | GGUF needs ~+1 bpw for EXL3's KLD; NVFP4 W4A16 at ~4.77 bpw has KLD 0.0100 vs EXL3 at 4.05 bpw 0.0067 (exllamav3 `qb_kld.png`) |
| int8, 8.5 bpw | 1.06 | DRAM-bound, ~4× the bytes | ~4× the bytes | ~25 GiB | near lossless |
| fp16 | 2.0 | FP16 math at 1/64 rate on GP106; 7× the bytes | 7× the bytes | ~48 GiB | lossless |
| LUT/VQ codes (QTIP HYB, AQLM, CPTC) | 0.27–0.28 | ≥1 LDS/w at 3.125 slots + bank conflicts, ≥7.5 slots/w (`REPORT.md:71`, B3 = 16.8 slots/w full loop) | fine: smem LUTs are cheap on sm_86+ | ~6.7 GiB | HYB MSE 0.071 vs 1MAD 0.069 (QTIP T1); AQLM 2-bit Llama-2-7B ppl 6.14 vs QTIP 5.86 and 81.5 vs 188 tok/s (QTIP T4/T5) |

On the 6 GB card, int4 decodes 15–45% more weights per second than C4, but it doubles expert bytes per token to ~1.7 GiB and halves the resident share. Each extra 100 MiB of misses costs ~12 ms at 8.6 GB/s, so the token gets slower. On RTX cards int4 loses outright. Neither result depends on whose int4 kernel is used.

### 3.3 The sm_61 kernel: an exact instruction diet, then gated approximations

All figures are issued slots/weight from REPORT (`REPORT.md:57-66,133-145`).

| stage | change | slots/w | exactness | RTX effect |
|---|---|---|---|---|
| today | `sgemv_run` (BFE, 3 XMAD, IDP4A, I2F, FFMA) | 9.47 issued (8.50 per REPORT recount) | gold ~2e-6 | n/a |
| A1+A2/A3 | load-time ring rotation (0% bloat) and paired 32-bit window extraction | −1.3 to −2.5 | bit-identical weights | fewer INT ops: helps sm_86 |
| B1 | direct `.H1/.H0` mul16 | 0 BFE/LOP | bit-identical weights | n/a (sm_86 has native IMAD) |
| C1c | `dp4a(P,1s,0x4B000000) − 8389118.f` replaces I2F | removes 3.57-slot I2F for a 1-slot FADD; total 6.73 | exact weights; accumulation error 1.8e-6 vs 1.2e-4 today; md5 changes | helps everywhere: conversions are below FP32 rate on sm_61 through sm_89 |
| wave shaping | GU/D grids are 144 blocks over ~60 resident: 2.4 waves, last wave 40% full | tail only | exact | retune per SM count |
| C4 | int16 x (per-16 scale), `IDP4A(−510)` + `XMAD.S16` | 5.69 | 2e-5 rel. error, needs KLD gate | **sm_61 only**: on sm_86/89 it moves the MAC from the full-rate FFMA to the half-rate INT32 path |
| C6 (option) | int8 x, one IDP4A does decode-sum and MAC | ~4.7 (estimate: C4 minus 0.97) | 4.5e-3 rel. error, needs KLD gate | sm_61 only |
| native 2-XMAD | cubin peephole on the hash multiply | −0.97 (C4 → 4.73) | exact | Pascal only |

**Codebook fork (option; owner decides).**
- The idea: a Pascal-friendly hash with a single 16-bit XMAD instead of mul1's 32-bit constant. Estimate: 4.6–5.0 slots/w under stock ptxas, about −3.5 ms against C4.
- Risk: a 16-bit multiplier mixes the upper bytes weakly, which can lose codebook quality.
- Gate it before any requant. A CPU Viterbi encode of i.i.d. Gaussian data at K=2 must stay within 1% of mul1's MSE (0.069, QTIP T1).
- On RTX it is neutral, because IMAD costs one instruction there either way.
- It ends compatibility with exllamav3's quantizer and kernels.

### 3.4 RTX 30–50: same file, different priorities

Expected time per token on cards that hold all experts:
- With every expert resident, a token reads ~1.7 GiB: experts 833 MiB, dense 581, unembed 221, BF16 ctrl/router ~70 (`opt/THEORY.md:67-75`).
- The DRAM floor is 5.1 ms on a 3060, 1.9 ms on a 3090, 1.8 ms on a 4090 and 1.0 ms on a 5090.
- The ~3,500 dependent launches per token cost 3.5–4.4 ms even at a quarter of the 1060's 4–5 µs median bubble. That equals the 3060's whole weight floor and exceeds the others.
- So steps 1–2 below (device token loop, glue fusion) are the top lever on RTX. The exact instruction diet (step 3) matters on the 3060 and 3090 and not on the 4090 or 5090.

**VRAM fit for the text model (~6.7 GiB):**

| VRAM | examples | fits? | runtime need |
|---|---|---|---|
| 6 GB | GTX 1060 | no | ~2.8 GiB resident plus caches (today) |
| 8 GB | RTX 3070, 4060 | marginal once context, KV and display are added | 1–3 layers on caches; headless at short context fits |
| 12 GB | RTX 3060 12 GB, 4070, 5070 | yes, ~4 GiB left for KV (128K at K4V2 = 864 MiB) | no staging; launch-bound |
| 24–32 GB | RTX 3090, 4090, 5090 | yes | launch-bound |

**m>1 (verify and prefill) on sm_80+.** Decode the trellis to fp16 in registers and use tensor-core `mma.sync`, which is what exllamav3 does. Pascal has no tensor cores. Speculative decoding pays more on RTX: with all experts resident, a 4-row verify costs launches and INT32 issue, not PCIe.

### 3.5 The other dense formats

- **Attention:** keep the trellis.
  - Attention GEMVs run ~7–9 ms of wall time, after mz stream overlap.
  - Today's attention trellis averages 3.63 bpw (278.7 MiB for 644.09M params). The spec's 4–5 bpw plan costs +83 MiB.
  - Moving them to int4/int5 g32 saves an estimated 1–2.5 ms on the 1060, for +70–150 MiB of VRAM (25–50 cache slots).
  - It loses on RTX (2× the bytes).
- **Ctrl and router: store them as BF16, not F32.**
  - `load_st` widens them with `bf16_to_f32` (`runner.rs:331`, `runner.rs:696-708`): 148 MiB on device (ctrl 108, router 27, misc 12.6).
  - BF16 frees ~74 MiB, about one more k=20 cache, and halves their read bytes. The BF16→F32 widen is a 16-bit shift.
- **Unembed:** keep the 221.3 MiB trellis. NariQ's 192.4 MiB head would save 28.9 MiB.

## 4. Ordered plan

Gains are ms/tok on the GTX 1060 from the 56.43 ms baseline. Gains overlap: fusion removes bubbles that would otherwise hide staging. Their sum (~30 ms) therefore overstates the end result. The estimated end state is 28–34 ms/tok. THEORY's practical ceiling with no misses is ~30 ms (`opt/THEORY.md:98`).

Every A/B below uses the existing protocol: ≥8 interleaved pairs in both orders, BEST env with graphs, the median, and wins against neighbouring controls counted.

| # | step | expected gain | validation | RTX |
|---|---|---|---|---|
| 0 | **Measure first.** (a) A timing-only run with ccopy skipped (output invalid) bounds steps 4–5 and 7. (b) Bench v2: 5 prompts (prose, code, technical, chat, long) × 256 tokens at ctx 4K and 32K, with per-layer miss rates from `INKWELL_PF_STAT`. (c) A KLD/top-1 gate against stored reference logits over ~2–4K tokens. | 0 | — | same harness |
| 1 | **Device-resident token loop.** Argmax on the device, embedding-row gather from mapped pinned memory inside the graph, a device-side position counter, and relaunch without rebuild. Remove the ~800 KB logits DtoH. | −3 to −3.5 (the 3.94 ms hole) | md5 identical; A/B; nsys boundary gap < 0.3 ms | same ms, larger share |
| 2 | **Glue fusion.** One kernel per visit for norm + residual + clip + scale; fold conv/mix/router into neighbouring custom ops; a BF16 ctrl/router GEMV fused with its consumer in place of 304 `mul_mat_vec_f` launches. Target: −60% of the ~2,540 glue launches. | −5 to −8 (pool 13 ms + bubbles) | md5 identical where FP32 op order is kept, otherwise KLD gate; launch count from nsys; A/B | largest lever |
| 3 | **Exact GEMV diet:** A1 + A2/A3 + B1 + C1c, then wave shaping of the MoE grids. Issued/w 9.47 → ~6.75. | −5 to −7 | kbench gold rel. error ≤ 2e-6 vs fp64; KLD gate ΔKLD within run-to-run noise; record the new md5; A/B | helps sm_86 (no I2F, fewer INT ops) |
| 4 | **Cache slots sized per K class.** 1.50× slots in the same bytes (k≈30 vs 20). | −1.5 to −3 (k 20 → 24 measured −1.7, inside noise) | miss rate from `INKWELL_PF_STAT`; md5 identical; A/B on bench v2 | 6–8 GB cards only |
| 5 | **Next-visit prefetch from a pre-gate:** apply visit L+1's router to visit L's hidden state and copy predicted misses during visit L. Build it only if offline recall of the top-8 is ≥ 70% (ids from `INKWELL_DUMP_IDS`, hidden states from `INKWELL_DUMP`). | −2 to −4 | offline recall first; md5 identical (prefetch changes no math); A/B | 6–8 GB cards only |
| 6 | **C4 int16 x, sm_61 build only.** Issued/w 6.75 → 5.69. | −2 to −3 | kbench rel. error ≤ 3e-5; KLD gate; A/B | do not ship on sm_86/89 |
| 7 | **2.2 bpw loop-fit requant (owner decides).** Loop experts 3.44 → ~3.13 GiB. All 12 loop layers resident; cached visits drop 18 → 12, so staged bytes fall to ~55 MiB. | −3 to −5 | KLD and top-1 against the source; A/B | ships as a 6 GB variant |
| 8 | **Pascal extras (owner decides):** native 2-XMAD cubin peephole (C4 → 4.73 slots/w); codebook fork, only if bundled with step 7's requant and only after it passes the Gaussian MSE gate. | −2 to −4 | kbench; KLD gate; A/B | neutral |
| 9 | **Re-evaluate speculative decoding** with 2–4-row blocks, the ntok kernel on the step-3 diet, and a drafter costing ≤ 5 ms. | 0 to −20% | (verify(m) + draft) / emit < serial ms on bench v2 | likely positive |

**Expected quality cost of step 7.**
- Lowering the loop layers from 2.42 to 2.2 bpw lowers the expert average by ~0.11 bpw.
- exllamav3's KLD curve falls by ln-slope ≈ 1.35 per bpw between 2.05 and 3.05 bpw. Extrapolating from that, KLD rises ~12–25%, from 0.0475 to ~0.053–0.060.
- The loop layers run 3×, so their error may count more than this uniform estimate assumes.
- Measure it. Do not ship on the extrapolation.

## 5. Open questions for the owner

1. **Context target on 6 GB.** The spec's 128K plan fits with zero expert caches and full non-loop streaming (§2.5). Which one?
   - 128K under the 5,120 MiB cap: 207 MB/tok staged, est. 49–61 ms/tok after the plan with DMA staging and 12 graph segments.
   - 128K plus non-loop caches in the 430–468 MiB the desktop leaves unused today: est. a further −9 to −12 ms, paid for with the display reserve (question 3).
   - A shorter context with caches: each 1K tokens of K4V2 global KV (6.75 MiB) buys ~3 expert slots. This review's plan reaches 28–34 ms/tok at short context with today's quant and caches.
2. **Gate for approximate kernels.** Keep md5 identity, or accept a KLD threshold? If a threshold, what number? Steps 3, 6 and 8 need the answer.
3. **Display reserve. DECIDED by the owner (2026-11-24): the 5,120 MiB process cap stays a hard safety margin. Do not spend the display reserve on caches.** Measured: the BEST config peaks at 5,010 MiB per process (nvidia-smi, short context), inside the cap.
4. **2.2 bpw loop-fit variant.** Ship it for 6 GB cards at an estimated +12–25% KLD, and keep the 2.26 bpw SAGE quant for ≥ 8–12 GB cards? What KLD is acceptable?
5. **Codebook fork.** Is ending exllamav3 compatibility acceptable for a Pascal-only win of ~3.5 ms?
6. **Lossy runtime knobs.** Are top-k 8→6 or expert skipping by routing weight in scope?
7. **RTX backend.** Reuse exllamav3's maintained sm_80+ kernels behind the Rust host, or port Inkwell's own?
8. **Bench definition.** Which prompts, how many tokens, what context, and desktop on or off? Every cache and placement decision is tuned to it.

## Sources

- Measurements: `opt/ledger.md` (lines cited inline), `opt/THEORY.md`, `opt/kbench/exl3_math/REPORT.md`, `opt/EXL3_PASCAL_REBUTTAL.md`.
- Sizes: safetensors headers of `Inkling-2-Mini-SAGE-REXL3-2.26bpw`. Experts 6.130 GiB (2.153 bpw average including suh/svh); loop experts 3.441 GiB; dense trellis 0.488 GiB; unembed 221.3 MiB.
- Quant quality: `opt/rexl3/README.md:26-35`.
- QTIP: Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing", arXiv:2406.11235. Tables 1, 3, 4, 5 and 17.
- exllamav3 `doc/qb_kld.png` (github.com/turboderp-org/exllamav3): KLD vs bpw for EXL3, GGUF and NVFP4.
- NVIDIA GA102 (Ampere) and AD102 (Ada) architecture whitepapers: INT32 on 64 of 128 FP32 lanes per SM. Blackwell: unified INT32/FP32 cores.
- Spec: `/tmp/inkling2-mini-spec.txt`, the master spec v2.2 / runbook v1.4 (line numbers cited inline).
- §2.5 measurements (2026-11-24): load logs `/tmp/place_runs/k20b.log`, `k24b.log`, `sweep_b3227_m800.log`, `sweep_loops_m600.log`; `nvidia-smi` memory and per-process use; a `cudaFree(0)` probe (6,057 MiB total, 5,480 MiB free with the desktop at 512 MiB, so the context takes 65 MiB); a pinned `cudaMemcpyAsync` HtoD probe (2/20/256 MiB). Probe sources are in `/tmp/review_arch_scripts/ctx/`.
