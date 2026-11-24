# REVIEW_SPEC_v3 — "Unified Master Spec & Execution Plan v3.0" against Inkwell's measurements

Reviewed: the v3.0 spec (owner attachment, cited as "spec Lnn" by line), against the safetensors headers in `~/models/inkling`, the code on master `3fa931c`, `opt/ledger.md`, `opt/QUANT_BRIEF.md`, `opt/REVIEW_ARCH.md` §2.5 and the kbench reports. New evidence for this review: a decode-attention context probe (`opt/kbench/attn_ctx/`) and the kslot cache replay (`opt/tools/kslot_replay.py sweep /tmp/ids_trace.bin`, 68 tokens, 3,264 visits). "Estimate" marks a derived number; everything else is measured or read from headers/code.

Unit conversion used for PCIe misses: **0.09–0.16 ms per MiB/token**, from the two A/Bs of the per-layer cache table (−29.4 MiB/tok gave −4.69 ms and −2.62 ms; `opt/ledger.md:1038,1049,1063`). Pure PCIe at 12.5 GB/s would be 0.084.

## Verdict

1. **Format and contracts: right.** EXL3 mul1 experts, F32 activations, BF16 ctrl/routers, the §5.4 INT4 dense gate, the §9 drafter file, the frequency-ranked 32K head and the canary-before-H200 rule all match QUANT_BRIEF. Most quoted runtime numbers match the repo (§7).
2. **Context: the spec never budgets 32K, and today's runtime cannot reach it.** Decode attention fails above **12,096 tokens** (one block per head holds all scores in shared memory; Pascal caps a block at 48 KiB; the probe fails at 12,160). Below the limit it is slow: measured per visit, attention costs ~13 ms/tok at 4K and ~29 ms/tok at 8K, against 0.93 ms in the short-prompt benchmark. At 32K, F16 KV adds **+1,008 MiB**, which is the whole 940 MiB expert cache plus the 68 MiB headroom. Caches shrink to **0 MiB** under the 5,120 MiB cap, and the loader does not enforce that cap. Estimate: 72–95 ms/tok at 32K as-is (after an attention fix), 57–68 ms/tok with runtime trims that cost no quality.
3. **Short-context speed: the v3 quant does not cut PCIe bytes; it raises them.** v3 makes the non-loop experts **+436 MiB** (1.888 → 2.188 bpw) and the dense weights **+58 MiB**. Estimated misses go from 53.6 to ~59 MiB/tok (default placement) or ~97–100 MiB/tok (spec's Profile 1, L16/L17 resident). "Smaller non-loop experts reduce PCIe miss bytes" (spec L177) is backwards, and "~24–27+ tok/s" (L178) is unsupported.
4. **Quality: seven lossy steps lack a hard ship gate.** The spec has targets (KLD ≤ 0.011) but no rule "never ship worse than today's 0.0475 / 94.66%". Under the owner's no-loss rule, each lossy step needs that gate (§5).
5. **Drafter: three departures from QUANT_BRIEF §7.** It trains multi-step k = 2..3, co-trains a D = 1024 drafter that is close to from-scratch, and dumps labels before the shipped quant is final (Option A/B, Window 2). Labels from a quant that does not ship lower p1.
6. **H200: the schedule is optimistic and the canary cannot calibrate it.** Estimate 6–17 H200-hours for Window 1 against the spec's 6.2–9.75 h; the 10 h window holds only if Stage 4 rollouts stay near 8M tokens at ≥ 1K tok/s. Canary 2C on an L4 cannot run the spec's recipe (it needs ~80 GB).

## 1. Findings, ranked by impact

### F1. Decode attention fails above 12,096 tokens and is slow from 1K tokens up (blocks the 32K test)
- `inkwell_attn_decode` runs one block per query head and keeps every score in dynamic shared memory: `scores = sm`, `qs = sm + n_kv`, plus a static `red[32]` (`crates/inkwell-cuda/src/kernels.cu:1478–1481`). The launch asks for `(n_kv + hd) * 4` bytes (`kernels.cu:1559–1560`). No `cudaFuncSetAttribute` exists in `crates/`, and sm_61 cannot opt in above 48 KiB per block anyway.
- Limit: (n_kv + 128) × 4 + 128 ≤ 49,152 → n_kv ≤ 12,128. Global visits pad n_kv to 64 (`crates/inkwell-run/src/runner.rs:1726–1728, 1893`), so the last working pad is **12,096**.
- Both decode (`runner.rs:1768`) and the 8-row block path (`runner.rs:2647`) use this kernel, so prefill and verify hit the same wall.
- Measured (`opt/kbench/attn_ctx/run.sh`, GTX 1060, F16 K/V, 24 q / 8 kv heads, hd 128, median of 15 launches):

  | n_kv | dynamic smem | launch | ms per visit |
  |---|---|---|---|
  | 1,024 | 4,608 B | ok | 0.183 |
  | 4,096 | 16,896 B | ok | 0.716 |
  | 8,192 | 33,280 B | ok | 2.754 |
  | 12,160 | 49,152 B | `invalid argument` | — |
  | 12,224 / 16,384 / 32,768 | 49,408 / 66,048 / 131,584 B | `invalid argument` | — |

  12,160 fails because the static `red[32]` adds 128 B on top of exactly 48 KiB of dynamic memory.
- **The same kernel is slow well below the limit.** Once the context passes 1,024 tokens, the 40 SWA visits cost 40 × 0.183 = 7.3 ms/tok. The 8 global visits add 5.7 ms at 4K and 22 ms at 8K. Total attention is **~13 ms/tok at 4K and ~29 ms/tok at 8K**, against 0.93 ms in the short-prompt benchmark (`opt/ledger.md:551`). The ~41 ms/tok baseline holds only for short contexts. The cost grows faster than linear (2× n_kv gives 3.85× time from 4K to 8K). The kernel reaches only 37–70 GB/s of logical K/V reads: it launches 24 blocks on 10 SMs, and in the V pass each thread walks all n_kv positions serially.
- The kernel also reads each KV head once per query head (24 q / 8 kv). At 32K one KV head is 8 MiB per tensor, far above the 1.5 MB L2, so K/V traffic is 3× the minimum.
- Fix (runtime, no quality change): a split-K, GQA-aware kernel with an online softmax. Its KV-read floor at 32K is 8 visits × 32,768 × 4 KiB = 1 GiB per token for the global visits plus 160 MiB for the SWA visits, **~7–8 ms/tok** at the 160–176 GB/s Inkwell's kernels reach (`opt/ledger.md:924–925`). The same kernel would cut attention at 4K from ~13 to ~2 ms/tok and at 8K from ~29 to ~3 ms/tok (estimate from the same floor).
- The spec's Milestone 2 prerequisites (L180) omit this kernel.

### F2. At 32K the F16 KV eats every expert cache, and the loader ignores the 5,120 MiB cap
- KV arenas are F16 `[hd, cap, hkv]` per visit slot; cap = `max_ctx` for the 8 global visits and 1,024 for the 40 SWA visits (`runner.rs:1176–1186`). At 32K: global 8 × 32,768 × 4 KiB = 1,024 MiB, SWA 160 MiB, conv 3.66 MiB → **1,187.7 MiB**, against 179.7 MiB at the benchmark's `MAX_CTX=512` (load log: `kv=179MiB`).
- The 5,052 MiB benchmark peak (`opt/ledger.md:1050`) + 1,008 MiB = 6,060 MiB. Only removing all 940 MiB of caches brings it to the cap, with zero headroom and before the 32K compute-buffer growth (the F32 rel-bias tensor is `nh × n_kv × 4` = 3 MiB per global visit, `runner.rs:1750–1753`).
- The loader does not stop at 5,120. The expert budget is a fixed 3,121,348,608 B whatever the context (`runner.rs:921–923`). Caches are then allocated while *physical* free VRAM exceeds size + margin (`runner.rs:1231–1233`; benchmarks set the margin to 10 MiB). At 32K they can fill into the 937 MiB display reserve, and the graph instantiate that follows needs ~96–209 MiB free (QUANT_BRIEF §3.3).
- Replay of the real routing trace: zero caches → **336.8 MiB/tok** of misses vs **53.6** today (the table's own figure; `opt/ledger.md:1038` has 53.2). That is +283 MiB/tok = **+25 to +45 ms/tok**.
- Runtime trims recover up to ~282 MiB without touching quality: the 123.6 MiB `Exl3Blk` 8-row verify scratch is always allocated (`runner.rs:1151–1155`), `dev_buf` reserves 160 MiB of headroom of which ~131.6 MiB is unused (`runner.rs:789`), and `gf_cat` copies the gain/forget weights while keeping the originals (~27 MiB, `runner.rs:1055–1072`). With 282 MiB of caches at 32K the replay gives **~168 MiB/tok** (+10 to +18 ms vs today).

### F3. The v3 quant raises PCIe bytes in the short-context layout
- v3 raises the non-loop experts: L0–5 1.83 → 2.0 bpw, L18–19 1.77 → 2.0, L20–23 2.03 → 2.5 (spec L111–121; today's sizes from the headers: 1,335.7 / 429.5 / 988.0 MiB). Total non-loop 2,753.2 → 3,189.4 MiB (+436 MiB, +15.8%).
- Dense weights: Option A attention 362.0 + head 192.4 + scales 3.6 = 558.0 MiB vs today's 499.96 (+58 MiB).
- Per-token cold bytes when nothing is cached: today 180.1 MB, v3 207.0 MB. The "207.0 MB cold experts" in spec L172 (and "at today's bpw" in QUANT_BRIEF §5.1) is v3's bpw, not today's.
- Estimated misses (replay with each layer's curve scaled by the new expert size; deltas are greedy allocation against greedy allocation, where today's greedy at 940 MiB gives 51.5 vs the shipped table's 53.6):

  | layout, short context | caches | misses MiB/tok | vs today |
  |---|---|---|---|
  | today | 940 | 53.6 | — |
  | v3, default placement (10 loop layers fit the fixed budget) | ~982 | ~59 | +0.5 to +0.9 ms |
  | v3 Profile 1, L6–17 resident (spec L177) | ~372–386 | ~97–100 | +3.9 to +7.5 ms |

- So the loop's −212.7 MiB is eaten by the bigger non-loop and dense weights. The v3 quant buys quality (if recovery works), not speed. The +16% non-loop bytes is the inverse trade of QUANT_BRIEF §5.1 (~1.1 ms per 0.1 bpw with ccopy). It is allowed under the owner's rule but must not be counted as a speed gain.

### F4. Lossy steps without a ship gate (owner rejects any quality loss)
See §5. The spec gates only §5.4 INT4 (L165, L344). The loop requant (−0.17 bpw, est. KLD +12–25% before recovery, QUANT_BRIEF §5.3), the head at K 3.5, Recipe A-MTP, K4V2, 4-bit embeddings, 5/6-bit vision and the reward term in Stage 4 have no rule tied to today's 0.0475 / 94.66%.

### F5. Drafter plan departs from QUANT_BRIEF §7
- **k.** Brief §7.2(2): draft 1 token, keep 1 mask row, drop the GRU. Spec L301 trains k = 1 *plus* multi-step k = 2..3 and decides k afterwards. The brief's model (§7.1) makes k = 1 best up to p1 ≈ 0.82; today's DSpark measures p1 0.50–0.75. Train k = 1 only; revisit multi-step if held-out p1 ≥ 0.85.
- **D = 1024.** Brief §7.2: "no pretraining"; D = 1024 from scratch is the fallback if surgery misses p1. The spec's "SVD-cut warm start" changes the width, the head layout (16q/4kv, hd 64 in `opt/kbench/drafter/REPORT.md:134`) and the tap projection. Little of DSpark survives, so this is the from-scratch fallback under another name. Co-training it is cheap on compute but is the plan's only pretraining-like step.
- **Labels.** Brief §7.2(5): taps and argmax labels from the exact quant Inkwell ships. The spec dumps them at the end of H200 Window 1 (L250–253), *before* the Option A/B decision (Phase 4, L343–344) and before a possible Window 2 (L258–261). If Option B or Window 2 ships, the labels are from the wrong model. The dump also runs in PyTorch, not with Inkwell's F32 kernels: check argmax agreement against Inkwell on a held-out sample before training.
- **Data volume.** 25–40 GB of 4 FP8 taps (9,216 B/token) + top-64 logits (~384 B) is 2.6–4.2M tokens (1.3–2.1M with BF16 taps). For 40K–100K sequences that is ≤ 105 tokens per sequence. Either the sequences are too short to teach long-context drafting or the size is off by ~10×.
- **128K.** Spec L12/L287 says Drafter B "fits 128K". Brief §7.2: "the 128K plan has no room for a drafter". It fits only with Recipe A-MTP, which is a lossy requant (F4), and the batched verifier it needs has no VRAM line (the plan deletes the 123.6 MiB verify scratch, L180).
- Right: V′ = 32K frequency-ranked, 256 bytes + EOS pinned, int8 per-row body, int4 g128 head only, §9 tensor names, p1 before/after on held-out text.

### F6. H200 schedule: optimistic, and the canary cannot calibrate it
- Canary 2C "full 26.4B via Unsloth in 21–29 GiB" on Colab L4/A100 (L321, L340). An L4 has 24 GB (22.4 GiB), so 29 GiB does not fit. The spec's own recipe needs an FP8 teacher (26.4 GB) plus a decoded student (~53 GB BF16) → ~80–88 GB (L243–244). An A100 has no FP8. So Canary 2C can time neither Stage 3A nor Stage 4 at production memory; at best an A100-80/H100 can.
- Stage 4 rollouts (L246–249): Unsloth has no vLLM path for Inkling (L222). At 1K tok/s, 250 steps × 64 prompts × 512 tokens = 8.2M tokens is 2.3 h of generation alone, before scoring and backprop. The spec's 2.75–3.75 h holds only for small rollouts.
- Estimates are in §4. Nothing in the H200 plan is pretraining: 15–30M tokens of KD and ~250 OPD steps are small post-training runs.

### F7. Option B "13.7 MiB smaller than all-EXL3" compares against v3's own Option A
Option B dense = 317.6 + 226.7 = 544.3 MiB. That is 13.7 MiB below v3 Option A (558.0), but **+44 MiB above today's** 499.96 MiB (QUANT_BRIEF §5.4: +43). The gate base also moves: INT4 g128 must now beat 4–5 bpw EXL3 attention, not today's 3.63 bpw, so it is less likely to pass. Gate it against both.

### F8. "−1.07 ms with the new mul1 decode core" is not a gain on master
That is the nsys kernel delta (`opt/ledger.md:937`). The A/B won 11/16 pairs twice and was not accepted; it stays opt-in (`opt/ledger.md:934–941, 1130–1133`).

### F9. The spec omits the largest no-quality-cost wins
- **H1 (VRAM descriptors for resident MoE layers):** −3.5 to −4.2 ms/tok, bit-exact, +120 KiB. The descriptor tables of resident layers live in mapped host memory (`runner.rs:1134–1144`; `opt/kbench/mmvf_had/REPORT.md:7, 163, 177`). Branch `t3/devdesc-vram-moe-descriptors` exists; not on master.
- **Runtime trims** of F2 (up to ~282 MiB). The spec lists only the 123.6 MiB scratch, and only for Milestone 2.

### F10. 128K: the arithmetic holds, the prerequisites are incomplete
REVIEW_ARCH §2.5 already verified the line items (5–25 MiB under the cap if the runtime is ≤ 200 MiB pooled). v3 adds no new evidence for the 180–200 MiB runtime line (L187). Missing prerequisites: the F1 attention kernel, a rel-bias that is fused or budgeted (12 MiB F32 per live global visit at 128K vs the spec's 3.00 MiB line, which is the bias *table*), and a verify-scratch line if speculation runs at 128K.

### F11. Byte reclamation buys nothing on this rig
4-bit embeddings (BF16 883.5 MiB today, host mmap) and 5/6-bit vision (EXL3 231.6 MiB today) save 0 VRAM and 0 decode ms. They only fit v3 under its ≤ 8 GB file target. The spec itself says 11.5–12 GB of host RAM stays free (L214). Under the no-loss rule, keep today's precision for both.

## 2. VRAM at 32K (owner's next test)

MiB. Measured columns: the 5,052 MiB peak at `MAX_CTX=512` (`opt/ledger.md:1050`) and the load log `/tmp/w4ab_5_ctl.log`. "Rest" = 5,052 minus the named lines (CUDA context 65, MoE staging ~15, ggml compute buffers, CUDA graphs). 32K columns are derived; the 32K compute-buffer growth (≥ 3 MiB per live global visit, ×8 rows in the block path) is not included.

| line | ctx 512 today (measured) | 32K, today's code | 32K + runtime trims | 32K, v3 quant (Option A, default placement) + trims |
|---|---|---|---|---|
| resident experts | 2,801.3 (L6–15) | 2,801.3 | 2,801.3 | ~2,701 (10 loop layers) |
| dense EXL3: attention + unembed | 500.0 | 500.0 | 500.0 | 558.0 |
| `dev_buf` (ctrl, routers, norms, convs, headroom) | 237.0 | 237.0 | 78.4 (−131.6 slack, −27 `gf_cat` copy) | 78.4 |
| GEMV scratch (`Exl3W` 15.7 + `Exl3Blk` 123.6) | 139.3 | 139.3 | 15.7 | 15.7 |
| KV + conv state, F16 | 179.7 | **1,187.7** | 1,187.7 | 1,187.7 |
| rest | 254.7 | 254.7 | 254.7 | 254.7 |
| **sum before caches** | 4,112.0 | **5,120.0** | 4,837.8 | ~4,795 |
| **expert caches under the 5,120 cap** | 940 (68 spare) | **0** (0 spare) | **~282** | **~325** |
| misses, MiB/tok (replay) | 53.6 | 336.8 | ~168 | ~169 (est.) |
| miss cost vs today | — | +25 to +45 ms | +10 to +18 ms | +10 to +18 ms |
| attention at 32K (needs F1 kernel) | 0.93 (all 48 visits) | +6 to +9 ms (est.) | +6 to +9 ms | +6 to +9 ms |
| decode, est. | ~41 ms (~24 tok/s) | ~72–95 ms (10.5–14 tok/s) | ~57–68 ms (15–17.5 tok/s) | ~57–68 ms |

- Today's code needs an explicit cache table (`INKWELL_CACHE_MB_L`, or `INKWELL_EXL3_CACHE_K=0`) to stay under the cap at 32K. Measure the process peak with nvidia-smi.
- Copy-engine DMA (12.4–12.6 vs 8.6 GB/s ccopy) would cut the miss time by ~31%: −9 to −17 ms in the "today's code" column.
- Quantized KV would free ~480 MiB (Q8_0) to ~736 MiB (Q4_0) at 32K, but it is a quality trade with no kernel today (`attn_decode` takes F16 or F32 only, `kernels.cu:1562`).

## 3. VRAM at 128K

| line | F16 KV, today's quant, trims, zero experts in VRAM | spec Config 1, Option A (L183–212) |
|---|---|---|
| runtime (context, compute, graphs) | 254.7 + `dev_buf` 78.4 + scratch 15.7 | 180–200 (unproven; today 387–518, REVIEW_ARCH §2.5) |
| dense weights | 500.0 | 663.0 (incl. ctrl/routers and scales) |
| resident loop experts | 0 | 3,280.5 |
| KV + state | 4,259.7 (global 4,096) | 933.41 (K4V2; no kernel, quality unmeasured) |
| staging / GEMV scratch | in "rest" | 20.0 + 15.7 |
| total | ~5,108 (12 spare) | 5,092.6–5,112.6 (7–27 spare) |
| decode, est. | every expert streamed: ~832 MiB/tok, +70 to +125 ms of misses over today, + ~27 ms attention → ~140–190 ms/tok (5–7 tok/s) | 12–13 tok/s today's kernels, 15–20 after kernel work (REVIEW_ARCH §2.5) |

- 128K with F16 KV technically fits only with zero experts in VRAM and is not useful. 128K needs quantized KV, which needs a kernel and a KLD check.
- On top of REVIEW_ARCH's conditions it needs the F1 kernel and a rel-bias budget (F10). Speculation at 128K needs a verify-scratch line and Recipe A-MTP (lossy).

## 4. H200 pipeline and drafter: GPU-hours

| phase | spec | estimate | basis |
|---|---|---|---|
| Canary 2A, 0.5–1.5B SLM | not given | 1–3 A100-h | small model, QAD + OPD smoke |
| Canary 2C, real 26.4B | "L4/A100, 21–29 GiB" | 2–4 h on an A100-80/H100; not possible on an L4 | spec recipe needs ~80–88 GB (F6) |
| §5.4 INT4 ladder + float64 KLD evals | 15–30 min | 2–5 A100-80-h | 3–5 candidates, each a full-vocab KLD run against the source; the BF16 source (~53 GB) does not fit an L4 |
| H200 Stage 0 setup | 0.35–0.5 | 0.35–0.5 | — |
| Stage 1, 3-pass YAQA | 0.25–0.75 | 0.5–1.5 | forward+backward over calibration tokens for H_O; 245 GB of Kronecker factors streamed to host |
| Stage 2, EXL3 encode | 0.10–0.25 | unmeasured | trellis encode time for 26B is in neither the spec nor the repo; time it in a canary |
| Stage 3A, KL-QAD | 2.0–3.25 | 1.0–4.0 | ~47 GFLOP/token (student fwd+recompute+bwd ≈ 3×10.7, FP8 teacher 10.7, full-vocab KL ~4) × 15–30M tokens = 0.7–1.4 EFLOP at 100–200 TFLOPS sustained |
| Stage 4, OPD | 2.75–3.75 | 3–9 | rollouts without vLLM (F6) |
| tap dump | 0.75–1.25 | 0.5–1.25 | I/O-bound |
| **Window 1** | **6.2–9.75** | **6–17** | sum; Stage 4 dominates the spread |
| Window 2, BCJR-QAT | 3.5–4.0 | ≥ 3.5–4.0 | if triggered; it also invalidates the tap dump (F5) |
| L4 drafters | ~3–4 h | 1–4 h | ≤ 0.75 GFLOP/token × ≤ 4.2M tokens is minutes of compute; I/O-bound |

Recommendations: make Stage 4 its own window, and gate shipping on KLD, not on finishing OPD. Dump taps and labels after the dense format and the trellis are final.

## 5. Quality gates

Owner's rule: no quality loss. Base for every gate: today's SAGE-REXL3 2.26 bpw, mean KLD 0.0475 and top-1 94.66% on the same float64 full-vocab harness (QUANT_BRIEF §1.5, §10).

| step | spec line | effect | gate in spec | needed gate |
|---|---|---|---|---|
| loop 2.42 → 2.25 bpw | L106 | lossy; est. KLD +12–25% before recovery | target only (L348) | ship only if final KLD ≤ 0.0475 and top-1 ≥ 94.66% |
| lm_head K 4 → 3.5 bpw (221.3 → 192.4 MiB) | L131 | lossy; K 3.5 is also off the faster core (brief §5.3) | none | per-tensor KLD delta ≤ 0, else keep K 4 |
| Recipe A-MTP, one core layer 2.5 → 2.0 | L164 | lossy | none | KLD gate; only to fit a drafter at 128K |
| K4V2 KV | L195 | unmeasured, V at ~2 bits | "fake-quant in Stage 1" | KLD at 32K and 128K against F16 KV |
| embeddings BF16 → 4-bit | L136 | lossy, saves 0 VRAM, 0 ms | none | drop (F11) |
| vision ~6.1 → 5/6 bpw | L141, L351 | allows −1.0% on vision benchmarks | ±1% | keep today's precision |
| Stage 4 reverse-KL + verifier reward | L246–247 | reward can move the model away from the source | none | measure forward KLD/top-1 after Stage 4; keep the Stage 3A checkpoint if worse |
| §5.4 INT4 dense | L165 | gated | ✓ vs Option A | also vs today's EXL3 dense |
| speculative decode | — | exact only if verify argmax equals serial | none | md5 parity with serial greedy (the ledger's gate), including B = 2 verify kernels |

## 6. Right in the spec (or required by it) but not done yet, ranked by gain

| # | item | gain | VRAM | quality | status |
|---|---|---|---|---|---|
| 1 | attention split-K GQA kernel (not in spec) | at contexts ≥ 1K: −6 ms/tok (SWA alone), −11 ms at 4K, −26 ms at 8K (est. from measured per-visit times); enables > 12,096 tokens (~7–8 ms/tok floor at 32K) | ~0 | exact up to float order | not started |
| 2 | H1 VRAM descriptors (not in spec) | −3.5 to −4.2 ms/tok | +120 KiB | bit-exact | measured in kbench; branch exists |
| 3 | runtime trims: lazy `Exl3Blk`, `dev_buf` slack, `gf_cat` copy | +282 MiB of caches: −1.8 to −3.2 ms at short context (53.6 → ~34 MiB/tok); −15 to −27 ms at 32K (337 → ~168 MiB/tok) | −282 | none | spec lists only the 123.6 MiB, for Milestone 2 |
| 4 | §5.4 INT4 dense | −3.26 ms (kernel); ~−2.6 ms net after +43 MiB of cache loss | +43 | gated | kernel measured; needs the quant |
| 5 | copy-engine DMA staging | −31% of miss time: ~−1.5 to −2.7 ms short context, −9 to −17 ms at 32K with zero caches | small | none | bandwidth measured only |
| 6 | drafter k = 1 + batched verifier | +11–17% (~−4 to −6 ms) if p1 ≥ 0.75 | 81.6 (~+1.2 to +2.0 ms of cache loss) + verify scratch | exact if md5-parity holds | verifier not built |
| 7 | pw16 decode core | −1.07 ms kernel | 0 | exact | A/B 11/16 twice, not accepted |
| 8 | loop requant for L16/L17 residency | L16/L17 misses are 11.1 MiB/tok today (≤ −1.0 to −1.8 ms) | +510 resident | lossy, gated | under v3 the net is negative (F3) |

## 7. Spec claims checked

Right (match the repo): ~41 ms/tok short context; 940 MiB caches, ~81% hit, ~53 MiB/tok misses; 9.47 → 6.75 instructions/weight; CPTC ≥ 7.5; INT8 activations reverted; 10.24 → 6.98 ms dense chain (−3.26); 6,057 / 87 / 937 MiB and 494–512 MiB desktop; 65 MiB bare context; graph-instantiate OOM at ~96 MiB and pass at ~209 MiB free; DMA 12.4–12.6 vs ccopy 8.6 GB/s; ~1.1 ms per 0.1 bpw of cold params; drafter 81.6 / 29.9 MiB, 0.55 / 0.24 ms; union 12.9 / 16.6 experts; p1 0.50–0.75, q 0.18–0.46; 32K head coverage 94.8%; " the" = 149876; §9 tensor names; 7,962,624 params/expert; loop 3,280.5 / 3,310.9 MiB; Recipe A-MTP −60.75 MiB; 128K line arithmetic (REVIEW_ARCH §2.5).

Wrong or misleading:
- "smaller non-loop experts reduce PCIe miss bytes" (L177): they grow 15.8% (F3).
- "~24–27+ tok/s" Milestone 1 (L178, L351): the measured speed is ~24 tok/s; the +3 tok/s depends on INT4 passing its gate and on the pw16 core, which is not accepted (F8); the v3 experts add bytes (F3).
- "Option B 544.2 MiB, 13.7 MiB smaller than all-EXL3" (L10): +44 MiB vs today's dense weights (F7).
- "−1.07 ms/tok with the new mul1 decode core" (L178): not accepted (F8).
- "207.0 MB cold experts" as a measured baseline (L172): v3's bpw; today's is 180.1 MB (F3).
- "2.907 GiB resident L6–15" (L173): that is the budget; the resident bytes are 2,801.3 MiB (2.736 GiB).
- Drafter B "fits 128K" (L12, L287): only with a lossy requant and an unbudgeted verifier (F5).
- "full 26.4B via Unsloth in 21–29 GiB" on an L4 (L321): does not fit 24 GB, and is not the spec's recipe (F6).
- "high-yield quality upgrades: byte reclamation" (L11): saves 0 VRAM and 0 ms; it is a quality cut (F11).

Unsupported (no evidence in the spec or repo): the Unsloth speedups and memory figures (L219, 1.81× / 3.10×) and "plain HF would exceed 10 h"; the B200 depth figures (L12); the EAR fit 1 − EAR ≈ 0.29√KL (L92); KLD ≤ 0.005–0.011 after recovery (L90); the 180–200 MiB runtime line (L187); every H200 stage time (L229–261); "15–30 min" for the INT4 ladder (L165); K4V2 quality.

Missing: the 32K budget (§2); the 12,096-token attention limit and the attention cost at 4K–8K (F1); H1 (F9); runtime trims (F2); a no-regression ship gate (§5).

## Sources

- Spec: `/home/ash/.t3/userdata/attachments/52f673f3-91c5-4506-aa67-433bf37f924d-a32530b9-ced6-44fc-9913-3fd97352e5ce-txt.txt`
- Code: `crates/inkwell-cuda/src/kernels.cu:1467–1573`; `crates/inkwell-run/src/runner.rs:789, 885–941, 1055–1072, 1113–1155, 1157–1195, 1203–1234, 1726–1768, 1886–1894, 2647`; `crates/inkwell-run/src/ops.rs:155–163`
- Probe: `opt/kbench/attn_ctx/probe.cu`, `opt/kbench/attn_ctx/run.sh`
- Replay: `python3 opt/tools/kslot_replay.py sweep /tmp/ids_trace.bin 0,8,16,...,240`
- `opt/QUANT_BRIEF.md` §3, §5, §7–§9; `opt/REVIEW_ARCH.md` §2.5; `opt/ledger.md:924–925, 934–941, 1038–1064, 1130–1133, 1253`; `opt/kbench/mmvf_had/REPORT.md`; `opt/kbench/drafter/REPORT.md`
