# THEORY — ceilings para Inkwell no GTX 1060 6GB

Medido nesta máquina (2026-11-22, desktop ligado):

| métrica | valor |
|---|---|
| D2D copy (read+write) | ~163 GB/s best, ~131-138 típico |
| kernel read-only payload grande (~86 MiB, lm_head iq4_xs) | ~217 GB/s |
| kernel payload médio (11-22 MiB/launch) | ~50-70 GB/s |
| PCIe zero-copy read (pinned host) | ~8.6 GB/s |

## Bytes/token de decode (contados dos headers dos arquivos)

IQ2_XXS GGUF (8.49 GB, 26.4B → ~2.57 bpw médio):

| componente | MiB/token |
|---|---|
| experts (8/128 × 48 visits) | 878 |
| dense attn (Q8_0) | 1249 |
| loop ctrl (F32) | 325 |
| lm_head (IQ4_XS) | 235 |
| embd row | ~0 |
| **total** | **~2686** |

SAGE-REXL3-2.26bpw (7.80 GiB + mtp.safetensors 226 MiB):

| componente | MiB/token |
|---|---|
| experts (8/128 × 48 visits) | 833 |
| dense attn (trellis ~2.26bpw) | ~634 |
| ctrl (BF16) + routers | ~140 |
| unembed trellis | 221 |
| **total** | **~1830** (−32%) |

KV read: desprezível no bench de 64 tok.

## Ceiling de decode (memória-bound)

| cenário | ms/tok | t/s |
|---|---|---|
| GGUF @ 217 GB/s (piso teórico) | 12.4 | 81 |
| GGUF @ ~60 GB/s efetivo (payload-bound) | 44.8 | 22 |
| REXL3 @ ~60 GB/s efetivo | 30.5 | 33 |
| **medido agora** | **115** | **8.7** |

Meta 15 t/s = 66.7 ms/tok → precisa −42% do tempo atual. Viável dentro do ceiling; REXL3 dá −32% de bytes (≈ ×1.5 em efficiência constante) + MTP drafter embutido (block_size 8, markov_rank 256) para spec decode.

## Prefill / startup / TTFT

(não medidos nesta rodada — decode c1 é a métrica-alvo)

---

## 2026-11-24 — ceiling refresh @ ~58 ms/tok (HEAD 0921399)

Measured this round (medians, BEST env, graphs ON): **58.49 ms/tok**
(runs 58.49 / 59.28 / 58.44). Graphs OFF: **66.15 ms/tok** (67.64 / 65.85 /
66.15). CUDA graphs save ~7.7 ms/tok. GGML_OP_PROF run (graphs off):
73.4 ms/tok incl. profiler overhead; GPU span/token = 65.6 ms.

### Bytes/token — decode (SAGE-REXL3, 48 visits, top-8/128)

Expert residency at load (INKWELL_DEV_MARGIN_MB=1300 default):
layers 0-11 fully VRAM-resident (on_dev), layers 12-23 pinned + per-layer
LFU cache k=20 (INKWELL_EXL3_CACHE_K=20, all 12 fit, ~760 MiB total).

| component | MiB/tok | bus |
|---|---|---|
| dense attn exl3 (6 projs x 48 visits) | 581 | VRAM |
| resident experts (24 visits x 8) | 391 | VRAM |
| cached experts, demand (24 visits x 8) | 441.6 | VRAM hits / PCIe misses |
| ctrl gemvs (4 ctrl x 12 layers x 2 passes) | 108 | VRAM |
| router gate_inp (48) | 27 | VRAM |
| unembed | 221 | VRAM |
| misc (norms, conv, res-q, KV, activations) | ~15 | VRAM |
| **VRAM fixed** | **~1343** | |
| + cached-expert VRAM traffic | 441.6 x (hit + 2x miss) | hit=read, miss=write+read |
| **PCIe** | 441.6 x miss_frac | staged ccopy |

Measured cache (INKWELL_MOE_PREFETCH + INKWELL_PF_STAT, layers 12-16,
steady state): **miss rate 18.6%** (22.3 misses/tok measured; L17-23
extrapolated at same rate +13.4) -> **~36 misses/tok, ~83 MiB staged**.
Bank-prefetch hits ~0 (hist = last visit's ids, wrong pass) — prefetch
as implemented is dead weight for serial decode.

At miss=18.6%: cached-expert VRAM traffic = 441.6 x 1.186 = 524 MiB.
**VRAM total ~1867 MiB/tok. PCIe ~83 MiB/tok.**

### Floors (measured bandwidths: 217 GB/s large-payload, ~60 GB/s
### mid-payload ~1-2 MiB, PCIe zc ~8.6 GB/s, staging copy ~8 GB/s)

| scenario | ms/tok |
|---|---|
| VRAM 1867 MiB @ 217 GB/s (DRAM limit, unreachable w/ cur kernels) | 8.6 |
| VRAM 1867 MiB @ 60 GB/s (achieved mid-payload) | 31.1 |
| PCIe 83 MiB @ 8.6 GB/s (partly hidden under hit-gemvs) | 9.7 |
| **practical ceiling (60 GB/s + ~70% PCIe exposed)** | **~38** |
| practical ceiling, misses -> 0 (all experts resident) | ~30 |
| **measured now (graphs on)** | **58.5** |
| **current % of practical ceiling** | **~65%** |

Component floors @ achieved rates: unembed 221 MiB -> 1.9 ms (measured
117 GB/s); dense attn 581 -> 9.7 ms @60; experts VRAM 915 -> 15.3 ms @60;
ctrl+router+misc 150 -> 2.5 ms @60; staging 83 MiB PCIe -> ~9.7 ms.

Gap vs floor today is NOT bandwidth: ~12 ms of elementwise/glue ops
(~2900 tiny kernels, avg ~4 us) + ~10 ms PCIe staging + ~6 ms host/graph
overhead sit on top of the ~30 ms weight-read floor.
