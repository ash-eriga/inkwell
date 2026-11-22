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
