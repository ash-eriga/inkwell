# Quantized KV cache — working report (DRAFT, not final)

## Real 4K teacher-forced logit KLD (64 rows; controls perfect in all runs)

| format / query mode | mean KLD | p95 | max | top1 | PPL vs ref 1.89604 |
|---|---|---|---|---|---|
| K8V6 fp32 q | **0.00261552** | 0.00827 | 0.0259 | 0.984 | 1.89380 |
| K10V6 dp4a3 | 0.00296233 | 0.00792 | 0.0489 | 1.0 | 1.88842 |
| K10V6 fp32 q | 0.00313035 | 0.00690 | 0.0636 | 0.984 | 1.89522 |
| K10V6 dp4a4 | 0.00334793 | 0.01381 | 0.0478 | 0.984 | 1.89501 |
| K10V6 dp4a2 | 0.00345090 | 0.00870 | 0.0605 | 0.984 | 1.89136 |
| INT8 (K8V8) | ~0.0021 (historical; re-measurement pending seq58) | | | | |

Target: mean ≤ 1e-3, top1 = 1.0. Best current (K8V6) misses by 2.6x.

## Key negative results

- Query quantization precision is irrelevant to real KLD: dp4a2 (near-exact q) is no better than fp32 q or grouped modes. CPU attention-KL proxy (1.3e-5) does not predict real logit KLD (3e-3).
- K10V6 is worse than K8V6 in real quality despite 16x better K precision in the proxy. Roundtrip test pending (seq58) to distinguish encoder bug vs format noise.
- Sink-token fp16 preservation barely helps in the CPU proxy (0.00677 vs 0.00683 rel); error is distributed across all tokens.
- Oracle clipping catastrophic; per-channel RMS scaling minor.
- V bit depth dominates CPU output error: v6 0.00565, v7 0.00283, v8 0.00141; groups g32/g16 help modestly (0.00478/0.00372).

## Position analysis

KLD is present from the first generated token (first8 positions 1e-3..6e-3) and grows slowly (first16 mean 0.0023 → last16 0.0033 for K8V6) with occasional large spikes (pos61 0.026). Per-layer quantization error is simply too large for ≤1e-3; not a slow-compounding-only effect.

## Speed status

- 64K K8V4 dp4a smoke (not interleaved): baseline 46.373, dp4a1 41.827, dp4a2 41.949 ms/tok. dp4a2 costs +0.12ms vs dp4a1 and is query-exact — the right mode if dp4a is used.
- Target 36 ms/tok. Gap 5.9ms on smoke numbers; no validated interleaved measurement yet.
- Attention is instruction/latency-bound at 64K (24 GB/s of 150+ ceiling in old kernel nsys; 8 full visits/token ~2.3ms each).
- qg3+dp4a (shares one quantized load/decode across 3 GQA heads) built; smoke pending seq57.
- K8V6 rows 1888B/token vs INT8/K10 2144 vs K8V4 1600.

## Memory

All ctx512 smokes peak 4940-5020 MiB < 5120 cap. 64K peaks pending validation.

## Open measurements (queued)

- seq57 qg3smoke: qg3+dp4a correctness/coherence at ctx512.
- seq58 k10test: K10V6 write-roundtrip + attention consistency; INT8 4K quality with current harness.
- seq61 hpq: K8V6+INKWELL_KV_HP=4:64 4K quality.

## Remaining work

1. Read queued results; decide format direction.
2. 64K interleaved speed validation for best candidate (run_v6_speed64.sh prepared).
3. 16K quality vs INT8.
4. Prefill attention-share profile; prefill A/B ≥5 rounds.
5. Default F16 gates: MD5 5cbe8e06, peak 5046, short decode 26.5 interleaved.
6. Honest gap arithmetic for quality target.
