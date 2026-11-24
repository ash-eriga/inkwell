#!/usr/bin/env python3
"""
Question 1: Analytical + trace-driven speculative decoding cost model for
Inkling-2-Mini (SAGE REXL3 2.26 bpw) on GTX 1060 6 GB (sm_61).

Anchors:
  - Serial decode (master 952e1ff): 46.8 ms/tok (21.37 tok/s)
  - Real MoE routing traces: opt/kbench/ids_block.bin (384 layer visits, 8 rows x 8 experts)
  - Batched EXL3 GEMV scaling: opt/kbench/exl3_math2/REPORT.md (M=16-Batch C0c exact FP32)
      B=1: 1.00x tput (1.000x time)
      B=2: 1.68x tput (1.190x time)
      B=3: 2.11x tput (1.422x time)
      B=4: 2.53x tput (1.581x time)
      B=5: 2.86x tput (1.748x time, linear extrapolation of slope 0.193/row)
  - Acceptance curves:
      (A) Prompt baseline: P(d1)=0.75, P(d_{i}|d_{i-1})=0.18
      (B) Ledger empirical (opt/ledger.md:130-139): P(d1)=0.75, q=0.46 -> emit ~1.75, 2.10, 2.25, 2.31
      (C) Trained EAGLE-class target: P(d1)=0.82, q=0.65 -> emit ~1.82, 2.35, 2.70, 2.92
"""

import array
import json
import os
import sys

# --- 1. Measured Serial Baseline Breakdown (master 952e1ff = 46.8 ms/tok) ---
# Reconciled from opt/ledger.md:541-552 (nsys @ 58.5 ms) minus landed gains:
#   gemv_sf (-1.7 ms), step_gcache (-6.7 ms), ctrl/relbias fusion (-3.5 ms), kslot (-1.1 ms)
SERIAL_MS = 46.8
SERIAL_BREAKDOWN = {
    "moe_gemv":     19.8,  # gemv_moe_s (48 visits x 8 experts)
    "moe_staging":   7.7,  # exposed PCIe ccopy (8.8 ms pre-kslot - 1.1 ms kslot)
    "moe_overhead":  2.5,  # had_in/out_moe, resolve, wsum (wall, net of overlap)
    "dense_exl3":   11.1,  # 48-visit concurrent mz + wo (~9.2 ms) + unembed (1.9 ms)
    "non_exl3":      4.9,  # ctrl mul_mat, router, mix/conv/clip, attn, fused glue
    "host_graph":    0.8,  # gcache boundary hole (measured 0.83 ms) + launch residual
}
assert abs(sum(SERIAL_BREAKDOWN.values()) - SERIAL_MS) < 1e-6

# --- 2. Batched EXL3 GEMV Time Multipliers (M=16-Batch C0c, exl3_math2/REPORT.md) ---
# t_exl3(B) = time(B rows) / time(1 row) = B / throughput_speedup(B)
EXL3_TPUT_SPEEDUP = {
    1: 1.00,
    2: 1.68,   # measured: 487.9 / 288.0 (interleaved 1.68x)
    3: 2.11,   # measured: 608.2 / 288.0 (interleaved 2.11x)
    4: 2.53,   # measured: 734.0 / 288.0 (interleaved 2.53x)
    5: 2.86,   # extrapolated: t(5) = t(4) + 0.167 = 1.748 -> 5/1.748 = 2.86x
    6: 3.13,   # extrapolated: t(6) = 1.915 -> 6/1.915
    7: 3.36,   # extrapolated: t(7) = 2.082 -> 7/2.082
    8: 3.56,   # extrapolated: t(8) = 2.249 -> 8/2.249
}
EXL3_TIME_MULT = {b: b / s for b, s in EXL3_TPUT_SPEEDUP.items()}

# --- 3. Load Real Router Trace (opt/kbench/ids_block.bin) ---
def load_router_stats(path="opt/kbench/ids_block.bin"):
    a = array.array("i")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    nvis = len(a) // 65  # 1 int32 layer_id + 64 int32 (8 rows x 8 experts)
    stats = {}
    for B in [1, 2, 3, 4, 5, 8]:
        unions = []
        mult_hist = {m: 0 for m in range(1, B + 1)}
        tot_u = 0
        for v in range(nvis):
            base = v * 65 + 1
            cnt = {}
            for r in range(B):
                for e in a[base + r * 8 : base + r * 8 + 8]:
                    cnt[e] = cnt.get(e, 0) + 1
            unions.append(len(cnt))
            tot_u += len(cnt)
            for c in cnt.values():
                mult_hist[c] += 1
        mean_u = tot_u / nvis
        # Ragged batched MoE multiplier: each distinct expert runs at its own multiplicity m
        # Exact time per visit relative to B=1 (8 experts at m=1):
        #   sum_{e in union} t(m_e) / 8 = (U(B)/8) * sum_m P(m|B) * t(m)
        ragged_per_exp = sum((mult_hist[m] / tot_u) * EXL3_TIME_MULT[m] for m in range(1, B + 1))
        stats[B] = {
            "mean_union": mean_u,
            "union_ratio": mean_u / 8.0,
            "avg_mult": (8.0 * B) / mean_u,
            "ragged_per_exp": ragged_per_exp,
            "ragged_moe_mult": (mean_u / 8.0) * ragged_per_exp,
            "padded_moe_mult": (mean_u / 8.0) * EXL3_TIME_MULT[B],
            "mult_dist": {m: mult_hist[m] / tot_u for m in range(1, B + 1)},
        }
    return stats, nvis

# --- 4. Verify Cost Model for k = 1..4 (B = k + 1 verify rows) ---
def verify_cost_ms(k, rstats, mode="ragged", vram_stolen_mib=0.0, staging_overlap_bonus=0.0):
    """
    Model verify(k) for B = k + 1 rows: [anchor, d_1, ..., d_k].
    Modes:
      - 'ragged': ideal per-multiplicity batched MoE kernel (each expert runs at its hit count m_e)
      - 'padded': uniform B-row batched MoE kernel (every expert in union runs at batch B)
    vram_stolen_mib: drafter VRAM taken from the 14 expert caches.
      Empirically from opt/ledger.md:436-438: k=20->16 (-162 MiB) added +2.39 ms/tok at B=1
      => +0.0148 ms/MiB at B=1, scaling with U(B)/8 for B verify rows.
    staging_overlap_bonus: fraction (0..1) of incremental staging hidden by longer hit-GEMV.
    """
    B = k + 1
    st = rstats[B]
    u_ratio = st["union_ratio"]

    # 1. MoE EXL3 GEMV
    moe_mult = st["ragged_moe_mult"] if mode == "ragged" else st["padded_moe_mult"]
    t_moe_gemv = SERIAL_BREAKDOWN["moe_gemv"] * moe_mult

    # 2. MoE PCIe Staging (exposed)
    # Distinct expert misses scale with U(B)/8; overlap_bonus can hide part of the delta > 1
    eff_stage_mult = 1.0 + (u_ratio - 1.0) * (1.0 - staging_overlap_bonus)
    t_staging = SERIAL_BREAKDOWN["moe_staging"] * eff_stage_mult
    # VRAM theft penalty (cache slots lost -> extra PCIe misses)
    t_vram_pen = (0.0148 * vram_stolen_mib) * u_ratio

    # 3. MoE overhead (had_in/out_moe scales with B for input/output + U(B)/8 for resolve/wsum)
    t_moe_ovh = SERIAL_BREAKDOWN["moe_overhead"] * (0.5 * u_ratio + 0.5 * B * 0.65)

    # 4. Dense EXL3 GEMV (all B rows share exact same weights -> M=16-Batch C0c scaling)
    t_dense = SERIAL_BREAKDOWN["dense_exl3"] * EXL3_TIME_MULT[B]

    # 5. Non-EXL3 compute (ctrl/router weight-bound ~1.15-1.35x; conv/attn/glue scales mildly)
    # Empirically weight-bound (~60% of 4.9 ms) scales as 1 + 0.12*(B-1); compute/act (~40%) scales as B
    non_exl3_mult = 0.60 * (1.0 + 0.12 * (B - 1)) + 0.40 * (1.0 + 0.55 * (B - 1))
    t_non_exl3 = SERIAL_BREAKDOWN["non_exl3"] * non_exl3_mult

    # 6. Host turnaround / graph launch (nearly fixed, slight logits DtoH growth for B rows)
    t_host = SERIAL_BREAKDOWN["host_graph"] + 0.15 * (B - 1)

    total = t_moe_gemv + t_staging + t_vram_pen + t_moe_ovh + t_dense + t_non_exl3 + t_host
    return {
        "k": k, "B": B,
        "moe_gemv": t_moe_gemv,
        "moe_staging": t_staging,
        "vram_pen": t_vram_pen,
        "moe_overhead": t_moe_ovh,
        "dense_exl3": t_dense,
        "non_exl3": t_non_exl3,
        "host_graph": t_host,
        "total_ms": total,
        "vs_serial": total / SERIAL_MS,
    }

# --- 5. Expected Emitted Tokens emit(k) ---
def emit_tokens(k, p1, q):
    """
    Exact expected tokens emitted per verify round with k drafts:
      P(accept >= 1) = p1
      P(accept >= j) = p1 * q^(j-1)  for j = 1..k
      emit(k) = 1 + sum_{j=1}^k P(accept >= j)
              = 1 + p1 * (1 - q^k) / (1 - q)   (for q != 1)
    """
    if k <= 0:
        return 1.0
    if abs(1.0 - q) < 1e-9:
        return 1.0 + p1 * k
    return 1.0 + p1 * (1.0 - q ** k) / (1.0 - q)

def main():
    rstats, nvis = load_router_stats()
    print(f"=== Router Trace Stats (opt/kbench/ids_block.bin, {nvis} layer visits) ===")
    print(f"{'B':>2} {'k':>2} | {'U(B)':>6} {'U/8':>6} {'avg_m':>6} | {'t_EXL3(B)':>9} {'ragged_MoE':>10} {'padded_MoE':>10}")
    for B in [1, 2, 3, 4, 5, 8]:
        s = rstats[B]
        print(f"{B:2d} {B-1:2d} | {s['mean_union']:6.2f} {s['union_ratio']:6.3f} {s['avg_mult']:6.3f} | "
              f"{EXL3_TIME_MULT[B]:9.3f} {s['ragged_moe_mult']:10.3f} {s['padded_moe_mult']:10.3f}")

    print("\n=== Verify Cost Model verify(k) (anchored to 46.8 ms serial, 0 MiB VRAM theft) ===")
    for mode, ov in [("ragged", 0.0), ("ragged", 0.35), ("padded", 0.0)]:
        label = f"{mode}, stage_overlap_bonus={int(ov*100)}%"
        print(f"\n-- Mode: {label} --")
        print(f"{'k':>2} {'B':>2} | {'MoE_GV':>7} {'Stage':>6} {'MoE_ov':>6} {'Dense':>6} {'NonEX':>6} {'Host':>5} | {'verify':>7} {'xSer':>6}")
        for k in [1, 2, 3, 4]:
            v = verify_cost_ms(k, rstats, mode=mode, staging_overlap_bonus=ov)
            print(f"{k:2d} {v['B']:2d} | {v['moe_gemv']:7.2f} {v['moe_staging']:6.2f} {v['moe_overhead']:6.2f} "
                  f"{v['dense_exl3']:6.2f} {v['non_exl3']:6.2f} {v['host_graph']:5.2f} | {v['total_ms']:7.2f} {v['vs_serial']:5.2f}x")

    print("\n=== Drafter Time Budgets (ms/round) to Beat Serial (46.8 ms/tok) by 0%, +10%, +20% ===")
    regimes = [
        ("A. Prompt baseline (p1=0.75, q=0.18)", 0.75, 0.18),
        ("B. Ledger empirical (p1=0.75, q=0.46)", 0.75, 0.46),
        ("C. Strong trained EAGLE (p1=0.82, q=0.65)", 0.82, 0.65),
    ]
    targets = [("0% (46.8ms)", 1.0), ("+10% (42.5ms)", 1.10), ("+20% (39.0ms)", 1.20)]

    summary = {"serial_ms": SERIAL_MS, "router_stats": rstats, "regimes": {}}
    for name, p1, q in regimes:
        print(f"\n--- {name} ---")
        print(f"{'k':>2} {'B':>2} | {'emit(k)':>7} | {'ver_rag':>7} {'B_0%':>7} {'B_10%':>7} {'B_20%':>7} | "
              f"{'ver_ov35':>8} {'B0%_ov':>7} {'B10%_ov':>7} | {'ver_pad':>7} {'B0%_pad':>7}")
        reg_out = []
        for k in [1, 2, 3, 4]:
            em = emit_tokens(k, p1, q)
            vr = verify_cost_ms(k, rstats, mode="ragged", staging_overlap_bonus=0.0)["total_ms"]
            vo = verify_cost_ms(k, rstats, mode="ragged", staging_overlap_bonus=0.35)["total_ms"]
            vp = verify_cost_ms(k, rstats, mode="padded", staging_overlap_bonus=0.0)["total_ms"]
            b0 = em * (SERIAL_MS / 1.00) - vr
            b10 = em * (SERIAL_MS / 1.10) - vr
            b20 = em * (SERIAL_MS / 1.20) - vr
            b0o = em * (SERIAL_MS / 1.00) - vo
            b10o = em * (SERIAL_MS / 1.10) - vo
            b0p = em * (SERIAL_MS / 1.00) - vp
            print(f"{k:2d} {k+1:2d} | {em:7.3f} | {vr:7.2f} {b0:+7.2f} {b10:+7.2f} {b20:+7.2f} | "
                  f"{vo:8.2f} {b0o:+7.2f} {b10o:+7.2f} | {vp:7.2f} {b0p:+7.2f}")
            reg_out.append({
                "k": k, "B": k+1, "emit": em,
                "verify_ragged": vr, "budget_0": b0, "budget_10": b10, "budget_20": b20,
                "verify_ragged_ov35": vo, "budget_0_ov35": b0o, "budget_10_ov35": b10o,
                "verify_padded": vp, "budget_0_padded": b0p,
            })
        summary["regimes"][name] = reg_out

    print("\n=== Required emit(k) for 0%, +10%, +20% with a 1.5 ms Drafter (ragged verify) ===")
    for k in [1, 2, 3, 4]:
        vr = verify_cost_ms(k, rstats, mode="ragged")["total_ms"]
        tot = vr + 1.5
        e0 = tot / SERIAL_MS
        e10 = tot / (SERIAL_MS / 1.10)
        e20 = tot / (SERIAL_MS / 1.20)
        print(f"k={k} (B={k+1}): verify={vr:.2f} + 1.50 = {tot:.2f} ms => needs emit >= {e0:.2f} (0%), {e10:.2f} (+10%), {e20:.2f} (+20%)")

    out_path = "opt/kbench/drafter/cost_model_out.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
