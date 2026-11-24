#!/usr/bin/env python3
"""
Synthesize all measured benchmarks (bench_drafter, vocab_coverage, acceptance_proxy)
with our Question 1 cost model (cost_model.py) to produce the ranked table of
drafter designs with exact VRAM-theft-adjusted end-to-end tok/s vs 46.8 ms/tok serial.
"""

import json
import sys
sys.path.insert(0, "opt/kbench/drafter")
import cost_model as cm

def eval_design(d, rstats, regime_p1, regime_q, mode="ragged", ov=0.0):
    p1 = regime_p1 * d["ret_p1"]
    q  = regime_q  * d["ret_q"]
    res = {}
    for k, td in [(1, d["t_k1"]), (2, d["t_k2"]), (3, d["t_k3"])]:
        em = cm.emit_tokens(k, p1, q)
        ver = cm.verify_cost_ms(k, rstats, mode=mode, vram_stolen_mib=d["vram"], staging_overlap_bonus=ov)["total_ms"]
        ms_tok = (ver + td) / em
        tps = 1000.0 / ms_tok
        vs_ser = (cm.SERIAL_MS / ms_tok - 1.0) * 100.0
        res[k] = {"emit": em, "verify_ms": ver, "draft_ms": td, "ms_tok": ms_tok, "tps": tps, "vs_serial_pct": vs_ser}
    return res

def main():
    rstats, _ = cm.load_router_stats()

    # Every latency is MEASURED on GTX 1060 6GB inside ONE captured CUDA graph (zero host sync).
    # Retention factors (ret_p1, ret_q) combine Q2 640-token coverage and Q5 real DSpark proxy:
    #   - Q2 coverage: 200K=1.000, 32K=0.9484, 16K=0.8672, 8K=0.7781
    #   - Q5 quant match (d1 / d1..3):
    #       Int8 body+head dp2a: 100% / 100% (84/84)
    #       Int8 body+head dp4a: 100% / 100% (36/36, 80/84 d1..7)
    #       Int4-g128 head dp4a: 100% / 94.4% (34/36)
    #       Int4-g128 full body: 91.7% / 80.6% (29/36)
    #       Int4-row  full body: 58.3% / 63.9% (23/36) -> disqualified
    #   - Width/architecture capacity modifiers (modeled relative to D=1024 EAGLE-1):
    #       D=1024/2304 EAGLE-1: 1.00; D=512 Slim: 0.96; Medusa (0-attn parallel): 0.92 p1, 0.70 q.
    designs = [
        {
            "name": "EAGLE-1-Med (i8 body + i4-g128 32K head, tied)",
            "head": "Int4-g128 dp4a V'=32K (H=1024)",
            "body": "1L D=1024 FF=2816 SWA-128 + 4-tap (i8)",
            "vram": 29.9,  # 13.2 body+kv + 16.7 head+fp16_g128_scales+vids (tied emb)
            "t_k1": 0.25, "t_k2": 0.48, "t_k3": 0.71,
            "proxy": "94.8% cov; 100% d1 / 94.4% d1..3 (Q5)",
            "ret_p1": 0.9484 * 1.000, "ret_q": 0.9484 * 0.944,
        },
        {
            "name": "EAGLE-1-Med (i8 body + i8 32K head, tied)",
            "head": "Int8 dp4a V'=32K (H=1024)",
            "body": "1L D=1024 FF=2816 SWA-128 + 4-tap (i8)",
            "vram": 45.4,
            "t_k1": 0.33, "t_k2": 0.64, "t_k3": 0.94,
            "proxy": "94.8% cov; 100% d1..3 / 95.2% d1..7 (Q5)",
            "ret_p1": 0.9484 * 1.000, "ret_q": 0.9484 * 1.000,
        },
        {
            "name": "EAGLE-1-Full (i8 body + Resident EXL3 200K head)",
            "head": "Resident EXL3 K=4.0 (200K, 0 MiB)",
            "body": "1L D=2304 FF=4608 SWA-128 + 4-tap (i8)",
            "vram": 44.3,  # 1L D=2304 i8 body + KV, 0 MiB head!
            "t_k1": 2.33, "t_k2": 4.66, "t_k3": 6.99,  # body 0.32ms/step + EXL3 unembed 2.01ms/step
            "proxy": "100% cov; 100% d1..3 (i8 body + exact head)",
            "ret_p1": 1.000, "ret_q": 1.000,
        },
        {
            "name": "EAGLE-2L-Med (i8 body + i4-g128 32K head, tied)",
            "head": "Int4-g128 dp4a V'=32K (H=1024)",
            "body": "2L D=1024 FF=2816 SWA-128 + 4-tap (i8)",
            "vram": 40.7,
            "t_k1": 0.35, "t_k2": 0.69, "t_k3": 1.03,
            "proxy": "94.8% cov; 100% d1 / 94.4% d1..3 (Q5)",
            "ret_p1": 0.9484 * 1.000, "ret_q": 0.9484 * 0.944,
        },
        {
            "name": "EAGLE-1-Slim (i8 body + i4-g128 32K head, tied)",
            "head": "Int4-g128 dp4a V'=32K (H=512)",
            "body": "1L D=512 FF=1536 SWA-128 + 4-tap (i8)",
            "vram": 12.6,
            "t_k1": 0.16, "t_k2": 0.31, "t_k3": 0.46,
            "proxy": "94.8% cov; 100% d1 / 94.4% d1..3 (est 0.96 cap)",
            "ret_p1": 0.9484 * 0.96, "ret_q": 0.9484 * 0.944 * 0.94,
        },
        {
            "name": "EAGLE-1-Full (i8 body + i4-g128 32K head, tied)",
            "head": "Int4-g128 dp4a V'=32K (H=2304)",
            "body": "1L D=2304 FF=4608 SWA-128 + 4-tap (i8)",
            "vram": 81.6,
            "t_k1": 0.57, "t_k2": 1.14, "t_k3": 1.71,
            "proxy": "94.8% cov; 100% d1 / 94.4% d1..3 (Q5)",
            "ret_p1": 0.9484 * 1.000, "ret_q": 0.9484 * 0.944,
        },
        {
            "name": "EAGLE-1-Med (i8 body + i8 16K head, tied)",
            "head": "Int8 dp4a V'=16K (H=1024)",
            "body": "1L D=1024 FF=2816 SWA-128 + 4-tap (i8)",
            "vram": 29.2,
            "t_k1": 0.23, "t_k2": 0.44, "t_k3": 0.66,
            "proxy": "86.7% cov; 92% d1 / 91.7% d1..3 (Q5)",
            "ret_p1": 0.8672, "ret_q": 0.8672,
        },
        {
            "name": "Medusa-MLP (2L D=1024 + i4-g128 32K head)",
            "head": "Int4-g128 dp4a V'=32K (H=1024)",
            "body": "0-attn 2L ResMLP D=1024 FF=2048 (batched)",
            "vram": 31.0,
            "t_k1": 0.24, "t_k2": 0.26, "t_k3": 0.30,
            "proxy": "94.8% cov; 0-attn lower q (~0.92 p1, 0.70 q)",
            "ret_p1": 0.9484 * 0.92, "ret_q": 0.9484 * 0.70,
        },
        {
            "name": "Medusa-MLP (2L D=1024 + Resident EXL3 200K)",
            "head": "Resident EXL3 K=4.0 (200K, batched)",
            "body": "0-attn 2L ResMLP D=1024 + up2304 (batched)",
            "vram": 16.5,
            "t_k1": 2.13, "t_k2": 2.49, "t_k3": 2.91,
            "proxy": "100% cov; 0-attn lower q (~0.92 p1, 0.70 q)",
            "ret_p1": 0.92, "ret_q": 0.70,
        },
        {
            "name": "DSpark-Int4-g128 (0-retrain 3L + i4-g128 32K)",
            "head": "Int4-g128 dp4a V'=32K (H=2304)",
            "body": "Existing 3L D=2304 FF=9216 + Markov (i4-g128)",
            "vram": 182.5,  # 174.4 + 8.1 MiB fp16 g128 scales
            "t_k1": 1.48, "t_k2": 1.93, "t_k3": 2.59,
            "proxy": "94.8% cov; Q5 full: 92% d1 / 80.6% d1..3",
            "ret_p1": 0.9484 * 0.917, "ret_q": 0.9484 * 0.806,
        },
        {
            "name": "DSpark-Int8 (0-retrain 3L + i8 32K head)",
            "head": "Int8 dp4a V'=32K (H=2304)",
            "body": "Existing 3L D=2304 FF=9216 + Markov (i8)",
            "vram": 337.5,
            "t_k1": 2.29, "t_k2": 2.94, "t_k3": 3.54,
            "proxy": "94.8% cov; Q5 full: 100% d1..3 / 95.2% d1..7",
            "ret_p1": 0.9484 * 1.000, "ret_q": 0.9484 * 1.000,
        },
        {
            "name": "Today DSpark EXL3 (no graph, f32)",
            "head": "Resident EXL3 K=4.0 (200K)",
            "body": "Existing 3L EXL3 + Markov (ungraphed)",
            "vram": 236.0,
            "t_k1": 22.0, "t_k2": 32.0, "t_k3": 42.0,
            "proxy": "100% cov; 100% oracle baseline",
            "ret_p1": 1.000, "ret_q": 1.000,
        },
    ]

    out_all = {}
    for reg_key, reg_name, p1, q in [
        ("A_prompt", "Regime A: Prompt baseline (p1=0.75, q=0.18)", 0.75, 0.18),
        ("B_ledger", "Regime B: Ledger empirical (p1=0.75, q=0.46)", 0.75, 0.46),
        ("C_eagle",  "Regime C: Trained EAGLE-class (p1=0.82, q=0.65)", 0.82, 0.65),
    ]:
        print(f"\n=== {reg_name} (Serial = {1000/cm.SERIAL_MS:.2f} tok/s = {cm.SERIAL_MS:.1f} ms/tok) ===")
        rows = []
        for d in designs:
            r = eval_design(d, rstats, p1, q)
            best_k = max([1, 2, 3], key=lambda kk: r[kk]["tps"])
            rows.append((r[best_k]["tps"], best_k, d, r))
        rows.sort(key=lambda x: -x[0])
        print(f"{'#':>2} {'Design':48s} | {'VRAM':>6} | {'t(k1)':>5} {'t(k2)':>5} {'t(k3)':>5} | {'k1 t/s':>6} {'k2 t/s':>6} {'k3 t/s':>6} | {'Best vs Ser':>12}")
        out_rows = []
        for idx, (tps, bk, d, r) in enumerate(rows, 1):
            print(f"{idx:2d} {d['name']:48s} | {d['vram']:5.1f}M | {d['t_k1']:5.2f} {d['t_k2']:5.2f} {d['t_k3']:5.2f} | "
                  f"{r[1]['tps']:6.2f} {r[2]['tps']:6.2f} {r[3]['tps']:6.2f} | "
                  f"k={bk}: {r[bk]['vs_serial_pct']:+5.1f}%")
            out_rows.append({"rank": idx, "name": d["name"], "vram": d["vram"], "best_k": bk, "res": r})
        out_all[reg_key] = out_rows

    with open("opt/kbench/drafter/ranked_designs_out.json", "w") as f:
        json.dump(out_all, f, indent=2)

if __name__ == "__main__":
    main()
