#!/usr/bin/env python3
"""Exact per-layer kslot cache allocation for a fixed routing trace.

Unlike opt/tools/kslot_alloc.py's greedy marginal walk, this computes each
streamed layer's independent timed-miss-byte curve and solves the total-byte
knapsack exactly by dynamic programming.  The warm prefix is replayed (so it
warms cache state) but only the trailing timed tokens count in the objective.

Usage: kslot_dp_alloc.py <trace> <budget_mib> [warm_tokens] [step_mib]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from kslot_replay import load_gexp, load_trace, KslotSim, GRAN

ON_DEV = {int(x) for x in os.environ.get(
    "ON_DEV", "3,6,7,8,10,11,12,14,16,17").split(",")}


def split_tokens(trace):
    n0 = sum(1 for il, _ in trace if il == 0)
    if not n0 or len(trace) % n0:
        raise ValueError(f"cannot infer token count: len={len(trace)} layer0={n0}")
    per = len(trace) // n0
    return [trace[i * per:(i + 1) * per] for i in range(n0)]


def main():
    trace = load_trace(sys.argv[1])
    budget_mib = int(sys.argv[2])
    warm = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    step_mib = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    gexp, max_slot, esz3 = load_gexp()
    toks = split_tokens(trace)
    timed = max(1, len(toks) - warm)
    layers = sorted({il for il, _ in trace} - ON_DEV)
    # Rebuild per-layer sequences with their token index.  A loop layer has three
    # visits inside one token; all three share that token's warm/timed label.
    seq = {l: [] for l in layers}
    for ti, tok in enumerate(toks):
        for il, ids in tok:
            if il in seq:
                seq[il].append((ti, ids))

    def eval_seq(l, arena_bytes):
        sim = KslotSim(gexp[l], arena_bytes)
        hits = misses = miss_bytes = direct = 0
        for ti, ids in seq[l]:
            h, m, b, d = sim.visit(ids)
            if ti >= warm:
                hits += h; misses += m; miss_bytes += b; direct += d
        return hits, misses, miss_bytes, direct, sim.kmax

    step = step_mib << 20
    units = budget_mib // step_mib
    curves = {}
    caps = {}
    for l in layers:
        cap_gran = sum(sorted((sum(gexp[l][e]) for e in range(128)), reverse=True)[:64]) * GRAN
        cap_mib = min(cap_gran >> 20, budget_mib)
        caps[l] = cap_mib
        c = {}
        for mb in range(0, cap_mib + 1, step_mib):
            c[mb] = eval_seq(l, mb << 20)
        curves[l] = c
        print(f"# L{l} curve 0..{cap_mib}MiB done", file=sys.stderr, flush=True)

    # dp[u] = maximum timed bytes saved vs zero cache after processing some layers.
    NEG = -1 << 120
    dp = [NEG] * (units + 1)
    dp[0] = 0
    choose = []
    for l in layers:
        zero = curves[l][0][2]
        ndp = [NEG] * (units + 1)
        ch = {}
        opts = []
        for mb, val in curves[l].items():
            u = mb // step_mib
            if u <= units:
                opts.append((u, zero - val[2], mb))
        for oldu, oldv in enumerate(dp):
            if oldv == NEG:
                continue
            for u, gain, mb in opts:
                nu = oldu + u
                if nu <= units:
                    nv = oldv + gain
                    if nv > ndp[nu]:
                        ndp[nu] = nv
                        ch[nu] = (oldu, mb)
        dp = ndp
        choose.append((l, ch))

    bestu = max(range(units + 1), key=lambda u: dp[u])
    alloc = {l: 0 for l in layers}
    u = bestu
    for l, ch in reversed(choose):
        oldu, mb = ch[u]
        alloc[l] = mb
        u = oldu

    def report(name, table):
        th = tm = tb = td = 0
        rows = []
        for l in layers:
            mb = table.get(l, 0)
            if mb in curves[l]:
                h, m, b, d, kmax = curves[l][mb]
            else:
                h, m, b, d, kmax = eval_seq(l, mb << 20)
            th += h; tm += m; tb += b; td += d
            rows.append((l, mb, h, m, b, d, kmax))
        print(f"{name}: total={sum(table.get(l,0) for l in layers)}MiB "
              f"timed_hits={th} timed_misses={tm} hit%={100*th/max(1,th+tm):.1f} "
              f"timed_missMiB={tb/1048576:.1f} timed_missMiB/tok={tb/1048576/timed:.2f} direct={td}")
        for l, mb, h, m, b, d, kmax in rows:
            print(f"  L{l:2d} {mb:3d}MiB kmax={kmax:2d} hit%={100*h/max(1,h+m):5.1f} "
                  f"miss={m:4d} missMiB/tok={b/1048576/timed:6.2f} direct={d}")
        return tb

    print(f"# trace_tokens={len(toks)} warm={warm} timed={timed} step={step_mib}MiB budget={budget_mib}MiB")
    t568 = {0:8,1:48,2:40,4:32,5:12,9:64,13:56,15:84,18:32,19:36,20:32,21:40,22:44,23:40}
    if set(t568) == set(layers) and sum(t568.values()) <= budget_mib + step_mib:
        report("T568J-reference", t568)
    report("DP-candidate", alloc)
    arr = [alloc.get(l, 0) for l in range(24)]
    print("table24=[" + ",".join(map(str, arr)) + "]")
    print("override=" + ",".join(f"{l}:{alloc[l]}" for l in layers))
    print(f"# dp_used={bestu*step_mib}MiB dp_saved_vs_zero={dp[bestu]/1048576:.1f}MiB")


if __name__ == "__main__":
    main()
