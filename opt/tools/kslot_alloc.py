#!/usr/bin/env python3
"""Per-layer cache allocation: compute each streamed layer's miss-byte curve
(4 MiB steps) by replaying the routing trace through the exact kslot policy,
then greedily spend a total MiB budget by marginal miss-bytes saved.

Usage: kslot_alloc.py <trace> <budget_mib> [<trace2>]
Prints the allocation table and totals vs uniform ck=20.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kslot_replay import load_gexp, load_trace, run, GRAN

STEP = 4 << 20
ON_DEV = {int(x) for x in os.environ.get(
    "ON_DEV", "6,7,8,9,10,11,12,13,14,15").split(",")}


def curve(trace, gexp, l, cap):
    """miss bytes at each arena size 0,STEP..cap (bytes)."""
    out = {}
    for mb in range(0, (cap >> 20) + STEP // (1 << 20), STEP >> 20):
        b = mb << 20
        out[b] = run(trace, gexp, lambda x: b, only={l})[l][2]
    return out


def main():
    # argv: script <trace> <budget_mib> [trace2 ...]
    gexp, max_slot, esz3 = load_gexp()
    budget = int(sys.argv[2]) << 20
    traces = [load_trace(p) for p in [sys.argv[1]] + sys.argv[3:]]
    nvis = {}
    for il, _ in traces[0]:
        nvis[il] = nvis.get(il, 0) + 1
    streamed = [l for l in sorted(nvis) if l not in ON_DEV]

    caps, curves = {}, {}
    for l in streamed:
        c = sum(sorted((sum(gexp[l][e]) for e in range(128)), reverse=True)[:64]) * GRAN
        caps[l] = c
        curves[l] = curve(traces[0], gexp, l, c)
        print(f"# L{l} curve done cap={c>>20}MiB", file=sys.stderr, flush=True)

    def mb(l, b):
        return curves[l][b]

    # uniform ck=20 baseline (exact bytes, simulated separately)
    base = {l: 60 * max_slot[l] for l in streamed}
    base_alloc = sum(base.values())

    # greedy from empty in STEP units
    alloc = {l: 0 for l in streamed}
    tot = 0
    while True:
        best_g, best_l = 0.0, None
        for l in streamed:
            nb = alloc[l] + STEP
            if nb > caps[l] or tot + STEP > budget:
                continue
            g = mb(l, alloc[l]) - mb(l, nb)
            if g > best_g:
                best_g, best_l = g, l
        if best_l is None:
            break
        alloc[best_l] += STEP
        tot += STEP

    # evaluate on all traces
    for ti, tr in enumerate(traces):
        res = run(tr, gexp, lambda l: alloc.get(l, 0))
        res0 = run(tr, gexp, lambda l: 60 * max_slot[l])
        tb = sum(v[2] for v in res.values())
        tm = sum(v[1] for v in res.values())
        tb0 = sum(v[2] for v in res0.values())
        tm0 = sum(v[1] for v in res0.values())
        tag = f"trace{ti}"
        print(f"{tag}: alloc miss={tm} missMiB={tb/1048576:.1f} | "
              f"uniform({sum(base.values())>>20}MiB) miss={tm0} missMiB={tb0/1048576:.1f}")

    print(f"# budget={budget>>20}MiB used={sum(alloc.values())>>20}MiB "
          f"(uniform total {base_alloc>>20}MiB)")
    for l in sorted(alloc):
        u = run(traces[0], gexp, lambda x: base[l], only={l})[l][2]
        print(f"L{l:2d} {alloc[l]>>20:4d} MiB (unif {base[l]>>20:3d}, cap {caps[l]>>20:3d}) "
              f"missMiB {mb(l, alloc[l])/1048576:6.1f} (unif {u/1048576:6.1f})")


if __name__ == "__main__":
    main()
