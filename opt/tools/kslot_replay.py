#!/usr/bin/env python3
"""Replay the kslot LFU expert-cache policy offline against a real routing
trace (INKWELL_IDSEQ file: [layer i32][8 i32] per visit, in visit order).

Mirrors inkwell_exl3_kslot_resolve in crates/inkwell-cuda/src/kernels.cu:
  - cnt[e] += 1 per selection (pass 1), hits claim their slot (u64 mask);
  - per miss: first free tag slot, then first-fit alloc of 3 extents
    (gexp[e*3+p] granules of 6912 B); evict global-least-cnt unclaimed
    resident (lowest slot idx on ties) until the extents fit; DIRECT when
    nothing is left to evict.
  - kmax = clamp(arena_gran / gmin3, 1, 64).

Usage:
  kslot_replay.py gexp                -> per-layer expert-size table
  kslot_replay.py sim <trace> [uniform_ck] -> per-layer misses at ck sizing
  kslot_replay.py sweep <trace>       -> miss-byte curve per layer vs MiB
  kslot_replay.py alloc <trace> <budget_mib> [base_mib_per_layer...]
"""
import json, struct, sys, os

MODEL = os.environ.get("INKWELL_MODEL", os.path.expanduser("~/models/inkling"))
GRAN = 6912
NL, NE, NP = 24, 128, 3
PROJS = ["gate_proj", "up_proj", "down_proj"]


def load_gexp():
    """gexp[L][e][p] = granules of expert e proj p in layer L; max_slot[L]
    = max blob bytes; esz3[L][e] = 3-proj footprint bytes."""
    gexp = [[[0] * NP for _ in range(NE)] for _ in range(NL)]
    for fn in os.listdir(MODEL):
        if not fn.startswith("rexl3-") or not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(MODEL, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            parts = name.split(".")
            if len(parts) != 9 or parts[5] != "experts" or parts[7] not in PROJS \
               or parts[8] != "trellis":
                continue
            il = int(parts[3]); e = int(parts[6]); p = PROJS.index(parts[7])
            k16 = meta["shape"][2]
            w = (k16 // 16) * 8 + (4 if k16 % 16 else 0)
            gexp[il][e][p] = 6 * w + 1
    max_slot = [max(GRAN * max(gexp[l][e]) for e in range(NE)) for l in range(NL)]
    esz3 = [[GRAN * sum(gexp[l][e]) for e in range(NE)] for l in range(NL)]
    return gexp, max_slot, esz3


def load_trace(path):
    recs = []
    data = open(path, "rb").read()
    assert len(data) % 36 == 0, f"bad trace len {len(data)}"
    for off in range(0, len(data), 36):
        il = struct.unpack_from("<i", data, off)[0]
        ids = struct.unpack_from("<8i", data, off + 4)
        recs.append((il, ids))
    return recs


class KslotSim:
    """Exact re-implementation of inkwell_exl3_kslot_resolve."""

    def __init__(self, gexp_l, arena_bytes):
        self.g = gexp_l  # [128][3] granule counts
        self.arena = arena_bytes // GRAN
        gmin3 = max(1, min(sum(g) for g in gexp_l))
        self.kmax = max(1, min(64, self.arena // gmin3))
        self.tag = [-1] * self.kmax
        self.soff = [[0] * 3 for _ in range(self.kmax)]
        self.cnt = [0] * NE
        self.fl = [[0, self.arena]]  # (base, len) sorted by base
        self.hits = self.misses = self.direct = 0
        self.miss_bytes = 0

    def _fit(self, need):
        for i, (b, l) in enumerate(self.fl):
            if l >= need:
                return i
        return -1

    def _alloc(self, i, need):
        b, l = self.fl[i]
        if l == need:
            self.fl.pop(i)
        else:
            self.fl[i][0] = b + need
            self.fl[i][1] = l - need
        return b

    def _free(self, base, ln):
        fl = self.fl
        q = 0
        while q < len(fl) and fl[q][0] < base:
            q += 1
        fl.insert(q, [base, ln])
        if q > 0 and fl[q - 1][0] + fl[q - 1][1] == base:
            fl[q - 1][1] += fl[q][1]
            fl.pop(q)
            q -= 1
        if q + 1 < len(fl) and fl[q][0] + fl[q][1] == fl[q + 1][0]:
            fl[q][1] += fl[q + 1][1]
            fl.pop(q + 1)

    def visit(self, ids):
        """One visit = 8 selections, in order. Returns (hits, misses, miss_bytes,
        directs) for this visit."""
        kmax, tag, cnt = self.kmax, self.tag, self.cnt
        claimed = 0
        miss = [False] * 8
        vh = vm = vb = vd = 0
        # pass 1
        for s in range(8):
            e = ids[s]
            cnt[e] += 1
            v = -1
            for j in range(kmax):
                if tag[j] == e:
                    v = j
                    break
            if v >= 0:
                claimed |= 1 << v
                vh += 1
            else:
                miss[s] = True
                vm += 1
        # pass 2
        for s in range(8):
            if not miss[s]:
                continue
            e = ids[s]
            vb += GRAN * sum(self.g[e])
            v = -1
            for j in range(kmax):
                if tag[j] < 0:
                    v = j
                    break
            got = [0, 0, 0]
            ng = 0
            while True:
                while v >= 0 and ng < 3:
                    need = self.g[e][ng]
                    fi = self._fit(need)
                    if fi < 0:
                        break
                    got[ng] = self._alloc(fi, need)
                    ng += 1
                if v >= 0 and ng == 3:
                    break
                best, bestc = -1, 0x7FFFFFFF
                for j in range(kmax):
                    if claimed & (1 << j) or tag[j] < 0:
                        continue
                    c = cnt[tag[j]]
                    if c < bestc:
                        bestc, best = c, j
                if best < 0:
                    break
                tv = tag[best]
                for p in range(3):
                    self._free(self.soff[best][p] // GRAN, self.g[tv][p])
                tag[best] = -1
                if v < 0:
                    v = best
            if v < 0 or ng < 3:
                while ng > 0:
                    ng -= 1
                    self._free(got[ng], self.g[e][ng])
                vd += 1
                continue
            claimed |= 1 << v
            tag[v] = e
            for p in range(3):
                self.soff[v][p] = got[p] * GRAN
        self.hits += vh
        self.misses += vm
        self.miss_bytes += vb
        self.direct += vd
        return vh, vm, vb, vd


def run(trace, gexp, arena_of, only=None):
    """arena_of: layer -> arena bytes. Returns per-layer (hits,misses,MB,direct)."""
    sims = {}
    for il, ids in trace:
        if only is not None and il not in only:
            continue
        if il not in sims:
            sims[il] = KslotSim(gexp[il], arena_of(il))
        sims[il].visit(ids)
    return {l: (s.hits, s.misses, s.miss_bytes, s.direct)
            for l, s in sims.items()}


def main():
    gexp, max_slot, esz3 = load_gexp()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "gexp"

    if cmd == "gexp":
        for l in range(NL):
            gmin = min(sum(gexp[l][e]) for e in range(NE))
            cap64 = sum(sorted((sum(gexp[l][e]) for e in range(NE)), reverse=True)[:64])
            print(f"L{l:2d} max_slot={max_slot[l]/1048576:6.2f}MiB "
                  f"uniform_ck20={60*max_slot[l]/1048576:6.1f}MiB "
                  f"esz3 avg={sum(esz3[l])/128/1048576:5.2f}MiB "
                  f"gmin3={gmin} cap64={cap64*GRAN/1048576:6.1f}MiB")
        return

    trace = load_trace(sys.argv[2])
    nvis = {}
    for il, _ in trace:
        nvis[il] = nvis.get(il, 0) + 1

    if cmd == "sim":
        ck = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        res = run(trace, gexp, lambda l: ck * 3 * max_slot[l])
        th = tm = tb = 0
        for l in sorted(res):
            h, m, b, d = res[l]
            th += h; tm += m; tb += b
            print(f"L{l:2d} vis={nvis[l]:3d} sel={8*nvis[l]:4d} hits={h:4d} "
                  f"miss={m:4d} ({100*m/(8*nvis[l]):5.1f}%) missMiB={b/1048576:6.1f} "
                  f"direct={d} arena={ ( ck*3*max_slot[l]) /1048576:5.1f}MiB")
        ntok = min(nvis.values())
        print(f"TOT hits={th} miss={tm} hit%={100*th/(th+tm):.1f} "
              f"missMiB/tok={tb/1048576/ntok:.1f} over {ntok} tok")
        return

    if cmd == "sweep":
        # per-layer: miss count / miss MiB / directs at a list of arena MiB
        on_dev = {int(x) for x in os.environ.get(
            "ON_DEV", "6,7,8,9,10,11,12,13,14,15").split(",")}
        sizes = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else \
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
        for l in sorted(nvis):
            if l in on_dev:
                continue
            row = []
            for mb in sizes:
                res = run(trace, gexp, lambda x, mb=mb: mb << 20, only={l})
                row.append(res[l][1:])
            pts = " ".join(f"{mb}:{m}/{b/1048576:.1f}" for (m, b, d), mb in zip(row, sizes))
            print(f"L{l:2d} vis={nvis[l]} {pts}")
        return

    if cmd == "alloc":
        budget = int(sys.argv[3]) << 20
        # streamed layers only (on_dev layers have no cache)
        on_dev = {int(x) for x in os.environ.get(
            "ON_DEV", "6,7,8,9,10,11,12,13,14,15").split(",")}
        streamed = [l for l in sorted(nvis) if l not in on_dev]
        # cap each layer at the arena that holds its 64 fattest experts
        cap = {}
        for l in streamed:
            c = sum(sorted((sum(gexp[l][e]) for e in range(NE)), reverse=True)[:64])
            cap[l] = c * GRAN
        # greedy from empty: marginal miss-bytes saved per added MiB
        alloc = {l: 0 for l in streamed}
        start = {l: 60 * max_slot[l] for l in streamed}  # ck=20 for reference
        # per-layer miss-byte curve memoized; each (l, size) sim runs once.
        cache = {}
        def mb_of(l, bytes_):
            key = (l, bytes_)
            if key not in cache:
                res = run(trace, gexp, lambda x: bytes_, only={l})
                cache[key] = res[l][2]
            return cache[key]
        cur_mb = {l: mb_of(l, alloc[l]) for l in streamed}
        tot = sum(alloc.values())
        step = 4 << 20
        while True:
            best_gain, best_l, best_v = 0.0, None, 0
            for l in streamed:
                if alloc[l] + step > cap[l] or tot + step > budget:
                    continue
                g = (cur_mb[l] - mb_of(l, alloc[l] + step)) / step
                if g > best_gain:
                    best_gain, best_l, best_v = g, l, alloc[l] + step
            if best_l is None:
                break
            cur_mb[best_l] = mb_of(best_l, best_v)
            tot += best_v - alloc[best_l]
            alloc[best_l] = best_v
        # report vs uniform ck=20
        res = run(trace, gexp, lambda l: alloc.get(l, 0))
        res0 = run(trace, gexp, lambda l: 60 * max_slot[l])
        tb = sum(v[2] for v in res.values())
        tm = sum(v[1] for v in res.values())
        tb0 = sum(v[2] for v in res0.values())
        tm0 = sum(v[1] for v in res0.values())
        print(f"# budget={budget>>20}MiB used={sum(alloc.values())>>20}MiB "
              f"miss={tm} missMiB={tb/1048576:.1f} "
              f"| uniform ck=20 ({sum(start.values())>>20}MiB): miss={tm0} missMiB={tb0/1048576:.1f}")
        for l in sorted(alloc):
            h, m, b, d = res[l]
            m0, b0 = res0[l][1], res0[l][2]
            print(f"L{l:2d} alloc={alloc[l]>>20:4d}MiB (unif {start[l]>>20:4d}) "
                  f"cap={cap[l]>>20:4d} miss={m:4d}(was {m0:4d}) "
                  f"missMiB={b/1048576:6.1f}(was {b0/1048576:6.1f}) direct={d}")
        return

    print(__doc__)
    sys.exit(1)


if __name__ == "__main__":
    main()
