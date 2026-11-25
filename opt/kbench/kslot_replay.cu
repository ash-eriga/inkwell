// replay a dumped kslot_gold mismatch: /tmp/kslot_dump.bin has
// hdr[8]={kmax,ns,key_base,arena,nslots_arg,use_pnexp,force_direct,pready},
// state[STW] words, ids[64], ptag[64], gexp[GENN].
// Runs serial vs warp kernel once on the pre-state and diffs, plus a
// host serial reference that prints every alloc/free/evict op.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

#define TAGN 80
#define CNTN 8192
#define GENN 16384
#define FLW 513
static const int OFF_FL = 0, OFF_TAG = 513, OFF_SOFF = 594, OFF_CNT = 1074,
                 OFF_MAP = 9266, OFF_MISS = 9330, OFF_PF = 9394, OFF_HM = 9396,
                 STW = 9398;

// host serial reference — a direct transcription of kslot_resolve_serial
// with op tracing
struct HS {
    int kmax;
    int tag[TAGN];
    uint64_t soff[3 * TAGN];
    int* cnt;
    const int* gexp;
    int fl[FLW];
    int map[64], miss[64], pf_hit[2], hm[2];
};

static void host_serial(HS& h, const int* ids, int nslots,
                        const int* ptag, int pready, int key_base,
                        int force_direct, bool trace) {
    int* tag = h.tag;
    int* cnt = h.cnt;
    int* map = h.map;
    int* miss = h.miss;
    int* fl = h.fl;
    const int* gexp = h.gexp;
    const int kmax = h.kmax;
    const int pf = pready;
    unsigned long long claimed = 0;
    for (int s = 0; s < nslots; ++s) {
        const int e = ids[s] + key_base;
        ++cnt[e];
        int v = -1;
        for (int j = 0; j < kmax; ++j) if (tag[j] == e) { v = j; break; }
        if (v < 0 && pf) {
            for (int p2 = 0; p2 < nslots; ++p2)
                if (ptag[p2] == e) { v = EXL3_PF_BASE + p2; ++h.pf_hit[0]; break; }
        }
        if (v >= 0) {
            miss[s] = 0; map[s] = v;
            if (v < kmax) claimed |= 1ULL << v;
            ++h.hm[0];
        } else { miss[s] = 1; map[s] = -1; ++h.hm[1]; }
    }
    int nfl = fl[0];
    int* fb = fl + 1;
    int* fn = fl + 1 + KSLOT_FLMAX;
    auto FIT = [&](int need) {
        for (int q = 0; q < nfl; ++q) if (fn[q] >= need) return q;
        return -1;
    };
    auto ALLOC = [&](int idx, int need) {
        const int base = fb[idx];
        fb[idx] += need; fn[idx] -= need;
        if (fn[idx] == 0) {
            for (int q = idx; q + 1 < nfl; ++q) { fb[q] = fb[q+1]; fn[q] = fn[q+1]; }
            --nfl;
        }
        return base;
    };
    auto FREE = [&](int base, int len) {
        if (trace) printf("    FREE (%d,%d) nfl=%d\n", base, len, nfl);
        int q = 0; while (q < nfl && fb[q] < base) ++q;
        for (int r = nfl; r > q; --r) { fb[r] = fb[r-1]; fn[r] = fn[r-1]; }
        fb[q] = base; fn[q] = len; ++nfl;
        if (q > 0 && fb[q-1] + fn[q-1] == fb[q]) {
            fn[q-1] += fn[q];
            for (int r = q; r + 1 < nfl; ++r) { fb[r] = fb[r+1]; fn[r] = fn[r+1]; }
            --nfl; --q;
        }
        if (q + 1 < nfl && fb[q] + fn[q] == fb[q+1]) {
            fn[q] += fn[q+1];
            for (int r = q + 1; r + 1 < nfl; ++r) { fb[r] = fb[r+1]; fn[r] = fn[r+1]; }
            --nfl;
        }
        if (trace) {
            printf("      ->");
            for (int i = 0; i < nfl; ++i) printf(" (%d,%d)", fb[i], fn[i]);
            printf("\n");
        }
    };
    for (int s = 0; s < nslots; ++s) {
        if (!miss[s]) continue;
        const int e = ids[s] + key_base;
        if (trace) printf("  miss s=%d e=%d\n", s, e);
        int v = -1;
        for (int j = 0; j < kmax; ++j) if (tag[j] < 0) { v = j; break; }
        int got[3]; int ng = 0;
        for (;;) {
            while (v >= 0 && ng < 3) {
                const int need = gexp[ids[s] * 3 + ng];
                int fi = FIT(need);
                if (fi < 0) break;
                got[ng++] = ALLOC(fi, need);
                if (trace) printf("    ALLOC need=%d fi=%d base=%d\n", need, fi, got[ng-1]);
            }
            if (v >= 0 && ng == 3) break;
            int best = -1, bestc = 0x7fffffff;
            for (int j = 0; j < kmax; ++j) {
                if (claimed & (1ULL << j)) continue;
                const int t2 = tag[j];
                if (t2 < 0) continue;
                const int c = cnt[t2];
                if (c < bestc) { bestc = c; best = j; }
            }
            if (best < 0) break;
            const int tv = tag[best];
            if (trace) printf("    EVICT slot %d key=%d cnt=%d\n", best, tv, cnt[tv]);
            for (int p = 0; p < 3; ++p)
                FREE((int)(h.soff[best * 3 + p] / KSLOT_GRAN), gexp[tv * 3 + p]);
            tag[best] = -1;
            if (v < 0) v = best;
        }
        if (v < 0 || ng < 3 || force_direct) {
            if (trace) printf("    DIRECT v=%d ng=%d\n", v, ng);
            while (ng > 0) { --ng; FREE(got[ng], gexp[ids[s] * 3 + ng]); }
            map[s] = EXL3_DIRECT;
            continue;
        }
        claimed |= 1ULL << v;
        tag[v] = e;
        for (int p = 0; p < 3; ++p) h.soff[v * 3 + p] = (uint64_t)got[p] * KSLOT_GRAN;
        map[s] = v;
        ++h.pf_hit[1];
        if (trace) printf("    -> slot %d soff=(%d,%d,%d)\n", v, got[0], got[1], got[2]);
    }
    fl[0] = nfl;
}

int main() {
    FILE* f = fopen("/tmp/kslot_dump.bin", "rb");
    if (!f) { printf("no dump\n"); return 1; }
    int hdr[8];
    std::vector<uint32_t> hst(STW), ids(64), ptag(64), gexp(GENN);
    if (fread(hdr, 4, 8, f) != 8 || fread(hst.data(), 4, STW, f) != STW ||
        fread(ids.data(), 4, 64, f) != 64 || fread(ptag.data(), 4, 64, f) != 64 ||
        fread(gexp.data(), 4, GENN, f) != GENN) { printf("short read\n"); return 1; }
    fclose(f);
    const int kmax = hdr[0], ns = hdr[1], kb = hdr[2], nsarg = hdr[4],
              use_pnexp = hdr[5], fd = hdr[6], pready = hdr[7];

    // host serial ref
    HS h; memset(&h, 0, sizeof h);
    h.kmax = kmax;
    h.cnt = new int[CNTN];
    h.gexp = (const int*)gexp.data();
    memcpy(h.fl, hst.data() + OFF_FL, FLW * 4);
    memcpy(h.tag, hst.data() + OFF_TAG, TAGN * 4);
    memcpy(h.soff, hst.data() + OFF_SOFF, 3 * TAGN * 8);
    memcpy(h.cnt, hst.data() + OFF_CNT, CNTN * 4);
    host_serial(h, (const int*)ids.data(), ns, (const int*)ptag.data(),
                pready, kb, fd, true);
    printf("host nfl=%d fl[1]=%d\n", h.fl[0], h.fl[1]);

    // device kernels on the dumped pre-state
    uint32_t *dA, *dB;
    int *d_ids, *d_gexp, *d_ptag, *d_pready, *d_pnexp;
    CK(cudaMalloc(&dA, STW * 4)); CK(cudaMalloc(&dB, STW * 4));
    CK(cudaMalloc(&d_ids, 256)); CK(cudaMalloc(&d_gexp, GENN * 4));
    CK(cudaMalloc(&d_ptag, 256)); CK(cudaMalloc(&d_pready, 4));
    CK(cudaMalloc(&d_pnexp, 4));
    CK(cudaMemcpy(dA, hst.data(), STW * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hst.data(), STW * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ids, ids.data(), 256, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gexp, gexp.data(), GENN * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ptag, ptag.data(), 256, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pready, &pready, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pnexp, &ns, 4, cudaMemcpyHostToDevice));
    const int* pnx = use_pnexp ? d_pnexp : nullptr;
    const int* pt = pready ? d_ptag : nullptr;
    const int* pr = pready ? d_pready : nullptr;
    int *tA = (int*)(dA + OFF_TAG), *tB = (int*)(dB + OFF_TAG);
    uint64_t *oA = (uint64_t*)(dA + OFF_SOFF), *oB = (uint64_t*)(dB + OFF_SOFF);
    int *mA = (int*)(dA + OFF_MAP), *mB = (int*)(dB + OFF_MAP);
    int *xA = (int*)(dA + OFF_MISS), *xB = (int*)(dB + OFF_MISS);
    int *cA = (int*)(dA + OFF_CNT), *cB = (int*)(dB + OFF_CNT);
    int *fA = (int*)(dA + OFF_FL), *fB = (int*)(dB + OFF_FL);
    int *pA = (int*)(dA + OFF_PF), *pB = (int*)(dB + OFF_PF);
    int *hA = (int*)(dA + OFF_HM), *hB = (int*)(dB + OFF_HM);
    inkwell_exl3_kslot_resolve<<<1, 32>>>(d_ids, tA, oA, mA, xA, cA,
        nsarg, kmax, d_gexp, fA, pt, pr, pA, pnx, kb, fd, hA);
    inkwell_exl3_kslot_resolve_w<<<1, 32>>>(d_ids, tB, oB, mB, xB, cB,
        nsarg, kmax, d_gexp, fB, pt, pr, pB, pnx, kb, fd, hB);
    CK(cudaDeviceSynchronize());
    std::vector<uint32_t> ra(STW), rb(STW);
    CK(cudaMemcpy(ra.data(), dA, STW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rb.data(), dB, STW * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < STW; ++i)
        if (ra[i] != rb[i]) printf("diff word %d A=%08x B=%08x\n", i, ra[i], rb[i]);
    const int* fa = (const int*)(ra.data() + OFF_FL);
    const int* fb2 = (const int*)(rb.data() + OFF_FL);
    printf("devA nfl=%d fl[1]=%d devB nfl=%d fl[1]=%d\n",
           fa[0], fa[1], fb2[0], fb2[1]);
    return 0;
}
