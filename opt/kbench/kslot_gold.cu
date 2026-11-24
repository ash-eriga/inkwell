// gold test: inkwell_exl3_kslot_resolve (serial) vs _w (warp) must produce
// byte-identical outputs and persistent state for any input. Randomized
// state + id sequences, state carried across many calls per sequence.
// Covers: hits, prefetch-bank hits, misses, free-slot alloc, LFU evictions
// (incl. cnt == INT_MAX tie edge), DIRECT + rollback, kmax 24..80 (>64 takes
// the serial fallback inside _w), nslots 1..64 (incl. pnexp), key_base,
// force_direct, adversarial (overlapping/unsorted-len) free lists.
// Every mutable word (tag, soff, cnt, fl incl. stale tail, map, miss,
// pf_hit, hm) is compared exactly after each call.
// build+run: bash opt/kbench/build.sh kslot_gold && opt/kbench/kslot_gold
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint32_t rs_ = 0x9e3779b9u;
static uint32_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 17; rs_ ^= rs_ << 5; return rs_; }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// ---- state layout (u32 words) ----
#define TAGN 80
#define CNTN 8192
#define GENN 16384
#define FLW 513
static const int OFF_FL   = 0;
static const int OFF_TAG  = OFF_FL + FLW;            // 513
static const int OFF_SOFF = OFF_TAG + TAGN + 1;      // 594 (u64-aligned)
static const int OFF_CNT  = OFF_SOFF + 480;          // 1074
static const int OFF_MAP  = OFF_CNT + CNTN;          // 9266
static const int OFF_MISS = OFF_MAP + 64;            // 9330
static const int OFF_PF   = OFF_MISS + 64;           // 9394
static const int OFF_HM   = OFF_PF + 2;              // 9396
static const int STW      = OFF_HM + 2;              // 9398

struct State {
    int* fl;   int* tag; uint64_t* soff; int* cnt;
    int* map;  int* miss; int* pf_hit; int* hm;
};
static State st_of(uint32_t* b) {
    State s;
    s.fl    = (int*)(b + OFF_FL);
    s.tag   = (int*)(b + OFF_TAG);
    s.soff  = (uint64_t*)(b + OFF_SOFF);
    s.cnt   = (int*)(b + OFF_CNT);
    s.map   = (int*)(b + OFF_MAP);
    s.miss  = (int*)(b + OFF_MISS);
    s.pf_hit= (int*)(b + OFF_PF);
    s.hm    = (int*)(b + OFF_HM);
    return s;
}

__global__ void cmp_state(const uint32_t* a, const uint32_t* b, int n,
                          uint32_t* slot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i])
        atomicMin(slot, 0x80000000u | (unsigned)i);
}

// ---- host-side free list with the same semantics as KS_* ----
struct HFl { int n; int b[600], l[600]; };
static int hfit(HFl& f, int need) {
    for (int q = 0; q < f.n; ++q) if (f.l[q] >= need) return q;
    return -1;
}
static int halloc(HFl& f, int need) {
    const int i = hfit(f, need);
    if (i < 0) return -1;
    const int base = f.b[i];
    f.b[i] += need; f.l[i] -= need;
    if (f.l[i] == 0) {
        for (int q = i; q + 1 < f.n; ++q) { f.b[q] = f.b[q+1]; f.l[q] = f.l[q+1]; }
        --f.n;
    }
    return base;
}
static void hfree(HFl& f, int base, int len) {
    int q = 0; while (q < f.n && f.b[q] < base) ++q;
    for (int r = f.n; r > q; --r) { f.b[r] = f.b[r-1]; f.l[r] = f.l[r-1]; }
    f.b[q] = base; f.l[q] = len; ++f.n;
    if (q > 0 && f.b[q-1] + f.l[q-1] == f.b[q]) {
        f.l[q-1] += f.l[q];
        for (int r = q; r + 1 < f.n; ++r) { f.b[r] = f.b[r+1]; f.l[r] = f.l[r+1]; }
        --f.n; --q;
    }
    if (q + 1 < f.n && f.b[q] + f.l[q] == f.b[q+1]) {
        f.l[q] += f.l[q+1];
        for (int r = q + 1; r + 1 < f.n; ++r) { f.b[r] = f.b[r+1]; f.l[r] = f.l[r+1]; }
        --f.n;
    }
}

struct Seq {
    int kmax, key_base, arena, nslots_arg, use_pnexp;
    bool adversarial;
};

static void gen_state(const Seq& sc, const int* gexp, std::vector<uint32_t>& hw,
                      std::vector<int>& resident_keys) {
    State s = st_of(hw.data());
    memset(hw.data(), 0, hw.size() * 4);
    resident_keys.clear();
    for (int i = 0; i < CNTN; ++i) {
        const uint32_t r = rnd();
        if (r % 97 == 0) s.cnt[i] = 0x7fffffff;          // INT_MAX edge
        else if (r % 89 == 0) s.cnt[i] = 0x7ffffffe;     // near-edge
        else s.cnt[i] = (int)(r % 200);
    }
    if (!sc.adversarial) {
        // consistent: carve the arena into resident extents + a sorted fl
        HFl f; f.n = 1; f.b[0] = 0; f.l[0] = sc.arena;
        for (int j = 0; j < sc.kmax; ++j) {
            if (rnd() % 5 == 0) { s.tag[j] = -1; continue; }
            const int key = sc.key_base + rint(0, 127);
            int got[3] = {0, 0, 0}, ng = 0;
            bool ok = true;
            for (int p = 0; p < 3; ++p) {
                const int base = halloc(f, gexp[key * 3 + p]);
                if (base < 0) { ok = false; break; }
                got[ng++] = base;
            }
            if (!ok) {
                while (ng > 0) { --ng; hfree(f, got[ng], gexp[key * 3 + ng]); }
                s.tag[j] = -1;
                continue;
            }
            s.tag[j] = key;
            resident_keys.push_back(key);
            for (int p = 0; p < 3; ++p)
                s.soff[j * 3 + p] = (uint64_t)got[p] * KSLOT_GRAN;
        }
        s.fl[0] = f.n;
        for (int i = 0; i < f.n; ++i) {
            s.fl[1 + i] = f.b[i];
            s.fl[1 + KSLOT_FLMAX + i] = f.l[i];
        }
        // stale words beyond nfl: leave deterministic garbage
        for (int i = f.n; i < 2 * KSLOT_FLMAX + 1; ++i)
            if (1 + i < FLW) s.fl[1 + i] = (int)rnd();
    } else {
        // adversarial: sorted bases but arbitrary lens (overlap allowed),
        // random tags/soff — still fully deterministic for both kernels.
        const int nfl = rint(10, 60);
        std::vector<int> bs(nfl);
        for (auto& x : bs) x = rint(0, sc.arena);
        std::sort(bs.begin(), bs.end());
        s.fl[0] = nfl;
        for (int i = 0; i < nfl; ++i) {
            s.fl[1 + i] = bs[i];
            s.fl[1 + KSLOT_FLMAX + i] = rint(1, sc.arena / 2 + 64);
        }
        for (int i = nfl; i < 2 * KSLOT_FLMAX + 1 && 1 + i < FLW; ++i)
            s.fl[1 + i] = (int)rnd();
        for (int j = 0; j < sc.kmax; ++j) {
            if (rnd() % 10 < 3) { s.tag[j] = -1; continue; }
            const int key = sc.key_base + rint(0, 127);
            s.tag[j] = key;
            resident_keys.push_back(key);
            for (int p = 0; p < 3; ++p) {
                uint64_t v = (uint64_t)rint(0, sc.arena + 512) * KSLOT_GRAN;
                if (rnd() % 23 == 0) v += rint(1, KSLOT_GRAN - 1); // non-multiple
                s.soff[j * 3 + p] = v;
            }
        }
    }
}

int main(int argc, char** argv) {
    const long target_calls = argc > 1 ? atol(argv[1]) : 110000;
    const int BATCH = 256;
    uint32_t *dA, *dB, *dPrev, *d_flags;
    CK(cudaMalloc(&dA, STW * 4));
    CK(cudaMalloc(&dB, STW * 4));
    CK(cudaMalloc(&dPrev, STW * 4));
    CK(cudaMalloc(&d_flags, BATCH * 4));
    int *d_ids, *d_gexp, *d_ptag, *d_pready, *d_pnexp;
    CK(cudaMalloc(&d_ids, 64 * 4));
    CK(cudaMalloc(&d_gexp, GENN * 4));
    CK(cudaMalloc(&d_ptag, 64 * 4));
    CK(cudaMalloc(&d_pready, 4));
    CK(cudaMalloc(&d_pnexp, 4));

    std::vector<uint32_t> hw(STW);
    std::vector<int> gexp(GENN), ids(64), ptag(64), rk;
    // per-call input history for replay dumps
    std::vector<int> h_ids(BATCH * 64), h_ptag(BATCH * 64), h_meta(BATCH * 4);
    long calls = 0, pf_calls = 0, fd_calls = 0, fb_calls = 0, adv_calls = 0;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    while (calls < target_calls) {
        Seq sc;
        sc.kmax = (rnd() % 12 == 0) ? rint(65, TAGN) : rint(24, 64);
        const int kb_pick = rint(0, 9);
        sc.key_base = kb_pick < 8 ? 0 : (kb_pick < 9 ? 128 : (rnd() % 2 ? 512 : 3968));
        // adversarial states can grow nfl past FLMAX's designed bound — cap
        // kmax so initial nfl + max frees (3*kmax + 24) stays < 256
        sc.adversarial = rnd() % 7 == 0 && sc.kmax <= 32;
        sc.arena = rint(200, 40000);
        sc.use_pnexp = rnd() % 4 == 0;
        sc.nslots_arg = sc.use_pnexp ? 64 : 8;
        if (sc.adversarial) ++adv_calls;
        if (sc.kmax > 64) fb_calls += 1;

        // per-(key,proj) granule needs over the full key space
        for (int k2 = 0; k2 < GENN / 3; ++k2)
            for (int p = 0; p < 3; ++p) {
                const uint32_t r = rnd();
                int w = (r % 11 == 0) ? rint(37, 300) : rint(8, 36);
                if (r % 41 == 0) w = rint(1, 7);
                gexp[k2 * 3 + p] = 6 * w + 1;
            }
        CK(cudaMemcpy(d_gexp, gexp.data(), GENN * 4, cudaMemcpyHostToDevice));

        gen_state(sc, gexp.data(), hw, rk);
        CK(cudaMemcpy(dA, hw.data(), STW * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, hw.data(), STW * 4, cudaMemcpyHostToDevice));
        State sa = st_of((uint32_t*)dA), sb = st_of((uint32_t*)dB);

        const int seq_calls = rint(150, 400);
        for (int cs = 0; cs < seq_calls && calls < target_calls; ) {
            const int ncall = std::min((long)BATCH,
                                       std::min((long)seq_calls - cs, target_calls - calls));
            // snapshot batch-start state + inputs for replay on failure
            CK(cudaMemcpy(dPrev, dA, STW * 4, cudaMemcpyDeviceToDevice));
            CK(cudaMemsetAsync(d_flags, 0xFF, ncall * 4));
            const int batch_c0 = cs;
            for (int c = 0; c < ncall; ++c, ++cs, ++calls) {
                const int ns = sc.use_pnexp ? rint(1, 64)
                               : (rnd() % 5 == 0 ? rint(1, 16) : 8);
                for (int s2 = 0; s2 < 64; ++s2) {
                    const uint32_t r = rnd();
                    if (s2 > 0 && r % 10 == 0) { ids[s2] = ids[s2 - 1]; continue; }
                    if (!rk.empty() && r % 10 < 5)
                        ids[s2] = rk[r % rk.size()] - sc.key_base;
                    else
                        ids[s2] = rint(0, 127);
                }
                const int pf_on = rnd() % 5 == 0;
                const int pready = pf_on ? rint(1, 3) : 0;
                for (int p2 = 0; p2 < 64; ++p2)
                    ptag[p2] = (pf_on && rnd() % 2 == 0)
                        ? ids[rnd() % (uint32_t)ns] + sc.key_base : -1;
                const int force_direct = rnd() % 9 == 0;
                pf_calls += pf_on; fd_calls += force_direct;
                memcpy(&h_ids[c * 64], ids.data(), 256);
                memcpy(&h_ptag[c * 64], ptag.data(), 256);
                h_meta[c * 4 + 0] = ns; h_meta[c * 4 + 1] = pready;
                h_meta[c * 4 + 2] = force_direct; h_meta[c * 4 + 3] = pf_on;
                CK(cudaMemcpyAsync(d_ids, ids.data(), 64 * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(d_ptag, ptag.data(), 64 * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(d_pready, &pready, 4, cudaMemcpyHostToDevice));
                if (sc.use_pnexp)
                    CK(cudaMemcpyAsync(d_pnexp, &ns, 4, cudaMemcpyHostToDevice));

                const int* pnx = sc.use_pnexp ? d_pnexp : nullptr;
                const int* pt  = pf_on ? d_ptag : nullptr;
                const int* pr  = pf_on ? d_pready : nullptr;
                const bool nohm = rnd() % 10 == 0;
                inkwell_exl3_kslot_resolve<<<1, 32>>>(
                    d_ids, sa.tag, sa.soff, sa.map, sa.miss, sa.cnt,
                    sc.nslots_arg, sc.kmax, d_gexp, sa.fl, pt, pr, sa.pf_hit,
                    pnx, sc.key_base, force_direct, nohm ? nullptr : sa.hm);
                CK(cudaGetLastError());
                inkwell_exl3_kslot_resolve_w<<<1, 32>>>(
                    d_ids, sb.tag, sb.soff, sb.map, sb.miss, sb.cnt,
                    sc.nslots_arg, sc.kmax, d_gexp, sb.fl, pt, pr, sb.pf_hit,
                    pnx, sc.key_base, force_direct, nohm ? nullptr : sb.hm);
                CK(cudaGetLastError());
                cmp_state<<<(STW + 255) / 256, 256>>>(dA, dB, STW, d_flags + c);
                CK(cudaGetLastError());
            }
            std::vector<uint32_t> flags(ncall);
            CK(cudaMemcpy(flags.data(), d_flags, ncall * 4, cudaMemcpyDeviceToHost));
            long bad = -1; uint32_t w = 0;
            for (int c = 0; c < ncall; ++c)
                if (flags[c] != 0xffffffffu) { bad = c; w = flags[c] & 0x7fffffffu; break; }
            if (bad >= 0) {
                std::vector<uint32_t> ha(STW), hb(STW);
                CK(cudaMemcpy(ha.data(), dA, STW * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(hb.data(), dB, STW * 4, cudaMemcpyDeviceToHost));
                const long call = calls - ncall + bad;
                printf("MISMATCH call %ld (batch slot %ld) word %u: A=%08x B=%08x\n",
                       call, bad, w, ha[w], hb[w]);
                const char* reg = w < (uint32_t)OFF_TAG ? "fl" :
                    w < (uint32_t)OFF_SOFF ? "tag" :
                    w < (uint32_t)OFF_CNT ? "soff" :
                    w < (uint32_t)OFF_MAP ? "cnt" :
                    w < (uint32_t)OFF_MISS ? "map" :
                    w < (uint32_t)OFF_PF ? "miss" :
                    w < (uint32_t)OFF_HM ? "pf_hit" : "hm";
                printf("region=%s off=%u kmax=%d kb=%d arena=%d adv=%d "
                       "pnexp=%d nsarg=%d\n", reg,
                       w - (w < (uint32_t)OFF_TAG ? OFF_FL :
                            w < (uint32_t)OFF_SOFF ? OFF_TAG :
                            w < (uint32_t)OFF_CNT ? OFF_SOFF :
                            w < (uint32_t)OFF_MAP ? OFF_CNT :
                            w < (uint32_t)OFF_MISS ? OFF_MAP :
                            w < (uint32_t)OFF_PF ? OFF_MISS :
                            w < (uint32_t)OFF_HM ? OFF_PF : OFF_HM),
                       sc.kmax, sc.key_base, sc.arena, sc.adversarial,
                       sc.use_pnexp, sc.nslots_arg);
                printf("call meta: ns=%d pready=%d fd=%d pf=%d\n",
                       h_meta[bad * 4], h_meta[bad * 4 + 1],
                       h_meta[bad * 4 + 2], h_meta[bad * 4 + 3]);
                int* fa = (int*)(ha.data() + OFF_FL);
                int* fb = (int*)(hb.data() + OFF_FL);
                printf("  nfl A=%d B=%d\n", fa[0], fb[0]);
                for (int i = 0; i < std::max(fa[0], fb[0]) + 2 && i < 40; ++i)
                    printf("  fl[%d] A=(%d,%d) B=(%d,%d)\n", i,
                           fa[1 + i], fa[1 + KSLOT_FLMAX + i],
                           fb[1 + i], fb[1 + KSLOT_FLMAX + i]);
                for (int j = 0; j < sc.kmax; ++j)
                    if (((int*)(ha.data() + OFF_TAG))[j] != ((int*)(hb.data() + OFF_TAG))[j])
                        printf("  tag[%d] A=%d B=%d\n", j,
                               ((int*)(ha.data() + OFF_TAG))[j],
                               ((int*)(hb.data() + OFF_TAG))[j]);
                for (int j = 0; j < sc.kmax * 3; ++j)
                    if (((uint64_t*)(ha.data() + OFF_SOFF))[j] !=
                        ((uint64_t*)(hb.data() + OFF_SOFF))[j])
                        printf("  soff[%d] A=%llu B=%llu\n", j,
                               (unsigned long long)((uint64_t*)(ha.data() + OFF_SOFF))[j],
                               (unsigned long long)((uint64_t*)(hb.data() + OFF_SOFF))[j]);
                std::vector<uint32_t> hp(STW);
                CK(cudaMemcpy(hp.data(), dPrev, STW * 4, cudaMemcpyDeviceToHost));
                FILE* f = fopen("/tmp/kslot_dump.bin", "wb");
                int hdr[8] = {sc.kmax, (int)bad + 1, sc.key_base, sc.arena,
                              sc.nslots_arg, sc.use_pnexp, 0, 0};
                fwrite(hdr, 4, 8, f);
                fwrite(hp.data(), 4, STW, f);
                for (int c = 0; c <= bad; ++c) {
                    fwrite(&h_meta[c * 4], 4, 4, f);
                    fwrite(&h_ids[c * 64], 4, 64, f);
                    fwrite(&h_ptag[c * 64], 4, 64, f);
                }
                fwrite(gexp.data(), 4, GENN, f);
                fclose(f);
                printf("  dumped /tmp/kslot_dump.bin (batch-start state + "
                       "%ld calls' inputs)\n", bad + 1);
                (void)batch_c0;
                return 1;
            }
        }
        if (calls % 10000 < 400)
            printf("... %ld calls\n", calls);
    }

    // per-kernel timing on the last state (idempotent-ish; state keeps
    // evolving but stays valid)
    State sa = st_of((uint32_t*)dA), sb = st_of((uint32_t*)dB);
    for (int s2 = 0; s2 < 64; ++s2) ids[s2] = rint(0, 127);
    CK(cudaMemcpy(d_ids, ids.data(), 64 * 4, cudaMemcpyHostToDevice));
    const int REP = 400;
    float ms;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < REP; ++i)
        inkwell_exl3_kslot_resolve<<<1, 32>>>(d_ids, sa.tag, sa.soff, sa.map,
            sa.miss, sa.cnt, 8, 40, d_gexp, sa.fl, nullptr, nullptr,
            sa.pf_hit, nullptr, 0, 0, sa.hm);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double t_old = ms * 1e3 / REP;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < REP; ++i)
        inkwell_exl3_kslot_resolve_w<<<1, 32>>>(d_ids, sb.tag, sb.soff, sb.map,
            sb.miss, sb.cnt, 8, 40, d_gexp, sb.fl, nullptr, nullptr,
            sb.pf_hit, nullptr, 0, 0, sb.hm);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double t_new = ms * 1e3 / REP;

    printf("OK %ld calls bit-identical\n", calls);
    printf("  mix: pf-calls %ld fd-calls %ld adv-seqs %ld fb-seqs(kmax>64) %ld\n",
           pf_calls, fd_calls, adv_calls, fb_calls);
    printf("  kernel time: serial %.2f us/launch, warp %.2f us/launch\n",
           t_old, t_new);
    return 0;
}
