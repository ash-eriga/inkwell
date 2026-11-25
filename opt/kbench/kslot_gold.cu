// gold test: inkwell_exl3_kslot_resolve (serial) vs _w (warp) must produce
// byte-identical outputs and persistent state after every call. Random valid
// states plus the real Inkling routing trace and real expert-size table.
// Covers hits, prefetch-bank hits, LFU evictions, arena-full eviction,
// nothing-evictable DIRECT, force_direct rollback, pnexp, hm == nullptr,
// and the kmax > 64 serial fallback in _w.
// build+run: bash opt/kbench/build.sh kslot_gold && opt/kbench/kslot_gold
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../../crates/inkwell-cuda/src/kernels.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint32_t rs_ = 0x9e3779b9u;
static uint32_t rnd() { rs_ ^= rs_ << 13; rs_ ^= rs_ >> 17; rs_ ^= rs_ << 5; return rs_; }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

#define TAGN 80
#define CNTN 8192
#define GENN 16384
#define FLW 513
static const int OFF_FL   = 0;
static const int OFF_TAG  = OFF_FL + FLW;
static const int OFF_SOFF = OFF_TAG + TAGN + 1;
static const int OFF_CNT  = OFF_SOFF + 480;
static const int OFF_MAP  = OFF_CNT + CNTN;
static const int OFF_MISS = OFF_MAP + 64;
static const int OFF_PF   = OFF_MISS + 64;
static const int OFF_HM   = OFF_PF + 2;
static const int STW      = OFF_HM + 2;

struct State {
    int* fl;
    int* tag;
    uint64_t* soff;
    int* cnt;
    int* map;
    int* miss;
    int* pf_hit;
    int* hm;
};

static State st_of(uint32_t* b) {
    State s;
    s.fl     = (int*)(b + OFF_FL);
    s.tag    = (int*)(b + OFF_TAG);
    s.soff   = (uint64_t*)(b + OFF_SOFF);
    s.cnt    = (int*)(b + OFF_CNT);
    s.map    = (int*)(b + OFF_MAP);
    s.miss   = (int*)(b + OFF_MISS);
    s.pf_hit = (int*)(b + OFF_PF);
    s.hm     = (int*)(b + OFF_HM);
    return s;
}

__global__ void cmp_state(const uint32_t* a, const uint32_t* b, int n,
                          uint32_t* slot) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i])
        atomicMin(slot, 0x80000000u | (unsigned)i);
}

struct HFl { int n; int b[600], l[600]; };
static int hfit(HFl& f, int need) {
    for (int q = 0; q < f.n; ++q) if (f.l[q] >= need) return q;
    return -1;
}
static int halloc(HFl& f, int need) {
    const int i = hfit(f, need);
    if (i < 0) return -1;
    const int base = f.b[i];
    f.b[i] += need;
    f.l[i] -= need;
    if (f.l[i] == 0) {
        for (int q = i; q + 1 < f.n; ++q) { f.b[q] = f.b[q + 1]; f.l[q] = f.l[q + 1]; }
        --f.n;
    }
    return base;
}
static void hfree(HFl& f, int base, int len) {
    int q = 0;
    while (q < f.n && f.b[q] < base) ++q;
    for (int r = f.n; r > q; --r) { f.b[r] = f.b[r - 1]; f.l[r] = f.l[r - 1]; }
    f.b[q] = base;
    f.l[q] = len;
    ++f.n;
    if (q > 0 && f.b[q - 1] + f.l[q - 1] == f.b[q]) {
        f.l[q - 1] += f.l[q];
        for (int r = q; r + 1 < f.n; ++r) { f.b[r] = f.b[r + 1]; f.l[r] = f.l[r + 1]; }
        --f.n;
        --q;
    }
    if (q + 1 < f.n && f.b[q] + f.l[q] == f.b[q + 1]) {
        f.l[q] += f.l[q + 1];
        for (int r = q + 1; r + 1 < f.n; ++r) { f.b[r] = f.b[r + 1]; f.l[r] = f.l[r + 1]; }
        --f.n;
    }
}

struct Ctx {
    std::vector<uint32_t> pre;
    uint32_t* dA;
    uint32_t* dB;
    uint32_t* dflag;
    int* d_ids;
    int* d_gexp;
    int* d_ptag;
    int* d_pready;
    int* d_pnexp;
};

struct CallInput {
    std::array<int, 64> ids{};
    std::array<int, 64> ptag{};
    int ns = 8;
    int ns_arg = 8;
    int pready = 0;
    int force_direct = 0;
    bool pf_on = false;
    bool use_pnexp = false;
    bool nohm = false;
};

static void sync_stage(const char* stage, long call) {
    cudaError_t e = cudaStreamSynchronize(0);
    if (e != cudaSuccess) {
        printf("CUDA error %s in %s at call %ld\n", cudaGetErrorString(e), stage, call);
        exit(1);
    }
}

static void print_diff(Ctx& c, uint32_t w, long call, const CallInput& in,
                       int kmax, int key_base, int arena) {
    std::vector<uint32_t> ha(STW), hb(STW), ge(GENN);
    CK(cudaMemcpy(ha.data(), c.dA, STW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), c.dB, STW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ge.data(), c.d_gexp, GENN * 4, cudaMemcpyDeviceToHost));
    FILE* df = fopen("/tmp/kslot_dump.bin", "wb");
    if (df) {
        int hdr[8] = {kmax, in.ns, key_base, arena, in.ns_arg,
                      (int)in.use_pnexp, in.force_direct, in.pready};
        fwrite(hdr, 4, 8, df);
        fwrite(c.pre.data(), 4, STW, df);
        fwrite(in.ids.data(), 4, 64, df);
        fwrite(in.ptag.data(), 4, 64, df);
        fwrite(ge.data(), 4, GENN, df);
        fclose(df);
    }
    const char* reg = w < (uint32_t)OFF_TAG ? "fl" :
        w < (uint32_t)OFF_SOFF ? "tag" :
        w < (uint32_t)OFF_CNT ? "soff" :
        w < (uint32_t)OFF_MAP ? "cnt" :
        w < (uint32_t)OFF_MISS ? "map" :
        w < (uint32_t)OFF_PF ? "miss" :
        w < (uint32_t)OFF_HM ? "pf_hit" : "hm";
    printf("MISMATCH call %ld word %u: A=%08x B=%08x region=%s\n",
           call, w, ha[w], hb[w], reg);
    printf("  kmax=%d key_base=%d arena=%d ns=%d ns_arg=%d pready=%d fd=%d pf=%d pnexp=%d nohm=%d\n",
           kmax, key_base, arena, in.ns, in.ns_arg, in.pready, in.force_direct,
           (int)in.pf_on, (int)in.use_pnexp, (int)in.nohm);
    int* fa = (int*)(ha.data() + OFF_FL);
    int* fb = (int*)(hb.data() + OFF_FL);
    printf("  nfl A=%d B=%d\n", fa[0], fb[0]);
    for (int i = 0; i < std::max(fa[0], fb[0]) + 2 && i < 32; ++i)
        printf("  fl[%d] A=(%d,%d) B=(%d,%d)\n", i,
               fa[1 + i], fa[1 + KSLOT_FLMAX + i],
               fb[1 + i], fb[1 + KSLOT_FLMAX + i]);
    for (int j = 0; j < kmax; ++j) {
        const int ta = ((int*)(ha.data() + OFF_TAG))[j];
        const int tb = ((int*)(hb.data() + OFF_TAG))[j];
        if (ta != tb) printf("  tag[%d] A=%d B=%d\n", j, ta, tb);
    }
    for (int j = 0; j < kmax * 3; ++j) {
        const uint64_t oa = ((uint64_t*)(ha.data() + OFF_SOFF))[j];
        const uint64_t ob = ((uint64_t*)(hb.data() + OFF_SOFF))[j];
        if (oa != ob) printf("  soff[%d] A=%llu B=%llu\n", j,
                             (unsigned long long)oa, (unsigned long long)ob);
    }
}

static bool run_one(Ctx& c, const CallInput& in, int kmax, int key_base,
                    int arena, long call, bool check_sync) {
    State sa = st_of(c.dA), sb = st_of(c.dB);
    c.pre.resize(STW);
    CK(cudaMemcpy(c.pre.data(), c.dA, STW * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c.d_ids, in.ids.data(), 64 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_ptag, in.ptag.data(), 64 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.d_pready, &in.pready, 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(c.dflag, 0xff, 4));

    int ns_dev = in.ns;
    if (in.use_pnexp) {
        ns_dev = 0;
        for (int i = 0; i < in.ns; ++i) {
            bool seen = false;
            for (int j = 0; j < i; ++j) seen |= in.ids[j] == in.ids[i];
            ns_dev += !seen;
        }
        CK(cudaMemcpy(c.d_pnexp, &ns_dev, 4, cudaMemcpyHostToDevice));
    }

    const int* pnx = in.use_pnexp ? c.d_pnexp : nullptr;
    const int* pt = in.pf_on ? c.d_ptag : nullptr;
    const int* pr = in.pf_on ? c.d_pready : nullptr;
    inkwell_exl3_kslot_resolve<<<1, 32>>>(
        c.d_ids, sa.tag, sa.soff, sa.map, sa.miss, sa.cnt,
        in.ns_arg, kmax, c.d_gexp, sa.fl, pt, pr, sa.pf_hit,
        pnx, key_base, in.force_direct, in.nohm ? nullptr : sa.hm);
    CK(cudaGetLastError());
    if (check_sync) sync_stage("serial", call);
    inkwell_exl3_kslot_resolve_w<<<1, 32>>>(
        c.d_ids, sb.tag, sb.soff, sb.map, sb.miss, sb.cnt,
        in.ns_arg, kmax, c.d_gexp, sb.fl, pt, pr, sb.pf_hit,
        pnx, key_base, in.force_direct, in.nohm ? nullptr : sb.hm);
    CK(cudaGetLastError());
    if (check_sync) sync_stage("warp", call);
    cmp_state<<<(STW + 255) / 256, 256>>>(c.dA, c.dB, STW, c.dflag);
    CK(cudaGetLastError());
    if (check_sync) sync_stage("compare", call);

    uint32_t flag = 0;
    CK(cudaMemcpy(&flag, c.dflag, 4, cudaMemcpyDeviceToHost));
    if (flag != 0xffffffffu) {
        print_diff(c, flag & 0x7fffffffu, call, in, kmax, key_base, arena);
        return false;
    }
    return true;
}

struct Seq {
    int kmax;
    int key_base;
    int arena;
    bool use_pnexp;
};

static void fill_stale_fl(int* fl, int n) {
    for (int i = n; i < KSLOT_FLMAX; ++i) {
        fl[1 + i] = (int)rnd();
        fl[1 + KSLOT_FLMAX + i] = (int)rnd();
    }
}

static void gen_state(const Seq& sc, const int* gexp, std::vector<uint32_t>& hw,
                      std::vector<int>& resident_keys) {
    State s = st_of(hw.data());
    memset(hw.data(), 0, hw.size() * 4);
    resident_keys.clear();
    for (int i = 0; i < TAGN; ++i) s.tag[i] = -1;
    for (int i = 0; i < CNTN; ++i) {
        const uint32_t r = rnd();
        if (r % 89 == 0) s.cnt[i] = 0x7ffffffe;
        else s.cnt[i] = (int)(r % 200);
    }
    std::vector<char> used(CNTN, 0);
    HFl f;
    f.n = 1;
    f.b[0] = 0;
    f.l[0] = sc.arena;
    for (int j = 0; j < sc.kmax; ++j) {
        if (rnd() % 5 == 0) continue;
        int key = sc.key_base + rint(0, 127);
        for (int t = 0; t < 8 && used[key]; ++t) key = sc.key_base + rint(0, 127);
        int got[3] = {0, 0, 0};
        int ng = 0;
        bool ok = true;
        for (int p = 0; p < 3; ++p) {
            const int base = halloc(f, gexp[key * 3 + p]);
            if (base < 0) { ok = false; break; }
            got[ng++] = base;
        }
        if (!ok) {
            while (ng > 0) { --ng; hfree(f, got[ng], gexp[key * 3 + ng]); }
            continue;
        }
        used[key] = 1;
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
    fill_stale_fl(s.fl, f.n);
}

static void upload_state(Ctx& c, const std::vector<uint32_t>& hw) {
    CK(cudaMemcpy(c.dA, hw.data(), STW * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.dB, hw.data(), STW * 4, cudaMemcpyHostToDevice));
}

static bool random_phase(Ctx& c, long target, long& calls, long& pf_calls,
                         long& fd_calls, long& nohm_calls, long& pnexp_calls,
                         long& fallback_calls) {
    std::vector<uint32_t> hw(STW);
    std::vector<int> gexp(GENN), rk;
    long done = 0;
    while (done < target) {
        Seq sc;
        sc.kmax = (rnd() % 12 == 0) ? rint(65, TAGN) : rint(24, 64);
        sc.key_base = 0;
        sc.arena = rint(200, 40000);
        sc.use_pnexp = rnd() % 4 == 0;

        for (int k2 = 0; k2 < GENN / 3; ++k2)
            for (int p = 0; p < 3; ++p) {
                const uint32_t r = rnd();
                int w = (r % 11 == 0) ? rint(37, 300) : rint(8, 36);
                if (r % 41 == 0) w = rint(1, 7);
                gexp[k2 * 3 + p] = 6 * w + 1;
            }
        CK(cudaMemcpy(c.d_gexp, gexp.data(), GENN * 4, cudaMemcpyHostToDevice));
        gen_state(sc, gexp.data(), hw, rk);
        upload_state(c, hw);

        const int seq_calls = rint(100, 220);
        for (int cs = 0; cs < seq_calls && done < target; ++cs, ++done, ++calls) {
            CallInput in;
            in.use_pnexp = sc.use_pnexp;
            in.ns = sc.use_pnexp ? rint(1, 64) : (rnd() % 5 == 0 ? rint(1, 16) : 8);
            in.ns_arg = sc.use_pnexp ? 64 : in.ns;
            for (int s2 = 0; s2 < 64; ++s2) {
                const uint32_t r = rnd();
                if (s2 > 0 && r % 10 == 0) { in.ids[s2] = in.ids[s2 - 1]; continue; }
                if (!rk.empty() && r % 10 < 5)
                    in.ids[s2] = rk[r % rk.size()] - sc.key_base;
                else
                    in.ids[s2] = rint(0, 127);
            }
            if (in.use_pnexp) {
                int u = 0;
                for (int i = 0; i < in.ns; ++i) {
                    bool seen = false;
                    for (int j = 0; j < u; ++j) seen |= in.ids[j] == in.ids[i];
                    if (!seen) in.ids[u++] = in.ids[i];
                }
                in.ns = u;
            }
            in.pf_on = rnd() % 5 == 0;
            in.pready = in.pf_on ? rint(1, 3) : 0;
            for (int p2 = 0; p2 < 64; ++p2)
                in.ptag[p2] = (in.pf_on && rnd() % 2 == 0)
                    ? in.ids[rnd() % (uint32_t)in.ns] + sc.key_base : -1;
            in.force_direct = rnd() % 9 == 0;
            in.nohm = rnd() % 10 == 0;
            pf_calls += in.pf_on;
            fd_calls += in.force_direct != 0;
            nohm_calls += in.nohm;
            pnexp_calls += in.use_pnexp;
            fallback_calls += sc.kmax > 64;
            if (!run_one(c, in, sc.kmax, sc.key_base, sc.arena, calls, false)) return false;
        }
    }
    return true;
}

static void edge_full_state(std::vector<uint32_t>& hw) {
    State s = st_of(hw.data());
    memset(hw.data(), 0, hw.size() * 4);
    for (int i = 0; i < TAGN; ++i) s.tag[i] = -1;
    for (int i = 0; i < CNTN; ++i) s.cnt[i] = 1;
    for (int j = 0; j < 8; ++j) {
        s.tag[j] = j;
        for (int p = 0; p < 3; ++p)
            s.soff[j * 3 + p] = (uint64_t)(j * 21 + p * 7) * KSLOT_GRAN;
    }
    s.fl[0] = 0;
    fill_stale_fl(s.fl, 0);
}

static void edge_empty_state(std::vector<uint32_t>& hw, int arena) {
    State s = st_of(hw.data());
    memset(hw.data(), 0, hw.size() * 4);
    for (int i = 0; i < TAGN; ++i) s.tag[i] = -1;
    s.fl[0] = 1;
    s.fl[1] = 0;
    s.fl[1 + KSLOT_FLMAX] = arena;
    fill_stale_fl(s.fl, 1);
}

static bool edge_phase(Ctx& c, long& calls) {
    std::vector<int> gexp(GENN, 7);
    CK(cudaMemcpy(c.d_gexp, gexp.data(), GENN * 4, cudaMemcpyHostToDevice));
    std::vector<uint32_t> hw(STW);

    edge_full_state(hw);
    upload_state(c, hw);
    CallInput in;
    for (int i = 0; i < 7; ++i) in.ids[i] = i;
    in.ids[7] = 8;
    in.ns = in.ns_arg = 8;
    if (!run_one(c, in, 8, 0, 168, calls++, true)) return false;

    edge_full_state(hw);
    upload_state(c, hw);
    in = CallInput();
    for (int i = 0; i < 8; ++i) in.ids[i] = i;
    in.ids[8] = 8;
    in.ns = in.ns_arg = 9;
    if (!run_one(c, in, 8, 0, 168, calls++, true)) return false;

    edge_empty_state(hw, 168);
    upload_state(c, hw);
    in = CallInput();
    in.ids[0] = 3;
    in.ids[1] = 4;
    in.ns = in.ns_arg = 2;
    in.force_direct = 1;
    in.nohm = true;
    if (!run_one(c, in, 8, 0, 168, calls++, true)) return false;

    edge_empty_state(hw, 168);
    upload_state(c, hw);
    in = CallInput();
    in.ids[0] = 5;
    in.ids[1] = 6;
    in.ns = in.ns_arg = 2;
    in.pf_on = true;
    in.pready = 1;
    in.ptag[0] = 5;
    if (!run_one(c, in, 8, 0, 168, calls++, true)) return false;
    return true;
}

static bool read_file(const char* path, std::vector<char>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(n);
    const bool ok = n >= 0 && fread(out.data(), 1, n, f) == (size_t)n;
    fclose(f);
    return ok;
}

static int rd_i32(const char* p) {
    int v;
    memcpy(&v, p, 4);
    return v;
}

static void real_empty_state(std::vector<uint32_t>& hw, int arena) {
    State s = st_of(hw.data());
    memset(hw.data(), 0, hw.size() * 4);
    for (int i = 0; i < TAGN; ++i) s.tag[i] = -1;
    s.fl[0] = 1;
    s.fl[1] = 0;
    s.fl[1 + KSLOT_FLMAX] = arena;
    fill_stale_fl(s.fl, 1);
}

static bool real_phase(Ctx& c, long target, long& calls, long& real_calls) {
    std::vector<char> gb, tb;
    if (!read_file("opt/kbench/kslot_gexp.bin", gb) || gb.size() != 24 * 128 * 3 * 4) {
        printf("missing or bad opt/kbench/kslot_gexp.bin\n");
        return false;
    }
    if (!read_file("opt/kbench/kslot_idseq.bin", tb) || tb.size() % 36 != 0) {
        printf("missing or bad opt/kbench/kslot_idseq.bin\n");
        return false;
    }
    static const int cachel_mb[24] = {
        56, 56, 56, 48, 56, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        156, 156, 48, 44, 48, 52, 48, 52
    };
    struct RL { int layer, arena, kmax; std::vector<std::array<int, 8>> recs; };
    std::vector<RL> layers;
    const int* rg = (const int*)gb.data();
    for (int l = 0; l < 24; ++l) {
        if (!cachel_mb[l]) continue;
        RL r;
        r.layer = l;
        r.arena = (cachel_mb[l] << 20) / KSLOT_GRAN;
        int gmin3 = 1 << 30;
        for (int e = 0; e < 128; ++e) {
            const int* ge = rg + (l * 128 + e) * 3;
            gmin3 = std::min(gmin3, ge[0] + ge[1] + ge[2]);
        }
        r.kmax = std::max(1, std::min(64, r.arena / std::max(1, gmin3)));
        for (size_t off = 0; off < tb.size(); off += 36) {
            if (rd_i32(tb.data() + off) != l) continue;
            std::array<int, 8> ids{};
            for (int i = 0; i < 8; ++i) ids[i] = rd_i32(tb.data() + off + 4 + i * 4);
            r.recs.push_back(ids);
        }
        if (!r.recs.empty()) layers.push_back(r);
    }
    if (layers.empty()) {
        printf("no cached layers in real trace\n");
        return false;
    }

    std::vector<uint32_t> hw(STW);
    long done = 0;
    for (size_t li = 0; li < layers.size() && done < target; ++li) {
        const RL& r = layers[li];
        CK(cudaMemcpy(c.d_gexp, rg + r.layer * 128 * 3, 384 * 4, cudaMemcpyHostToDevice));
        real_empty_state(hw, r.arena);
        upload_state(c, hw);
        const long left_layers = (long)layers.size() - (long)li;
        const long layer_target = std::min((target - done + left_layers - 1) / left_layers,
                                           target - done);
        long lc = 0;
        while (lc < layer_target && done < target) {
            for (const auto& ids : r.recs) {
                CallInput in;
                for (int i = 0; i < 8; ++i) in.ids[i] = ids[i];
                in.ns = in.ns_arg = 8;
                if (!run_one(c, in, r.kmax, 0, r.arena, calls, false)) return false;
                ++calls;
                ++done;
                ++real_calls;
                ++lc;
                if (lc >= layer_target || done >= target) break;
            }
        }
    }
    return true;
}

static void timing_phase(Ctx& c) {
    std::vector<char> gb;
    if (!read_file("opt/kbench/kslot_gexp.bin", gb)) return;
    const int* rg = (const int*)gb.data();
    const int layer = 16;
    const int arena = (156 << 20) / KSLOT_GRAN;
    int gmin3 = 1 << 30;
    for (int e = 0; e < 128; ++e) {
        const int* ge = rg + (layer * 128 + e) * 3;
        gmin3 = std::min(gmin3, ge[0] + ge[1] + ge[2]);
    }
    const int kmax = std::max(1, std::min(64, arena / gmin3));
    CK(cudaMemcpy(c.d_gexp, rg + layer * 128 * 3, 384 * 4, cudaMemcpyHostToDevice));
    std::vector<uint32_t> hw(STW);
    real_empty_state(hw, arena);
    upload_state(c, hw);
    CallInput in;
    for (int i = 0; i < 8; ++i) in.ids[i] = rint(0, 127);
    CK(cudaMemcpy(c.d_ids, in.ids.data(), 64 * 4, cudaMemcpyHostToDevice));

    State sa = st_of(c.dA), sb = st_of(c.dB);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int REP = 400;
    float ms = 0;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < REP; ++i)
        inkwell_exl3_kslot_resolve<<<1, 32>>>(c.d_ids, sa.tag, sa.soff, sa.map,
            sa.miss, sa.cnt, 8, kmax, c.d_gexp, sa.fl, nullptr, nullptr,
            sa.pf_hit, nullptr, 0, 0, sa.hm);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double t_old = ms * 1e3 / REP;
    CK(cudaEventRecord(e0));
    for (int i = 0; i < REP; ++i)
        inkwell_exl3_kslot_resolve_w<<<1, 32>>>(c.d_ids, sb.tag, sb.soff, sb.map,
            sb.miss, sb.cnt, 8, kmax, c.d_gexp, sb.fl, nullptr, nullptr,
            sb.pf_hit, nullptr, 0, 0, sb.hm);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double t_new = ms * 1e3 / REP;
    printf("  kernel time: serial %.2f us/launch, warp %.2f us/launch (L16 kmax=%d)\n",
           t_old, t_new, kmax);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

int main(int argc, char** argv) {
    const long target = argc > 1 ? atol(argv[1]) : 22000;
    Ctx c;
    CK(cudaMalloc(&c.dA, STW * 4));
    CK(cudaMalloc(&c.dB, STW * 4));
    CK(cudaMalloc(&c.dflag, 4));
    CK(cudaMalloc(&c.d_ids, 64 * 4));
    CK(cudaMalloc(&c.d_gexp, GENN * 4));
    CK(cudaMalloc(&c.d_ptag, 64 * 4));
    CK(cudaMalloc(&c.d_pready, 4));
    CK(cudaMalloc(&c.d_pnexp, 4));

    long calls = 0, real_calls = 0;
    long pf_calls = 0, fd_calls = 0, nohm_calls = 0, pnexp_calls = 0, fallback_calls = 0;
    if (!edge_phase(c, calls)) return 1;
    const long edge_calls = calls;
    const long random_target = std::max(1L, target / 2);
    if (!random_phase(c, random_target, calls, pf_calls, fd_calls, nohm_calls,
                      pnexp_calls, fallback_calls)) return 1;
    const long random_calls = calls - edge_calls;
    const long real_target = std::max(1L, target - random_target);
    if (!real_phase(c, real_target, calls, real_calls)) return 1;

    printf("OK %ld calls bit-identical (%ld edge, %ld random, %ld real)\n",
           calls, edge_calls, random_calls, real_calls);
    printf("  mix: pf %ld fd %ld nohm %ld pnexp %ld fallback-kmax %ld\n",
           pf_calls, fd_calls, nohm_calls, pnexp_calls, fallback_calls);
    timing_phase(c);
    return 0;
}
