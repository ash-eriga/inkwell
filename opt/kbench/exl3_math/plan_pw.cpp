// Planner + exhaustive CPU verifier for the paired-window (A1/A2/A3) sgemv core.
// For each (KA,H) it builds the 32-window extraction plan on the lane's LOCAL
// stream (no load-time transform):
//   H=0: raw ring words R[0..NR-1] at per-lane w0; windows at E_j = D_j + o,
//        o = (KA+16) mod 32 (same for all lanes) -> compile-time offsets, 0 align.
//   H=1: per-tile aligned stream W[m] = shf(R[m+1],R[m],o), o = B&31 runtime;
//        windows at compile-time D_j.
// Greedy stages (same as test_cpu.cpp extract_chunk_opt):
//   1. aligned (off % 16 == 0)          -> direct .H1/.H0
//   2. pair (d, d+16) via 1 SHF         -> .H1 / .H0 (+bonus (U+U).H1 if d+1 a window)
//   3. d % 32 == 1                      -> IADD (W+W).H1
//   4. pair (d, d+15) via SHF(d-1)+IADD -> (U+U).H1 = d, U.H0 = d+15
//   5. singles                          -> SHF(d).H1
// Verifies EVERY lane's emitted op sequence bit-exact vs ref_window on random
// rings, then prints the CUDA body for kernels.cu.
//
// g++ -O2 -std=c++17 opt/kbench/exl3_math/plan_pw.cpp -o /tmp/plan_pw && /tmp/plan_pw
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

static constexpr uint32_t MUL1_C = 0x83DCD12Du;

// reference window (exact match to kernels.cu exl3_window)
static inline uint16_t ref_window(const uint32_t* ring, int i, int ka, int hk) {
    const int nw = 8 * ka + 4 * hk;
    const int R = nw * 32;
    const int s = hk ? (i + 1) * ka + ((i + 1) >> 1) : (i + 1) * ka;
    const int b0 = s - 16 + (s < 16 ? R : 0);
    const int i0 = b0 >> 5;
    const int s0 = 48 - (b0 & 31);
    const int i1 = (i0 + 1 == nw) ? 0 : i0 + 1;
    const uint64_t v = ((uint64_t)ring[i0] << 32) | (uint64_t)ring[i1];
    return (uint16_t)((v >> s0) & 0xffffu);
}

// 32-bit word at bit offset d of local array
static inline uint32_t shf_l(const uint32_t* W, int d) {
    int w = d >> 5, b = d & 31;
    return b == 0 ? W[w] : ((W[w] << b) | (W[w + 1] >> (32 - b)));
}

enum Kind { AL1, AL0, PR16, PR15, IAD1, SNGL };
struct Op { Kind kind; int w, b, j, j2, j3; };
// AL1/AL0: mac j from W[w] half
// PR16: U=shf(w,b): mac j .H1 now, stash U -> j2 .H0 (j3 = bonus d+1 or -1)
// PR15: U=shf(w,b)=shf(d-1): mac j via (U+U).H1, stash U -> j2 .H0
// IAD1: mac j via (W[w]+W[w]).H1
// SNGL: U=shf(w,b): mac j .H1

struct Plan {
    int ka, hk, nr, na, nshf, niadd;
    std::vector<Op> ops;
    uint16_t win_of_j[32];   // which op/half produced j -> filled during verify
};

static Plan plan_k(int ka, int hk) {
    const int M = 32;
    int D[32];
    for (int j = 0; j < M; ++j) D[j] = j * ka + hk * ((j + 1) >> 1);
    Plan pl; pl.ka = ka; pl.hk = hk; pl.nshf = pl.niadd = 0;
    int o = 0;
    if (!hk) o = (ka + 16) & 31;          // folded per-lane offset (all lanes share)
    int E[32];
    for (int j = 0; j < M; ++j) E[j] = D[j] + o;
    const int span = E[M - 1] + 16;
    const int nloc = (span + 31) / 32;
    pl.na = nloc;
    pl.nr = hk ? nloc + 1 : nloc;         // H=1 needs one extra raw word for align

    int idx_of[1024]; std::memset(idx_of, -1, sizeof idx_of);
    for (int j = 0; j < M; ++j) idx_of[E[j]] = j;
    bool done[32] = {};
    // op list in *planning* order; emission is re-sorted by j afterwards
    std::vector<Op> ops;
    // 1. aligned
    for (int j = 0; j < M; ++j)
        if ((E[j] & 15) == 0) { ops.push_back({(E[j] & 16) ? AL0 : AL1, E[j] >> 5, 0, j, -1, -1}); done[j] = true; }
    // 2. pairs (d, d+16), with d+1 bonus
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        int d = E[j], j2 = idx_of[d + 16];
        if (j2 >= 0 && !done[j2]) {
            int j3 = idx_of[d + 1];
            if (j3 >= 0 && !done[j3]) { ops.push_back({PR16, d >> 5, d & 31, j, j2, j3}); done[j3] = true; }
            else ops.push_back({PR16, d >> 5, d & 31, j, j2, -1});
            done[j] = done[j2] = true;
        }
    }
    // 3. d % 32 == 1 -> IADD
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        int d = E[j];
        if ((d & 31) == 1) { ops.push_back({IAD1, d >> 5, 0, j, -1, -1}); done[j] = true; }
    }
    // 4. pairs (d, d+15)
    for (int j = 0; j < M; ++j) {
        if (done[j]) continue;
        int d = E[j], j2 = idx_of[d + 15];
        if (d >= 1 && j2 >= 0 && !done[j2]) {
            ops.push_back({PR15, (d - 1) >> 5, (d - 1) & 31, j, j2, -1});
            done[j] = done[j2] = true;
        }
    }
    // 5. singles
    for (int j = 0; j < M; ++j)
        if (!done[j]) { int d = E[j]; ops.push_back({SNGL, d >> 5, d & 31, j, -1, -1}); done[j] = true; }
    // sort ops into j-emission order: an op's position = min(j it completes).
    // pair ops position at primary j; stash use is a pseudo-op at j2; bonus j3 sits right after j.
    std::vector<Op> emit;
    for (auto& op : ops) {
        emit.push_back(op);
        if (op.kind == PR16 || op.kind == PR15)
            emit.push_back({op.kind == PR16 ? AL0 : AL0, -1, -1, op.j2, -1, -1}); // placeholder USE op
    }
    // USE ops: kind==AL0 with w=-1 -> mark specially
    for (auto& op : emit) if (op.w == -1) op.kind = (Kind)6 /*USE0*/;
    std::stable_sort(emit.begin(), emit.end(), [](const Op& a, const Op& b) { return a.j < b.j; });
    pl.ops = emit;
    for (auto& op : pl.ops) {
        if (op.kind == PR16 || op.kind == PR15 || op.kind == SNGL) pl.nshf++;
        if (op.kind == PR15 || op.kind == IAD1 || (op.kind == PR16 && op.j3 >= 0)) pl.niadd++;
    }
    return pl;
}

// simulate one lane's extraction on CPU, verify vs ref_window
static int verify_lane(const Plan& pl, const uint32_t* ring, int g) {
    const int ka = pl.ka, hk = pl.hk;
    const int NW = 8 * ka + 4 * hk, R = NW * 32;
    // lane g chunk base bit
    int B = ka - 16 + g * (32 * ka + 16 * hk);
    if (B < 0) B += R;
    const int w0 = B >> 5, o = B & 31;
    uint32_t Rw[8], W[8];
    for (int k = 0; k < pl.nr; ++k) Rw[k] = ring[(w0 + k < NW) ? w0 + k : w0 + k - NW];
    if (hk) { int na = pl.na; for (int m = 0; m < na; ++m) W[m] = shf_l(Rw, 32 * m + o); }
    else for (int k = 0; k < pl.nr; ++k) W[k] = Rw[k];

    uint16_t got[32] = {};
    uint32_t stash[64]; // indexed by op position
    int si = 0;
    for (size_t pi = 0; pi < pl.ops.size(); ++pi) {
        const Op& op = pl.ops[pi];
        switch ((int)op.kind) {
        case AL1: got[op.j] = (uint16_t)(W[op.w] >> 16); break;
        case AL0: got[op.j] = (uint16_t)(W[op.w] & 0xffffu); break;
        case PR16: {
            uint32_t U = shf_l(W, op.w * 32 + op.b);
            got[op.j] = (uint16_t)(U >> 16);
            if (op.j3 >= 0) got[op.j3] = (uint16_t)((U + U) >> 16);
            stash[si++] = U;
            break; }
        case PR15: {
            uint32_t U = shf_l(W, op.w * 32 + op.b);
            got[op.j] = (uint16_t)((U + U) >> 16);
            stash[si++] = U;
            break; }
        case IAD1: { uint32_t V = W[op.w] + W[op.w]; got[op.j] = (uint16_t)(V >> 16); break; }
        case SNGL: { uint32_t U = shf_l(W, op.w * 32 + op.b); got[op.j] = (uint16_t)(U >> 16); break; }
        case 6: { /* USE0: partner window from next pending stash in order */ break; }
        }
    }
    // second pass: assign stash .H0 to USE0 ops in order
    si = 0;
    for (size_t pi = 0; pi < pl.ops.size(); ++pi) {
        const Op& op = pl.ops[pi];
        if (op.kind == PR16 || op.kind == PR15) { uint32_t U = stash[si++]; (void)U; }
        if ((int)op.kind == 6) {
            // find which stash: count USE0 index
            int use_i = 0;
            for (size_t q = 0; q < pi; ++q) if ((int)pl.ops[q].kind == 6) use_i++;
            // stashes were pushed in op order -> stash index = use_i
            got[op.j] = (uint16_t)(stash[use_i] & 0xffffu);
        }
    }
    int fails = 0;
    for (int j = 0; j < 32; ++j) {
        uint16_t rf = ref_window(ring, g * 32 + j, ka, hk);
        if (got[j] != rf) { ++fails; if (fails < 4) printf("    g%d j%d: got %04x want %04x\n", g, j, got[j], rf); }
    }
    return fails;
}

static uint32_t rngs = 999331u;
static uint32_t rnd() { rngs ^= rngs << 13; rngs ^= rngs >> 17; rngs ^= rngs << 5; return rngs; }

// ---- CUDA body emitter -----------------------------------------------------
// acc index / x operand per window j (tile_perm):
//   ac = 2*(j>>3) + ((j&4)>>2); r = 2*(j>>3) + (j&1) + ((j&2)?8:0)
// permuted xs (per 16): [x0,x1,x8,x9, x2,x3,x10,x11, x4,x5,x12,x13, x6,x7,x14,x15]
//   -> block t' loads float4 xq<t'> at xsk+4t' = {x[2t'],x[2t'+1],x[2t'+8],x[2t'+9]}
static const char* xop(int j) {
    static char b[8][16];
    static int rot = 0;
    const int t = j >> 3, jj = j & 7;
    const int r = (jj & 1) + ((jj & 2) ? 8 : 0);      // +2t' folded into xq index
    const char* c = r == 0 ? ".x" : r == 1 ? ".y" : r == 8 ? ".z" : ".w";
    snprintf(b[rot], 16, "xq%d%s", t, c);
    const char* s = b[rot]; rot = (rot + 1) & 7;
    return s;
}
static int acc_of(int j) { return 2 * (j >> 3) + ((j & 4) >> 2); }

static void emit_body(const Plan& pl) {
    printf("// K=%.1f (KA=%d,H=%d): NR=%d raw words%s, %d SHF + %d IADD extract\n",
           pl.ka + 0.5 * pl.hk, pl.ka, pl.hk, pl.nr, pl.hk ? " + align" : " (folded o)",
           pl.nshf, pl.niadd);
    printf("template <> __device__ __forceinline__ void pw_body<%d,%d>(const uint32_t* __restrict__ W, const float* __restrict__ xsk, float* __restrict__ A, uint32_t CM) {\n",
           pl.ka, pl.hk);
    int sid = 0;
    bool xq_em[4] = {};
    // map partner j -> stash var name
    int stash_var[32]; std::memset(stash_var, -1, sizeof stash_var);
    for (const Op& op : pl.ops) {
        // hoist this op's (and its bonus window's) x-block load
        int need_t = op.j >> 3;
        if (op.j3 >= 0 && (op.j3 >> 3) > need_t) need_t = op.j3 >> 3;
        for (int tt = 0; tt <= need_t; ++tt)
            if (!xq_em[tt]) { xq_em[tt] = true; printf("    const float4 xq%d = *(const float4*)(xsk + %d);\n", tt, 4 * tt); }
        const int ac = acc_of(op.j);
        switch ((int)op.kind) {
        case AL1:
        case AL0:
            printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh%d(W[%d], CM)), A[%d]);\n",
                   ac, xop(op.j), op.kind == AL1 ? 1 : 0, op.w, ac);
            break;
        case SNGL:
            if (op.b && op.b <= 16)
                printf("    { const uint32_t u_ = W[%d] << %d; A[%d] = fmaf(%s, pw_cvt(pw_mh1(u_, CM)), A[%d]); }\n",
                       op.w, op.b, ac, xop(op.j), ac);
            else if (op.b)
                printf("    { const uint32_t u_ = __funnelshift_l(W[%d], W[%d], %d); A[%d] = fmaf(%s, pw_cvt(pw_mh1(u_, CM)), A[%d]); }\n",
                       op.w + 1, op.w, op.b, ac, xop(op.j), ac);
            else
                printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh1(W[%d], CM)), A[%d]);\n",
                       ac, xop(op.j), op.w, ac);
            break;
        case PR16:
        case PR15: {
            const int v = sid++;
            stash_var[op.j2] = v;
            // end-of-stream: use SHL when every needed bit sits inside W[w]
            const int need = op.kind == PR16 ? 31 : 30;
            const bool shl = (op.w * 32 + op.b + need) < (op.w + 1) * 32;
            if (op.b == 0) printf("    const uint32_t u%d = W[%d];\n", v, op.w);
            else if (shl) printf("    const uint32_t u%d = W[%d] << %d;\n", v, op.w, op.b);
            else printf("    const uint32_t u%d = __funnelshift_l(W[%d], W[%d], %d);\n", v, op.w + 1, op.w, op.b);
            if (op.kind == PR16)
                printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh1(u%d, CM)), A[%d]);\n", ac, xop(op.j), v, ac);
            else
                printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh1(u%d + u%d, CM)), A[%d]);\n", ac, xop(op.j), v, v, ac);
            if (op.j3 >= 0)  // bonus (U+U).H1 = window at d+1, j3 = j+1
                printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh1(u%d + u%d, CM)), A[%d]);\n",
                       acc_of(op.j3), xop(op.j3), v, v, acc_of(op.j3));
            break; }
        case IAD1:
            printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh1(W[%d] + W[%d], CM)), A[%d]);\n",
                   ac, xop(op.j), op.w, op.w, ac);
            break;
        case 6:  // USE0: partner .H0 of stash
            printf("    A[%d] = fmaf(%s, pw_cvt(pw_mh0(u%d, CM)), A[%d]);\n", ac, xop(op.j), stash_var[op.j], ac);
            break;
        }
    }
    printf("}\n\n");
}

int main(int argc, char** argv) {
    const int dump = argc > 1 && !strcmp(argv[1], "-d");
    const int emit = argc > 1 && !strcmp(argv[1], "-e");
    const struct { int ka, hk; } khs[7] = {{1,1},{2,0},{2,1},{3,0},{3,1},{4,0},{5,0}};
    int total_fail = 0;
    for (int ki = 0; ki < 7; ++ki) {
        const int ka = khs[ki].ka, hk = khs[ki].hk;
        Plan pl = plan_k(ka, hk);
        int fails = 0, checks = 0;
        const int NW = 8 * ka + 4 * hk;
        for (int t = 0; t < 2000; ++t) {
            uint32_t ring[40];
            for (int w = 0; w < NW; ++w) ring[w] = rnd();
            for (int g = 0; g < 8; ++g) { fails += verify_lane(pl, ring, g); checks += 32; }
        }
        total_fail += fails;
        if (!emit)
            printf("K=%.1f  NR=%d NA=%d  ops=%zu  SHF=%d IADD=%d  extract-instr/w=%.3f  verify %s (%d checks, %d fail)\n",
                   ka + 0.5 * hk, pl.nr, pl.na, pl.ops.size(), pl.nshf, pl.niadd,
                   (pl.nshf + pl.niadd + (hk ? pl.na : 0)) / 32.0,
                   fails ? "FAIL" : "PASS", checks, fails);
        if (fails) { printf("  plan broken, not emitting\n"); continue; }
        if (emit) emit_body(pl);
        if (dump) {
            for (auto& op : pl.ops) {
                const char* kn = (int)op.kind == 6 ? "USE0" : (const char*[]){"AL1","AL0","PR16","PR15","IAD1","SNGL"}[(int)op.kind];
                printf("    %-4s w=%d b=%d j=%d j2=%d j3=%d\n", kn, op.w, op.b, op.j, op.j2, op.j3);
            }
        }
    }
    printf("\n%s\n", total_fail ? "PLAN FAIL" : "PLAN PASS");
    return total_fail ? 1 : 0;
}
