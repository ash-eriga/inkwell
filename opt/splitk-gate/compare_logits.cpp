#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Row {
    std::vector<float> x;
};

static bool read_row(std::ifstream& f, Row& r) {
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f) return false;
    r.x.resize(n);
    f.read(reinterpret_cast<char*>(r.x.data()), (std::streamsize)(4 * (size_t)n));
    return bool(f);
}

struct Acc {
    double sum = 0.0;
    double mx = 0.0;
    std::vector<double> vals;
    void add(double x) { sum += x; mx = std::max(mx, x); vals.push_back(x); }
    double mean() const { return vals.empty() ? 0.0 : sum / (double)vals.size(); }
    double p95() {
        if (vals.empty()) return 0.0;
        std::sort(vals.begin(), vals.end());
        return vals[std::min(vals.size() - 1, (size_t)(0.95 * (double)(vals.size() - 1)))];
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: compare_logits <a.bin> <b.bin> [rows.tsv]\n";
        return 2;
    }
    std::ifstream fa(argv[1], std::ios::binary), fb(argv[2], std::ios::binary);
    if (!fa || !fb) {
        std::cerr << "missing input\n";
        return 2;
    }
    std::ofstream rows;
    if (argc >= 4) rows.open(argv[3]);
    if (rows) rows << "row\tkld_ab\tkld_ba\tmax_abs\tagree\ttop_a\ttop_b\tmargin_a\tmargin_b\n";
    Row a, b;
    size_t nrows = 0, agree = 0, first_mismatch = SIZE_MAX;
    double max_abs_all = 0.0;
    Acc full_ab, full_ba, pre_ab, pre_ba;
    while (read_row(fa, a) && read_row(fb, b)) {
        if (a.x.size() != b.x.size() || a.x.empty()) {
            std::cerr << "row width mismatch\n";
            return 2;
        }
        const size_t n = a.x.size();
        size_t ta = 0, tb = 0;
        double ma = -1e300, mb = -1e300;
        for (size_t i = 0; i < n; ++i) {
            if ((double)a.x[i] > ma) { ma = a.x[i]; ta = i; }
            if ((double)b.x[i] > mb) { mb = b.x[i]; tb = i; }
        }
        double sa = 0.0, sb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sa += exp((double)a.x[i] - ma);
            sb += exp((double)b.x[i] - mb);
        }
        double kld_ab = 0.0, kld_ba = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double pa = exp((double)a.x[i] - ma) / sa;
            const double pb = exp((double)b.x[i] - mb) / sb;
            if (pa > 0.0 && pb > 0.0) {
                kld_ab += pa * log(pa / pb);
                kld_ba += pb * log(pb / pa);
            } else if (pa > 0.0) {
                kld_ab = std::numeric_limits<double>::infinity();
            } else if (pb > 0.0) {
                kld_ba = std::numeric_limits<double>::infinity();
            }
            max_abs = std::max(max_abs, fabs((double)a.x[i] - (double)b.x[i]));
        }
        double best2a = -1e300, best2b = -1e300;
        for (size_t i = 0; i < n; ++i) {
            if (i != ta) best2a = std::max(best2a, (double)a.x[i]);
            if (i != tb) best2b = std::max(best2b, (double)b.x[i]);
        }
        const bool same = ta == tb;
        agree += same;
        if (!same && first_mismatch == SIZE_MAX) first_mismatch = nrows;
        max_abs_all = std::max(max_abs_all, max_abs);
        full_ab.add(kld_ab);
        full_ba.add(kld_ba);
        if (first_mismatch == SIZE_MAX) {
            pre_ab.add(kld_ab);
            pre_ba.add(kld_ba);
        }
        if (rows) {
            rows << nrows << '\t' << kld_ab << '\t' << kld_ba << '\t' << max_abs << '\t' << same
                 << '\t' << ta << '\t' << tb << '\t' << ma - best2a << '\t' << mb - best2b << '\n';
        }
        ++nrows;
    }
    std::cout << "rows=" << nrows
              << " top1_agree=" << agree << "/" << nrows
              << " first_mismatch=" << (first_mismatch == SIZE_MAX ? -1 : (long long)first_mismatch)
              << " max_abs=" << max_abs_all
              << " pre_rows=" << pre_ab.vals.size()
              << " pre_kld_ab_mean=" << pre_ab.mean() << " pre_kld_ab_p95=" << pre_ab.p95() << " pre_kld_ab_max=" << pre_ab.mx
              << " pre_kld_ba_mean=" << pre_ba.mean() << " pre_kld_ba_p95=" << pre_ba.p95() << " pre_kld_ba_max=" << pre_ba.mx
              << " full_kld_ab_mean=" << full_ab.mean() << " full_kld_ab_p95=" << full_ab.p95() << " full_kld_ab_max=" << full_ab.mx
              << " full_kld_ba_mean=" << full_ba.mean() << " full_kld_ba_p95=" << full_ba.p95() << " full_kld_ba_max=" << full_ba.mx
              << "\n";
    return 0;
}
