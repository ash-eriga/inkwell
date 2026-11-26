//! Sampling over model logits: the llama.cpp sampler chain in Rust.
//!
//! Order matters (same as upstream): penalties first, then
//! top-k, typical, top-p, min-p, temperature, then the pick
//! (greedy, weighted or Mirostat is left to the caller for now).

/// Repetition/frequency/presence penalties.
#[derive(Debug, Clone, Copy, Default)]
pub struct Penalties {
    /// repeat divisor for seen tokens (1 = off)
    pub repeat: f32,
    /// subtracted per occurrence (0 = off)
    pub frequency: f32,
    /// subtracted once if present (0 = off)
    pub presence: f32,
    /// how many recent tokens the window covers
    pub last_n: usize,
}

/// Deterministic RNG (xorshift64*) — reproducible sampling, no deps.
#[derive(Debug)]
pub struct Rng(u64);

impl Rng {
    /// Seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Uniform float in [0, 1).
    #[expect(
        clippy::cast_precision_loss,
        reason = "intentional: 53-bit mantissa slice"
    )]
    pub fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let v = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (v >> 11) as f32 * (1.0 / (1u64 << 53) as f32)
    }
}

/// Apply penalties to logits over `recent` tokens.
///
/// Exact `==` comparisons are intentional: the fields are set
/// verbatim from CLI flags, never computed.
#[expect(
    clippy::float_cmp,
    reason = "sentinel values compared exactly as configured"
)]
#[expect(clippy::cast_precision_loss, reason = "occurrence counts stay small")]
pub fn apply_penalties(logits: &mut [f32], recent: &[u32], p: &Penalties) {
    if p.repeat == 1.0 && p.frequency == 0.0 && p.presence == 0.0 {
        return;
    }
    let start = recent.len().saturating_sub(p.last_n);
    // The penalty window is normally tens or hundreds of tokens. Sorting a
    // compact copy is cheaper than allocating and hashing a HashMap on every
    // generated token, and produces the same per-token occurrence counts.
    let mut ids: Vec<u32> = recent[start..].to_vec();
    ids.sort_unstable();
    let mut i = 0usize;
    while i < ids.len() {
        let t = ids[i];
        let mut c = 1u32;
        while i + (c as usize) < ids.len() && ids[i + c as usize] == t {
            c += 1;
        }
        i += c as usize;
        let l = &mut logits[t as usize];
        if *l > 0.0 {
            *l /= p.repeat;
        } else {
            *l *= p.repeat;
        }
        *l -= p.frequency * c as f32 + p.presence;
    }
}

/// Sampling configuration.
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    /// keep only the k best logits (0 = off)
    pub top_k: usize,
    /// nucleus threshold (1 = off)
    pub top_p: f32,
    /// min probability relative to the top token (0 = off)
    pub min_p: f32,
    /// temperature (0 = greedy)
    pub temperature: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 1.0,
        }
    }
}

impl Sampler {
    /// Sample a token id from `logits` (after penalties).
    ///
    /// Exact float comparisons are intentional (greedy/disable flags).
    ///
    /// # Panics
    /// If `logits` is empty.
    #[expect(clippy::cast_possible_truncation, reason = "vocab < u32::MAX")]
    #[must_use]
    pub fn sample(&self, logits: &[f32], rng: &mut Rng) -> u32 {
        assert!(!logits.is_empty(), "cannot sample an empty logits slice");
        // Greedy path: linear argmax, no sort. Keep the best value in a
        // register instead of reloading logits[best] for every comparison.
        if self.temperature == 0.0 {
            return argmax(logits) as u32;
        }

        let t = self.temperature;
        let explicit_k = self.top_k > 0 && self.top_k < logits.len();
        let (mut cand, sum) = if explicit_k || self.top_k >= logits.len() && self.top_k != 0 {
            // The common inference path has a small fixed top-k. Select it by
            // insertion into that small array instead of materializing and
            // partitioning the full vocabulary.
            let k = self.top_k.min(logits.len());
            let cand = top_k_candidates(logits, k);
            let max = cand[0].1;
            let sum = cand.iter().map(|c| ((c.1 - max) / t).exp()).sum();
            (cand, sum)
        } else if self.top_p < 1.0 {
            // Without top-k, top-p normally terminates after a tiny prefix of
            // the softmax. Select doubling prefixes until that prefix reaches
            // the nucleus instead of sorting the entire vocabulary. The full
            // softmax denominator is still computed over every logit.
            let mut mi = 0usize;
            let mut mv = logits[0];
            for (i, &l) in logits.iter().enumerate().skip(1) {
                if l.total_cmp(&mv) == std::cmp::Ordering::Greater {
                    mi = i;
                    mv = l;
                }
            }
            let _ = mi;
            let max = mv;
            let sum: f32 = logits.iter().map(|&l| ((l - max) / t).exp()).sum();
            let mut k = 128usize.min(logits.len());
            let cand = loop {
                let c = if k == logits.len() {
                    all_candidates(logits)
                } else {
                    top_k_candidates(logits, k)
                };
                let mut acc = 0.0f32;
                let mut reached = false;
                for &(_, l) in &c {
                    acc += ((l - max) / t).exp() / sum;
                    if acc >= self.top_p {
                        reached = true;
                        break;
                    }
                }
                if reached || k == logits.len() {
                    break c;
                }
                k = (k * 2).min(logits.len());
            };
            (cand, sum)
        } else {
            // top-p=1 with no nucleus constraint genuinely needs the full
            // ordering. Unstable sort avoids the stable sort's scratch memory;
            // logit ties have no semantic ordering in the sampler.
            let cand = all_candidates(logits);
            let max = cand[0].1;
            let sum = cand.iter().map(|c| ((c.1 - max) / t).exp()).sum();
            (cand, sum)
        };

        let max = cand[0].1;
        for c in &mut cand {
            c.1 = ((c.1 - max) / t).exp();
        }
        // top-p and min-p filters. The iteration order (descending logit) and
        // per-term division mirror the original sampler chain.
        let mut cutoff = 0;
        let mut acc = 0.0f32;
        for (i, c) in cand.iter().enumerate() {
            acc += c.1 / sum;
            cutoff = i + 1;
            if acc >= self.top_p {
                break;
            }
        }
        cand.truncate(cutoff);
        if self.min_p > 0.0 {
            let thresh = cand[0].1 / sum * self.min_p;
            cand.retain(|c| c.1 / sum >= thresh);
        }
        let sum2: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = rng.uniform() * sum2;
        for c in &cand {
            r -= c.1;
            if r <= 0.0 {
                return c.0;
            }
        }
        cand.last().map_or(0, |c| c.0)
    }
}

fn desc_cmp(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.total_cmp(&a.1)
}

fn all_candidates(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    cand.sort_unstable_by(desc_cmp);
    cand
}

/// Return the `k` highest logits in descending order without allocating or
/// partitioning a vocabulary-sized candidate vector.
fn top_k_candidates(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }
    if k >= logits.len() {
        return all_candidates(logits);
    }
    let mut cand: Vec<(u32, f32)> = logits[..k]
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    cand.sort_unstable_by(desc_cmp);
    scan_topk(logits, k, &mut cand);
    cand
}

#[inline]
fn better_total(l: f32, cutoff: f32) -> bool {
    l.total_cmp(&cutoff) == std::cmp::Ordering::Greater
}

#[inline]
fn insert_topk(cand: &mut Vec<(u32, f32)>, i: usize, l: f32) {
    // Descending order; partition_point inserts after equal values, so exact
    // ties retain input order.
    let pos = cand.partition_point(|c| l.total_cmp(&c.1) != std::cmp::Ordering::Greater);
    cand.insert(pos, (i as u32, l));
    cand.pop();
}

fn scan_topk(logits: &[f32], start: usize, cand: &mut Vec<(u32, f32)>) {
    #[cfg(target_arch = "x86_64")]
    if avx2() && !cand[cand.len() - 1].1.is_nan() {
        unsafe { scan_topk_avx2(logits, start, cand) };
        return;
    }
    scan_topk_scalar(logits, start, cand);
}

fn scan_topk_scalar(logits: &[f32], start: usize, cand: &mut Vec<(u32, f32)>) {
    let mut cutoff = cand[cand.len() - 1].1;
    for (i, &l) in logits.iter().enumerate().skip(start) {
        if better_total(l, cutoff) {
            insert_topk(cand, i, l);
            cutoff = cand[cand.len() - 1].1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2() -> bool {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// Scan eight logits at a time. The vector compare is only a filter: whenever
/// a lane may enter the small top-k array, that lane is inserted by the scalar
/// total-order path, preserving ordering and tie behavior exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_topk_avx2(logits: &[f32], start: usize, cand: &mut Vec<(u32, f32)>) {
    use core::arch::x86_64::{
        _mm256_cmp_ps, _mm256_loadu_ps, _mm256_movemask_ps, _mm256_set1_ps, _CMP_GT_OQ,
        _CMP_UNORD_Q,
    };
    let mut cutoff = cand[cand.len() - 1].1;
    let mut thresh = _mm256_set1_ps(cutoff);
    let mut i = start;
    while i + 8 <= logits.len() {
        let v = unsafe { _mm256_loadu_ps(logits.as_ptr().add(i)) };
        let gt = _mm256_movemask_ps(_mm256_cmp_ps(v, thresh, _CMP_GT_OQ));
        let unord = _mm256_movemask_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        if unord != 0 {
            // NaNs are rare and not ordered IEEE values, but total_cmp admits
            // them. Take the exact scalar path for a vector containing one.
            for lane in 0..8 {
                let j = i + lane;
                let l = logits[j];
                if better_total(l, cutoff) {
                    insert_topk(cand, j, l);
                    cutoff = cand[cand.len() - 1].1;
                }
            }
        } else if gt != 0 {
            let mut bits = gt;
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let j = i + lane;
                let l = logits[j];
                if l > cutoff {
                    insert_topk(cand, j, l);
                    cutoff = cand[cand.len() - 1].1;
                }
            }
        }
        thresh = _mm256_set1_ps(cutoff);
        i += 8;
    }
    while i < logits.len() {
        let l = logits[i];
        if better_total(l, cutoff) {
            insert_topk(cand, i, l);
            cutoff = cand[cand.len() - 1].1;
        }
        i += 1;
    }
}

fn argmax(logits: &[f32]) -> usize {
    #[cfg(target_arch = "x86_64")]
    if avx2() && logits.len() >= 8 {
        if logits[0].is_nan() {
            return 0;
        }
        return unsafe { argmax_avx2(logits) };
    }
    argmax_scalar(logits)
}

fn argmax_scalar(logits: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = logits[0];
    for (i, &l) in logits.iter().enumerate().skip(1) {
        if l > bv {
            bi = i;
            bv = l;
        }
    }
    bi
}

/// Lane-parallel argmax. Each lane keeps the first strict maximum on its
/// strided subsequence; the reduction breaks equal values by lower index,
/// matching the scalar first-maximum scan. NaNs after index zero never compare
/// greater and therefore cannot win; a NaN in a lane's first slot is treated as
/// vacant (-inf) rather than as a value that poisons that lane.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_avx2(logits: &[f32]) -> usize {
    use core::arch::x86_64::{
        _mm256_add_epi32, _mm256_blendv_epi8, _mm256_blendv_ps, _mm256_castps_si256, _mm256_cmp_ps,
        _mm256_loadu_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_epi32, _mm256_storeu_ps,
        _mm256_storeu_si256, _CMP_GT_OQ, _CMP_UNORD_Q,
    };
    let mut bestv = unsafe { _mm256_loadu_ps(logits.as_ptr()) };
    let unord = _mm256_cmp_ps(bestv, bestv, _CMP_UNORD_Q);
    bestv = _mm256_blendv_ps(bestv, _mm256_set1_ps(f32::NEG_INFINITY), unord);
    let step = _mm256_set1_epi32(8);
    let mut curi = _mm256_add_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), step);
    let mut besti = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut i = 8usize;
    while i + 8 <= logits.len() {
        let v = unsafe { _mm256_loadu_ps(logits.as_ptr().add(i)) };
        let mask = _mm256_cmp_ps(v, bestv, _CMP_GT_OQ);
        bestv = _mm256_blendv_ps(bestv, v, mask);
        besti = _mm256_blendv_epi8(besti, curi, _mm256_castps_si256(mask));
        curi = _mm256_add_epi32(curi, step);
        i += 8;
    }
    let mut vals = [0f32; 8];
    let mut idxs = [0i32; 8];
    unsafe {
        _mm256_storeu_ps(vals.as_mut_ptr(), bestv);
        _mm256_storeu_si256(idxs.as_mut_ptr().cast(), besti);
    }
    let mut bi = idxs[0] as usize;
    let mut bv = vals[0];
    for lane in 1..8 {
        let (v, x) = (vals[lane], idxs[lane] as usize);
        if v > bv || (v == bv && x < bi) {
            bv = v;
            bi = x;
        }
    }
    while i < logits.len() {
        if logits[i] > bv {
            bv = logits[i];
            bi = i;
        }
        i += 1;
    }
    bi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_logits(n: usize) -> Vec<f32> {
        let mut x = 0x243f6a8885a308d3u64;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 11) as f32 / (1u64 << 53) as f32) * 24.0 - 18.0
            })
            .collect()
    }

    #[test]
    fn penalties_count_repeated_tokens() {
        let mut logits = vec![10.0; 8];
        let recent = [2, 5, 2, 7, 2];
        apply_penalties(
            &mut logits,
            &recent,
            &Penalties {
                repeat: 2.0,
                frequency: 0.5,
                presence: 0.25,
                last_n: 4,
            },
        );
        assert_eq!(logits[2], 10.0 / 2.0 - (0.5 * 2.0 + 0.25));
        assert_eq!(logits[5], 10.0 / 2.0 - (0.5 + 0.25));
        assert_eq!(logits[7], 10.0 / 2.0 - (0.5 + 0.25));
    }

    #[test]
    fn greedy_is_first_strict_maximum() {
        let logits = [1.0, 4.0, 4.0, 2.0];
        let s = Sampler {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 0.0,
        };
        let mut rng = Rng::new(1);
        assert_eq!(s.sample(&logits, &mut rng), 1);
    }

    #[test]
    fn avx_argmax_matches_scalar_on_finite_inputs() {
        for n in [8usize, 9, 31, 32, 33, 257] {
            let logits = test_logits(n);
            assert_eq!(argmax(&logits), argmax_scalar(&logits));
        }
    }

    #[test]
    fn top_k_matches_sorted_reference_for_unique_logits() {
        for n in [17usize, 128, 1027] {
            let logits = test_logits(n);
            for k in [1usize, 3, 40] {
                let got = top_k_candidates(&logits, k.min(n));
                let mut all: Vec<(u32, f32)> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i as u32, l))
                    .collect();
                all.sort_unstable_by(desc_cmp);
                all.truncate(k.min(n));
                assert_eq!(got, all);
            }
        }
    }
}
