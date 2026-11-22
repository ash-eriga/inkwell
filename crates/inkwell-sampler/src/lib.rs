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
    #[expect(clippy::cast_precision_loss, reason = "intentional: 53-bit mantissa slice")]
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
#[expect(clippy::float_cmp, reason = "sentinel values compared exactly as configured")]
#[expect(clippy::cast_precision_loss, reason = "occurrence counts stay small")]
pub fn apply_penalties(logits: &mut [f32], recent: &[u32], p: &Penalties) {
    if p.repeat == 1.0 && p.frequency == 0.0 && p.presence == 0.0 {
        return;
    }
    let start = recent.len().saturating_sub(p.last_n);
    let mut count = std::collections::HashMap::new();
    for &t in &recent[start..] {
        *count.entry(t).or_insert(0u32) += 1;
    }
    for (t, c) in count {
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
        Self { top_k: 0, top_p: 1.0, min_p: 0.0, temperature: 1.0 }
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
        // top-k then temperature on survivors, then top-p/min-p.
        // Greedy path: linear argmax, no sort.
        if self.temperature == 0.0 {
            let mut bi = 0usize;
            for (i, &l) in logits.iter().enumerate() {
                if l > logits[bi] {
                    bi = i;
                }
            }
            return bi as u32;
        }
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        if self.top_k > 0 && self.top_k < cand.len() {
            // partial select: only the k best need sorting
            cand.select_nth_unstable_by(self.top_k, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(self.top_k);
        }
        cand.sort_by(|a, b| b.1.total_cmp(&a.1));
        let t = self.temperature;
        if t == 0.0 {
            return cand[0].0;
        }
        let max = cand[0].1;
        let mut sum = 0.0f32;
        for c in &mut cand {
            c.1 = ((c.1 - max) / t).exp();
            sum += c.1;
        }
        // top-p and min-p filters
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
