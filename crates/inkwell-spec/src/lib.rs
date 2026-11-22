//! Speculative-decode driver: the host-side verify/commit/rollback
//! loop shared by every drafter (`DSpark`, ngram, lookup).
//!
//! Per round, like the ingine spec loop:
//! 1. the drafter proposes up to `k` tokens (device-side in practice;
//!    here behind the `Draft` trait),
//! 2. the target model runs ONE verify step over `k+1` positions —
//!    one weight read amortizes the whole draft,
//! 3. the longest agreeing prefix is committed; the KV writes of the
//!    rejected suffix are rolled back (ring/capacity positions rewind),
//! 4. one bonus token (the target's prediction after the last accept)
//!    comes for free.

/// Anything that can propose `<= k` draft tokens given the committed
/// tail of context.
pub trait Draft {
    /// Propose draft tokens (already sampled).
    fn propose(&mut self, tail: &[u32], k: usize) -> Vec<u32>;
}

/// A trivial ngram/lookup drafter: repeats the last `n` - 1 tokens'
/// continuation if the pattern occurred before.
#[derive(Debug)]
pub struct NgramDraft {
    /// pattern length used for lookup
    pub n: usize,
    history: std::collections::HashMap<Vec<u32>, u32>,
}

impl NgramDraft {
    /// `n` = pattern length (e.g. 3).
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self { n, history: std::collections::HashMap::new() }
    }
}

impl Draft for NgramDraft {
    fn propose(&mut self, tail: &[u32], k: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(k);
        let mut window = tail.to_vec();
        for _ in 0..k {
            if window.len() < self.n {
                break;
            }
            let pat: Vec<u32> = window[window.len() - self.n + 1..].to_vec();
            match self.history.get(&pat) {
                Some(&t) => {
                    out.push(t);
                    window.push(t);
                }
                None => break,
            }
        }
        if let Some(&last) = tail.last() {
            if tail.len() >= self.n {
                let pat: Vec<u32> = tail[tail.len() - self.n..].to_vec();
                self.history.insert(pat, last);
            }
        }
        let _ = &mut window;
        out
    }
}

/// How many leading draft tokens the target's argmax agreed with.
/// `verify[i]` is the target's chosen token at the position where the
/// drafter proposed `draft[i]` — computed from logits of the verify
/// step. Greedy semantics (upstream's `--draft-max` verify).
#[must_use]
pub fn accept_len(draft: &[u32], verify: &[u32]) -> usize {
    let mut n = 0;
    for (d, v) in draft.iter().zip(verify.iter()) {
        if d != v {
            break;
        }
        n += 1;
    }
    n
}

/// Rolling acceptance statistics.
#[derive(Debug, Default)]
pub struct SpecStats {
    /// rounds run
    pub rounds: u64,
    /// total drafted
    pub drafted: u64,
    /// total accepted (excludes bonus token)
    pub accepted: u64,
    /// tokens emitted (accepted + bonus)
    pub emitted: u64,
}

#[expect(clippy::cast_precision_loss, reason = "acceptance counters stay small vs 2^53")]
impl SpecStats {
    /// Mean tokens emitted per verify round (the weight-read multiplier).
    #[must_use]
    pub fn tokens_per_read(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.emitted as f64 / self.rounds as f64
        }
    }

    /// Acceptance rate over drafted tokens.
    #[must_use]
    pub fn rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}

/// Position bookkeeping for one verify round: slots each arena wrote.
#[derive(Debug)]
pub struct Round {
    /// draft tokens to verify
    pub draft: Vec<u32>,
    /// absolute position of `draft[0]`
    pub pos: usize,
}

/// The driver: tracks position and stats; the backend calls
/// `begin_round`, runs verify, then `commit`.
#[derive(Debug)]
pub struct SpecLoop {
    /// next uncommitted position
    pub pos: usize,
    /// stats
    pub stats: SpecStats,
}

impl SpecLoop {
    /// New driver at `pos`.
    #[must_use]
    pub fn new(pos: usize) -> Self {
        Self { pos, stats: SpecStats::default() }
    }

    /// Start a round: ask the drafter for up to `k` tokens.
    pub fn begin_round<D: Draft>(&mut self, draft: &mut D, tail: &[u32], k: usize) -> Round {
        let draft_tokens = draft.propose(tail, k);
        self.stats.drafted += draft_tokens.len() as u64;
        self.stats.rounds += 1;
        Round { draft: draft_tokens, pos: self.pos }
    }

    /// Commit: `verify` holds the target's argmax at each draft
    /// position (one extra entry for the bonus token after the last
    /// accept). Returns tokens emitted this round (accepted + bonus).
    pub fn commit(&mut self, round: &Round, verify: &[u32]) -> Vec<u32> {
        let n = accept_len(&round.draft, verify);
        let mut out: Vec<u32> = round.draft[..n].to_vec();
        if let Some(&bonus) = verify.get(n) {
            out.push(bonus);
        }
        self.stats.accepted += n as u64;
        self.stats.emitted += out.len() as u64;
        self.pos += out.len();
        out
    }
}
