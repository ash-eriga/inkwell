//! Expert pager: the RAM→VRAM cache for `MoE` expert weights.
//!
//! Port of `llama-moe-cache.cpp` (the dp4a/strata stack): a
//! frequency-class LRU over expert slots. The GPU holds `n_slots`
//! expert-sized banks; `slot_map[layer][expert]` gives the bank or
//! `NONE`. Hits are counted in usage classes
//! `class = floor(log2(uses))`, and evictions always take the LRU of
//! the lowest populated class — frequently reused experts survive.
//!
//! Two fill paths, like upstream:
//! - demand fills: experts selected for the current token, batched so
//!   consecutive experts land in consecutive slots (one H2D upload);
//! - speculative prefetch (STRATA-style): during the draft stage, the
//!   `PCIe` link is idle, so the pager fills the *predicted* experts of
//!   upcoming layers (`prev_sel` of the previous token) into free/cold
//!   slots, bounded by a byte budget.

/// Slot value meaning "not cached".
pub const NONE: u32 = u32::MAX;

/// Per-layer expert byte geometry.
#[derive(Debug, Clone)]
pub struct ExpertLayout {
    /// bytes of one expert on a given layer (`gate_up` + `down` + scales) / `n_expert`
    pub bytes_per_expert: Vec<usize>,
    /// experts per layer
    pub n_expert: usize,
    /// experts used per token per layer
    pub n_expert_used: usize,
}

impl ExpertLayout {
    /// Maximum expert size over all layers — the bank/slot size.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.bytes_per_expert.iter().copied().max().unwrap_or(0)
    }
}

/// One slot's bookkeeping in the class-LRU.
#[derive(Debug, Clone, Copy)]
struct Slot {
    key: u32,        // layer * n_expert + expert, NONE if free
    lru_prev: u32,   // intrusive linked list within its class
    lru_next: u32,
    uses: u32,       // demand hits — drives the class
}

const N_CLASS: usize = 8;

/// Frequency-class LRU over expert slots.
///
/// Slot/key indices are `u32` — bounded by VRAM bank count and
/// `n_layers * n_expert`, both far under `u32::MAX`.
#[derive(Debug)]
pub struct Pager {
    slots: Vec<Slot>,
    /// per-class head (LRU end) — index into slots, NONE if empty
    head: [u32; N_CLASS],
    /// per-class tail (MRU end)
    tail: [u32; N_CLASS],
    /// `slot_map[layer][expert]` -> slot or NONE
    slot_map: Vec<Vec<u32>>,
    layout: ExpertLayout,
    /// fill plan for the current batch: (layer, expert, slot)
    fills: Vec<(u32, u32, u32)>,
    /// stats
    pub hits: u64,
    /// demand misses
    pub misses: u64,
    /// bytes moved by speculative prefetches
    pub prefetch_bytes: u64,
}

fn class_of(uses: u32) -> usize {
    uses.checked_ilog2().map_or(0, |v| usize::try_from(v).unwrap_or(0)).min(N_CLASS - 1)
}

#[expect(clippy::cast_possible_truncation, reason = "slot/key indices bounded well below u32::MAX")]
impl Pager {
    /// `cache_bytes` of VRAM for banks; `n_layers` covered.
    #[must_use]
    pub fn new(layout: ExpertLayout, n_layers: usize, cache_bytes: usize) -> Self {
        let n_slots = cache_bytes / layout.slot_bytes().max(1);
        let slot_map = vec![vec![NONE; layout.n_expert]; n_layers];
        Self {
            slots: vec![Slot { key: NONE, lru_prev: NONE, lru_next: NONE, uses: 0 }; n_slots],
            head: [NONE; N_CLASS],
            tail: [NONE; N_CLASS],
            slot_map,
            layout,
            fills: Vec::new(),
            hits: 0,
            misses: 0,
            prefetch_bytes: 0,
        }
    }

    /// Number of VRAM banks.
    #[must_use]
    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    fn unlink(&mut self, s: u32) {
        let (p, n) = (self.slots[s as usize].lru_prev, self.slots[s as usize].lru_next);
        let c = class_of(self.slots[s as usize].uses);
        if p == NONE { self.head[c] = n; } else { self.slots[p as usize].lru_next = n; }
        if n == NONE { self.tail[c] = p; } else { self.slots[n as usize].lru_prev = p; }
    }

    fn link_mru(&mut self, s: u32) {
        let c = class_of(self.slots[s as usize].uses);
        self.slots[s as usize].lru_prev = self.tail[c];
        self.slots[s as usize].lru_next = NONE;
        if self.tail[c] == NONE { self.head[c] = s; } else { self.slots[self.tail[c] as usize].lru_next = s; }
        self.tail[c] = s;
    }

    /// Victim: the LRU slot of the lowest populated class, or a free
    /// slot. `protected` (a layer id) is never evicted.
    fn victim(&mut self, protected: u32) -> Option<u32> {
        // free slot first
        if let Some(s) = self.slots.iter().position(|s| s.key == NONE) {
            return Some(s as u32);
        }
        for head in self.head {
            let mut s = head;
            while s != NONE {
                let key = self.slots[s as usize].key;
                if key / self.layout.n_expert as u32 != protected {
                    return Some(s);
                }
                s = self.slots[s as usize].lru_next;
            }
        }
        None
    }

    fn evict_into(&mut self, slot: u32, key: u32) {
        let old = self.slots[slot as usize].key;
        if old != NONE {
            let l = (old / self.layout.n_expert as u32) as usize;
            let e = (old % self.layout.n_expert as u32) as usize;
            self.slot_map[l][e] = NONE;
        }
        if old != NONE {
            self.unlink(slot);
        }
        self.slots[slot as usize].key = key;
        self.slots[slot as usize].uses = 0;
        self.link_mru(slot);
        let l = (key / self.layout.n_expert as u32) as usize;
        let e = (key % self.layout.n_expert as u32) as usize;
        self.slot_map[l][e] = slot;
    }

    /// Demand-resolve a layer's selected experts. Returns slots
    /// aligned with `ids`; misses are queued in `fills` for the caller
    /// to batch into H2D uploads.
    ///
    /// # Panics
    /// When every slot is pinned to `layer` (impossible in practice:
    /// `n_expert_used` < slots).
    pub fn resolve(&mut self, layer: u32, ids: &[u32]) -> Vec<u32> {
        let l = layer as usize;
        let mut out = Vec::with_capacity(ids.len());
        for &e in ids {
            let slot = self.slot_map[l][e as usize];
            if slot == NONE {
                self.misses += 1;
                let v = self.victim(layer).expect("pager full of protected slots");
                self.evict_into(v, layer * self.layout.n_expert as u32 + e);
                self.fills.push((layer, e, v));
                out.push(v);
            } else {
                self.hits += 1;
                self.unlink(slot);
                self.slots[slot as usize].uses += 1;
                self.link_mru(slot);
                out.push(slot);
            }
        }
        out
    }

    /// STRATA-style prefetch of predicted experts while the link is
    /// idle. Only touches free slots and cold (class-0, non-protected)
    /// slots; bounded by `budget` bytes. Returns slots filled.
    pub fn prefetch(&mut self, layer: u32, ids: &[u32], budget: &mut usize) -> u32 {
        let l = layer as usize;
        let mut n = 0;
        for &e in ids {
            if self.slot_map[l][e as usize] != NONE {
                continue; // already cached
            }
            let need = self.layout.bytes_per_expert[l];
            if *budget < need {
                break;
            }
            let Some(v) = self.victim(layer) else { break };
            // prefetch may only take free or class-0 slots
            if self.slots[v as usize].key != NONE && class_of(self.slots[v as usize].uses) > 0 {
                continue;
            }
            self.evict_into(v, layer * self.layout.n_expert as u32 + e);
            *budget -= need;
            self.prefetch_bytes += need as u64;
            self.fills.push((layer, e, v));
            n += 1;
        }
        n
    }

    /// Pending fill count (for splitting prefetch vs demand fills).
    #[must_use]
    pub fn fills_len(&self) -> usize {
        self.fills.len()
    }

    /// Consume the fill plan, merged into runs of consecutive
    /// (layer, expert) → consecutive slots, like upstream's single
    /// H2D upload per run.
    pub fn take_fills(&mut self) -> Vec<(u32, u32, u32)> {
        std::mem::take(&mut self.fills)
    }
}
