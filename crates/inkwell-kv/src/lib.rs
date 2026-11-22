//! KV cache layout for the Inkling-2 architecture.
//!
//! The model has 24 physical layers producing 48 attention visits
//! (layers 0-5 pre-loop, 6-17 × `n_loop` passes, 18-23 post-loop).
//! Four layers (every 6th: 5, 11, 17, 23) are full-attention with a
//! `relative_extent`-sized bias table; the rest are sliding-window
//! (1024) — verified against `attention.sliding_window_pattern` in the
//! checkpoint.
//!
//! Keys are clamped to `qk_clip` and values RMS-normed, so E4M3 never
//! saturates: the full-attention KV can be stored scale-free in FP8
//! (the trick the ingine engine uses for its 512 MiB/visit full KV).

use inkwell_core::InklingConfig;

/// Element type of a KV arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    /// scale-free E4M3 — legal because of qk_clip/rel_clip bounds
    F8E4m3,
    /// brain float 16
    Bf16,
    /// IEEE fp32 (reference/debug)
    F32,
}

impl KvType {
    /// Bytes per stored element.
    pub const fn elem_size(self) -> usize {
        match self {
            Self::F8E4m3 => 1,
            Self::Bf16 | Self::F32 => 2, /* bf16 stored as u16 */
        }
    }
}

/// Kind of KV arena a visit writes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaKind {
    /// Sliding-window ring of `window` slots (pos % window).
    Sliding {
        /// ring capacity in tokens
        window: usize,
    },
    /// Position-indexed full-attention cache.
    Full {
        /// allocated capacity in tokens
        capacity: usize,
    },
}

/// One physical KV arena (possibly shared by several loop passes).
#[derive(Debug)]
pub struct Arena {
    /// arena id == index into `KvPlan::arenas`
    pub id: usize,
    /// layout
    pub kind: ArenaKind,
    /// element type
    pub ty: KvType,
    /// bytes per token (K+V over all kv heads)
    pub bytes_per_token: usize,
    /// total arena bytes
    pub bytes: usize,
}

impl Arena {
    /// Ring/capacity slot for a given absolute position.
    #[must_use]
    pub fn slot_for(&self, pos: usize) -> usize {
        match self.kind {
            ArenaKind::Sliding { window } => pos % window,
            ArenaKind::Full { .. } => pos,
        }
    }
}

/// One of the 48 attention visits mapped to its arena.
#[derive(Debug, Clone, Copy)]
pub struct Visit {
    /// visit index in execution order (0..48)
    pub index: usize,
    /// physical layer this visit belongs to
    pub layer: u32,
    /// loop pass (0 for non-loop layers)
    pub pass: u32,
    /// arena this visit reads/writes
    pub arena: usize,
    /// whether this visit is full-attention
    pub full: bool,
}

/// Complete KV layout for a model + context budget.
#[derive(Debug)]
pub struct KvPlan {
    /// physical arenas (deduplicated when `share_loop`)
    pub arenas: Vec<Arena>,
    /// all 48 visits in execution order
    pub visits: Vec<Visit>,
    /// whether loop passes share one arena per layer
    pub share_loop: bool,
    /// sliding window from config
    pub swa_window: usize,
}

impl KvPlan {
    /// Build the layout for `max_ctx` tokens.
    ///
    /// `share_loop`: when true, the `n_loop` passes of each loop layer
    /// write to the same arena (upstream `loop_share_cache` PR); when
    /// false each pass gets its own arena like upstream llama.cpp.
    #[must_use]
    pub fn new(cfg: &InklingConfig, max_ctx: usize, ty: KvType, share_loop: bool) -> Self {
        let n_kv = cfg.n_head_kv as usize;
        let head_dim = cfg.head_dim as usize;
        let bytes_per_token = 2 * n_kv * head_dim * ty.elem_size();

        let mut arenas = Vec::new();
        // arena id per (layer, pass-or-shared)
        let mut map: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
        let mut arena_for = |layer: u32, pass: u32, full: bool| -> usize {
            let key = (layer, if share_loop { 0 } else { pass });
            *map.entry(key).or_insert_with(|| {
                let kind = if full {
                    ArenaKind::Full { capacity: max_ctx }
                } else {
                    ArenaKind::Sliding { window: 1024 }
                };
                let cap = match kind {
                    ArenaKind::Sliding { window } => window,
                    ArenaKind::Full { capacity } => capacity,
                };
                let id = arenas.len();
                arenas.push(Arena { id, kind, ty, bytes_per_token, bytes: cap * bytes_per_token });
                id
            })
        };

        let mut visits = Vec::with_capacity(cfg.n_visits() as usize);
        for pass in 0..=cfg.n_loop {
            // loop passes 1..=n_loop cover layers loop_start..loop_end
            let (a, b) = if pass == 0 {
                (0, cfg.loop_start)
            } else {
                (cfg.loop_start, cfg.loop_end)
            };
            for layer in a..b {
                let full = layer % 6 == 5;
                let arena = arena_for(layer, pass, full);
                visits.push(Visit { index: visits.len(), layer, pass, arena, full });
            }
        }
        // post-loop layers loop_end..n_layer with pass 0
        for layer in cfg.loop_end..cfg.n_layer {
            let full = layer % 6 == 5;
            let arena = arena_for(layer, 0, full);
            visits.push(Visit { index: visits.len(), layer, pass: 0, arena, full });
        }

        Self { arenas, visits, share_loop, swa_window: 1024 }
    }

    /// Total KV bytes across all arenas.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.arenas.iter().map(|a| a.bytes).sum()
    }

    /// Bytes held by full-attention arenas (the part that grows with ctx).
    #[must_use]
    pub fn full_bytes(&self) -> usize {
        self.arenas
            .iter()
            .filter(|a| matches!(a.kind, ArenaKind::Full { .. }))
            .map(|a| a.bytes)
            .sum()
    }

    /// Bytes held by sliding rings (context-independent).
    #[must_use]
    pub fn swa_bytes(&self) -> usize {
        self.total_bytes() - self.full_bytes()
    }
}
