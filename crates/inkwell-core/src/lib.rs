//! Inkwell core: model config, weight map, graph builder (host orchestration).
//! Kernels live in CUDA C++ behind FFI — this crate owns scheduling,
//! expert paging (RAM<->VRAM), KV layout, and the spec-decode loop.

use inkwell_gguf::{GgufFile, MetaValue};

/// Static config of an `inkling2_loop` checkpoint, read from GGUF metadata.
#[derive(Debug)]
pub struct InklingConfig {
    /// logical layers (24)
    pub n_layer: u32,
    /// loop passes for middle layers (3)
    pub n_loop: u32,
    /// 6
    pub loop_start: u32,
    /// 18 (exclusive)
    pub loop_end: u32,
    /// 64
    pub loop_gate_rank: u32,
    /// 2304
    pub n_embd: u32,
    /// 24
    pub n_head: u32,
    /// 8
    pub n_head_kv: u32,
    /// 128
    pub head_dim: u32,
    /// 16
    pub n_rel: u32,
    /// 1024
    pub rel_extent: u32,
    /// 1024
    pub swa: u32,
    /// 128
    pub n_expert: u32,
    /// 8
    pub n_expert_used: u32,
    /// 1152
    pub expert_ff: u32,
    /// 201024
    pub vocab: u32,
    /// 1048576
    pub ctx_train: u32,
    /// 22.627
    pub qk_clip: f32,
    /// 8.0
    pub rel_clip: f32,
    /// 30.0
    pub softcap: f32,
    /// layers between block-stack splits (6)
    pub attn_res_block_size: u32,
    /// SITU activation beta (4.0)
    pub situ_beta: f32,
    /// SITU linear beta (25.0)
    pub situ_linear_beta: f32,
    /// expert weight scale (8.0)
    pub expert_weights_scale: f32,
    /// log-position alpha for full-attn layers (0.1)
    pub log_scale_alpha: f32,
    /// log-position floor (8192)
    pub log_scale_floor: u32,
    /// shortconv kernel/state length (4)
    pub sconv_l_cache: u32,
    /// real vocab without padding (200058)
    pub vocab_unpadded: u32,
    /// rms norm epsilon (1e-6)
    pub rms_eps: f32,
}

fn get<'a>(m: &'a std::collections::HashMap<String, MetaValue>, k: &str) -> &'a MetaValue {
    m.get(k).unwrap_or_else(|| panic!("missing gguf key {k}"))
}
fn u32v(v: &MetaValue) -> u32 {
    match v {
        MetaValue::U32(x) => *x,
        MetaValue::I32(x) => u32::try_from(*x).expect("i32 meta fits u32"),
        _ => panic!("not u32"),
    }
}
fn f32v(v: &MetaValue) -> f32 {
    match v {
        MetaValue::F32(x) => *x,
        _ => panic!("not f32"),
    }
}
fn arr(m: &std::collections::HashMap<String, MetaValue>, k: &str, i: usize) -> u32 {
    match get(m, k) {
        MetaValue::Arr(a) => u32v(&a[i]),
        _ => panic!("not arr {k}"),
    }
}

impl InklingConfig {
    /// Read `inkling2_loop` config from GGUF metadata.
    ///
    /// # Panics
    ///
    /// On missing or mistyped metadata keys.
    #[must_use]
    pub fn from_gguf(g: &GgufFile) -> Self {
        let m = &g.meta;
        let p = "inkling2_loop.";
        let k = |s: &str| format!("{p}{s}");
        InklingConfig {
            n_layer: u32v(get(m, &k("block_count"))),
            n_loop: u32v(get(m, &k("loop.count"))),
            loop_start: u32v(get(m, &k("loop.start"))),
            loop_end: u32v(get(m, &k("loop.end"))),
            loop_gate_rank: u32v(get(m, &k("loop.gate_rank"))),
            n_embd: u32v(get(m, &k("embedding_length"))),
            n_head: arr(m, &k("attention.head_count"), 0),
            n_head_kv: arr(m, &k("attention.head_count_kv"), 0),
            head_dim: u32v(get(m, &k("attention.key_length"))),
            n_rel: u32v(get(m, &k("attention.relative_length"))),
            rel_extent: u32v(get(m, &k("attention.relative_extent"))),
            swa: u32v(get(m, &k("attention.sliding_window"))),
            n_expert: u32v(get(m, &k("expert_count"))),
            n_expert_used: u32v(get(m, &k("expert_used_count"))),
            expert_ff: u32v(get(m, &k("expert_feed_forward_length"))),
            vocab: u32v(get(m, &k("vocab_size"))),
            ctx_train: u32v(get(m, &k("context_length"))),
            qk_clip: f32v(get(m, &k("attention.qk_clip"))),
            rel_clip: f32v(get(m, &k("attention.relative_clip"))),
            softcap: f32v(get(m, &k("final_logit_softcapping"))),
            attn_res_block_size: u32v(get(m, &k("attn_res.block_size"))),
            situ_beta: f32v(get(m, &k("activation.situ_beta"))),
            situ_linear_beta: f32v(get(m, &k("activation.situ_linear_beta"))),
            expert_weights_scale: f32v(get(m, &k("expert_weights_scale"))),
            log_scale_alpha: f32v(get(m, &k("attention.log_scale_alpha"))),
            log_scale_floor: u32v(get(m, &k("attention.log_scale_floor"))),
            sconv_l_cache: u32v(get(m, &k("shortconv.l_cache"))),
            vocab_unpadded: u32v(get(m, &k("vocab_size_unpadded"))),
            rms_eps: f32v(get(m, &k("attention.layer_norm_rms_epsilon"))),
        }
    }

    /// Layer-visits per token: 0..6, then (6..18) x `n_loop`, then 18..24.
    #[must_use]
    pub fn n_visits(&self) -> u32 {
        self.loop_start
            + (self.loop_end - self.loop_start) * self.n_loop
            + (self.n_layer - self.loop_end)
    }
}
