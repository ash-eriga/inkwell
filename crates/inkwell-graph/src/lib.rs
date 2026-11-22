//! Host-side execution schedule for `inkling2_loop`, ported from
//! `src/models/inkling2.cpp` (llama.cpp). This is the op *sequence*
//! the backend executes per forward step — the tensors themselves
//! live behind the pager/FFI, the graph only names them.
//!
//! Per-layer dataflow (verified against `build_layer`):
//! 1. `DepthMix` — attention over the running block stack, queries
//!    `attn_res_query` (or `ffn_res_query`); absent on blk.0.
//! 2. `LoopControl` × 4 — only inside the loop region: mix previous
//!    pass's message into the Q/K/V and FFN inputs
//!    ``(1 - forget)·cur + gain·message`` through `ctrl_down/gain/forget`
//!    `LoRA` heads, driven by `loop_msg_query`.
//! 3. Attention inputs: SWA layers share one normed input; conv-gated
//!    layers (l % 6 == 5) have separate `q/k/v_inp_norm`.
//! 4. `Q = clip(q·q_clip)`; `K,V = shortconv(+ dynamic taps on conv
//!    layers)`; `R = clip(rel)`; full-attention layers scale Q and R
//!    by the log-position `tau`.
//! 5. Rel-bias ``table[head, extent+1, tok] = rel_bias·R − sink``; the
//!    attention gathers it by distance — the upstream OOM bug; our
//!    plan is the indexed-bias kernel (PR #32740 path).
//! 6. ``attn_out = (attn ⊙ σ(gate)) → wo → post_conv``; residual split
//!    every `attn_res_block_size` layers pushes `inpSA` to `blocks`.
//! 7. `DepthMix(ffn_res_query) → ffn_norm → route` (SITU sigmoid +
//!    `exp_probs_b`) → expert GEMV (pager) → `ffn_post_conv` → residual.

use inkwell_core::InklingConfig;

/// Reference to a weight tensor by its GGUF short name.
#[derive(Debug, Clone, Copy)]
pub struct W(pub &'static str);

/// One scheduled operation. Inputs are implicit from the schedule's
/// dataflow position; only weights and parameters are named.
#[derive(Debug)]
pub enum Op {
    /// `token_embd` gather + `token_embd_norm`
    Embed,
    /// `rms_norm(cur) ∥ rms_norm(msg) → ctrl_down → tanh(gain),tanh(forget) → mix`
    LoopControl {
        /// which controller input: 0=q, 1=k, 2=v, 3=ffn
        ctrl: u32,
    },
    /// depth-attention mixing earlier block outputs, per query row
    DepthMix {
        /// `attn_res_query`/`ffn_res_query`/`output_res_query`
        query: W,
        /// query rows: 1 for SWA attention, 3 for conv-gated (q|k|v)
        rows: u32,
    },
    /// rms norm with weight tensor
    Norm {
        /// norm weight
        w: W,
    },
    /// weight-less rms norm over V per head (upstream `ggml_rms_norm` on V)
    VNorm,
    /// matmul / `LoRA` projection producing an activation
    Proj {
        /// weight tensor
        w: W,
        /// semantic tag for the backend kernel picker
        tag: ProjTag,
    },
    /// L2 clip to `threshold` with optional per-channel scale
    Clip {
        /// optional per-channel scale (e.g. `attn_q_clip`)
        scale: Option<W>,
        /// clamp value (`qk_clip`/`rel_clip`)
        threshold: f32,
    },
    /// causal shortconv (`l_cache` taps) + optional dynamic per-head taps
    /// + residual + `1/sqrt(n_conv*(dyn?2:1)+1)` scale
    ShortConv {
        /// conv weight (`attn_k_conv`/`attn_v_conv`/`attn_post_conv`/`ffn_post_conv`)
        w: W,
        /// dynamic tap projection (`attn_*_conv_proj`, conv layers only)
        dynamic: Option<W>,
        /// recurrent-state channel offset
        offset: u32,
    },
    /// multiply by the log-position factor (full-attn layers only)
    TauScale,
    /// `table = rel_bias·R − sink` — feeds the indexed-bias attention
    RelBiasTable,
    /// attention over the visit's arena (SWA ring or full + sinks)
    Attn {
        /// full-attention (indexed rel-bias gather) vs sliding window
        full: bool,
    },
    /// sigmoid gate multiply (`wqkv_gate`)
    GateMul {
        /// gate projection weight
        w: W,
    },
    /// push current `inpSA` to the block stack (every
    /// `attn_res_block_size` layers) — schedule marker, no math
    PushBlock,
    /// residual add `cur += prev`
    Residual,
    /// `MoE` routing: `ffn_gate_inp` scores + `exp_probs_b` + sigmoid,
    /// top-`n_expert_used` selection → pager resolve
    MoeRoute,
    /// expert GEMV `gate_up`/`down` (+scale) through the pager
    MoeExperts,
    /// produce the next pass's cross-pass message via `loop_msg_query`
    LoopMessage,
    /// `output_norm → output` head + `tanh`-softcap + unpad vocab
    Head,
}

/// Projection tags so the backend can pick fused kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjTag {
    /// Q projection
    Q,
    /// K projection
    K,
    /// V projection
    V,
    /// relative-position head
    Rel,
    /// attention sink projection
    Sink,
    /// dynamic conv tap projections
    ConvTap,
    /// attention output
    AttnOut,
    /// the other LoRA/misc projection
    Other,
}

/// One visit of the 48-visit schedule.
#[derive(Debug)]
pub struct Visit {
    /// physical layer
    pub layer: u32,
    /// loop pass (0 outside the loop region)
    pub pass: u32,
    /// ordered ops for this visit
    pub ops: Vec<Op>,
}

/// Build the per-layer op sequence.
fn layer_ops(cfg: &InklingConfig, layer: u32, pass: u32) -> Vec<Op> {
    let conv = layer % 6 == 5;
    let in_loop = layer >= cfg.loop_start && layer < cfg.loop_end;
    let mut ops = Vec::with_capacity(40);

    // 1. depth mix for attention inputs (all layers except blk.0)
    if layer > 0 {
        ops.push(Op::DepthMix {
            query: W("attn_res_query"),
            rows: if conv { 3 } else { 1 },
        });
    }

    // 2. loop control mixing the previous pass's message (q,k,v)
    if in_loop && pass > 0 {
        for ctrl in 0..3 {
            ops.push(Op::LoopControl { ctrl });
        }
    }

    // 3. input norms
    if conv {
        for w in ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"] {
            ops.push(Op::Norm { w: W(w) });
        }
    } else {
        // in-loop SWA norms apply per q/k/v input after control;
        // same op, the backend sees one weight
        ops.push(Op::Norm { w: W("attn_norm") });
    }

    // 4. projections
    ops.push(Op::Proj { w: W("attn_q"), tag: ProjTag::Q });
    ops.push(Op::Proj { w: W("attn_k"), tag: ProjTag::K });
    ops.push(Op::Proj { w: W("attn_v"), tag: ProjTag::V });
    ops.push(Op::Proj { w: W("attn_rel"), tag: ProjTag::Rel });
    ops.push(Op::Proj { w: W("attn_sink_proj"), tag: ProjTag::Sink });

    // 5. clips + conv
    ops.push(Op::Clip { scale: Some(W("attn_q_clip")), threshold: cfg.qk_clip });
    ops.push(Op::Clip { scale: None, threshold: cfg.qk_clip });
    ops.push(Op::ShortConv {
        w: W("attn_k_conv"),
        dynamic: conv.then_some(W("attn_k_conv_proj")),
        offset: 0,
    });
    ops.push(Op::ShortConv {
        w: W("attn_v_conv"),
        dynamic: conv.then_some(W("attn_v_conv_proj")),
        offset: cfg.n_head_kv * cfg.head_dim,
    });
    ops.push(Op::VNorm);
    ops.push(Op::Clip { scale: Some(W("attn_rel_clip")), threshold: cfg.rel_clip });
    if conv {
        ops.push(Op::TauScale);
    }

    // 6. rel-bias table + attention + gate
    ops.push(Op::RelBiasTable);
    ops.push(Op::Attn { full: conv });
    ops.push(Op::GateMul { w: W("attn_gate") });
    ops.push(Op::Proj { w: W("attn_output"), tag: ProjTag::AttnOut });
    ops.push(Op::ShortConv {
        w: W("attn_post_conv"),
        dynamic: None,
        offset: 2 * cfg.n_head_kv * cfg.head_dim,
    });

    // 7. residual/block-stack split
    if layer.is_multiple_of(cfg.attn_res_block_size) {
        ops.push(Op::PushBlock);
    } else {
        ops.push(Op::Residual);
    }

    // 8. MoE
    ops.push(Op::DepthMix { query: W("ffn_res_query"), rows: 1 });
    if in_loop && pass > 0 {
        ops.push(Op::LoopControl { ctrl: 3 });
    }
    ops.push(Op::Norm { w: W("ffn_norm") });
    ops.push(Op::MoeRoute);
    ops.push(Op::MoeExperts);
    ops.push(Op::ShortConv {
        w: W("ffn_post_conv"),
        dynamic: None,
        offset: 2 * cfg.n_head_kv * cfg.head_dim + cfg.n_embd,
    });
    ops.push(Op::Residual);

    ops
}

/// The full forward schedule: embed → 48 visits → head.
#[derive(Debug)]
pub struct Schedule {
    /// embedding + norm op
    pub embed: Op,
    /// 48 visits in execution order
    pub visits: Vec<Visit>,
    /// cross-pass message op, inserted between loop passes
    pub message: Op,
    /// output head op
    pub head: Op,
}

/// Build the decode/prefill schedule for `cfg`.
#[must_use]
pub fn build_schedule(cfg: &InklingConfig) -> Schedule {
    let mut visits = Vec::with_capacity(cfg.n_visits() as usize);
    // pre-loop layers
    for layer in 0..cfg.loop_start {
        visits.push(Visit { layer, pass: 0, ops: layer_ops(cfg, layer, 0) });
    }
    // loop passes 1..=n_loop
    for pass in 1..=cfg.n_loop {
        for layer in cfg.loop_start..cfg.loop_end {
            visits.push(Visit { layer, pass, ops: layer_ops(cfg, layer, pass) });
        }
    }
    // post-loop layers
    for layer in cfg.loop_end..cfg.n_layer {
        visits.push(Visit { layer, pass: 0, ops: layer_ops(cfg, layer, 0) });
    }
    Schedule {
        embed: Op::Embed,
        visits,
        message: Op::LoopMessage,
        head: Op::Head,
    }
}
