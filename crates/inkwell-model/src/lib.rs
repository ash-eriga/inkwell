//! Per-layer weight map for `inkling2_loop`, resolved from a `GgufFile`'s
//! tensor table. Host-side only: gives each kernel/FFI call typed views into
//! the mmap'd checkpoint, no copies.

use inkwell_core::InklingConfig;
use inkwell_gguf::{GgufFile, TensorInfo};
use thiserror::Error;

/// Errors resolving the model's tensor table.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ModelError {
    /// a tensor the architecture requires is absent from the file
    #[error("missing tensor {0}")]
    MissingTensor(String),
    /// malformed architecture metadata
    #[error("invalid model config: {0}")]
    Config(#[from] inkwell_core::ConfigError),
}

/// Tensor names that repeat per logical layer (`blk.N.` prefix).
/// Tensors every logical layer has (22).
const LAYER_TENSORS: &[&str] = &[
    "attn_gate.weight",
    "attn_q.weight",
    "attn_q_clip.weight",
    "attn_k.weight",
    "attn_k_conv.weight",
    "attn_v.weight",
    "attn_v_conv.weight",
    "attn_rel.weight",
    "attn_rel_bias.weight",
    "attn_rel_clip.weight",
    "attn_sink_proj.weight",
    "attn_output.weight",
    "attn_post_conv.weight",
    "ffn_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_up_exps.weight",
    "ffn_down_exps.weight",
    "ffn_res_query.weight",
    "ffn_post_conv.weight",
    "exp_probs_b.bias",
];

/// Norm variant on non-conv layers (`blk.0` additionally lacks
/// `attn_res_query`; every other layer has both).
const STD_ATTN_TENSORS: &[&str] = &["attn_norm.weight", "attn_res_query.weight"];

/// Expert scale tensors, present only on layers 0..20 (layers 20-23
/// carry unscaled expert weights).
const EXPERT_SCALE_TENSORS: &[&str] = &["ffn_gate_up_exps.scale", "ffn_down_exps.scale"];

/// Conv-gated attention tensors on layers with `l % 6 == 5` (they carry
/// `attn_res_query` and input norms/conv projections instead of `attn_norm`).
const CONV_ATTN_TENSORS: &[&str] = &[
    "attn_res_query.weight",
    "attn_q_inp_norm.weight",
    "attn_k_inp_norm.weight",
    "attn_v_inp_norm.weight",
    "attn_k_conv_proj.weight",
    "attn_v_conv_proj.weight",
];

/// Loop-controller tensors, only on layers in `[loop_start, loop_end)` (13).
const LOOP_TENSORS: &[&str] = &[
    "loop_msg_query.weight",
    "loop_q_ctrl_down.weight",
    "loop_q_ctrl_forget.weight",
    "loop_q_ctrl_gain.weight",
    "loop_k_ctrl_down.weight",
    "loop_k_ctrl_forget.weight",
    "loop_k_ctrl_gain.weight",
    "loop_v_ctrl_down.weight",
    "loop_v_ctrl_forget.weight",
    "loop_v_ctrl_gain.weight",
    "loop_ffn_ctrl_down.weight",
    "loop_ffn_ctrl_forget.weight",
    "loop_ffn_ctrl_gain.weight",
];

/// One logical layer's tensors. Order: `LAYER_TENSORS`, then
/// `attn_res_query` when present, then `LOOP_TENSORS` when present.
#[derive(Debug)]
pub struct LayerWeights<'a> {
    /// resolved tensor views
    pub tensors: Vec<&'a TensorInfo>,
    /// short names aligned with `tensors` (`LAYER_TENSORS` + variants)
    pub names: Vec<&'static str>,
}

/// Non-layered (global) weights.
#[derive(Debug)]
pub struct GlobalWeights<'a> {
    /// `token_embd.weight` — [vocab, embd]
    pub token_embd: &'a TensorInfo,
    /// `token_embd_norm.weight`
    pub token_embd_norm: &'a TensorInfo,
    /// `output_norm.weight`
    pub output_norm: &'a TensorInfo,
    /// `output.weight` — LM head, [vocab, embd]
    pub output: &'a TensorInfo,
    /// `output_res_query.weight`
    pub output_res_query: &'a TensorInfo,
}

/// Full model: global tensors + one `LayerWeights` per logical layer.
#[derive(Debug)]
pub struct ModelWeights<'a> {
    /// static arch config
    pub cfg: InklingConfig,
    /// global tensors
    pub global: GlobalWeights<'a>,
    /// `blk.N.` tensors per layer (24)
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> ModelWeights<'a> {
    /// Resolve all tensors for `inkling2_loop`.
    ///
    /// # Errors
    /// `MissingTensor` if any required tensor is absent, or `Config` if architecture metadata is malformed.
    ///
    /// # Panics
    /// If `LAYER_TENSORS` grows past the per-layer search path (won't).
    pub fn from_gguf(g: &'a GgufFile) -> Result<Self, ModelError> {
        let cfg = InklingConfig::from_gguf(g)?;
        let find = |name: &str| -> Result<&'a TensorInfo, ModelError> {
            g.tensors
                .get(name)
                .ok_or_else(|| ModelError::MissingTensor(name.to_owned()))
        };
        let global = GlobalWeights {
            token_embd: find("token_embd.weight")?,
            token_embd_norm: find("token_embd_norm.weight")?,
            output_norm: find("output_norm.weight")?,
            output: find("output.weight")?,
            output_res_query: find("output_res_query.weight")?,
        };
        let mut layers = Vec::with_capacity(cfg.n_layer as usize);
        for l in 0..cfg.n_layer {
            let mut names: Vec<&'static str> = LAYER_TENSORS.to_vec();
            if l < 20 {
                names.extend(EXPERT_SCALE_TENSORS);
            }
            // layer classes (verified against the checkpoint):
            // - blk.0: attn_norm only
            // - every 6th (l % 6 == 5): conv-gated, no attn_norm
            // - the rest: attn_norm + attn_res_query
            // - loop layers [6, 18): + 13 loop_* controller tensors
            if l % 6 == 5 {
                names.extend(CONV_ATTN_TENSORS);
            } else {
                names.extend(STD_ATTN_TENSORS.iter().take(if l == 0 { 1 } else { 2 }));
            }
            if l >= cfg.loop_start && l < cfg.loop_end {
                names.extend(LOOP_TENSORS);
            }
            let mut tensors = Vec::with_capacity(names.len());
            for &short in &names {
                tensors.push(find(&format!("blk.{l}.{short}"))?);
            }
            layers.push(LayerWeights { tensors, names });
        }
        Ok(Self { cfg, global, layers })
    }

    /// Total weight bytes (all tensors in the file).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.global.token_embd.nbytes
            + self.global.token_embd_norm.nbytes
            + self.global.output_norm.nbytes
            + self.global.output.nbytes
            + self.global.output_res_query.nbytes
            + self
                .layers
                .iter()
                .flat_map(|l| l.tensors.iter())
                .map(|t| t.nbytes)
                .sum::<u64>()
    }

    /// Bytes of the `MoE` expert tensors only (the RAM-resident stream set).
    #[must_use]
    pub fn expert_bytes(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|l| l.tensors.iter())
            .filter(|t| t.name.contains("exps."))
            .map(|t| t.nbytes)
            .sum()
    }
}
