//! Inkwell core: model config, weight map, graph builder (host orchestration).
//! Kernels live in CUDA C++ behind FFI — this crate owns scheduling,
//! expert paging (RAM<->VRAM), KV layout, and the spec-decode loop.

use inkwell_gguf::{GgufFile, MetaValue};
use thiserror::Error;

/// Static config of an `inkling2_loop` checkpoint, read from GGUF metadata.
#[derive(Debug, Clone)]
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

/// Errors reading malformed `inkling2_loop` config metadata.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ConfigError {
    /// a required metadata key is absent
    #[error("missing gguf key {0}")]
    MissingKey(String),
    /// a metadata key has an unexpected value type
    #[error("gguf key {0} has wrong type: expected {1}")]
    WrongType(String, &'static str),
    /// a metadata integer cannot be represented as `u32`
    #[error("gguf key {0} integer does not fit u32")]
    IntegerRange(String),
    /// a required metadata array is empty
    #[error("gguf key {0} array is empty")]
    EmptyArray(String),
    /// a required metadata array ends before the required index
    #[error("gguf key {0} array has {1} entries, need index {2}")]
    ArrayTooShort(String, usize, usize),
}

type MetaMap = std::collections::HashMap<String, MetaValue>;

fn get<'a>(m: &'a MetaMap, k: &str) -> Result<&'a MetaValue, ConfigError> {
    m.get(k).ok_or_else(|| ConfigError::MissingKey(k.to_owned()))
}

fn u32v(v: &MetaValue, k: &str) -> Result<u32, ConfigError> {
    match v {
        MetaValue::U32(x) => Ok(*x),
        MetaValue::I32(x) => u32::try_from(*x)
            .map_err(|_| ConfigError::IntegerRange(k.to_owned())),
        _ => Err(ConfigError::WrongType(k.to_owned(), "u32")),
    }
}

fn f32v(v: &MetaValue, k: &str) -> Result<f32, ConfigError> {
    match v {
        MetaValue::F32(x) => Ok(*x),
        _ => Err(ConfigError::WrongType(k.to_owned(), "f32")),
    }
}

fn meta_u32(m: &MetaMap, k: &str) -> Result<u32, ConfigError> {
    u32v(get(m, k)?, k)
}

fn meta_f32(m: &MetaMap, k: &str) -> Result<f32, ConfigError> {
    f32v(get(m, k)?, k)
}

fn arr(m: &MetaMap, k: &str, i: usize) -> Result<u32, ConfigError> {
    match get(m, k)? {
        MetaValue::Arr(a) => {
            if a.is_empty() {
                return Err(ConfigError::EmptyArray(k.to_owned()));
            }
            let v = a.get(i).ok_or_else(|| {
                ConfigError::ArrayTooShort(k.to_owned(), a.len(), i)
            })?;
            u32v(v, k)
        }
        _ => Err(ConfigError::WrongType(k.to_owned(), "array")),
    }
}

impl InklingConfig {
    /// Read `inkling2_loop` config from GGUF metadata.
    ///
    /// # Errors
    /// `ConfigError` when required metadata is missing, mistyped, or malformed.
    pub fn from_gguf(g: &GgufFile) -> Result<Self, ConfigError> {
        let m = &g.meta;
        let p = "inkling2_loop.";
        let k = |s: &str| format!("{p}{s}");
        Ok(InklingConfig {
            n_layer: meta_u32(m, &k("block_count"))?,
            n_loop: meta_u32(m, &k("loop.count"))?,
            loop_start: meta_u32(m, &k("loop.start"))?,
            loop_end: meta_u32(m, &k("loop.end"))?,
            loop_gate_rank: meta_u32(m, &k("loop.gate_rank"))?,
            n_embd: meta_u32(m, &k("embedding_length"))?,
            n_head: arr(m, &k("attention.head_count"), 0)?,
            n_head_kv: arr(m, &k("attention.head_count_kv"), 0)?,
            head_dim: meta_u32(m, &k("attention.key_length"))?,
            n_rel: meta_u32(m, &k("attention.relative_length"))?,
            rel_extent: meta_u32(m, &k("attention.relative_extent"))?,
            swa: meta_u32(m, &k("attention.sliding_window"))?,
            n_expert: meta_u32(m, &k("expert_count"))?,
            n_expert_used: meta_u32(m, &k("expert_used_count"))?,
            expert_ff: meta_u32(m, &k("expert_feed_forward_length"))?,
            vocab: meta_u32(m, &k("vocab_size"))?,
            ctx_train: meta_u32(m, &k("context_length"))?,
            qk_clip: meta_f32(m, &k("attention.qk_clip"))?,
            rel_clip: meta_f32(m, &k("attention.relative_clip"))?,
            softcap: meta_f32(m, &k("final_logit_softcapping"))?,
            attn_res_block_size: meta_u32(m, &k("attn_res.block_size"))?,
            situ_beta: meta_f32(m, &k("activation.situ_beta"))?,
            situ_linear_beta: meta_f32(m, &k("activation.situ_linear_beta"))?,
            expert_weights_scale: meta_f32(m, &k("expert_weights_scale"))?,
            log_scale_alpha: meta_f32(m, &k("attention.log_scale_alpha"))?,
            log_scale_floor: meta_u32(m, &k("attention.log_scale_floor"))?,
            sconv_l_cache: meta_u32(m, &k("shortconv.l_cache"))?,
            vocab_unpadded: meta_u32(m, &k("vocab_size_unpadded"))?,
            rms_eps: meta_f32(m, &k("attention.layer_norm_rms_epsilon"))?,
        })
    }

    /// Layer-visits per token: 0..6, then (6..18) x `n_loop`, then 18..24.
    #[must_use]
    pub fn n_visits(&self) -> u32 {
        self.loop_start
            + (self.loop_end - self.loop_start) * self.n_loop
            + (self.n_layer - self.loop_end)
    }
}

/// `serde_json` available via inkwell-st? no — parse config.json with a tiny
/// reader: fields are fixed constants of the Inkling-2-Mini SAGE checkpoint.
impl InklingConfig {
    /// Build config for the REXL3 safetensors checkpoint (`config.json` parsed
    /// by the caller is unnecessary — the checkpoint fixes every field).
    #[must_use]
    pub fn rexl3() -> Self {
        InklingConfig {
            n_layer: 24, n_loop: 3, loop_start: 6, loop_end: 18, loop_gate_rank: 64,
            n_embd: 2304, n_head: 24, n_head_kv: 8, head_dim: 128, n_rel: 16,
            rel_extent: 1024, swa: 1024, n_expert: 128, n_expert_used: 8,
            expert_ff: 1152, vocab: 201024, ctx_train: 1048576,
            qk_clip: 22.627416997969522, rel_clip: 8.0, softcap: 30.0,
            attn_res_block_size: 6, situ_beta: 4.0, situ_linear_beta: 25.0,
            expert_weights_scale: 8.0, log_scale_alpha: 0.1, log_scale_floor: 8192,
            sconv_l_cache: 4, vocab_unpadded: 200058, rms_eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FIXTURE_SEQ: AtomicUsize = AtomicUsize::new(0);

    enum TestMeta {
        U32(u32),
        F32(f32),
        U32Arr(Vec<u32>),
    }

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_meta(out: &mut Vec<u8>, key: &str, value: &TestMeta) {
        push_str(out, key);
        match value {
            TestMeta::U32(x) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            TestMeta::F32(x) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
            }
            TestMeta::U32Arr(xs) => {
                out.extend_from_slice(&9u32.to_le_bytes());
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                for x in xs {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }

    fn write_fixture(name: &str, entries: &[(&str, TestMeta)]) -> std::path::PathBuf {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            push_meta(&mut bytes, key, value);
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        let seq = FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inkwell-core-config-test-{name}-{}-{seq}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).expect("write config fixture");
        path
    }

    fn valid_entries() -> Vec<(&'static str, TestMeta)> {
        let mut entries: Vec<(&'static str, TestMeta)> = [
            "inkling2_loop.block_count",
            "inkling2_loop.loop.count",
            "inkling2_loop.loop.start",
            "inkling2_loop.loop.end",
            "inkling2_loop.loop.gate_rank",
            "inkling2_loop.embedding_length",
            "inkling2_loop.attention.key_length",
            "inkling2_loop.attention.relative_length",
            "inkling2_loop.attention.relative_extent",
            "inkling2_loop.attention.sliding_window",
            "inkling2_loop.expert_count",
            "inkling2_loop.expert_used_count",
            "inkling2_loop.expert_feed_forward_length",
            "inkling2_loop.vocab_size",
            "inkling2_loop.context_length",
            "inkling2_loop.attn_res.block_size",
            "inkling2_loop.attention.log_scale_floor",
            "inkling2_loop.shortconv.l_cache",
            "inkling2_loop.vocab_size_unpadded",
        ]
        .into_iter()
        .map(|k| (k, TestMeta::U32(1)))
        .collect();
        entries.extend(
            [
                "inkling2_loop.attention.qk_clip",
                "inkling2_loop.attention.relative_clip",
                "inkling2_loop.final_logit_softcapping",
                "inkling2_loop.activation.situ_beta",
                "inkling2_loop.activation.situ_linear_beta",
                "inkling2_loop.expert_weights_scale",
                "inkling2_loop.attention.log_scale_alpha",
                "inkling2_loop.attention.layer_norm_rms_epsilon",
            ]
            .into_iter()
            .map(|k| (k, TestMeta::F32(1.0))),
        );
        entries.push((
            "inkling2_loop.attention.head_count",
            TestMeta::U32Arr(vec![1]),
        ));
        entries.push((
            "inkling2_loop.attention.head_count_kv",
            TestMeta::U32Arr(vec![1]),
        ));
        entries
    }

    fn config_error(name: &str, entries: &[(&str, TestMeta)]) -> ConfigError {
        let path = write_fixture(name, entries);
        let gguf = GgufFile::open(&path).expect("open config fixture");
        let err = InklingConfig::from_gguf(&gguf).expect_err("config must be rejected");
        let _ = std::fs::remove_file(path);
        err
    }

    #[test]
    fn valid_minimal_config_metadata_is_accepted() {
        let entries = valid_entries();
        let path = write_fixture("valid", &entries);
        let gguf = GgufFile::open(&path).expect("open config fixture");
        InklingConfig::from_gguf(&gguf).expect("valid config");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_config_key_is_an_error() {
        let err = config_error("missing", &[]);
        assert!(
            matches!(err, ConfigError::MissingKey(k) if k == "inkling2_loop.block_count")
        );
    }

    #[test]
    fn wrong_config_value_type_is_an_error() {
        let mut entries = valid_entries();
        for (key, value) in &mut entries {
            if *key == "inkling2_loop.block_count" {
                *value = TestMeta::F32(1.0);
            }
        }
        let err = config_error("wrong-type", &entries);
        assert!(
            matches!(err, ConfigError::WrongType(k, "u32") if k == "inkling2_loop.block_count")
        );
    }

    #[test]
    fn empty_config_array_is_an_error() {
        let mut entries = valid_entries();
        for (key, value) in &mut entries {
            if *key == "inkling2_loop.attention.head_count" {
                *value = TestMeta::U32Arr(Vec::new());
            }
        }
        let err = config_error("empty-array", &entries);
        assert!(
            matches!(err, ConfigError::EmptyArray(k) if k == "inkling2_loop.attention.head_count")
        );
    }
}
