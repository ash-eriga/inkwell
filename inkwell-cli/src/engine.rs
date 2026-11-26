//! High-level inference engine wrapping `Runner`, `Tokenizer`, and `Sampler`.

use crate::chat_template::{InklingStreamParser, StreamChannel, TokenAction};
use inkwell_core::InklingConfig;
use inkwell_gguf::GgufFile;
use inkwell_run::runner::Runner;
use inkwell_sampler::{apply_penalties, Penalties, Rng, Sampler};
use inkwell_tokenizer::Tokenizer;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Sampling and stop controls for one generation request.
#[derive(Debug, Clone)]
pub struct GenParams {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature (`0.0` = fast greedy path).
    pub temperature: f32,
    /// Nucleus sampling threshold (`1.0` = disabled).
    pub top_p: f32,
    /// Top-k sampling cap (`0` = disabled).
    pub top_k: usize,
    /// Min-p threshold (`0.0` = disabled).
    pub min_p: f32,
    /// Repetition penalty (`1.0` = disabled).
    pub repeat_penalty: f32,
    /// Optional deterministic RNG seed.
    pub seed: Option<u64>,
    /// Stop strings (generation halts when output ends with any stop string).
    pub stop: Vec<String>,
    /// When true, parse Inkling-2 `<|content_thinking|>` / `<|content_text|>` framing.
    pub chat_mode: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.0,
            seed: None,
            stop: Vec::new(),
            chat_mode: true,
        }
    }
}

/// One streamed delta piece.
#[derive(Debug, Clone)]
pub enum GenChunk {
    /// Visible assistant / completion text.
    Content(String),
    /// Reasoning text (`<|content_thinking|>`).
    Reasoning(String),
}

/// Completed generation summary.
#[derive(Debug, Clone)]
pub struct GenStats {
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// Wall-clock prefill time in milliseconds.
    pub prefill_ms: f64,
    /// Wall-clock decode time in milliseconds.
    pub decode_ms: f64,
    /// OpenAI `finish_reason`: `"stop"` or `"length"`.
    pub finish_reason: &'static str,
}

impl GenStats {
    /// Prefill throughput in tokens/sec.
    #[must_use]
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_ms > 0.0 {
            self.prompt_tokens as f64 / (self.prefill_ms * 1e-3)
        } else {
            0.0
        }
    }

    /// Decode throughput in tokens/sec.
    #[must_use]
    pub fn decode_tps(&self) -> f64 {
        if self.decode_ms > 0.0 {
            self.completion_tokens as f64 / (self.decode_ms * 1e-3)
        } else {
            0.0
        }
    }
}

/// Incremental UTF-8 decoder that buffers incomplete multi-byte sequences across tokens.
#[derive(Debug, Default)]
struct Utf8StreamDecoder {
    buf: Vec<u8>,
}

impl Utf8StreamDecoder {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.buf[..valid]).expect("valid prefix"),
                        );
                    }
                    match e.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.buf.drain(..valid + bad);
                        }
                        None => {
                            self.buf.drain(..valid);
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    fn flush(&mut self) -> String {
        if self.buf.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }
}

/// Loaded model runner + tokenizer ready for interactive or HTTP serving.
#[derive(Debug)]
pub struct Engine {
    runner: Runner,
    tokenizer: Tokenizer,
    model_id: String,
    max_ctx: usize,
    /// Exact sequence of token ids currently committed in the runner's KV cache + conv state.
    cached_ids: Vec<u32>,
}

impl Engine {
    /// Load a model directory (EXL3 safetensors + GGUF tokenizer sidecar) or standalone `.gguf`.
    pub fn load(model_path: &str, max_ctx: usize) -> Result<Self, String> {
        inkwell_run::prof::init();
        let p = Path::new(model_path);
        let is_dir = p.is_dir();
        let gguf_path = if is_dir {
            if let Ok(env_gg) = std::env::var("INKWELL_TOK_GGUF") {
                env_gg
            } else {
                let local_gg = format!("{model_path}/Inkling-2-Mini-UD-IQ2_XXS.gguf");
                if Path::new(&local_gg).is_file() {
                    local_gg
                } else {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_owned());
                    let fallback = format!("{home}/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf");
                    if Path::new(&fallback).is_file() {
                        fallback
                    } else {
                        local_gg
                    }
                }
            }
        } else {
            model_path.to_owned()
        };
        let gg = {
            let _g = inkwell_run::prof::guard("gguf_open");
            GgufFile::open(&gguf_path)
                .map_err(|e| format!("failed to open GGUF {gguf_path}: {e}"))?
        };
        let tokenizer = {
            let _g = inkwell_run::prof::guard("tokenizer");
            Tokenizer::from_gguf(&gg).map_err(|e| format!("failed to load tokenizer: {e}"))?
        };
        inkwell_run::prof::mark("tokenizer_done");
        let cfg = if is_dir {
            InklingConfig::rexl3()
        } else {
            InklingConfig::from_gguf(&gg)
        };
        let model_id = p
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("inkling-2-mini")
            .to_owned();

        let max_ctx_i64 = i64::try_from(max_ctx).map_err(|_| "max_ctx overflow".to_owned())?;
        let runner = if is_dir {
            let repo = {
                let _g = inkwell_run::prof::guard("strepo_open");
                inkwell_st::StRepo::open(model_path)
                    .map_err(|e| format!("failed to open safetensors repo {model_path}: {e}"))?
            };
            inkwell_run::prof::mark("strepo_done");
            Runner::load_st(&repo, &cfg, max_ctx_i64)
        } else {
            Runner::load(&gg, &cfg, max_ctx_i64)
        };

        inkwell_run::prof::mark("engine_ready");
        Ok(Self {
            runner,
            tokenizer,
            model_id,
            max_ctx,
            cached_ids: Vec::new(),
        })
    }

    /// Model identifier exposed in CLI headers and `/v1/models`.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Configured maximum context length.
    #[must_use]
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Access the loaded tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Run prefill + decode for `prompt_ids`, invoking `on_chunk` for each decoded piece.
    /// Return `false` from `on_chunk` to abort early (e.g., client disconnected).
    pub fn generate<F>(
        &mut self,
        prompt_ids: &[u32],
        params: &GenParams,
        mut on_chunk: F,
    ) -> Result<GenStats, String>
    where
        F: FnMut(GenChunk) -> bool,
    {
        if prompt_ids.is_empty() {
            return Err("prompt cannot be empty".to_owned());
        }
        if prompt_ids.len() >= self.max_ctx {
            return Err(format!(
                "prompt length ({}) exceeds max_ctx ({})",
                prompt_ids.len(),
                self.max_ctx
            ));
        }

        let prompt_len = prompt_ids.len();
        // Reuse existing KV + conv state when the new prompt strictly extends the
        // exact token sequence already committed on the GPU (multi-turn chat / serve).
        let can_reuse = !self.cached_ids.is_empty()
            && self.cached_ids.len() < prompt_len
            && prompt_ids.starts_with(&self.cached_ids);

        let start_pos = if can_reuse {
            self.cached_ids.len()
        } else {
            self.runner.reset();
            self.cached_ids.clear();
            0
        };

        let t_prefill = Instant::now();
        let mut pos: i64 = start_pos as i64;
        if start_pos < prompt_len - 1 {
            let delta = &prompt_ids[start_pos..prompt_len - 1];
            let use_chunked =
                std::env::var_os("INKWELL_NO_PREFILL").is_none() && delta.len() >= 64;
            if use_chunked {
                let _ = self.runner.prefill_chunked(delta, pos);
                pos += delta.len() as i64;
            } else {
                for &tid in delta {
                    self.runner.step_prompt_state(tid, pos);
                    pos += 1;
                }
            }
            self.cached_ids.extend_from_slice(delta);
            self.runner.clear_step_cache();
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        let prefilled_tokens = (prompt_len - 1).saturating_sub(start_pos) + 1;

        let t_decode = Instant::now();
        let max_gen = params
            .max_tokens
            .min(self.max_ctx.saturating_sub(prompt_len));
        let seed = params.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(42, |d| d.as_nanos() as u64)
        });
        let mut rng = Rng::new(seed);
        let sampler = Sampler {
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            temperature: params.temperature,
        };
        let penalties = Penalties {
            repeat: params.repeat_penalty,
            frequency: 0.0,
            presence: 0.0,
            last_n: 64,
        };
        let fast_greedy = params.temperature == 0.0 && (params.repeat_penalty - 1.0).abs() < 1e-6;
        let mut parser = InklingStreamParser::new(&self.tokenizer);
        let eos_id = self.tokenizer.eos_id();
        let eot_id = self.tokenizer.token_id("<|endoftext|>");

        let mut content_dec = Utf8StreamDecoder::default();
        let mut reasoning_dec = Utf8StreamDecoder::default();
        let mut recent: Vec<u32> = Vec::with_capacity(max_gen.min(256));
        let mut content_tail = String::new();
        let max_stop_len = params.stop.iter().map(String::len).max().unwrap_or(0);

        let mut cur_id = prompt_ids[prompt_len - 1];
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length";

        for _ in 0..max_gen {
            let next_id = if fast_greedy {
                self.runner.step_id(cur_id, pos)
            } else {
                let mut logits = self.runner.step(cur_id, pos);
                apply_penalties(&mut logits, &recent, &penalties);
                sampler.sample(&logits, &mut rng)
            };
            self.cached_ids.push(cur_id);
            pos += 1;
            cur_id = next_id;
            completion_tokens += 1;
            recent.push(next_id);

            let action = if params.chat_mode {
                parser.feed(next_id)
            } else if next_id == eos_id || eot_id == Some(next_id) {
                TokenAction::Stop
            } else {
                TokenAction::Emit(StreamChannel::Content)
            };

            match action {
                TokenAction::Stop => {
                    finish_reason = "stop";
                    break;
                }
                TokenAction::Skip => continue,
                TokenAction::Emit(channel) => {
                    let bytes = self
                        .tokenizer
                        .decode_bytes(next_id)
                        .map_err(|e| format!("decode error: {e}"))?;
                    let text = match channel {
                        StreamChannel::Content => content_dec.push(&bytes),
                        StreamChannel::Reasoning => reasoning_dec.push(&bytes),
                    };
                    if text.is_empty() {
                        continue;
                    }
                    if channel == StreamChannel::Content && !params.stop.is_empty() {
                        content_tail.push_str(&text);
                        if let Some(stop_s) = params
                            .stop
                            .iter()
                            .find(|s| !s.is_empty() && content_tail.ends_with(s.as_str()))
                        {
                            let keep = text.len().saturating_sub(stop_s.len());
                            if keep > 0 && !on_chunk(GenChunk::Content(text[..keep].to_owned())) {
                                finish_reason = "stop";
                                break;
                            }
                            finish_reason = "stop";
                            break;
                        }
                        if content_tail.len() > max_stop_len + 64 {
                            let trim = content_tail.len() - (max_stop_len + 32);
                            let idx = content_tail
                                .char_indices()
                                .find(|(i, _)| *i >= trim)
                                .map_or(0, |(i, _)| i);
                            content_tail.drain(..idx);
                        }
                    }
                    let keep_going = match channel {
                        StreamChannel::Content => on_chunk(GenChunk::Content(text)),
                        StreamChannel::Reasoning => on_chunk(GenChunk::Reasoning(text)),
                    };
                    if !keep_going {
                        finish_reason = "stop";
                        break;
                    }
                }
            }
        }

        let rem_r = reasoning_dec.flush();
        if !rem_r.is_empty() {
            let _ = on_chunk(GenChunk::Reasoning(rem_r));
        }
        let rem_c = content_dec.flush();
        if !rem_c.is_empty() {
            let _ = on_chunk(GenChunk::Content(rem_c));
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
        Ok(GenStats {
            prompt_tokens: prefilled_tokens,
            completion_tokens,
            prefill_ms,
            decode_ms,
            finish_reason,
        })
    }
}
