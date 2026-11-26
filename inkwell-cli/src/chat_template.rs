//! Inkling-2 chat template rendering and structured output stream parsing.

use inkwell_tokenizer::Tokenizer;

/// One chat turn in an OpenAI-compatible conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Message text content.
    pub content: String,
    /// Optional prior assistant reasoning trace.
    pub reasoning_content: Option<String>,
}

/// Parse `reasoning_effort` (`"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`, or float in `[0.0, 0.99]`).
pub fn parse_effort(raw: Option<&str>) -> Result<f32, String> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(0.0);
    };
    let val = match s.to_ascii_lowercase().as_str() {
        "none" | "off" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        other => other
            .parse::<f32>()
            .map_err(|_| format!("unknown reasoning_effort: {s}"))?,
    };
    if !(0.0..=0.99).contains(&val) {
        return Err(format!("reasoning_effort must be in [0.0, 0.99], got {val}"));
    }
    Ok(val)
}

fn format_effort(effort: f32) -> String {
    if effort == 0.0 {
        "0".to_owned()
    } else {
        let s = format!("{effort:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

/// Render chat messages using the Inkling-2 chat template from GGUF metadata.
pub fn render_chat_prompt(messages: &[ChatMessage], effort: f32) -> Result<String, String> {
    let mut out = String::new();
    let mut effort_emitted = false;
    let effort_block = format!(
        "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>",
        format_effort(effort)
    );

    for msg in messages {
        let role_tok = match msg.role.as_str() {
            "system" | "developer" => "<|message_system|>",
            "user" => "<|message_user|>",
            "assistant" => "<|message_model|>",
            "tool" => "<|message_tool|>",
            other => return Err(format!("unsupported message role: {other}")),
        };
        if !effort_emitted && msg.role != "system" && msg.role != "developer" {
            out.push_str(&effort_block);
            effort_emitted = true;
        }
        if msg.role == "assistant" {
            if let Some(rc) = &msg.reasoning_content {
                if !rc.is_empty() {
                    out.push_str(role_tok);
                    out.push_str("<|content_thinking|>");
                    out.push_str(rc);
                    out.push_str("<|end_message|>");
                }
            }
        }
        if !msg.content.is_empty() || msg.role != "assistant" {
            out.push_str(role_tok);
            out.push_str("<|content_text|>");
            out.push_str(&msg.content);
            out.push_str("<|end_message|>");
        }
        if msg.role == "assistant" {
            out.push_str("<|content_model_end_sampling|>");
        }
    }

    if !effort_emitted {
        out.push_str(&effort_block);
    }

    if effort == 0.0 {
        out.push_str("<|message_model|><|content_text|>");
    }

    Ok(out)
}

/// Output channel for structured Inkling-2 generation tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamChannel {
    /// Assistant reasoning trace (`<|content_thinking|>`).
    Reasoning,
    /// Assistant visible text (`<|content_text|>`).
    Content,
}

/// Action produced when feeding one generated token id to [`InklingStreamParser`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenAction {
    /// Emit this token's decoded bytes on the given channel.
    Emit(StreamChannel),
    /// Control token consumed internally (do not display).
    Skip,
    /// End of assistant turn (`<|content_model_end_sampling|>` or `<|endoftext|>`).
    Stop,
}

/// Stateful token-level parser for Inkling-2's `<|message_model|>` / `<|content_*|>` framing.
#[derive(Debug)]
pub struct InklingStreamParser {
    eos_id: u32,
    eot_id: Option<u32>,
    msg_model_id: Option<u32>,
    content_text_id: Option<u32>,
    content_thinking_id: Option<u32>,
    end_message_id: Option<u32>,
    channel: StreamChannel,
}

impl InklingStreamParser {
    /// Construct a parser bound to the tokenizer's special token ids.
    #[must_use]
    pub fn new(tok: &Tokenizer) -> Self {
        Self {
            eos_id: tok.eos_id(),
            eot_id: tok.token_id("<|endoftext|>"),
            msg_model_id: tok.token_id("<|message_model|>"),
            content_text_id: tok.token_id("<|content_text|>"),
            content_thinking_id: tok.token_id("<|content_thinking|>"),
            end_message_id: tok.token_id("<|end_message|>"),
            channel: StreamChannel::Content,
        }
    }

    /// Classify one generated token id.
    pub fn feed(&mut self, id: u32) -> TokenAction {
        if id == self.eos_id || self.eot_id == Some(id) {
            return TokenAction::Stop;
        }
        if self.msg_model_id == Some(id) || self.end_message_id == Some(id) {
            return TokenAction::Skip;
        }
        if self.content_thinking_id == Some(id) {
            self.channel = StreamChannel::Reasoning;
            return TokenAction::Skip;
        }
        if self.content_text_id == Some(id) {
            self.channel = StreamChannel::Content;
            return TokenAction::Skip;
        }
        TokenAction::Emit(self.channel)
    }
}
