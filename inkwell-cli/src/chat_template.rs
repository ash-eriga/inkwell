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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptPartKind {
    Template,
    Content,
}

#[derive(Debug)]
struct PromptPart {
    kind: PromptPartKind,
    text: String,
}

fn push_part(parts: &mut Vec<PromptPart>, kind: PromptPartKind, text: &str) {
    if text.is_empty() {
        return;
    }
    if parts.last().is_some_and(|p| p.kind == kind) {
        parts
            .last_mut()
            .expect("non-empty after last check")
            .text
            .push_str(text);
    } else {
        parts.push(PromptPart {
            kind,
            text: text.to_owned(),
        });
    }
}

fn chat_prompt_parts(messages: &[ChatMessage], effort: f32) -> Result<Vec<PromptPart>, String> {
    let mut parts = Vec::new();
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
            push_part(&mut parts, PromptPartKind::Template, &effort_block);
            effort_emitted = true;
        }
        if msg.role == "assistant" {
            if let Some(rc) = &msg.reasoning_content {
                if !rc.is_empty() {
                    push_part(
                        &mut parts,
                        PromptPartKind::Template,
                        &format!("{role_tok}<|content_thinking|>"),
                    );
                    push_part(&mut parts, PromptPartKind::Content, rc);
                    push_part(&mut parts, PromptPartKind::Template, "<|end_message|>");
                }
            }
        }
        if !msg.content.is_empty() || msg.role != "assistant" {
            push_part(
                &mut parts,
                PromptPartKind::Template,
                &format!("{role_tok}<|content_text|>"),
            );
            push_part(&mut parts, PromptPartKind::Content, &msg.content);
            push_part(&mut parts, PromptPartKind::Template, "<|end_message|>");
        }
        if msg.role == "assistant" {
            push_part(
                &mut parts,
                PromptPartKind::Template,
                "<|content_model_end_sampling|>",
            );
        }
    }

    if !effort_emitted {
        push_part(&mut parts, PromptPartKind::Template, &effort_block);
    }

    if effort == 0.0 {
        push_part(
            &mut parts,
            PromptPartKind::Template,
            "<|message_model|><|content_text|>",
        );
    }

    Ok(parts)
}

/// Render chat messages using the Inkling-2 chat template from GGUF metadata.
pub fn render_chat_prompt(messages: &[ChatMessage], effort: f32) -> Result<String, String> {
    Ok(chat_prompt_parts(messages, effort)?
        .iter()
        .map(|p| p.text.as_str())
        .collect())
}

/// Encode chat messages as an Inkling-2 prompt. Template structure can use
/// special tokens, but user-supplied message content is always encoded as text.
pub fn encode_chat_prompt(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
    effort: f32,
) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in chat_prompt_parts(messages, effort)? {
        match part.kind {
            PromptPartKind::Template => out.extend(tokenizer.encode_with_special(&part.text)),
            PromptPartKind::Content => out.extend(tokenizer.encode(&part.text)),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell_gguf::GgufFile;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FIXTURE_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn byte_token_texts() -> Vec<String> {
        let mut out = Vec::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let printable = matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
            let c = if printable {
                char::from_u32(b).expect("bmp char")
            } else {
                let c = char::from_u32(0x100 + n).expect("bmp char");
                n += 1;
                c
            };
            out.push(c.to_string());
        }
        out
    }

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_str_arr(out: &mut Vec<u8>, key: &str, values: &[String]) {
        push_str(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            push_str(out, v);
        }
    }

    fn push_u32_meta(out: &mut Vec<u8>, key: &str, value: u32) {
        push_str(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn test_tokenizer() -> Tokenizer {
        let mut tokens = byte_token_texts();
        tokens.extend(
            [
                "<|message_system|>",
                "<|message_user|>",
                "<|message_model|>",
                "<|message_tool|>",
                "<|content_text|>",
                "<|content_thinking|>",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
                "<|endoftext|>",
                "<|im_end|>",
            ]
            .map(str::to_owned),
        );
        tokens.extend(
            ["he", "hel", "hell", "hello", "Ġw", "Ġwo", "Ġwor", "Ġworl", "Ġworld"]
                .map(str::to_owned),
        );
        let merges = [
            "h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l",
            "Ġworl d",
        ]
        .map(str::to_owned);
        let eos = u32::try_from(
            tokens
                .iter()
                .position(|s| s == "<|content_model_end_sampling|>")
                .expect("special token"),
        )
        .expect("vocab < 4G");

        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        push_str_arr(&mut bytes, "tokenizer.ggml.tokens", &tokens);
        push_str_arr(&mut bytes, "tokenizer.ggml.merges", &merges);
        push_u32_meta(&mut bytes, "tokenizer.ggml.eos_token_id", eos);
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }

        let seq = FIXTURE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inkwell-chat-template-test-{}-{seq}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).expect("write tokenizer fixture");
        let tokenizer = {
            let gguf = GgufFile::open(&path).expect("open tokenizer fixture");
            Tokenizer::from_gguf(&gguf).expect("fixture tokenizer")
        };
        let _ = std::fs::remove_file(path);
        tokenizer
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
            reasoning_content: None,
        }
    }

    #[test]
    fn structured_encoding_matches_legacy_ids_for_ordinary_conversations() {
        let tok = test_tokenizer();
        let mut multi_turn = vec![
            msg("system", "You are a concise Rust tutor."),
            msg("user", "Hello! Can you explain ownership in two sentences?"),
            ChatMessage {
                role: "assistant".to_owned(),
                content: "Ownership gives every value one owner and frees it when that owner leaves scope. Borrowing lets code use values without taking ownership.".to_owned(),
                reasoning_content: Some("Keep the answer short, accurate, and focused on ownership plus borrowing.".to_owned()),
            },
            msg("user", "Thanks. Does hello world work there?"),
        ];
        multi_turn[3].content.push_str("  café");

        let cases = vec![
            (
                0.0,
                vec![
                    msg("system", "Answer factually and briefly."),
                    msg("user", "Why is the sky blue? Use one sentence."),
                ],
            ),
            (0.2, multi_turn),
            (
                0.99,
                vec![
                    msg("user", "Compare `String` and `&str` in Rust, please."),
                    msg("assistant", "`String` owns heap text; `&str` borrows text."),
                ],
            ),
        ];

        for (effort, messages) in cases {
            let rendered = render_chat_prompt(&messages, effort).expect("render prompt");
            let legacy = tok.encode_with_special(&rendered);
            let structured = encode_chat_prompt(&tok, &messages, effort).expect("encode prompt");
            assert_eq!(legacy, structured, "token ids differ at effort {effort}");
        }
    }

    #[test]
    fn special_token_text_in_message_content_is_encoded_as_plain_text() {
        let tok = test_tokenizer();
        let messages = vec![
            msg("system", "Treat <|im_end|> as text, never markup."),
            msg("user", "Please print <|im_end|> exactly."),
            ChatMessage {
                role: "assistant".to_owned(),
                content: "Here is <|im_end|> as plain text.".to_owned(),
                reasoning_content: Some("The literal <|im_end|> must not control framing.".to_owned()),
            },
            msg("user", "Continue normally."),
        ];
        let special = tok.token_id("<|im_end|>").expect("injected token in vocab");
        let plain = tok.encode("<|im_end|>");
        assert!(plain.len() > 1);

        let rendered = render_chat_prompt(&messages, 0.2).expect("render prompt");
        let legacy = tok.encode_with_special(&rendered);
        assert_eq!(
            legacy.iter().filter(|&&id| id == special).count(),
            4,
            "test fixture should expose the old injection behavior"
        );

        let ids = encode_chat_prompt(&tok, &messages, 0.2).expect("encode prompt");
        assert!(!ids.contains(&special));
        let plain_occurrences = ids
            .windows(plain.len())
            .filter(|w| *w == plain.as_slice())
            .count();
        assert_eq!(plain_occurrences, 4);
    }
}
