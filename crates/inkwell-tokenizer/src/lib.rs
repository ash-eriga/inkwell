//! Byte-level BPE tokenizer for `inkling2` (`pre` type: GPT-4o splitter).
//! Ports llama.cpp `llama-vocab.cpp` BPE + `unicode.cpp` GPT-4o regex split.

use fancy_regex::Regex;
use inkwell_gguf::{GgufFile, MetaValue};
use std::collections::HashMap;
use thiserror::Error;

/// Errors from building or running the tokenizer.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TokError {
    /// required metadata key missing or mistyped
    #[error("missing/bad tokenizer key {0}")]
    BadKey(&'static str),
    /// the pre-tokenizer regex failed to compile
    #[error("regex: {0}")]
    Regex(Box<fancy_regex::Error>),
    /// a token id outside the vocabulary was decoded
    #[error("token id {0} out of range")]
    BadId(u32),
}

// The active llama.cpp GPT4O branch (llama-vocab.cpp): the o200k case-aware
// pattern, not the HF "gpt-4o" simplification.
const GPT4O_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Byte-level BPE tokenizer (GPT-2 byte map, GPT-4o pre-split, rank merges).
#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    byte_to_char: HashMap<u8, char>,
    char_to_byte: HashMap<char, u8>,
    split_re: Regex,
}

fn meta_arr<'a>(g: &'a GgufFile, key: &'static str) -> Result<&'a [MetaValue], TokError> {
    match g.meta.get(key) {
        Some(MetaValue::Arr(a)) => Ok(a),
        _ => Err(TokError::BadKey(key)),
    }
}

/// The GPT-2 bytes-to-unicode map: printable bytes map to themselves,
/// others (whitespace, controls, >=0x80) map to 0x100+n in insertion order.
fn byte_map() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut b2c = HashMap::with_capacity(256);
    let mut c2b = HashMap::with_capacity(256);
    let mut n: u32 = 0;
    for b in 0u32..256 {
        let printable = matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
        let c = if printable {
            char::from_u32(b).expect("bmp char")
        } else {
            let c = char::from_u32(0x100 + n).expect("bmp char");
            n += 1;
            c
        };
        let b = u8::try_from(b).expect("0..255");
        b2c.insert(b, c);
        c2b.insert(c, b);
    }
    (b2c, c2b)
}

impl Tokenizer {
    /// Build from GGUF metadata (`tokenizer.ggml.{tokens,merges,token_type}`).
    ///
    /// # Errors
    /// `BadKey` when a required array is absent or has the wrong element type.
    ///
    /// # Panics
    /// If the vocabulary exceeds `u32::MAX` tokens (never in practice).
    pub fn from_gguf(g: &GgufFile) -> Result<Self, TokError> {
        let toks = meta_arr(g, "tokenizer.ggml.tokens")?;
        let mut tokens = Vec::with_capacity(toks.len());
        let mut token_to_id = HashMap::with_capacity(toks.len());
        for (i, t) in toks.iter().enumerate() {
            match t {
                MetaValue::Str(s) => {
                    let id = u32::try_from(i).expect("vocab < 4G");
                    token_to_id.insert(s.clone(), id);
                    tokens.push(s.clone());
                }
                _ => return Err(TokError::BadKey("tokenizer.ggml.tokens")),
            }
        }
        let merges = meta_arr(g, "tokenizer.ggml.merges")?;
        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            match m {
                MetaValue::Str(s) => {
                    // "first second" — split on the space after the first byte
                    // (byte 1 may sit inside a multi-byte char; search bytes).
                    let sep = s.as_bytes()[1..]
                        .iter()
                        .position(|&b| b == b' ')
                        .map(|p| p + 1);
                    if let Some(sep) = sep {
                        let (a, b) = (s[..sep].to_owned(), s[sep + 1..].to_owned());
                        merge_ranks.insert((a, b), u32::try_from(i).expect("merges < 4G"));
                    }
                }
                _ => return Err(TokError::BadKey("tokenizer.ggml.merges")),
            }
        }
        let (byte_to_char, char_to_byte) = byte_map();
        Ok(Self {
            tokens,
            token_to_id,
            merge_ranks,
            byte_to_char,
            char_to_byte,
            split_re: Regex::new(GPT4O_PATTERN).map_err(|e| TokError::Regex(Box::new(e)))?,
        })
    }

    /// Vocabulary size.
    ///
    /// # Panics
    /// If the vocabulary exceeds `u32::MAX` tokens (never in practice).
    #[must_use]
    pub fn n_tokens(&self) -> u32 {
        u32::try_from(self.tokens.len()).expect("vocab < 4G")
    }

    /// Encode text to token ids (no BOS — inkling2 has none by default).
    ///
    /// # Panics
    /// If BPE produces a merged symbol missing from the vocabulary — that
    /// would mean the merges list is inconsistent with the tokens list.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // GPT-4o pre-split runs on the RAW text (llama.cpp order): the
        // byte-level mapping happens per word afterwards, because the
        // space byte maps to the letter 'Ġ' and would break the regex.
        let mut out = Vec::new();
        for m in self.split_re.find_iter(text) {
            let word = m.expect("regex matches").as_str();
            let mapped: String = word.bytes().map(|b| self.byte_to_char[&b]).collect();
            out.extend(self.bpe_word(&mapped));
        }
        out
    }

    /// Decode token ids back to text (lossy on invalid UTF-8).
    ///
    /// # Errors
    /// `BadId` when a token id is out of range.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = self.tokens.get(id as usize).ok_or(TokError::BadId(id))?;
            bytes.extend(tok.chars().filter_map(|c| self.char_to_byte.get(&c)));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The string form of one token id (byte-level space).
    #[must_use]
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// BPE-merge one pre-token word into token ids.
    fn bpe_word(&self, word: &str) -> Vec<u32> {
        if word.len() == 1 {
            return self.token_to_id.get(word).into_iter().copied().collect();
        }
        // symbols = the word's unicode chars (already byte-level chars)
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            // find the min-rank adjacent pair
            let mut best: Option<(usize, u32)> = None;
            for i in 0..syms.len() - 1 {
                if let Some(&r) = self
                    .merge_ranks
                    .get(&(syms[i].clone(), syms[i + 1].clone()))
                {
                    if best.is_none_or(|(_, br)| r < br) {
                        best = Some((i, r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        syms.iter()
            .map(|s| {
                self.token_to_id
                    .get(s)
                    .copied()
                    .unwrap_or_else(|| panic!("merged symbol not in vocab: {s:?}"))
            })
            .collect()
    }
}
