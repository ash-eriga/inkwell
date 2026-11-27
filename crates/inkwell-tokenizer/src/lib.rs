//! Byte-level BPE tokenizer for `inkling2` (`pre` type: GPT-4o splitter).
//! Ports llama.cpp `llama-vocab.cpp` BPE + `unicode.cpp` GPT-4o regex split.

use fancy_regex::Regex;
use inkwell_gguf::{GgufFile, MetaValue};
use rustc_hash::FxHashMap;
use std::sync::Arc;
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
    /// the merge table refers to symbols absent from the token table
    #[error("inconsistent tokenizer merge table")]
    BadMerge,
    /// a byte-level base token is absent from the token table
    #[error("incomplete byte-level tokenizer vocabulary")]
    MissingByteToken,
}

// The active llama.cpp GPT4O branch (llama-vocab.cpp): the o200k case-aware
// pattern, not the HF "gpt-4o" simplification.
const GPT4O_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Byte-level BPE tokenizer (GPT-2 byte map, GPT-4o pre-split, rank merges).
#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<Arc<str>>,
    token_to_id: FxHashMap<Arc<str>, u32>,
    /// `(left_token, right_token) -> (merge_rank, merged_token)`.
    /// Interning merge strings as token ids avoids nearly a million small
    /// allocations for the 424k-merge inkling vocabulary and makes every hot
    /// BPE lookup two integer hashes instead of two string hashes.
    merge_ranks: FxHashMap<(u32, u32), (u32, u32)>,
    byte_token: [u32; 256],
    char_to_byte: [u16; 512],
    split_re: Regex,
    eos_id: u32,
    special_tokens: Vec<(Arc<str>, u32)>,
}

fn meta_arr<'a>(g: &'a GgufFile, key: &'static str) -> Result<&'a [MetaValue], TokError> {
    match g.meta.get(key) {
        Some(MetaValue::Arr(a)) => Ok(a),
        _ => Err(TokError::BadKey(key)),
    }
}

/// The GPT-2 bytes-to-unicode map: printable bytes map to themselves,
/// others (whitespace, controls, >=0x80) map to 0x100+n in insertion order.
/// Also build the reverse table. Mapped values never reach U+0200, so a
/// fixed 512-entry table replaces two hash lookups per decoded character.
fn byte_maps() -> ([char; 256], [u16; 512]) {
    let mut b2c = ['\0'; 256];
    let mut c2b = [u16::MAX; 512];
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
        b2c[usize::from(b)] = c;
        c2b[c as usize] = u16::from(b);
    }
    (b2c, c2b)
}

fn parse_merge_chunk(
    token_to_id: &FxHashMap<Arc<str>, u32>,
    chunk: &[MetaValue],
    base: usize,
) -> Result<Vec<((u32, u32), (u32, u32))>, TokError> {
    let mut entries = Vec::with_capacity(chunk.len());
    let mut merged = String::new();
    for (j, m) in chunk.iter().enumerate() {
        match m {
            MetaValue::Str(s) => {
                // "first second" — split on the space after the first byte
                // (byte 1 may sit inside a multi-byte char; search bytes).
                let bytes = s.as_bytes();
                if bytes.len() < 3 {
                    return Err(TokError::BadMerge);
                }
                let sep = bytes[1..]
                    .iter()
                    .position(|&b| b == b' ')
                    .map(|p| p + 1)
                    .ok_or(TokError::BadMerge)?;
                let (a, b) = (&s[..sep], &s[sep + 1..]);
                if b.is_empty() {
                    return Err(TokError::BadMerge);
                }
                let aid = *token_to_id.get(a).ok_or(TokError::BadMerge)?;
                let bid = *token_to_id.get(b).ok_or(TokError::BadMerge)?;
                merged.clear();
                merged.push_str(a);
                merged.push_str(b);
                let out = *token_to_id.get(merged.as_str()).ok_or(TokError::BadMerge)?;
                entries.push((
                    (aid, bid),
                    (u32::try_from(base + j).expect("merges < 4G"), out),
                ));
            }
            _ => return Err(TokError::BadKey("tokenizer.ggml.merges")),
        }
    }
    Ok(entries)
}

impl Tokenizer {
    /// Build from GGUF metadata (`tokenizer.ggml.{tokens,merges,token_type}`).
    ///
    /// # Errors
    /// `BadKey` when a required array is absent or has the wrong element type,
    /// `BadMerge`/`MissingByteToken` when the vocabulary is internally
    /// inconsistent, or `Regex` when the pre-tokenizer fails to compile.
    ///
    /// # Panics
    /// If the vocabulary exceeds `u32::MAX` tokens (never in practice).
    pub fn from_gguf(g: &GgufFile) -> Result<Self, TokError> {
        let toks = meta_arr(g, "tokenizer.ggml.tokens")?;
        let mut tokens = Vec::with_capacity(toks.len());
        let mut token_to_id = FxHashMap::default();
        token_to_id.reserve(toks.len());
        for (i, t) in toks.iter().enumerate() {
            match t {
                MetaValue::Str(s) => {
                    let id = u32::try_from(i).expect("vocab < 4G");
                    // One shared allocation serves both the id-indexed table
                    // and the reverse lookup key. The old representation made
                    // two additional copies of every vocabulary string.
                    let tok: Arc<str> = Arc::from(s.as_str());
                    token_to_id.insert(tok.clone(), id);
                    tokens.push(tok);
                }
                _ => return Err(TokError::BadKey("tokenizer.ggml.tokens")),
            }
        }

        let merges = meta_arr(g, "tokenizer.ggml.merges")?;
        let mut merge_ranks = FxHashMap::default();
        merge_ranks.reserve(merges.len());
        // Resolving merge strings into token ids is read-only against the
        // vocabulary and dominates tokenizer construction. Split it over a
        // few cores; insertion of the compact integer entries stays serial
        // and deterministic.
        let nth = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(4);
        if nth <= 1 || merges.len() < 16_384 {
            for x in parse_merge_chunk(&token_to_id, merges, 0)? {
                merge_ranks.insert(x.0, x.1);
            }
        } else {
            let chunk_len = merges.len().div_ceil(nth);
            let mut parsed = Vec::with_capacity(nth);
            std::thread::scope(|scope| {
                let mut jobs = Vec::with_capacity(nth);
                for (ci, chunk) in merges.chunks(chunk_len).enumerate() {
                    let tid = &token_to_id;
                    jobs.push(scope.spawn(move || parse_merge_chunk(tid, chunk, ci * chunk_len)));
                }
                for job in jobs {
                    parsed.push(job.join().expect("merge parser panicked"));
                }
            });
            for part in parsed {
                for x in part? {
                    merge_ranks.insert(x.0, x.1);
                }
            }
        }

        let (byte_to_char, char_to_byte) = byte_maps();
        let mut byte_token = [u32::MAX; 256];
        let mut one = [0u8; 4];
        for (b, id) in byte_token.iter_mut().enumerate() {
            let s = byte_to_char[b].encode_utf8(&mut one);
            *id = *token_to_id.get(&*s).ok_or(TokError::MissingByteToken)?;
        }

        let eos_id = match g.meta.get("tokenizer.ggml.eos_token_id") {
            Some(MetaValue::U32(id)) => *id,
            _ => token_to_id
                .get("<|content_model_end_sampling|>")
                .copied()
                .unwrap_or(200_006),
        };
        let mut special_tokens: Vec<(Arc<str>, u32)> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if s.starts_with("<|") && s.ends_with("|>") {
                    Some((s.clone(), u32::try_from(i).expect("vocab < 4G")))
                } else {
                    None
                }
            })
            .collect();
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Self {
            tokens,
            token_to_id,
            merge_ranks,
            byte_token,
            char_to_byte,
            split_re: Regex::new(GPT4O_PATTERN).map_err(|e| TokError::Regex(Box::new(e)))?,
            eos_id,
            special_tokens,
        })
    }

    /// End-of-sequence / end-of-turn token id.
    #[must_use]
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Look up an exact vocabulary token string (including special tokens).
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Encode text while recognizing `<|...|>` special tokens verbatim.
    #[must_use]
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 4 + 8);
        let mut rest = text;
        while !rest.is_empty() {
            // Scan once for the next `<|` instead of running one full-string
            // search per special token (60 tokens on inkling). At each marker,
            // the length-sorted table gives the longest verbatim match.
            let mut next_match: Option<(usize, usize, u32)> = None;
            let mut search = 0usize;
            while let Some(rel) = rest[search..].find("<|") {
                let pos = search + rel;
                if let Some((tok, id)) = self
                    .special_tokens
                    .iter()
                    .find(|(tok, _)| rest[pos..].starts_with(tok.as_ref()))
                {
                    next_match = Some((pos, tok.len(), *id));
                    break;
                }
                search = pos + 2;
                if search >= rest.len() {
                    break;
                }
            }
            match next_match {
                Some((pos, len, id)) => {
                    if pos > 0 {
                        out.extend(self.encode(&rest[..pos]));
                    }
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    out.extend(self.encode(rest));
                    break;
                }
            }
        }
        out
    }

    /// Decode a single token id into raw UTF-8 bytes (for streaming decoders).
    ///
    /// # Errors
    /// `BadId` when `id` is out of range.
    pub fn decode_bytes(&self, id: u32) -> Result<Vec<u8>, TokError> {
        let tok = self.tokens.get(id as usize).ok_or(TokError::BadId(id))?;
        Ok(tok
            .chars()
            .filter_map(|c| self.decode_byte_char(c))
            .collect())
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
        // GPT-4o pre-split runs on the RAW text (llama.cpp order). Initial
        // byte-level token ids are table lookups, so no per-word byte-mapped
        // Unicode string is materialized at all.
        let mut out = Vec::with_capacity(text.len() / 4 + 8);
        for m in self.split_re.find_iter(text) {
            let word = m.expect("regex matches").as_str();
            self.bpe_word_into(word.as_bytes(), &mut out);
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
            bytes.extend(tok.chars().filter_map(|c| self.decode_byte_char(c)));
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The string form of one token id (byte-level space).
    #[must_use]
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_ref())
    }

    #[inline]
    fn decode_byte_char(&self, c: char) -> Option<u8> {
        let x = self
            .char_to_byte
            .get(c as usize)
            .copied()
            .unwrap_or(u16::MAX);
        (x != u16::MAX).then_some(u8::try_from(x).expect("byte table stores u8"))
    }

    /// BPE-merge one pre-token word directly into `out`.
    fn bpe_word_into(&self, bytes: &[u8], out: &mut Vec<u32>) {
        const STACK: usize = 64;
        if bytes.len() == 1 {
            out.push(self.byte_token[usize::from(bytes[0])]);
            return;
        }
        if bytes.len() <= STACK {
            let mut buf = [0u32; STACK];
            for (dst, &b) in buf.iter_mut().zip(bytes) {
                *dst = self.byte_token[usize::from(b)];
            }
            self.bpe_symbols(&mut buf[..bytes.len()], out);
        } else {
            let mut buf: Vec<u32> = bytes
                .iter()
                .map(|&b| self.byte_token[usize::from(b)])
                .collect();
            self.bpe_symbols(&mut buf, out);
        }
    }

    /// Merge a symbol buffer in place and append the final token ids.
    /// Symbol equality is exactly the old string equality because constructor
    /// interning maps every distinct byte-level string to its canonical id.
    fn bpe_symbols(&self, syms: &mut [u32], out: &mut Vec<u32>) {
        let mut len = syms.len();
        while len > 1 {
            // Find the minimum-rank adjacent pair. Strict comparison keeps the
            // first pair on equal rank, matching the old left-to-right scan.
            let mut best: Option<(usize, u32, u32)> = None;
            for i in 0..len - 1 {
                if let Some(&(rank, merged)) = self.merge_ranks.get(&(syms[i], syms[i + 1])) {
                    if best.is_none_or(|(_, br, _)| rank < br) {
                        best = Some((i, rank, merged));
                    }
                }
            }
            let Some((i, _, merged)) = best else { break };
            syms[i] = merged;
            syms.copy_within(i + 2..len, i + 1);
            len -= 1;
        }
        out.extend_from_slice(&syms[..len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_round_trips_all_bytes() {
        let (b2c, c2b) = byte_maps();
        for b in 0usize..256 {
            let c = b2c[b];
            assert_eq!(c2b[c as usize], b as u16);
        }
    }

    #[test]
    fn malformed_short_merge_is_error_not_panic() {
        let map = FxHashMap::default();
        for bad in ["", "x", "x "] {
            let err = parse_merge_chunk(&map, &[MetaValue::Str(bad.to_owned())], 0)
                .expect_err("short merge must be rejected");
            assert!(matches!(err, TokError::BadMerge));
        }
    }
}
