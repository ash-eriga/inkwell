//! Minimal safetensors reader: mmap + header JSON + tensor index.
//! Shards stay on disk; tensor bytes are `mmap[data_start + begin .. end]`.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;
use thiserror::Error;

/// Errors opening or indexing a safetensors file.
#[derive(Debug, Error)]
pub enum StError {
    /// filesystem failure
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// file too short or corrupt
    #[error("truncated at offset {0}")]
    Truncated(usize),
    /// header is not valid JSON
    #[error("bad header json: {0}")]
    Json(#[from] serde_json::Error),
    /// tensor entry missing a required field
    #[error("malformed tensor entry: {0}")]
    Malformed(String),
}

/// One tensor's location and shape in the file.
#[derive(Debug, Clone)]
pub struct StTensor {
    /// tensor name, e.g. `model.llm.layers.0.mlp.experts.3.gate_proj.trellis`
    pub name: String,
    /// safetensors dtype string (`BF16`, `I16`, ...)
    pub dtype: String,
    /// row-major shape as written in the header
    pub shape: Vec<u64>,
    /// byte range relative to `data_start` `[begin, end)`
    pub begin: u64,
    /// byte range relative to `data_start` `[begin, end)`
    pub end: u64,
}

impl StTensor {
    /// `end - begin`
    #[must_use]
    pub fn nbytes(&self) -> u64 {
        self.end - self.begin
    }
    /// element count
    #[must_use]
    pub fn elems(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// An opened safetensors file: mmap'd, header parsed, tensor index built.
#[derive(Debug)]
pub struct StFile {
    /// shared memory map
    pub mmap: Mmap,
    /// tensor index by name
    pub tensors: HashMap<String, StTensor>,
    /// absolute offset of the data region (`8 + header_len`)
    pub data_start: u64,
}

impl StFile {
    /// Open and index a safetensors file.
    ///
    /// # Errors
    /// `Io` on filesystem failures, `Truncated`/`Json`/`Malformed` on bad files.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StError> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        if mmap.len() < 8 {
            return Err(StError::Truncated(mmap.len()));
        }
        let hlen = u64::from_le_bytes(mmap[0..8].try_into().expect("8-byte slice")) as usize;
        let hstart = 8usize;
        let hend = hstart
            .checked_add(hlen)
            .ok_or(StError::Truncated(hstart))?;
        if hend > mmap.len() {
            return Err(StError::Truncated(hend));
        }
        let hdr: Value = serde_json::from_slice(&mmap[hstart..hend])?;
        let obj = hdr
            .as_object()
            .ok_or_else(|| StError::Malformed("header not an object".into()))?;

        let mut tensors = HashMap::with_capacity(obj.len());
        for (name, ent) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = ent
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| StError::Malformed(name.clone()))?
                .to_owned();
            let shape: Vec<u64> = ent
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| StError::Malformed(name.clone()))?
                .iter()
                .filter_map(Value::as_u64)
                .collect();
            let offs = ent
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| StError::Malformed(name.clone()))?;
            let begin = offs.first().and_then(Value::as_u64).unwrap_or(0);
            let end = offs.get(1).and_then(Value::as_u64).unwrap_or(0);
            tensors.insert(
                name.clone(),
                StTensor {
                    name: name.clone(),
                    dtype,
                    shape,
                    begin,
                    end,
                },
            );
        }
        Ok(StFile {
            mmap,
            tensors,
            data_start: hend as u64,
        })
    }

    /// Tensor bytes (absolute mmap slice).
    #[must_use]
    pub fn data(&self, t: &StTensor) -> &[u8] {
        let lo = (self.data_start + t.begin) as usize;
        let hi = (self.data_start + t.end) as usize;
        &self.mmap[lo..hi]
    }
}

/// A sharded safetensors repo: `model.safetensors.index.json` name → shard file.
/// Single-file checkpoints pass the file itself.
#[derive(Debug)]
pub struct StRepo {
    /// one opened file per shard
    pub files: Vec<StFile>,
    /// tensor name → (shard idx, tensor)
    pub tensors: HashMap<String, (usize, StTensor)>,
}

impl StRepo {
    /// Open a repo from a single `.safetensors` file or a directory holding
    /// `*.safetensors` shards + `model.safetensors.index.json`.
    ///
    /// # Errors
    /// `Io`/`Json`/`Malformed`/`Truncated` per `StFile::open`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StError> {
        let path = path.as_ref();
        if path.is_file() {
            let f = StFile::open(path)?;
            let tensors = f.tensors.iter().map(|(n, t)| (n.clone(), (0usize, t.clone()))).collect();
            return Ok(Self { files: vec![f], tensors });
        }
        let idx_p = path.join("model.safetensors.index.json");
        let idx: Value = serde_json::from_slice(&std::fs::read(&idx_p)?)?;
        let wm = idx
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| StError::Malformed("weight_map".into()))?;
        let mut shards: Vec<String> = wm.values().filter_map(Value::as_str).map(str::to_owned).collect();
        shards.sort();
        shards.dedup();
        let mut files = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::with_capacity(wm.len());
        for (i, s) in shards.iter().enumerate() {
            let f = StFile::open(path.join(s))?;
            for (n, t) in &f.tensors {
                tensors.insert(n.clone(), (i, t.clone()));
            }
            files.push(f);
        }
        Ok(Self { files, tensors })
    }

    /// Tensor bytes.
    #[must_use]
    pub fn data(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|(i, t)| self.files[*i].data(t))
    }
    /// Tensor metadata.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&StTensor> {
        self.tensors.get(name).map(|(_, t)| t)
    }
}
