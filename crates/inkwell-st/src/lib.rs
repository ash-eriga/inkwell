//! Minimal safetensors reader: mmap + header JSON + tensor index.
//! Shards stay on disk; tensor bytes are `mmap[data_start + begin .. end]`.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Component, Path};

use memmap2::Mmap;
use serde::de::{self, MapAccess, Visitor};
use serde::Deserialize;
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

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug)]
struct TensorHeader(Vec<(String, HeaderTensor)>);

impl<'de> Deserialize<'de> for TensorHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = TensorHeader;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a safetensors header object")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    if key == "__metadata__" {
                        map.next_value::<de::IgnoredAny>()?;
                    } else {
                        entries.push((key, map.next_value::<HeaderTensor>()?));
                    }
                }
                Ok(TensorHeader(entries))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
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

fn dtype_size(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    })
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
        let hend = hstart.checked_add(hlen).ok_or(StError::Truncated(hstart))?;
        if hend > mmap.len() {
            return Err(StError::Truncated(hend));
        }
        // Parse directly into tensor entries rather than a serde_json::Value
        // DOM. The first shard's 4.5 MB header has ~37k entries; avoiding the
        // DOM removes millions of temporary allocations at startup.
        let hdr: TensorHeader = serde_json::from_slice(&mmap[hstart..hend])?;
        let data_len = (mmap.len() - hend) as u64;
        let mut tensors = HashMap::with_capacity(hdr.0.len());
        for (name, ent) in hdr.0 {
            let begin = ent.data_offsets[0];
            let end = ent.data_offsets[1];
            if end < begin {
                return Err(StError::Malformed(format!(
                    "{name}: end offset before begin"
                )));
            }
            let nbytes = end - begin;
            if nbytes > data_len || begin > data_len {
                return Err(StError::Malformed(format!(
                    "{name}: data range outside file"
                )));
            }
            let elem_size = dtype_size(&ent.dtype).ok_or_else(|| {
                StError::Malformed(format!("{name}: unsupported dtype {}", ent.dtype))
            })?;
            let elems = ent
                .shape
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| {
                    StError::Malformed(format!("{name}: shape element count overflow"))
                })?;
            let expected = elems
                .checked_mul(elem_size)
                .ok_or_else(|| StError::Malformed(format!("{name}: byte size overflow")))?;
            if expected != nbytes {
                return Err(StError::Malformed(format!(
                    "{name}: shape/dtype implies {expected} bytes, range has {nbytes}"
                )));
            }
            tensors.insert(
                name.clone(),
                StTensor {
                    name,
                    dtype: ent.dtype,
                    shape: ent.shape,
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
            let tensors = f
                .tensors
                .iter()
                .map(|(n, t)| (n.clone(), (0usize, t.clone())))
                .collect();
            return Ok(Self {
                files: vec![f],
                tensors,
            });
        }
        let idx_p = path.join("model.safetensors.index.json");
        let idx: Value = serde_json::from_slice(&std::fs::read(&idx_p)?)?;
        let wm = idx
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| StError::Malformed("weight_map".into()))?;
        let mut shards: Vec<String> = wm
            .values()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        shards.sort();
        shards.dedup();
        let mut files = Vec::with_capacity(shards.len());
        let mut tensors = HashMap::with_capacity(wm.len());
        for (i, s) in shards.iter().enumerate() {
            let rel = Path::new(s);
            if rel.is_absolute()
                || rel.components().any(|c| {
                    matches!(
                        c,
                        Component::ParentDir | Component::RootDir | Component::Prefix(_)
                    )
                })
            {
                return Err(StError::Malformed(format!(
                    "unsafe shard path in index: {s}"
                )));
            }
            let f = StFile::open(path.join(rel))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn st_bytes(shape: &[u64], offsets: [u64; 2], data: &[u8]) -> Vec<u8> {
        let shape_s = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let hdr = format!(
            "{{\"x\":{{\"dtype\":\"U8\",\"shape\":[{shape_s}],\"data_offsets\":[{},{}]}}}}",
            offsets[0], offsets[1]
        );
        let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(hdr.as_bytes());
        out.extend_from_slice(data);
        out
    }

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("inkwell-st-test-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).expect("write test safetensors");
        path
    }

    #[test]
    fn accepts_valid_tensor() {
        let path = write_tmp("valid", &st_bytes(&[2], [0, 2], b"ab"));
        let f = StFile::open(&path).expect("valid safetensors");
        assert_eq!(f.data(&f.tensors["x"]), b"ab");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_shape_size_mismatch() {
        let path = write_tmp("mismatch", &st_bytes(&[3], [0, 2], b"ab"));
        let err = StFile::open(&path).expect_err("mismatched byte length must be rejected");
        assert!(matches!(err, StError::Malformed(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_data_range_outside_file() {
        let path = write_tmp("bounds", &st_bytes(&[2], [0, 3], b"ab"));
        let err = StFile::open(&path).expect_err("out-of-bounds range must be rejected");
        assert!(matches!(err, StError::Malformed(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_parent_directory_shard_paths() {
        let dir = std::env::temp_dir().join(format!("inkwell-st-repo-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create repo test dir");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"../outside.safetensors"}}"#,
        )
        .expect("write index");
        let err = StRepo::open(&dir).expect_err("parent shard path must be rejected");
        assert!(matches!(err, StError::Malformed(_)));
        let _ = std::fs::remove_dir_all(dir);
    }
}
