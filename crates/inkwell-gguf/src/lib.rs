//! GGUF v3 reader: metadata + tensor map over mmap. No deserialization copies.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// Errors from opening or parsing a GGUF file.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum GgufError {
    /// underlying I/O failure
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// file does not start with `GGUF`
    #[error("bad magic")]
    BadMagic,
    /// GGUF format version we do not read
    #[error("unsupported version {0}")]
    Version(u32),
    /// file ended before a field was complete
    #[error("truncated at offset {0}")]
    Truncated(usize),
    /// `GGML_TYPE_*` id with no known block layout
    #[error("unknown ggml type {0}")]
    UnknownType(u32),
    /// element count not a multiple of the block size
    #[error("unaligned tensor {0}")]
    Unaligned(String),
    /// size arithmetic overflowed while indexing the file
    #[error("size overflow in {0}")]
    SizeOverflow(String),
    /// tensor data range extends outside the file
    #[error("tensor {0} data range extends outside the file")]
    TensorOutOfBounds(String),
}

/// A GGUF metadata value (single scalar or homogeneous array).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MetaValue {
    /// `uint8`
    U8(u8),
    /// `int8`
    I8(i8),
    /// `uint16`
    U16(u16),
    /// `int16`
    I16(i16),
    /// `uint32`
    U32(u32),
    /// `int32`
    I32(i32),
    /// `float32`
    F32(f32),
    /// `uint64`
    U64(u64),
    /// `int64`
    I64(i64),
    /// `float64`
    F64(f64),
    /// `bool`
    Bool(bool),
    /// UTF-8 string
    Str(String),
    /// homogeneous array
    Arr(Vec<MetaValue>),
}

/// One tensor's location and shape in the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// tensor name, e.g. `blk.0.attn_gate.weight`
    pub name: String,
    /// dimensions, fastest-varying first (ggml order)
    pub dims: Vec<u64>,
    /// `GGML_TYPE_*` numeric id
    pub ggml_type: u32,
    /// byte offset relative to `data_start`
    pub offset: u64,
    /// total size in bytes
    pub nbytes: u64,
}

/// An opened GGUF file: mmap'd, metadata and tensor index parsed.
#[derive(Debug)]
pub struct GgufFile {
    /// shared memory map of the file
    pub mmap: Mmap,
    /// key-value metadata
    pub meta: HashMap<String, MetaValue>,
    /// tensor index by name
    pub tensors: HashMap<String, TensorInfo>,
    /// absolute offset of the tensor data region
    pub data_start: u64,
}

/// `(elems_per_block, bytes_per_block)` per `GGML_TYPE_*` id.
#[must_use]
pub fn type_block(ggml_type: u32) -> Option<(u64, u64)> {
    Some(match ggml_type {
        0 | 26 => (1, 4),      // F32 | I32
        1 | 25 | 30 => (1, 2), // F16 | I16 | BF16
        2 | 20 => (32, 18),    // Q4_0 | IQ4_NL
        3 => (32, 20),         // Q4_1
        6 => (32, 22),         // Q5_0
        7 => (32, 24),         // Q5_1
        8 => (32, 34),         // Q8_0
        9 => (32, 40),         // Q8_1
        10 => (256, 84),       // Q2_K
        11 => (256, 140),      // Q3_K
        12 => (256, 144),      // Q4_K
        13 => (256, 176),      // Q5_K
        14 => (256, 210),      // Q6_K
        15 => (256, 292),      // Q8_K
        16 | 35 => (256, 66),  // IQ2_XXS | TQ2_0
        17 => (256, 74),       // IQ2_XS
        18 => (256, 98),       // IQ3_XXS
        19 => (256, 50),       // IQ1_S
        21 => (256, 110),      // IQ3_S
        22 => (256, 82),       // IQ2_S
        23 => (256, 136),      // IQ4_XS
        29 => (256, 56),       // IQ1_M
        24 | 43 => (1, 1),     // I8 | F8_E4M3
        27 | 28 => (1, 8),     // I64 | F64
        34 => (256, 54),       // TQ1_0
        39 => (32, 17),        // MXFP4
        40 => (64, 36),        // NVFP4
        41 => (128, 18),       // Q1_0
        42 => (64, 18),        // Q2_0
        _ => return None,
    })
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(GgufError::Truncated(self.pos))?;
        if end > self.b.len() {
            return Err(GgufError::Truncated(self.pos));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed-width slice"),
        ))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("fixed-width slice"),
        ))
    }
    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("fixed-width slice"),
        ))
    }
    fn str(&mut self) -> Result<String, GgufError> {
        let n = usize::try_from(self.u64()?).map_err(|_| GgufError::Truncated(self.pos))?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

impl GgufFile {
    /// Open and index a GGUF file: mmaps it, parses metadata and the tensor table.
    ///
    /// # Errors
    ///
    /// `BadMagic`/`Version` on wrong files, `Truncated`/`UnknownType`/`Unaligned`
    /// on malformed contents, `Io` on filesystem failures.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let mmap = unsafe { Mmap::map(&File::open(path)?)? };
        let mut r = Reader { b: &mmap, pos: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(GgufError::BadMagic);
        }
        let ver = r.u32()?;
        if ver != 3 {
            return Err(GgufError::Version(ver));
        }
        let n_tensors = usize::try_from(r.u64()?).map_err(|_| GgufError::Truncated(r.pos))?;
        let n_kv = usize::try_from(r.u64()?).map_err(|_| GgufError::Truncated(r.pos))?;

        // Count fields are attacker-controlled when opening an untrusted
        // checkpoint. Do not preallocate their full claimed count: every KV
        // entry needs at least a string length, type tag, and scalar payload,
        // so a truncated small file cannot legitimately contain many entries.
        let meta_cap = n_kv.min((r.b.len().saturating_sub(r.pos)) / 12 + 1);
        let mut meta = HashMap::with_capacity(meta_cap);
        for _ in 0..n_kv {
            let key = r.str()?;
            meta.insert(key, Self::value(&mut r)?);
        }

        let tensor_cap = n_tensors.min((r.b.len().saturating_sub(r.pos)) / 32 + 1);
        let mut tensor_list = Vec::with_capacity(tensor_cap);
        for _ in 0..n_tensors {
            let name = r.str()?;
            let n_dims = r.u32()? as usize;
            let dim_cap = n_dims.min((r.b.len().saturating_sub(r.pos)) / 8 + 1);
            let mut dims = Vec::with_capacity(dim_cap);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            let Some((blk, bpb)) = type_block(ggml_type) else {
                return Err(GgufError::UnknownType(ggml_type));
            };
            let elems = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| GgufError::SizeOverflow(format!("tensor {name} element count")))?;
            if !elems.is_multiple_of(blk) {
                return Err(GgufError::Unaligned(name));
            }
            let nbytes = (elems / blk)
                .checked_mul(bpb)
                .ok_or_else(|| GgufError::SizeOverflow(format!("tensor {name} byte size")))?;
            tensor_list.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
                nbytes,
            });
        }
        let data_start = (r.pos as u64)
            .checked_add(31)
            .ok_or_else(|| GgufError::SizeOverflow("tensor data alignment".to_owned()))?
            & !31;
        if data_start > mmap.len() as u64 {
            return Err(GgufError::Truncated(r.pos));
        }
        let mut tensors = HashMap::with_capacity(tensor_list.len());
        for t in tensor_list {
            let rel_end = t
                .offset
                .checked_add(t.nbytes)
                .ok_or_else(|| GgufError::SizeOverflow(format!("tensor {} data range", t.name)))?;
            let abs_end = data_start
                .checked_add(rel_end)
                .ok_or_else(|| GgufError::SizeOverflow(format!("tensor {} data range", t.name)))?;
            if abs_end > mmap.len() as u64 {
                return Err(GgufError::TensorOutOfBounds(t.name));
            }
            tensors.insert(t.name.clone(), t);
        }
        Ok(GgufFile {
            mmap,
            meta,
            tensors,
            data_start,
        })
    }

    fn value(r: &mut Reader) -> Result<MetaValue, GgufError> {
        let v = match r.u32()? {
            0 => MetaValue::U8(r.u8()?),
            1 => MetaValue::I8(i8::from_ne_bytes(
                r.take(1)?.try_into().expect("1-byte slice"),
            )),
            2 => MetaValue::U16(u16::from_le_bytes(
                r.take(2)?.try_into().expect("fixed-width slice"),
            )),
            3 => MetaValue::I16(i16::from_le_bytes(
                r.take(2)?.try_into().expect("fixed-width slice"),
            )),
            4 => MetaValue::U32(r.u32()?),
            5 => MetaValue::I32(r.i32()?),
            6 => MetaValue::F32(f32::from_le_bytes(
                r.take(4)?.try_into().expect("fixed-width slice"),
            )),
            7 => MetaValue::Bool(r.u8()? != 0),
            8 => MetaValue::Str(r.str()?),
            10 => MetaValue::U64(r.u64()?),
            11 => MetaValue::I64(i64::from_le_bytes(
                r.take(8)?.try_into().expect("fixed-width slice"),
            )),
            12 => MetaValue::F64(f64::from_le_bytes(
                r.take(8)?.try_into().expect("fixed-width slice"),
            )),
            9 => {
                let elem_ty = r.u32()?;
                let n = usize::try_from(r.u64()?).map_err(|_| GgufError::Truncated(r.pos))?;
                // Every scalar is at least one byte. Capping the reserve at
                // the remaining file size keeps a forged length from causing
                // a huge allocation before the first bounds check.
                let cap = n.min(r.b.len().saturating_sub(r.pos) + 1);
                let mut items = Vec::with_capacity(cap);
                for _ in 0..n {
                    items.push(Self::scalar(r, elem_ty)?);
                }
                MetaValue::Arr(items)
            }
            t => return Err(GgufError::Version(t)),
        };
        Ok(v)
    }

    fn scalar(r: &mut Reader, ty: u32) -> Result<MetaValue, GgufError> {
        Ok(match ty {
            0 => MetaValue::U8(r.u8()?),
            1 => MetaValue::I8(i8::from_ne_bytes(
                r.take(1)?.try_into().expect("1-byte slice"),
            )),
            2 => MetaValue::U16(u16::from_le_bytes(
                r.take(2)?.try_into().expect("fixed-width slice"),
            )),
            3 => MetaValue::I16(i16::from_le_bytes(
                r.take(2)?.try_into().expect("fixed-width slice"),
            )),
            4 => MetaValue::U32(r.u32()?),
            5 => MetaValue::I32(r.i32()?),
            6 => MetaValue::F32(f32::from_le_bytes(
                r.take(4)?.try_into().expect("fixed-width slice"),
            )),
            7 => MetaValue::Bool(r.u8()? != 0),
            8 => MetaValue::Str(r.str()?),
            10 => MetaValue::U64(r.u64()?),
            11 => MetaValue::I64(i64::from_le_bytes(
                r.take(8)?.try_into().expect("fixed-width slice"),
            )),
            12 => MetaValue::F64(f64::from_le_bytes(
                r.take(8)?.try_into().expect("fixed-width slice"),
            )),
            t => return Err(GgufError::Version(t)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn gguf_with_tensor(dims: &[u64], ggml_type: u32, offset: u64, data: &[u8]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        push_str(&mut out, "x");
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&ggml_type.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(data);
        out
    }

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("inkwell-gguf-test-{name}-{}", std::process::id()));
        std::fs::write(&path, bytes).expect("write test gguf");
        path
    }

    #[test]
    fn rejects_tensor_element_count_overflow() {
        let bytes = gguf_with_tensor(&[u64::MAX, u64::MAX], 0, 0, &[]);
        let path = write_tmp("overflow", &bytes);
        let err =
            GgufFile::open(&path).expect_err("overflowing tensor dimensions must be rejected");
        assert!(matches!(err, GgufError::SizeOverflow(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_tensor_data_past_end_of_file() {
        let bytes = gguf_with_tensor(&[1024], 0, 0, &[]);
        let path = write_tmp("out-of-bounds", &bytes);
        let err = GgufFile::open(&path).expect_err("tensor outside file must be rejected");
        assert!(matches!(err, GgufError::TensorOutOfBounds(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn accepts_minimal_valid_tensor() {
        let bytes = gguf_with_tensor(&[1], 0, 0, &0f32.to_le_bytes());
        let path = write_tmp("valid", &bytes);
        let g = GgufFile::open(&path).expect("valid gguf");
        assert_eq!(g.tensors["x"].nbytes, 4);
        let _ = std::fs::remove_file(path);
    }
}
