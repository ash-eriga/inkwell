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

        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.str()?;
            meta.insert(key, Self::value(&mut r)?);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.str()?;
            let n_dims = r.u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            let Some((blk, bpb)) = type_block(ggml_type) else {
                return Err(GgufError::UnknownType(ggml_type));
            };
            let elems: u64 = dims.iter().product();
            if !elems.is_multiple_of(blk) {
                return Err(GgufError::Unaligned(name));
            }
            let nbytes = elems / blk * bpb;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ggml_type,
                    offset,
                    nbytes,
                },
            );
        }
        let data_start = (r.pos as u64 + 31) & !31;
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
                let mut items = Vec::with_capacity(n);
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
