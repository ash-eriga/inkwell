use std::cmp::Ordering;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

pub const DIM: usize = 2304;
pub const VOCAB: usize = 200058;
pub const SOFTCAP: f32 = 30.0;
const RECORD_LEN: usize = 8 + DIM * 4 + 16 * 8;

#[derive(Debug)]
struct HeadDump {
    file: File,
}

static DUMP: OnceLock<Option<Mutex<HeadDump>>> = OnceLock::new();

fn init() -> Option<Mutex<HeadDump>> {
    let path = std::env::var_os("INKWELL_DUMP_HEAD")?;
    let mut file = File::create(&path).expect("create INKWELL_DUMP_HEAD");
    let mut header = Vec::with_capacity(24);
    header.extend_from_slice(b"INKWHEAD");
    header.extend_from_slice(&1u32.to_le_bytes());
    header.extend_from_slice(&u32::try_from(DIM).expect("dim fits u32").to_le_bytes());
    header.extend_from_slice(&u32::try_from(VOCAB).expect("vocab fits u32").to_le_bytes());
    header.extend_from_slice(&SOFTCAP.to_bits().to_le_bytes());
    file.write_all(&header)
        .expect("write INKWELL_DUMP_HEAD header");
    Some(Mutex::new(HeadDump { file }))
}

fn dump() -> &'static Option<Mutex<HeadDump>> {
    DUMP.get_or_init(init)
}

pub fn enabled() -> bool {
    dump().is_some()
}

fn better(a: (u32, f32), b: (u32, f32)) -> bool {
    match a.1.total_cmp(&b.1) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => a.0 > b.0,
    }
}

fn top16(logits: &[f32], chosen: u32) -> [(u32, f32); 16] {
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(16);
    for (i, &v) in logits.iter().enumerate() {
        let e = (u32::try_from(i).expect("vocab index fits u32"), v);
        if top.len() < 16 || better(e, top[top.len() - 1]) {
            let p = top.iter().position(|&q| better(e, q)).unwrap_or(top.len());
            top.insert(p, e);
            top.truncate(16);
        }
    }
    if let Some(p) = top.iter().position(|e| e.0 == chosen) {
        if p != 0 {
            let e = top.remove(p);
            top.insert(0, e);
        }
    } else {
        let v = logits[usize::try_from(chosen).expect("chosen index fits usize")];
        top.insert(0, (chosen, v));
        top.truncate(16);
    }
    let mut out = [(0u32, 0f32); 16];
    for (dst, src) in out.iter_mut().zip(top) {
        *dst = src;
    }
    out
}

pub fn write_record(step: u32, token: u32, x: &[f32], logits: &[f32]) {
    assert_eq!(x.len(), DIM, "head dump x length");
    assert_eq!(logits.len(), VOCAB, "head dump logits length");
    let Some(dump) = dump() else { return };
    let top = top16(logits, token);
    let mut rec = Vec::with_capacity(RECORD_LEN);
    rec.extend_from_slice(&step.to_le_bytes());
    rec.extend_from_slice(&token.to_le_bytes());
    for v in x {
        rec.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    for (i, v) in top {
        rec.extend_from_slice(&i.to_le_bytes());
        rec.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    debug_assert_eq!(rec.len(), RECORD_LEN);
    dump.lock()
        .expect("head dump mutex")
        .file
        .write_all(&rec)
        .expect("write INKWELL_DUMP_HEAD record");
}
