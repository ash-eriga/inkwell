//! Custom-op MoE path: expert GEMVs run on the GPU over **zero-copy pinned
//! host memory** (the Strata trick) from inside a `GGML_OP_CUSTOM` callback.
//!
//! The op lands on the CPU backend (the only one that supports custom ops);
//! its small srcs (x, ids, weights, scales) arrive on host via scheduler
//! copies, get uploaded, and the result is written into `dst`.

use inkwell_cuda as cu;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const T_IQ2_XXS: i32 = 16;
const T_IQ2_XS: i32 = 17;
const T_IQ1_S: i32 = 19;
const T_IQ2_S: i32 = 22;
const T_NVFP4: i32 = 40;
const T_Q2_0: i32 = 42;

const N_EMBD: usize = 2304;
const N_FF: usize = 1152;
const N_SLOTS: usize = 8;

/// Per-visit weight bindings + shared staging. `userdata` of the custom op.
#[derive(Debug)]
pub struct MoeArgs {
    /// Device ptr to mapped gate_up weights (all experts contiguous).
    pub gu_w: *mut c_void,
    /// Bytes of one expert's gate_up matrix (`nb[2]` of the tensor).
    pub gu_stride: usize,
    /// ggml type of gate_up (IQ2_XXS | IQ1_S).
    pub gu_type: i32,
    /// Same trio for down.
    pub dn_w: *mut c_void,
    /// Bytes of one expert's down matrix.
    pub dn_stride: usize,
    /// ggml type of down (Q2_0 | NVFP4).
    pub dn_type: i32,
    /// Whether the custom op's args include gate_up/down scale tensors.
    pub has_sgu: bool,
    /// See `has_sgu`.
    pub has_sdn: bool,
    /// Host base pointer for gate_up table.
    pub gu_host: *mut u8,
    /// Host base pointer for down table.
    pub dn_host: *mut u8,
    /// Optional VRAM LRU caches (gate_up, down).
    pub caches: std::sync::Mutex<Option<(Cache, Cache)>>,
}

/// Hit/miss stats (debug).
pub static mut HITS: u64 = 0;
/// miss count.
pub static mut MISSES: u64 = 0;

/// LRU cache of resident experts in VRAM: `map[expert] -> slot | -1`.
#[derive(Debug)]
pub struct Cache {
    /// VRAM arena of `k * esz` bytes.
    pub dev: cu::DevBuf,
    /// expert -> slot index (-1 absent).
    pub map: Vec<i32>,
    /// per-slot last-used stamp.
    pub lru: Vec<u64>,
    /// monotonically increasing stamp.
    pub stamp: u64,
    /// bytes per expert.
    pub esz: usize,
    /// per-expert use count (promote at >=2).
    pub uses: Vec<u8>,
}

impl Cache {
    /// New cache holding `k` experts of `esz` bytes.
    #[must_use]
    pub fn new(k: usize, esz: usize) -> Self {
        Self { dev: cu::DevBuf::new_bytes(k * esz), map: vec![-1; 128], lru: vec![0; k], stamp: 0, esz, uses: vec![0; 128] }
    }

    /// `n_slots`.
    fn k(&self) -> usize { self.lru.len() }

    /// Resolve `ptrs` for the selected experts; returns miss list (slot_idx, expert).
    fn resolve(&mut self, ids_v: &[i32], host: *mut u8, ptrs: &mut [usize; N_SLOTS], misses: &mut Vec<(usize, i32)>) {
        for (s, &e) in ids_v.iter().enumerate() {
            let e = e as usize;
            let slot = self.map[e];
            if slot >= 0 {
                unsafe { HITS += 1 };
                self.uses[e] = self.uses[e].saturating_add(1);
                ptrs[s] = unsafe { self.dev.dev_ptr().cast::<u8>().add(slot as usize * self.esz) } as usize;
                self.stamp += 1;
                self.lru[slot as usize] = self.stamp;
            } else {
                unsafe { MISSES += 1 };
                ptrs[s] = unsafe { host.add(e * self.esz) } as usize;
                self.uses[e] = self.uses[e].saturating_add(1);
                if self.uses[e] >= 2 { misses.push((s, e as i32)); }
            }
        }
    }

    /// Promote misses into VRAM (evict LRU); copies are stream-ordered.
    /// Re-points `ptrs` at the cache slot so this call's GEMV reads VRAM
    /// instead of re-streaming the same expert over PCIe (halves miss traffic).
    fn promote(&mut self, misses: &[(usize, i32)], host: *mut u8, ptrs: &mut [usize; N_SLOTS], side: bool) {
        for &(s, e) in misses {
            if self.k() == 0 { break; }
            let e = e as usize;
            let (mut victim, mut oldest) = (0usize, u64::MAX);
            for (i, &t) in self.lru.iter().enumerate() {
                if t < oldest { oldest = t; victim = i; }
            }
            if let Some(old) = self.map.iter().position(|&m| m == victim as i32) {
                self.map[old] = -1;
            }
            unsafe {
                let d = self.dev.dev_ptr().cast::<u8>().add(victim * self.esz).cast();
                let h = host.add(e * self.esz).cast_const().cast();
                if side { cu::memcpy_h2d_async_side(d, h, self.esz) } else { cu::memcpy_h2d_async(d, h, self.esz) }
            }
            ptrs[s] = unsafe { self.dev.dev_ptr().cast::<u8>().add(victim * self.esz) } as usize;
            self.map[e] = victim as i32;
            self.stamp += 1;
            self.lru[victim] = self.stamp;
        }
    }
}

/// Device staging buffers shared by all MoE visits of a step.
struct Staging {
    x: Mutex<cu::DevBuf>,       // [N_EMBD] f32
    ids: Mutex<cu::DevBuf>,     // [N_SLOTS] i32
    w: Mutex<cu::DevBuf>,       // [N_SLOTS] f32
    sgu: Mutex<cu::DevBuf>,     // [N_SLOTS] f32
    sdn: Mutex<cu::DevBuf>,     // [N_SLOTS] f32
    gu_ptrs: Mutex<cu::DevBuf>, // [N_SLOTS] u64
    dn_ptrs: Mutex<cu::DevBuf>, // [N_SLOTS] u64
    outp: Mutex<cu::DevBuf>,    // [N_EMBD] f32
    gu: Mutex<cu::DevBuf>,      // [N_SLOTS][2*N_FF] f32
    hid: Mutex<cu::DevBuf>,     // [N_SLOTS][N_FF] f32
    dn: Mutex<cu::DevBuf>,      // [N_SLOTS][N_EMBD] f32
}

static STAGING: OnceLock<Staging> = OnceLock::new();

fn staging() -> &'static Staging {
    STAGING.get_or_init(|| Staging {
        x: Mutex::new(cu::DevBuf::new_bytes(N_EMBD * 4)),
        ids: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 4)),
        w: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 4)),
        sgu: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 4)),
        sdn: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 4)),
        gu_ptrs: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 8)),
        dn_ptrs: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 8)),
        gu: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 2 * N_FF * 4)),
        hid: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * N_FF * 4)),
        dn: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * N_EMBD * 4)),
        outp: Mutex::new(cu::DevBuf::new_bytes(N_EMBD * 4)),
    })
}

/// `ggml_custom_op_t` — runs on a CPU-backend thread inside `sched_compute`.
///
/// # Safety
/// Called by ggml with `dst->src[]` = the op's args and `userdata = &MoeArgs`.
#[expect(clippy::cast_ptr_alignment, reason = "ggml tensors are f32/i32 aligned")]
unsafe extern "C" fn moe_cb(dst: *mut c_void, _ith: i32, _nth: i32, ud: *mut c_void) {
    let a = unsafe { &*ud.cast::<MoeArgs>() };
    let st = staging();
    // src data ptrs (host; scheduler copied them to the CPU backend)
    let src = |i: i32| unsafe {
        let t = inkwell_ggml::ink_tensor_src(dst, i);
        inkwell_ggml::ink_tensor_data(t)
    };
    let x_h = src(0) as *const f32;
    let ids_h = src(1) as *const i32;
    let w_h = src(2) as *const f32;
    let mut sgu_h: *const f32 = std::ptr::null();
    let mut sdn_h: *const f32 = std::ptr::null();
    if a.has_sgu && a.has_sdn {
        sgu_h = src(3).cast::<f32>();
        sdn_h = src(4).cast::<f32>();
    } else if a.has_sgu {
        sgu_h = src(3).cast::<f32>();
    } else if a.has_sdn {
        sdn_h = src(3).cast::<f32>();
    }

    let t_up = std::time::Instant::now();
    // uploads
    let mut x = st.x.lock().expect("x");
    x.write_bytes_async(unsafe { std::slice::from_raw_parts(x_h.cast::<u8>(), N_EMBD * 4) });
    let ids_v: Vec<i32> = unsafe { std::slice::from_raw_parts(ids_h, N_SLOTS).to_vec() };
    let mut w = st.w.lock().expect("w");
    w.write_bytes_async(unsafe { std::slice::from_raw_parts(w_h.cast::<u8>(), N_SLOTS * 4) });
    let mut sgu = st.sgu.lock().expect("sgu");
    if !sgu_h.is_null() {
        sgu.write_bytes_async(unsafe { std::slice::from_raw_parts(sgu_h.cast::<u8>(), N_SLOTS * 4) });
    }
    let mut sdn = st.sdn.lock().expect("sdn");
    if !sdn_h.is_null() {
        sdn.write_bytes_async(unsafe { std::slice::from_raw_parts(sdn_h.cast::<u8>(), N_SLOTS * 4) });
    }
    // resolve per-slot weight pointers via the VRAM LRU caches
    let mut gu_misses: Vec<(usize, i32)> = Vec::new();
    let mut dn_misses: Vec<(usize, i32)> = Vec::new();
    let mut gu_ptrs = st.gu_ptrs.lock().expect("gu_ptrs");
    let mut dn_ptrs = st.dn_ptrs.lock().expect("dn_ptrs");
    {
        let mut ptrs = [0usize; N_SLOTS];
        let mut caches = a.caches.lock().expect("caches");
        if let Some((gc, dc)) = caches.as_mut() {
            gc.resolve(&ids_v, a.gu_host, &mut ptrs, &mut gu_misses);
            gc.promote(&gu_misses, a.gu_host, &mut ptrs, false);
            gu_ptrs.write_bytes_async(unsafe { std::slice::from_raw_parts(ptrs.as_ptr().cast::<u8>(), N_SLOTS * 8) });
            dc.resolve(&ids_v, a.dn_host, &mut ptrs, &mut dn_misses);
            // dn promotion + ptr upload run on the side stream, overlapped with
            // the gu GEMV; stream 0 waits before the dn GEMV launch
            dc.promote(&dn_misses, a.dn_host, &mut ptrs, true);
            dn_ptrs.write_bytes_async_side(unsafe { std::slice::from_raw_parts(ptrs.as_ptr().cast::<u8>(), N_SLOTS * 8) });
        } else {
            for (s, &e) in ids_v.iter().enumerate() {
                ptrs[s] = unsafe { a.gu_host.add(e as usize * a.gu_stride) } as usize;
            }
            gu_ptrs.write_bytes_async(unsafe { std::slice::from_raw_parts(ptrs.as_ptr().cast::<u8>(), N_SLOTS * 8) });
            for (s, &e) in ids_v.iter().enumerate() {
                ptrs[s] = unsafe { a.dn_host.add(e as usize * a.dn_stride) } as usize;
            }
            dn_ptrs.write_bytes_async(unsafe { std::slice::from_raw_parts(ptrs.as_ptr().cast::<u8>(), N_SLOTS * 8) });
        }
    }

    let t_mid = std::time::Instant::now();
    let mut gu = st.gu.lock().expect("gu");
    let mut hid = st.hid.lock().expect("hid");
    let mut dn = st.dn.lock().expect("dn");
    let mut outp = st.outp.lock().expect("outp");

    if std::env::var("INKWELL_MOE_DUMP").is_ok() {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let v = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if v == 0 {
            let f = std::fs::File::create("/tmp/moe_dump.bin").expect("dump");
            use std::io::Write as _;
            let mut f = std::io::BufWriter::new(f);
            f.write_all(unsafe { std::slice::from_raw_parts(x_h.cast::<u8>(), N_EMBD * 4) }).expect("w");
            f.write_all(unsafe { std::slice::from_raw_parts(ids_h.cast::<u8>(), 32) }).expect("w");
            f.write_all(unsafe { std::slice::from_raw_parts(w_h.cast::<u8>(), 32) }).expect("w");
            if !sgu_h.is_null() { f.write_all(unsafe { std::slice::from_raw_parts(sgu_h.cast::<u8>(), 32) }).expect("w"); }
            if !sdn_h.is_null() { f.write_all(unsafe { std::slice::from_raw_parts(sdn_h.cast::<u8>(), 32) }).expect("w"); }
            eprintln!("DUMPED gu_t={} dn_t={} sgu={} sdn={}", a.gu_type, a.dn_type, a.has_sgu, a.has_sdn);
        }
    }
    if std::env::var("INKWELL_MOE_DBG").is_ok() {
        let ids_v = unsafe { std::slice::from_raw_parts(ids_h, 8) };
        let w_v = unsafe { std::slice::from_raw_parts(w_h, 8) };
        let x_v = unsafe { std::slice::from_raw_parts(x_h, 4) };
        eprintln!("MOECB ids={ids_v:?} w={w_v:?} x={x_v:?} gu_t={} dn_t={}", a.gu_type, a.dn_type);
    }

    let t0 = std::time::Instant::now();
    static SKIP: OnceLock<bool> = OnceLock::new();
    if *SKIP.get_or_init(|| std::env::var("INKWELL_MOE_SKIP").is_ok()) {
        unsafe { std::ptr::write_bytes(inkwell_ggml::ink_tensor_data(dst).cast::<u8>(), 0, N_EMBD * 4) };
        return;
    }
    unsafe {
        match a.gu_type {
            T_IQ2_XXS => cu::mmvq_iq2_xxs_moe_async(gu_ptrs.dev_ptr().cast_const(), x.dev_ptr().cast_const(), &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ1_S => cu::mmvq_iq1_s_moe_async(gu_ptrs.dev_ptr().cast_const(), x.dev_ptr().cast_const(), &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ2_XS => cu::mmvq_iq2_xs_moe_async(gu_ptrs.dev_ptr().cast_const(), x.dev_ptr().cast_const(), &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ2_S => cu::mmvq_iq2_s_moe_async(gu_ptrs.dev_ptr().cast_const(), x.dev_ptr().cast_const(), &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            t => panic!("gate_up type {t}"),
        }
        cu::situ_pair_async(&mut hid, &gu, if sgu_h.is_null() { std::ptr::null() } else { sgu.dev_ptr().cast_const() }, 4.0, 25.0, N_FF, N_SLOTS);
        cu::side_sync_to_main();
        match a.dn_type {
            T_Q2_0 => cu::mmvq_q2_0_moe_i8_async(dn_ptrs.dev_ptr().cast_const(), hid.dev_ptr().cast_const(), &mut dn, N_EMBD, N_FF, N_FF, N_SLOTS),
            T_NVFP4 => cu::mmvq_nvfp4_moe_i8_async(dn_ptrs.dev_ptr().cast_const(), hid.dev_ptr().cast_const(), &mut dn, N_EMBD, N_FF, N_FF, N_SLOTS),
            t => panic!("down type {t}"),
        }
        cu::expert_wsum_async(outp.dev_ptr(), &dn, w.dev_ptr().cast_const(), if sdn_h.is_null() { std::ptr::null() } else { sdn.dev_ptr().cast_const() }, N_EMBD, N_SLOTS);
    }
    if std::env::var("INKWELL_MOE_T").is_ok() {
        unsafe { eprintln!("MOET kern={:?} hits={HITS} misses={MISSES}", t0.elapsed()) };
    }
    let mut v = [0u8; N_EMBD * 4];
    outp.read_bytes(&mut v);
    if std::env::var("INKWELL_MOE_DUMP").is_ok() {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let v = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if v == 0 {
            let f = std::fs::File::create("/tmp/moe_dump.bin").expect("dump");
            use std::io::Write as _;
            let mut f = std::io::BufWriter::new(f);
            f.write_all(unsafe { std::slice::from_raw_parts(x_h.cast::<u8>(), N_EMBD * 4) }).expect("w");
            f.write_all(unsafe { std::slice::from_raw_parts(ids_h.cast::<u8>(), 32) }).expect("w");
            f.write_all(unsafe { std::slice::from_raw_parts(w_h.cast::<u8>(), 32) }).expect("w");
            if !sgu_h.is_null() { f.write_all(unsafe { std::slice::from_raw_parts(sgu_h.cast::<u8>(), 32) }).expect("w"); }
            if !sdn_h.is_null() { f.write_all(unsafe { std::slice::from_raw_parts(sdn_h.cast::<u8>(), 32) }).expect("w"); }
            eprintln!("DUMPED gu_t={} dn_t={} sgu={} sdn={}", a.gu_type, a.dn_type, a.has_sgu, a.has_sdn);
        }
    }
    if std::env::var("INKWELL_MOE_DBG").is_ok() {
        let f: &[f32] = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<f32>(), 8) };
        eprintln!("MOECB out={f:?}");
    }
    unsafe {
        std::ptr::copy_nonoverlapping(v.as_ptr(), inkwell_ggml::ink_tensor_data(dst).cast::<u8>(), v.len());
    }
}

/// Cast the callback to the `usize` the FFI boundary expects.
#[must_use]
pub fn moe_cb_addr() -> usize {
    moe_cb as usize
}
