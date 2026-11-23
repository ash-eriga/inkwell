//! Custom-op MoE path — **device-side dispatch**: the op now runs on the CUDA
//! backend (fork supports GGML_OP_CUSTOM), so `dst->src[]` data pointers are
//! device pointers and everything below is a stream-ordered enqueue:
//! resolve(ids→ptrs) → alloc(LRU evict, on device) → gather(host→VRAM) →
//! GEMVs → wsum into `dst`. Zero host syncs, zero backend splits, and the
//! whole thing is CUDA-graph capturable.

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
    /// Optional VRAM LRU caches (gate_up, down) — device-resident state.
    pub caches: std::sync::Mutex<Option<(Cache, Cache)>>,
}

/// LRU expert cache kept entirely in device memory: the resolve/alloc/gather
/// kernels read and mutate this state inside the compute stream.
#[derive(Debug)]
pub struct Cache {
    /// VRAM arena of `k * esz` bytes.
    pub dev: cu::DevBuf,
    /// `map_dev[expert] -> slot | -1` (i32[128]).
    pub map_dev: cu::DevBuf,
    /// `lru_dev[slot]` last-used stamp (u64[k]).
    pub lru_dev: cu::DevBuf,
    /// `stamp_dev` monotonic counter (u64[1]).
    pub stamp_dev: cu::DevBuf,
    /// `rev_dev[slot] -> expert | -1` (i32[k]).
    pub rev_dev: cu::DevBuf,
    /// `missq`: i32[18] — [0]=count, then (slot, expert) pairs.
    pub missq_dev: cu::DevBuf,
    /// `work`: u64[16] — (src, dst) pairs for the gather kernel.
    pub work_dev: cu::DevBuf,
    /// `k` resident slots.
    pub k: usize,
    /// bytes per expert.
    pub esz: usize,
}

impl Cache {
    /// New cache holding `k` experts of `esz` bytes.
    #[must_use]
    pub fn new(k: usize, esz: usize) -> Self {
        let map_dev = cu::DevBuf::new_bytes(128 * 4);
        map_dev.write_bytes_async(&[0xFFu8; 128 * 4]);
        let rev_dev = cu::DevBuf::new_bytes(k * 4);
        rev_dev.write_bytes_async(&vec![0xFFu8; k * 4]);
        Self {
            dev: cu::DevBuf::new_bytes(k * esz),
            map_dev,
            lru_dev: cu::DevBuf::new_bytes(k * 8),
            stamp_dev: cu::DevBuf::new_bytes(8),
            rev_dev,
            missq_dev: cu::DevBuf::new_bytes(72),
            work_dev: cu::DevBuf::new_bytes(128),
            k,
            esz,
        }
    }
}

/// Device staging buffers shared by all MoE visits of a step.
struct Staging {
    stats: cu::DevBuf,
    gu_ptrs: Mutex<cu::DevBuf>, // [N_SLOTS] u64
    dn_ptrs: Mutex<cu::DevBuf>, // [N_SLOTS] u64
    gu: Mutex<cu::DevBuf>,      // [N_SLOTS][2*N_FF] f32
    hid: Mutex<cu::DevBuf>,     // [N_SLOTS][N_FF] f32
    dn: Mutex<cu::DevBuf>,      // [N_SLOTS][N_EMBD] f32
    xh: Mutex<cu::DevBuf>,      // [N_SLOTS][N_EMBD] f32 — EXL3 hadamard staging
    exl3_stage: Mutex<cu::DevBuf>, // [N_SLOTS×3][672768] staged expert weights
    sdescs_g: Mutex<cu::DevBuf>,   // staged desc tables (indexed by ids[slot])
    sdescs_u: Mutex<cu::DevBuf>,
    sdescs_d: Mutex<cu::DevBuf>,
    exl3_map: Mutex<cu::DevBuf>,   // [N_SLOTS] i32 — slot→cache slot (cached path)
    exl3_miss: Mutex<cu::DevBuf>,  // [N_SLOTS] i32 — miss flags (cached path)
}

static STAGING: OnceLock<Staging> = OnceLock::new();

/// Eagerly allocate staging buffers (must run before CUDA-graph capture).
pub fn warm_staging() { staging(); }

fn staging() -> &'static Staging {
    STAGING.get_or_init(|| Staging {
        stats: unsafe { let d = cu::DevBuf::new_bytes(16); d.zero(); d },
        gu_ptrs: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 8)),
        dn_ptrs: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 8)),
        gu: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 2 * N_FF * 4)),
        hid: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * N_FF * 4)),
        dn: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * N_EMBD * 4)),
        xh: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * N_EMBD * 4)),
        exl3_stage: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 3 * EXL3_STAGE_PROJ)),
        sdescs_g: Mutex::new(cu::DevBuf::new_bytes(128 * 32)),
        sdescs_u: Mutex::new(cu::DevBuf::new_bytes(128 * 32)),
        sdescs_d: Mutex::new(cu::DevBuf::new_bytes(128 * 32)),
        exl3_map: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 4)),
        exl3_miss: Mutex::new(cu::DevBuf::new_bytes(N_SLOTS * 4)),
    })
}

/// `ggml_custom_op_t` — runs on the CUDA backend inside `sched_compute`.
/// Everything enqueued on the default stream; returns immediately.
///
/// # Safety
/// Called by ggml with `dst->src[]` = the op's args and `userdata = &MoeArgs`.
#[expect(clippy::cast_ptr_alignment, reason = "ggml tensors are f32/i32 aligned")]
unsafe extern "C" fn moe_cb(dst: *mut c_void, _ith: i32, _nth: i32, ud: *mut c_void) {
    let a = unsafe { &*ud.cast::<MoeArgs>() };
    let st = staging();
    unsafe { cu::set_stream(inkwell_ggml::cuda_stream()) };
    if std::env::var("INKWELL_MOE_PTRS").is_ok() {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    }
    // src data ptrs — DEVICE pointers on the CUDA backend
    let src = |i: i32| unsafe {
        let t = inkwell_ggml::ink_tensor_src(dst, i);
        inkwell_ggml::ink_tensor_data(t)
    };
    let x_dev = src(0) as *const c_void;
    let ids_dev = src(1) as *const i32;
    let w_dev = src(2) as *const c_void;
    let mut sgu_dev: *const c_void = std::ptr::null();
    let mut sdn_dev: *const c_void = std::ptr::null();
    if a.has_sgu && a.has_sdn {
        sgu_dev = src(3);
        sdn_dev = src(4);
    } else if a.has_sgu {
        sgu_dev = src(3);
    } else if a.has_sdn {
        sdn_dev = src(3);
    }

    let gu_ptrs = st.gu_ptrs.lock().expect("gu_ptrs");
    let dn_ptrs = st.dn_ptrs.lock().expect("dn_ptrs");
    unsafe {
        let mut caches = a.caches.lock().expect("caches");
        if std::env::var("INKWELL_MOE_NOCACHE").is_ok() { *caches = None; }
        if let Some((gc, dc)) = caches.as_mut() {
            let dbg = std::env::var("INKWELL_MOE_CHECK").is_ok();
            for (c, ptrs, host) in [(&*gc, &*gu_ptrs, a.gu_host), (&*dc, &*dn_ptrs, a.dn_host)] {
                cu::memset_dev_async(c.missq_dev.dev_ptr(), 0, 4);
                cu::moe_resolve_async(ids_dev, c.map_dev.dev_ptr().cast(),
                    c.lru_dev.dev_ptr().cast(), c.stamp_dev.dev_ptr().cast_const().cast(),
                    c.dev.dev_ptr() as u64, host as u64, c.esz as i64,
                    ptrs.dev_ptr().cast(), c.missq_dev.dev_ptr().cast(), N_SLOTS);
                cu::moe_alloc_async(c.missq_dev.dev_ptr().cast_const().cast(),
                    c.map_dev.dev_ptr().cast(), c.lru_dev.dev_ptr().cast(),
                    c.stamp_dev.dev_ptr().cast(), c.rev_dev.dev_ptr().cast(),
                    c.dev.dev_ptr() as u64, host as u64, c.esz as i64,
                    ptrs.dev_ptr().cast(), c.work_dev.dev_ptr().cast(), c.k);
                if dbg { cu::sync_check("dispatch"); }
            }
            // gu gather inline on the main stream (its GEMV is next); dn
            // gather runs on the side stream, hidden under gu GEMV + situ,
            // and joined by wait_gather_done right before the dn GEMV.
            let stats = if std::env::var("INKWELL_MOE_STATS").is_ok() {
                st.stats.dev_ptr().cast()
            } else { std::ptr::null_mut() };
            // stats[0] = bytes copied, stats[1] = raw miss count (incl. >8)
            for c in [&*gc, &*dc] {
                cu::moe_gather_async(c.missq_dev.dev_ptr().cast_const().cast(),
                    c.work_dev.dev_ptr().cast_const().cast(), c.esz as i64, stats);
            }
        } else {
            cu::moe_resolve_async(ids_dev, std::ptr::null(), std::ptr::null_mut(), std::ptr::null(),
                0, a.gu_host as u64, a.gu_stride as i64,
                gu_ptrs.dev_ptr().cast(), std::ptr::null_mut(), N_SLOTS);
            cu::moe_resolve_async(ids_dev, std::ptr::null(), std::ptr::null_mut(), std::ptr::null(),
                0, a.dn_host as u64, a.dn_stride as i64,
                dn_ptrs.dev_ptr().cast(), std::ptr::null_mut(), N_SLOTS);
        }
    }

    if std::env::var("INKWELL_MOE_SKIP").is_ok() {
        unsafe { cu::memset_dev_async(inkwell_ggml::ink_tensor_data(dst), 0, N_EMBD * 4) };
        return;
    }
    let caches = a.caches.lock().expect("caches").is_some();
    unsafe {
        if std::env::var("INKWELL_MOE_STATS").is_ok() {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 24 == 0 {
                let mut b = [0u8; 16];
                st.stats.read_bytes(&mut b);
                let v = u64::from_le_bytes(b[..8].try_into().unwrap());
                let m = u64::from_le_bytes(b[8..].try_into().unwrap());
                eprintln!("MOESTATS visit={n} gathered={:.1}MiB misses_total={m}", v as f64/1048576.0);
            }
        }
        let mut gu = st.gu.lock().expect("gu");
        let mut hid = st.hid.lock().expect("hid");
        let mut dn = st.dn.lock().expect("dn");
        match a.gu_type {
            T_IQ2_XXS if std::env::var_os("INKWELL_IQ2XXS_FP").is_some() =>
                cu::mmvq_iq2_xxs_moe_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ2_XXS => cu::mmvq_iq2_xxs_moe_i8_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ1_S => cu::mmvq_iq1_s_moe_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ2_XS if std::env::var_os("INKWELL_IQ2XS_FP").is_some() => {
                cu::mmvq_iq2_xs_moe_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS)
            }
            T_IQ2_XS => cu::mmvq_iq2_xs_moe_i8_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            T_IQ2_S if std::env::var_os("INKWELL_IQ2S_FP").is_some() => {
                cu::mmvq_iq2_s_moe_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS)
            }
            T_IQ2_S => cu::mmvq_iq2_s_moe_i8_async(gu_ptrs.dev_ptr().cast_const(), x_dev, &mut gu, 2 * N_FF, N_EMBD, 0, N_SLOTS),
            t => panic!("gate_up type {t}"),
        }
        cu::situ_pair_async(&mut hid, &gu, sgu_dev, 4.0, 25.0, N_FF, N_SLOTS);
        match a.dn_type {
            T_Q2_0 => cu::mmvq_q2_0_moe_i8_async(dn_ptrs.dev_ptr().cast_const(), hid.dev_ptr().cast_const(), &mut dn, N_EMBD, N_FF, N_FF, N_SLOTS),
            T_NVFP4 => cu::mmvq_nvfp4_moe_i8_async(dn_ptrs.dev_ptr().cast_const(), hid.dev_ptr().cast_const(), &mut dn, N_EMBD, N_FF, N_FF, N_SLOTS),
            t => panic!("down type {t}"),
        }
        cu::expert_wsum_async(inkwell_ggml::ink_tensor_data(dst), &dn, w_dev, sdn_dev, N_EMBD, N_SLOTS);
        if std::env::var("INKWELL_MOE_CHECK").is_ok() { cu::sync_check("gemv"); }
        if std::env::var("INKWELL_MOE_PTRS").is_ok() {
            static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let mut g = [0u64; N_SLOTS];
                let mut d = [0u64; N_SLOTS];
                gu_ptrs.read_bytes(unsafe { std::slice::from_raw_parts_mut(g.as_mut_ptr().cast::<u8>(), N_SLOTS * 8) });
                dn_ptrs.read_bytes(unsafe { std::slice::from_raw_parts_mut(d.as_mut_ptr().cast::<u8>(), N_SLOTS * 8) });
                let caches = a.caches.lock().expect("c");
                let (gk, dk) = caches.as_ref().map_or((0, 0), |(x, y)| (x.k, y.k));
                eprintln!("PTRS gu_h={:p} gu_arena={:?} dn_h={:p} dn_arena={:?} k=({gk},{dk})",
                    a.gu_host, caches.as_ref().map(|c| c.0.dev.dev_ptr()),
                    a.dn_host, caches.as_ref().map(|c| c.1.dev.dev_ptr()));
                for i in 0..N_SLOTS {
                    eprintln!("  s{i}: gu={:#x} dn={:#x}", g[i], d[i]);
                }
            }
        }
    }
}

/// Cast the callback to the `usize` the FFI boundary expects.
#[must_use]
pub fn moe_cb_addr() -> usize {
    moe_cb as usize
}

/// Per-expert EXL3 descriptor tables for one layer visit: pinned u64[128][4]
/// `{trellis_ptr, suh_ptr, svh_ptr, ka | halfk<<8}` — kernels read the desc
/// then stream the trellis over PCIe zero-copy.
#[derive(Debug)]
pub struct MoeArgsExl3 {
    /// gate_proj descriptors
    pub descs_g: cu::PinnedBuf,
    /// up_proj descriptors
    pub descs_u: cu::PinnedBuf,
    /// down_proj descriptors
    pub descs_d: cu::PinnedBuf,
    /// descriptors already point into device memory — staging is a no-op
    pub on_dev: bool,
    /// per-layer VRAM cache of staged expert blobs (pinned layers only)
    pub cache: Option<Exl3Cache>,
    /// per-expert blob stride in bytes (max trel_b + suh+svh) for `cache` sizing
    pub max_slot: usize,
    /// ordinal of this MoE layer (for INKWELL_MOE_IDLOG histograms)
    pub layer: i32,
}

/// Expert-usage histograms: HIST[layer][expert] counts, only when
/// INKWELL_MOE_IDLOG=path is set and the cb runs eagerly (no cuda graphs).
static MOE_HIST: std::sync::Mutex<Option<Box<[[u64; 128]; 32]>>> = std::sync::Mutex::new(None);

/// Accumulate one visit's ids into MOE_HIST (host side, eager mode only).
fn moe_idlog(ids_dev: *const i32, layer: i32) {
    use std::io::Write as _;
    let path = match std::env::var("INKWELL_MOE_IDLOG") { Ok(v) => v, Err(_) => return };
    unsafe extern "C" { fn cudaMemcpy(d: *mut c_void, s: *const c_void, n: usize, k: i32) -> i32; }
    let mut h = [0i32; 16];
    unsafe { cudaMemcpy(h.as_mut_ptr().cast(), ids_dev.cast(), 64, 2) };
    let n = h[0..8].iter().filter(|&&e| (0..128).contains(&e)).count();
    let mut g = MOE_HIST.lock().expect("hist");
    let hist = g.get_or_insert_with(|| Box::new([[0u64; 128]; 32]));
    let l = layer as usize;
    if l < 32 { for &e in &h[..n] { hist[l][e as usize] += 1; } }
    let tokens: u64 = hist.iter().map(|r| r.iter().sum::<u64>()).sum::<u64>() / 8;
    if tokens % 64 == 0 && tokens > 0 {
        if let Ok(mut f) = std::fs::File::create(&path) {
            for (li, row) in hist.iter().enumerate() {
                let mut v: Vec<(u32, u64)> = row.iter().enumerate().filter(|(_, &c)| c > 0)
                    .map(|(e, &c)| (e as u32, c)).collect();
                v.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
                let _ = writeln!(f, "L{li}: {:?}", v.iter().map(|(e,c)| format!("{e}:{c}")).collect::<Vec<_>>());
            }
        }
    }
}

/// Per-layer VRAM cache: `k` slots × 3 projs × EXL3_STAGE_PROJ bytes,
/// round-robin replacement tagged by expert id.
#[derive(Debug)]
pub struct Exl3Cache {
    /// `k * 3 * EXL3_STAGE_PROJ` bytes of staged expert blobs.
    pub buf: cu::DevBuf,
    /// `k` i32 expert ids (init 0xFF).
    pub tag: cu::DevBuf,
    /// per-expert use counts (i32[128]) for LFU eviction.
    pub cnt: cu::DevBuf,
    /// slot count; must be ≥ N_SLOTS to keep per-visit slots distinct.
    pub k: i32,
    /// bytes per slot per proj = this layer's max expert blob.
    pub slot: usize,
    /// predictive prefetch state (side-stream copies of last-visit ids)
    pub prefetch: Option<Exl3Prefetch>,
}

/// Per-layer prefetch bank + seqlock for host-issued side-stream promotion.
#[derive(Debug)]
pub struct Exl3Prefetch {
    /// last visit's router ids (8 i32) — prediction source
    pub hist: cu::DevBuf,
    /// 8 slots × 3 projs × slot bytes of prefetched expert blobs
    pub bank: cu::DevBuf,
    /// expert id held in each bank slot (8 i32, init 0xFF)
    pub ptag: cu::DevBuf,
    /// 1 i32: 0 while the bank is being (re)written, 1 when complete
    pub pready: cu::DevBuf,
    /// private resolve outputs for the prefetch pass
    pub pf_map: cu::DevBuf,
    /// private miss flags
    pub pf_miss: cu::DevBuf,
}

impl Exl3Prefetch {
    /// Allocate the prefetch buffers (bank = `slot` bytes × 8 × 3).
    #[must_use]
    pub fn new(slot: usize) -> Option<Self> {
        Some(Self {
            hist: cu::DevBuf::try_new_bytes(32)?,
            bank: cu::DevBuf::try_new_bytes(8 * 3 * slot)?,
            ptag: { let mut d = cu::DevBuf::try_new_bytes(32)?; d.write_bytes(&[0xFFu8; 32]); d },
            pready: cu::DevBuf::try_new_bytes(4)?,
            pf_map: cu::DevBuf::try_new_bytes(32)?,
            pf_miss: { let mut d = cu::DevBuf::try_new_bytes(40)?; d.write_bytes(&[0u8; 40]); d },
        })
    }
}

/// Bytes per cache slot per projection (trellis+suh+svh, w≤32 worst case).
pub const EXL3_STAGE_PROJ: usize = 1334016;

/// `ggml_custom_op_t` for the EXL3 MoE path.
///
/// # Safety
/// Called by ggml with `dst->src[]` = `[x, ids, w]` and `userdata = &MoeArgsExl3`.
#[expect(clippy::cast_ptr_alignment, reason = "ggml tensors are f32/i32 aligned")]
unsafe extern "C" fn moe_cb_exl3(dst: *mut c_void, _ith: i32, _nth: i32, ud: *mut c_void) {
    let a = unsafe { &*ud.cast::<MoeArgsExl3>() };
    let st = staging();
    unsafe { cu::set_stream(inkwell_ggml::cuda_stream()) };
    let src = |i: i32| unsafe {
        let t = inkwell_ggml::ink_tensor_src(dst, i);
        inkwell_ggml::ink_tensor_data(t)
    };
    let x_dev = src(0) as *const f32;
    let ids_dev = src(1) as *const i32;
    let w_dev = src(2) as *const f32;
    if std::env::var("INKWELL_MOE_SKIP").is_ok() {
        unsafe { cu::memset_dev_async(inkwell_ggml::ink_tensor_data(dst), 0, N_EMBD * 4) };
        return;
    }
    let mut gu = st.gu.lock().expect("gu");
    let mut hid = st.hid.lock().expect("hid");
    let mut dn = st.dn.lock().expect("dn");
    let mut xh = st.xh.lock().expect("xh");
    unsafe {
        if let Some(c) = &a.cache {
            if std::env::var_os("INKWELL_MOE_IDLOG").is_some() { moe_idlog(ids_dev, a.layer); }
            if let Some(pf) = &c.prefetch {
                cu::memcpy_ids8_async(pf.hist.dev_ptr(), ids_dev.cast());
            }
            cu::exl3_moe_cached_async(a.descs_g.dev, a.descs_u.dev, a.descs_d.dev, ids_dev,
                x_dev, xh.dev_ptr().cast(), gu.dev_ptr().cast(), hid.dev_ptr().cast(),
                dn.dev_ptr().cast(), w_dev, inkwell_ggml::ink_tensor_data(dst).cast(),
                N_SLOTS as i32,
                c.buf.dev_ptr(), c.tag.dev_ptr().cast(),
                st.exl3_map.lock().expect("map").dev_ptr().cast(),
                st.exl3_miss.lock().expect("miss").dev_ptr().cast(),
                c.cnt.dev_ptr().cast(), c.k, c.slot as i64,
                st.sdescs_g.lock().expect("sg").dev_ptr(),
                st.sdescs_u.lock().expect("su").dev_ptr(),
                st.sdescs_d.lock().expect("sd").dev_ptr(),
                c.prefetch.as_ref().map_or(std::ptr::null(), |p| p.ptag.dev_ptr().cast()),
                c.prefetch.as_ref().map_or(std::ptr::null(), |p| p.pready.dev_ptr().cast()),
                c.prefetch.as_ref().map_or(std::ptr::null_mut(), |p| p.bank.dev_ptr()),
                c.prefetch.as_ref().map_or(std::ptr::null_mut(), |p| unsafe { p.pf_miss.dev_ptr().cast::<i32>().add(8) }));
        } else {
            if std::env::var_os("INKWELL_MOE_IDLOG").is_some() { moe_idlog(ids_dev, a.layer); }
            cu::exl3_moe_async(a.descs_g.dev, a.descs_u.dev, a.descs_d.dev, ids_dev,
                x_dev, xh.dev_ptr().cast(), gu.dev_ptr().cast(), hid.dev_ptr().cast(),
                dn.dev_ptr().cast(), w_dev, inkwell_ggml::ink_tensor_data(dst).cast(),
                N_SLOTS as i32,
                st.exl3_stage.lock().expect("stg").dev_ptr(),
                st.sdescs_g.lock().expect("sg").dev_ptr(),
                st.sdescs_u.lock().expect("su").dev_ptr(),
                st.sdescs_d.lock().expect("sd").dev_ptr(),
                a.on_dev);
        }
    }
}

/// Callback address for the EXL3 MoE custom op.
#[must_use]
pub fn moe_cb_exl3_addr() -> usize {
    moe_cb_exl3 as usize
}

/// Verify-path expanded scratch: 64 (expert,consumer) slots.
struct VScratch {
    xh_e: Mutex<cu::DevBuf>,   // [64][2304]
    gu_e: Mutex<cu::DevBuf>,   // [64][2304]
    hid_e: Mutex<cu::DevBuf>,  // [64][1152]
    dn_e: Mutex<cu::DevBuf>,   // [64][2304]
    uids: Mutex<cu::DevBuf>,   // [64] i32
    eids: Mutex<cu::DevBuf>,   // [64] i32
    euni: Mutex<cu::DevBuf>,   // [64] i32
    etok: Mutex<cu::DevBuf>,   // [64] i32
    estart: Mutex<cu::DevBuf>, // [65] i32
    epos: Mutex<cu::DevBuf>,   // [64] i32
    unexp: Mutex<cu::DevBuf>,  // [1] i32
    map: Mutex<cu::DevBuf>,    // [64] i32
    miss: Mutex<cu::DevBuf>,   // [64] i32
}

static VSCRATCH: OnceLock<VScratch> = OnceLock::new();

fn vscratch() -> &'static VScratch {
    VSCRATCH.get_or_init(|| VScratch {
        xh_e: Mutex::new(cu::DevBuf::new_bytes(64 * N_EMBD * 4)),
        gu_e: Mutex::new(cu::DevBuf::new_bytes(64 * N_EMBD * 4)),
        hid_e: Mutex::new(cu::DevBuf::new_bytes(64 * N_FF * 4)),
        dn_e: Mutex::new(cu::DevBuf::new_bytes(64 * N_EMBD * 4)),
        uids: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
        eids: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
        euni: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
        etok: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
        estart: Mutex::new(cu::DevBuf::new_bytes(65 * 4)),
        epos: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
        unexp: Mutex::new(cu::DevBuf::new_bytes(4)),
        map: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
        miss: Mutex::new(cu::DevBuf::new_bytes(64 * 4)),
    })
}

/// Union-batched MoE callback for the 8-token verify block.
/// src = [x8 (n_embd×8 f32), ids8 (8×8 i32), w8 (8×8 f32)]; dst = [n_embd, 8].
#[expect(clippy::cast_ptr_alignment, clippy::too_many_lines, reason = "custom op")]
unsafe extern "C" fn moe_cb_exl3_ntok(dst: *mut c_void, _ith: i32, _nth: i32, ud: *mut c_void) {
    let a = unsafe { &*ud.cast::<MoeArgsExl3>() };
    let st = staging();
    let v = vscratch();
    unsafe { cu::set_stream(inkwell_ggml::cuda_stream()) };
    let src = |i: i32| unsafe {
        let t = inkwell_ggml::ink_tensor_src(dst, i);
        inkwell_ggml::ink_tensor_data(t)
    };
    let x8 = src(0) as *const f32;
    let ids8 = src(1) as *const i32;
    let w8 = src(2) as *const f32;
    let out8 = inkwell_ggml::ink_tensor_data(dst) as *mut f32;
    unsafe {
        let (cache, tag, cnt, k, slot_proj) = if let Some(c) = &a.cache {
            (c.buf.dev_ptr(), c.tag.dev_ptr().cast::<i32>(), c.cnt.dev_ptr().cast::<i32>(),
             c.k, c.slot as i64)
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), 0, 0)
        };
        cu::exl3_moe_ntok_async(a.descs_g.dev, a.descs_u.dev, a.descs_d.dev,
            x8, ids8, w8, out8,
            v.xh_e.lock().expect("x").dev_ptr().cast(),
            v.gu_e.lock().expect("g").dev_ptr().cast(),
            v.hid_e.lock().expect("h").dev_ptr().cast(),
            v.dn_e.lock().expect("d").dev_ptr().cast(),
            v.uids.lock().expect("u").dev_ptr().cast(),
            v.eids.lock().expect("i").dev_ptr().cast(),
            v.euni.lock().expect("n").dev_ptr().cast(),
            v.etok.lock().expect("t").dev_ptr().cast(),
            v.estart.lock().expect("s").dev_ptr().cast(),
            v.epos.lock().expect("p").dev_ptr().cast(),
            v.unexp.lock().expect("e").dev_ptr().cast(),
            cache, tag,
            v.map.lock().expect("m").dev_ptr().cast(),
            v.miss.lock().expect("f").dev_ptr().cast(),
            cnt, k, slot_proj,
            st.sdescs_g.lock().expect("sg").dev_ptr(),
            st.sdescs_u.lock().expect("su").dev_ptr(),
            st.sdescs_d.lock().expect("sd").dev_ptr(),
            a.on_dev);
    }
}

/// Callback address for the union-batched EXL3 MoE custom op.
#[must_use]
pub fn moe_cb_exl3_ntok_addr() -> usize {
    moe_cb_exl3_ntok as usize
}
