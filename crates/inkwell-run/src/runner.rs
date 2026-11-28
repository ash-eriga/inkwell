//! Decode-time (n_tokens = 1, single sequence) forward runner.
//!
//! Weights: dense tensors on the CUDA buffer; `*_exps` expert tensors live in
//! pinned+mapped host memory and stream over PCIe into own GEMV kernels via a
//! `GGML_OP_CUSTOM` per MoE visit (Strata-style zero-copy offload).
//! KV + conv state live in per-visit device arenas.
mod madv {
    use std::ffi::c_void;
    unsafe extern "C" {
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }
    /// Drop a file-mapped range from the page cache (keeps the pinned copy only).
    pub unsafe fn dontneed(addr: *const u8, len: usize) {
        let a = addr as usize & !4095;
        let end = (addr as usize + len + 4095) & !4095;
        unsafe { madvise(a as *mut c_void, end - a, 4 /* MADV_DONTNEED */); }
    }
}

use crate::ops::{t, Buf, Ptr, Rt};
use inkwell_gguf::GgufFile;
use inkwell_core::InklingConfig;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

static VERBOSE: AtomicBool = AtomicBool::new(true);

/// Control internal diagnostic logging (`false` for clean interactive/server UX).
pub fn set_verbose(v: bool) {
    VERBOSE.store(v, AtomicOrdering::Relaxed);
}

#[inline]
fn is_verbose() -> bool {
    VERBOSE.load(AtomicOrdering::Relaxed)
}

macro_rules! vlog {
    ($($arg:tt)*) => {
        if is_verbose() {
            eprintln!($($arg)*);
        }
    };
}

const F32: i32 = 0;
const F16: i32 = 1;
const I32: i32 = 26;
const BF16: i32 = 30;

fn warm_file_pages(addr: *const u8, len: usize) {
    if len == 0 { return; }
    for off in (0..len).step_by(4096) {
        unsafe { std::ptr::read_volatile(addr.add(off)) };
    }
    unsafe { std::ptr::read_volatile(addr.add(len - 1)) };
}

fn kv_bucket(n: i64, max: i64) -> i64 {
    ((n + 63) & !63).min(max)
}

/// Device-scalar exact n_kv attention is the default decode path.  The
/// unfused rel-bias fallback and the pow2 attention dump keep the old shapes.
fn dyn_nkv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("INKWELL_NO_DYN_NKV").is_none()
        && std::env::var_os("INKWELL_NO_RELGATHER").is_none()
        && std::env::var_os("INKWELL_ATTN_DUMP_DIR").is_none())
}

/// `(no_split, split_min, chunk_override)`, mirroring the CUDA dispatcher.
fn attn_split_cfg() -> (bool, i64, i64) {
    static CFG: std::sync::OnceLock<(bool, i64, i64)> = std::sync::OnceLock::new();
    *CFG.get_or_init(|| {
        let no_split = std::env::var_os("INKWELL_NO_ATTNSPLIT").is_some();
        let split_min = std::env::var("INKWELL_ATTNSPLIT_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(320);
        let mut chunk = std::env::var("INKWELL_ATTNSPLIT_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if chunk != 0 && chunk < 32 { chunk = 32; }
        (no_split, split_min, chunk)
    })
}

/// Kernel-topology regime for the padded length master would have used.  Using
/// the padded length here (rather than raw n_kv) preserves master's exact kernel
/// choice at pos=256 while the dynamic kernels still skip padded work on device.
fn attn_regime_for_pad(n_kv_pad: i64) -> u8 {
    let (no_split, split_min, chunk_override) = attn_split_cfg();
    if no_split || n_kv_pad < split_min { return 0; }
    if chunk_override == 0 && n_kv_pad > 16384 { 2 } else { 1 }
}

fn attn_chunk_len(regime: u8) -> i32 {
    let (_, _, chunk_override) = attn_split_cfg();
    let v = if chunk_override != 0 { chunk_override } else if regime == 2 { 256 } else { 32 };
    i32::try_from(v).expect("attention chunk length fits i32")
}

/// per-layer weight handles resolved by name
#[derive(Debug, Default)]
struct L {
    attn_norm: Option<Ptr>, attn_q_inp_norm: Option<Ptr>, attn_k_inp_norm: Option<Ptr>, attn_v_inp_norm: Option<Ptr>,
    attn_res_query: Option<Ptr>, ffn_res_query: Option<Ptr>,
    wq: Option<Ptr>, wk: Option<Ptr>, wv: Option<Ptr>, wo: Option<Ptr>,
    attn_rel: Option<Ptr>, attn_rel_bias: Option<Ptr>, attn_rel_bias_pad: Option<Ptr>, attn_rel_clip: Option<Ptr>, attn_q_clip: Option<Ptr>,
    attn_sink_proj: Option<Ptr>, wqkv_gate: Option<Ptr>,
    // concat of same-type {q,gate,rel} projections sharing inputs[0]
    proj0_cat: Option<Ptr>, proj0_off: [i64; 3],
    attn_k_conv: Option<Ptr>, attn_v_conv: Option<Ptr>, attn_post_conv: Option<Ptr>, ffn_post_conv: Option<Ptr>,
    attn_k_conv_proj: Option<Ptr>, attn_v_conv_proj: Option<Ptr>,
    ffn_norm: Option<Ptr>, ffn_gate_inp: Option<Ptr>, ffn_exp_probs_b: Option<Ptr>,
    gate_up_scale: Option<Ptr>, down_scale: Option<Ptr>,
    moe: Option<usize>,
    /// EXL3 weight descriptor index per short name (REXL3 only)
    exl3: HashMap<&'static str, usize>,
    /// EXL3 MoE binding index (REXL3 only)
    moe_exl3: Option<usize>,
    msg_query: Option<Ptr>, ctrl_down: [Option<Ptr>; 4], ctrl_gain: [Option<Ptr>; 4], ctrl_forget: [Option<Ptr>; 4],
    /// concat of same-type {ctrl_gain, ctrl_forget} (shared input) -> 1 GEMV + slices
    ctrl_gf_cat: [Option<Ptr>; 4], ctrl_gf_off: [[i64; 2]; 4],
}

/// Everything the decode step needs.
#[derive(Debug)]
pub struct Runner {
    rt: Rt,
    sched: Ptr,
    /// (sched, cpu backend) for pinning expert-chain ops to CPU
    sched_cpu: Option<(Ptr, Ptr)>,
    /// pooled userdata for fused conv-step custom ops (stable addresses)
    conv_args: std::cell::UnsafeCell<Vec<Box<ConvStepArgs>>>,
    conv_arg_idx: std::cell::Cell<usize>,
    /// persistent ggml context (graph nodes), reset per step
    cur_ctx: Ptr,
    /// per-step cache of mix()'s block-stack prefix concat, keyed by block ptrs
    mix_prefix: std::cell::UnsafeCell<Option<(usize, Ptr)>>,
    dump_list: std::cell::UnsafeCell<Vec<(String, Ptr)>>,
    /// persistent device buffer for step inputs (ids/pos/onehot/sink)
    in_buf: Option<Buf>,
    #[expect(dead_code)]
    dev_buf: Buf,
    #[expect(dead_code)]
    cpu_buf: Buf,
    /// name → tensor
    w: HashMap<String, Ptr>,
    /// pinned+mapped expert weight copies (zero-copy GEMV over PCIe)
    pinned: Vec<inkwell_cuda::PinnedBuf>,
    /// VRAM-resident expert weight blobs (REXL3)
    expert_dev: Vec<inkwell_cuda::DevBuf>,
    /// per-layer MoE weight bindings consumed by the custom op
    moe_args: Vec<Box<crate::moe::MoeArgs>>,
    /// EXL3 weight descriptors (device-resident trellis+suh+svh)
    exl3_w: Vec<Box<Exl3W>>,
    /// 8-token variants sharing `exl3_w` weights (spec-decode verify)
    exl3_blk: Vec<Box<Exl3Blk>>,
    /// EXL3 MoE bindings per visit
    moe_args_exl3: Vec<Box<crate::moe::MoeArgsExl3>>,
    /// unembed as EXL3 (REXL3)
    output_exl3: Option<usize>,
    cfg: InklingConfig,
    layers: Vec<L>,
    // globals
    tok_embd: Ptr, embd_src: (*const u8, i32, usize), embd_dev: *mut std::ffi::c_void, embd_row: Ptr, tok_norm: Ptr, output: Ptr, output_norm: Ptr, output_res_query: Ptr,
    /// packed middle-cycle msg queries [n_embd, 48] (zero-copy per-layer views)
    msg_query_packed: Ptr,
    // own lm_head path: device ptr to output.weight + work buffers (Some iff type==IQ4_XS)
    out_head: Option<(inkwell_cuda::DevBuf, inkwell_cuda::DevBuf, inkwell_cuda::DevBuf)>,
    // per-visit KV arenas + conv states (visit → tensors)
    k_arena: Vec<Ptr>, v_arena: Vec<Ptr>, conv_state: Vec<Ptr>,
    /// per-position conv snapshots for spec rollback: [slot][NT][3*n_sconv].
    conv_snap: Vec<inkwell_cuda::DevBuf>,
    /// DSpark ctx taps captured by step_block: [NT][4][n_embd] f32
    /// (visits {5,23,41} inputs + final-norm output).
    block_feats: Option<inkwell_cuda::DevBuf>,
    /// boxed `*mut c_void` dsts for feat_tap ops; cleared per `step_block`.
    feat_ud: std::cell::RefCell<Vec<Box<*mut std::ffi::c_void>>>,
    /// tap-op tensors to expand into the `step_block` graph.
    feat_ops: std::cell::RefCell<Vec<Ptr>>,
    max_ctx: i64,
    swa: i64,
    swa_oracle: bool,
    swa_legacy_min: bool,
    visits: Vec<(usize /*layer*/, usize /*slot*/)>,
    n_full: usize,
    /// built step graph kept alive across tokens while padded n_kv shapes are
    /// stable (rebuild on each 64-token boundary). `None` after `step_block`.
    step_gcache: Option<StepCache>,
    /// pinned/device side buffers for the `step_id` decode path
    sid: Option<SidBufs>,
}

/// Live objects of a built step graph: the cgraph + its input tensors (via
/// `step`). Device addresses never change (inputs sit in `in_buf`,
/// intermediates in the sched's allocated splits, which are not reset while
/// this is alive).
#[derive(Copy, Clone, Debug)]
struct StepCache {
    step: Step,
    graph: Ptr,
    /// final tensor read back for logits (`cur` at graph end)
    out: Ptr,
    /// normalised hidden state entering the output head
    head_in: Ptr,
    /// device offsets of the input tensors inside `in_buf` + span bytes
    /// (indices 9/10 are the dynamic n_kv scalars when enabled)
    in_offs: [usize; 11],
    in_span: usize,
    /// embedding row offset inside the packed input span (`usize::MAX` when unpacked)
    embd_off: usize,
    /// packed input bytes needed by the pipelined device-resolver path
    input_copy_len: usize,
    /// whether the cached graph output includes the lm_head logits
    has_head: bool,
}

/// One cached-step host timing row, microseconds.
#[derive(Default, Debug, Clone, Copy)]
struct StepProfRow {
    marshal: u64,
    embd: u64,
    h2d: u64,
    prefetch: u64,
    sched: u64,
    tailq: u64,
    premarshal: u64,
    wait: u64,
    total: u64,
}

static STEP_PROF_ROWS: std::sync::Mutex<Vec<StepProfRow>> = std::sync::Mutex::new(Vec::new());

fn stepprof_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("INKWELL_STEPPROF").is_some())
}

fn stepprof_dur(a: std::time::Instant, b: std::time::Instant) -> u64 {
    u64::try_from(b.duration_since(a).as_micros()).unwrap_or(u64::MAX)
}

fn stepprof_push(row: StepProfRow) {
    STEP_PROF_ROWS.lock().expect("stepprof mutex").push(row);
}

/// Print aggregate cached-step host timings collected under `INKWELL_STEPPROF`.
pub fn stepprof_dump() {
    let rows = STEP_PROF_ROWS.lock().expect("stepprof mutex").clone();
    if rows.is_empty() { return; }
    eprintln!("STEPPROF rows={} us/token: field mean p50 p90 max", rows.len());
    let fields: [(&str, fn(StepProfRow) -> u64); 9] = [
        ("marshal", |r| r.marshal), ("embd", |r| r.embd), ("h2d", |r| r.h2d),
        ("prefetch", |r| r.prefetch), ("sched", |r| r.sched), ("tailq", |r| r.tailq),
        ("premarshal", |r| r.premarshal), ("wait", |r| r.wait), ("total", |r| r.total),
    ];
    for (name, get) in fields {
        let mut v: Vec<u64> = rows.iter().map(|&r| get(r)).collect();
        v.sort_unstable();
        let mean = v.iter().sum::<u64>() as f64 / v.len() as f64;
        eprintln!("STEPPROF {name:<10} {:.1} {} {} {}", mean, v[v.len() / 2], v[v.len() * 9 / 10], v[v.len() - 1]);
    }
}

/// tensor dims (ne0..) of a gguf tensor
fn gguf_dims(t: &inkwell_gguf::TensorInfo) -> Vec<i64> {
    t.dims.iter().map(|d| *d as i64).collect()
}

const CTRL_NAMES: [&str; 4] = ["q", "k", "v", "ffn"];

/// One EXL3 weight: device-resident trellis + suh + svh + scratch.
#[derive(Debug)]
pub struct Exl3W {
    /// trellis i16 payload on device
    pub trel: inkwell_cuda::DevBuf,
    /// suh fp16 on device
    pub suh: inkwell_cuda::DevBuf,
    /// svh fp16 on device
    pub svh: inkwell_cuda::DevBuf,
    /// integer K part
    pub ka: usize,
    /// half-K flag
    pub halfk: bool,
    /// input dim k / output dim n
    pub k: usize,
    pub n: usize,
    /// device scratch: xh[k] + t[n] f32
    pub scratch: inkwell_cuda::DevBuf,
}

/// `ggml_custom_op_t`: EXL3 trellis GEMV `y = svh . H (Wt^T H (suh . x))`.
///
/// # Safety
/// `userdata` = `&Exl3W`; `dst->src[0]` = x (f32 device).
unsafe extern "C" fn exl3_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let w = unsafe { &*ud.cast::<Exl3W>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let x = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let x_dev = unsafe { inkwell_ggml::ink_tensor_data(x) } as *const f32;
    let y = unsafe { inkwell_ggml::ink_tensor_data(dst) } as *mut f32;
    unsafe {
        inkwell_cuda::exl3_gemv_async(w.trel.dev_ptr(), w.suh.dev_ptr(), w.svh.dev_ptr(),
            x_dev, y, w.k, w.n, w.ka, w.halfk, w.scratch.dev_ptr().cast());
    }
}

/// Callback address for the EXL3 GEMV custom op.
#[must_use]
pub fn exl3_cb_addr() -> usize {
    exl3_cb as usize
}

/// 8-token EXL3 weight: shares `Exl3W` device buffers, owns a wider scratch.
#[derive(Debug)]
pub struct Exl3Blk {
    trel: *const std::ffi::c_void,
    suh: *const std::ffi::c_void,
    svh: *const std::ffi::c_void,
    k: usize,
    n: usize,
    ka: usize,
    halfk: bool,
    scratch_len: usize,
    scratch: std::sync::OnceLock<inkwell_cuda::DevBuf>,
}

impl Exl3Blk {
    /// Allocate the 8-row scratch on first use (the serial decode path never uses it).
    fn ensure_scratch(&self) {
        self.scratch.get_or_init(|| inkwell_cuda::DevBuf::new(self.scratch_len));
    }
}

/// `ggml_custom_op_t`: batched EXL3 GEMV — `dst->src[0]` = x `[k, 8]` f32 device.
unsafe extern "C" fn exl3_ntok_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let w = unsafe { &*ud.cast::<Exl3Blk>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let x = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let x_dev = unsafe { inkwell_ggml::ink_tensor_data(x) } as *const f32;
    let y = unsafe { inkwell_ggml::ink_tensor_data(dst) } as *mut f32;
    let scratch = w.scratch.get_or_init(|| inkwell_cuda::DevBuf::new(w.scratch_len));
    unsafe {
        inkwell_cuda::exl3_gemv_ntok_async(w.trel, w.suh, w.svh,
            x_dev, y, w.k, w.n, w.ka, w.halfk, scratch.dev_ptr().cast());
    }
}

/// Callback address for the batched EXL3 GEMV custom op.
#[must_use]
pub fn exl3_ntok_cb_addr() -> usize {
    exl3_ntok_cb as usize
}

/// `ggml_custom_op_t`: device→device copy of src[0] into `*(ud as *mut *mut c_void)`.
/// Captures a graph tensor's data into `block_feats` at compute time.
unsafe extern "C" fn feat_copy_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let s = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let sp = unsafe { inkwell_ggml::ink_tensor_data(s) };
    let n = (unsafe { inkwell_ggml::ink_tensor_ne(s, 0) } * unsafe { inkwell_ggml::ink_tensor_ne(s, 1) }) as usize * 4;
    let d = unsafe { *ud.cast::<*mut std::ffi::c_void>() };
    unsafe { inkwell_cuda::memcpy_d2d_async(d, sp, n) };
}

/// userdata for the fused conv-step custom op.
#[derive(Debug)]
struct ConvStepArgs {
    state: *mut f32,      // device ptr, conv_state[slot] base
    region_off: i32,      // channel offset inside [3, n_sconv]
    ch: i32,
    n_heads: i32,         // 0 = no dynamic taps
    scale: f32,
    snap: *mut f32,       // nullable per-position state save [3, n_sconv]
}

/// `ggml_custom_op_t`: fused conv shift + ssm_conv + taps + residual + scale.
/// src[0]=x_in [ch], src[1]=w [4,ch], src[2]=dyn [4,nh] (optional).
/// userdata for the fused per-column L2 clip (`clip()`).
struct ClipL2Args { thr: f32, has_w: i32 }

/// `ggml_custom_op_t`: `y = (x*w) * min(1, thr/(||x*w||_2 + eps))` per column.
/// src[0] = x `[d, h]`; src[1] = scale (only when `has_w`). dst = x-shaped.
struct RouterArgs { wscale: f32, wfloor: f32 }

unsafe extern "C" fn router_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*(ud as *const RouterArgs) };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let logits = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 0)) as *const f32 };
    let bias = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 1)) as *const f32 };
    unsafe { inkwell_cuda::router_async(logits.cast(), bias.cast(),
        inkwell_ggml::ink_tensor_data(dst).cast(), 8, 128, a.wscale, a.wfloor) };
}

/// `ggml_custom_op_t`: fused BF16 mlp.gate GEMV + router top-k.
/// srcs = [gate weight, input, bias]; dst[0..8]=weights, dst[8..16]=ids.
unsafe extern "C" fn gate_router_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*(ud as *const RouterArgs) };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let w = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let x = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 1)) as *const f32 };
    let bias = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 2)) as *const f32 };
    let nrows = unsafe { inkwell_ggml::ink_tensor_ne(w, 1) } as i32;
    let ncols2 = (unsafe { inkwell_ggml::ink_tensor_ne(w, 0) } / 2) as i32;
    unsafe { inkwell_cuda::gate_router_async(inkwell_ggml::ink_tensor_data(w), x, bias.cast(),
        inkwell_ggml::ink_tensor_data(dst).cast(), nrows, ncols2, 8, a.wscale, a.wfloor) };
}

struct MixArgs { embd: i32, rows: i32, qcols: i32, scale: f32 }

unsafe extern "C" fn mix_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*(ud as *const MixArgs) };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let st = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 0)) as *const f32 };
    let q = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 1)) as *const f32 };
    unsafe { inkwell_cuda::mix_async(st.cast(), q.cast(),
        inkwell_ggml::ink_tensor_data(dst).cast(), a.embd, a.rows, a.qcols, a.scale) };
}

struct MixPtrsArgs { embd: i32, rows: i32, qcols: i32, scale: f32 }

unsafe extern "C" fn mix_ptrs_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*(ud as *const MixPtrsArgs) };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let nrows = a.rows as usize;
    let mut rows = [std::ptr::null::<f32>(); 9];
    for (j, rp) in rows.iter_mut().enumerate().take(nrows) {
        *rp = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, j as i32)) as *const f32 };
    }
    let q = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, a.rows)) as *const f32 };
    unsafe { inkwell_cuda::mix_ptrs_async(rows.as_ptr(), q,
        inkwell_ggml::ink_tensor_data(dst).cast(), a.embd, a.rows, a.qcols, a.scale) };
}

struct QkvrPrepArgs {
    state: *mut f32,
    k_region_off: i32,
    v_region_off: i32,
    nmax: i32,
    hd: i32,
    nh: i32,
    hkv: i32,
    nrel: i32,
    conv_scale: f32,
    qk_thr: f32,
    rel_thr: f32,
    dyn_src: i32,
    kw_src: i32,
    vw_src: i32,
    qw_src: i32,
    rw_src: i32,
    tau_src: i32,
}

unsafe extern "C" fn qkvr_prep_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*(ud as *const QkvrPrepArgs) };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let src = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let opt_src = |i: i32| if i >= 0 { src(i) } else { std::ptr::null() };
    unsafe {
        inkwell_cuda::attn_qkvr_prep_async(src(0), a.nmax, opt_src(a.dyn_src),
            src(a.kw_src), src(a.vw_src), opt_src(a.qw_src), opt_src(a.rw_src), opt_src(a.tau_src),
            a.state, a.k_region_off, a.v_region_off,
            inkwell_ggml::ink_tensor_data(dst).cast(), a.hd, a.nh, a.hkv, a.nrel,
            a.conv_scale, a.qk_thr, a.rel_thr)
    };
}

/// `ggml_custom_op_t`: loop-controller tail —
/// y = (cur - tanh(g3[ng+i])*cur) + tanh(g3[i])*msg. srcs = [g3, cur, msg];
/// userdata = forget offset ng, carried as the pointer value.
unsafe extern "C" fn ctrl_apply_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let ng = ud as usize as i32;
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let n = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    unsafe {
        inkwell_cuda::ctrl_apply_async(s(0), s(1), s(2),
            inkwell_ggml::ink_tensor_data(dst).cast(), n, ng)
    };
}

/// `ggml_custom_op_t`: fused ctrl gain/forget GEMV + apply.
/// srcs = [gf_cat weight, gate, cur, msg]; dst [n]; userdata = forget offset fo.
unsafe extern "C" fn gf_apply_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let fo = ud as usize as i32;
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let w = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let n = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let is_bf16 = i32::from(unsafe { inkwell_ggml::ink_tensor_type(w) } == BF16);
    unsafe {
        inkwell_cuda::gf_apply_async(inkwell_ggml::ink_tensor_data(w), is_bf16,
            s(1), s(2), s(3), inkwell_ggml::ink_tensor_data(dst).cast(), n, fo)
    };
}

/// `ggml_custom_op_t`: batched k/v_sconv projection GEMV.
/// srcs = [wk, xk, wv, xv]; dst is [32,2].
unsafe extern "C" fn kv_sconv_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let t = |i: i32| unsafe { inkwell_ggml::ink_tensor_src(dst, i) };
    let d = |i: i32| unsafe { inkwell_ggml::ink_tensor_data(t(i)) };
    let wk = unsafe { t(0) };
    let nrows = unsafe { inkwell_ggml::ink_tensor_ne(wk, 1) } as i32;
    let ncols2 = (unsafe { inkwell_ggml::ink_tensor_ne(wk, 0) } / 2) as i32;
    let out = unsafe { inkwell_ggml::ink_tensor_data(dst) as *mut f32 };
    unsafe { inkwell_cuda::kv_sconv_batched_async(d(0), d(2), d(1) as *const f32, d(3) as *const f32,
        out, out.add(nrows as usize), nrows, ncols2) };
}

#[repr(C)]
struct CtrlTriadIo {
    cur: [*const f32; 3],
    msg: [*const f32; 3],
    down: [*const std::ffi::c_void; 3],
    gf: [*const std::ffi::c_void; 3],
    dst: *mut f32,
}

struct CtrlTriadArgs {
    down: [*const std::ffi::c_void; 3],
    gf: [*const std::ffi::c_void; 3],
    fo: i32,
}

/// `ggml_custom_op_t`: batched q/k/v controller triad.
/// srcs = [cur0,msg0,cur1,msg1,cur2,msg2]; dst is [n,3].
unsafe extern "C" fn ctrl_triad_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*(ud as *const CtrlTriadArgs) };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let io = CtrlTriadIo {
        cur: [s(0), s(2), s(4)],
        msg: [s(1), s(3), s(5)],
        down: a.down,
        gf: a.gf,
        dst: unsafe { inkwell_ggml::ink_tensor_data(dst) as *mut f32 },
    };
    let n = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    unsafe { inkwell_cuda::ctrl_triad_async((&io as *const CtrlTriadIo).cast(), n, 1e-6, a.fo) };
}

/// `ggml_custom_op_t`: fused rms2cat + ctrl_down GEMV.
/// srcs = [x0, x1, weight]; dst [nrows] where nrows = weight->ne[1].
unsafe extern "C" fn rms2down_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let w = unsafe { inkwell_ggml::ink_tensor_src(dst, 2) };
    let n = (unsafe { inkwell_ggml::ink_tensor_ne(w, 0) } / 2) as i32;
    let nrows = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let is_bf16 = i32::from(unsafe { inkwell_ggml::ink_tensor_type(w) } == BF16);
    unsafe {
        inkwell_cuda::rms2down_async(s(0), s(1), inkwell_ggml::ink_tensor_data(w),
            is_bf16, inkwell_ggml::ink_tensor_data(dst).cast(), n, nrows, 1e-6)
    };
}

/// `ggml_custom_op_t`: concat(rms_norm(x0), rms_norm(x1)) in one launch.
/// srcs = [x0, x1]; dst [2n] with x0 in [0..n).
unsafe extern "C" fn rms2cat_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let n = (unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } / 2) as i32;
    unsafe {
        inkwell_cuda::rms2cat_async(s(0), s(1),
            inkwell_ggml::ink_tensor_data(dst).cast(), n, 1e-6)
    };
}

/// `ggml_custom_op_t`: kq_b[j + h*nkv] = table[rel_pos[j] + h*ext] - sinks[h].
/// srcs = [table mm [ext, nh], sinks [nh], rel_pos i32 [nkv]].
unsafe extern "C" fn relbias_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let mm = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i))
    };
    let nkv = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let nh = unsafe { inkwell_ggml::ink_tensor_ne(mm, 1) } as i32;
    let ext = unsafe { inkwell_ggml::ink_tensor_ne(mm, 0) } as i32;
    unsafe {
        inkwell_cuda::relbias_gather_async(inkwell_ggml::ink_tensor_data(mm).cast(),
            s(1).cast(), s(2).cast(), inkwell_ggml::ink_tensor_data(dst).cast(),
            nkv, nh, ext)
    };
}

/// `ggml_custom_op_t`: fixed-capacity rel-bias gather; exact length is src[3].
unsafe extern "C" fn relbias_dyn_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let mm = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i))
    };
    let cap = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let nh = unsafe { inkwell_ggml::ink_tensor_ne(mm, 1) } as i32;
    let ext = unsafe { inkwell_ggml::ink_tensor_ne(mm, 0) } as i32;
    unsafe {
        inkwell_cuda::relbias_gather_dyn_async(inkwell_ggml::ink_tensor_data(mm).cast(),
            s(1).cast(), s(2).cast(), s(3).cast(), inkwell_ggml::ink_tensor_data(dst).cast(),
            cap, nh, ext)
    };
}

struct RelbiasDynPosArgs {
    mode: i32,
    swa: i32,
    rel_extent: i32,
}

struct SinkRelbiasArgs {
    mode: i32,
    swa: i32,
    rel_extent: i32,
    dynamic: i32,
}

/// `ggml_custom_op_t`: fixed-capacity rel-bias gather whose integer relative
/// positions are computed on device. srcs = [table, sinks, pos, n_kv].
unsafe extern "C" fn relbias_dynpos_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<RelbiasDynPosArgs>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let mm = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let s = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i))
    };
    let cap = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let nh = unsafe { inkwell_ggml::ink_tensor_ne(mm, 1) } as i32;
    let ext = unsafe { inkwell_ggml::ink_tensor_ne(mm, 0) } as i32;
    unsafe {
        inkwell_cuda::relbias_gather_dynpos_async(inkwell_ggml::ink_tensor_data(mm).cast(),
            s(1).cast(), s(2).cast(), s(3).cast(), inkwell_ggml::ink_tensor_data(dst).cast(),
            cap, nh, ext, a.mode, a.swa, a.rel_extent)
    };
}

/// `ggml_custom_op_t`: fused BF16 sink projection + rel-bias dot/gather.
/// srcs = [sink weight, input, BF16 padded rel-bias, r, pos/rel_pos, optional n_kv].
unsafe extern "C" fn sink_relbias_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<SinkRelbiasArgs>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let t = |i: i32| unsafe { inkwell_ggml::ink_tensor_src(dst, i) };
    let d = |i: i32| unsafe { inkwell_ggml::ink_tensor_data(t(i)) };
    let ws = unsafe { t(0) };
    let nh = unsafe { inkwell_ggml::ink_tensor_ne(ws, 1) } as i32;
    let ncols2 = (unsafe { inkwell_ggml::ink_tensor_ne(ws, 0) } / 2) as i32;
    let y_stride = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let d_nkv = if a.dynamic != 0 { d(5).cast() } else { std::ptr::null() };
    unsafe {
        inkwell_cuda::sink_relbias_fused_async(d(0), d(1).cast(), d(2), d(3).cast(),
            d(4), d_nkv, inkwell_ggml::ink_tensor_data(dst).cast(), y_stride, nh,
            y_stride, ncols2, a.mode, a.swa, a.rel_extent)
    };
}

unsafe extern "C" fn clip_l2_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<ClipL2Args>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let x = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 0)) as *const f32 };
    let w = if a.has_w != 0 {
        unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, 1)) as *const f32 }
    } else { std::ptr::null() };
    let y = unsafe { inkwell_ggml::ink_tensor_data(dst) } as *mut f32;
    let d = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) } as i32;
    let h = (unsafe { inkwell_ggml::ink_tensor_ne(dst, 1) } * unsafe { inkwell_ggml::ink_tensor_ne(dst, 2) }) as i32;
    unsafe { inkwell_cuda::clip_l2_async(x, w, y, d, h, a.thr) };
}

/// userdata for the multi-projection EXL3 gemv (wq/gate/rel/wk/wv on parallel streams).
struct Exl3MzArgs { ws: [*const Exl3W; 6], xidx: [i32; 6], nz: i32 }

/// `ggml_custom_op_t`: `nz` EXL3 projections on parallel side streams.
/// src[i] = the i-th distinct input tensor; per-proj input picked via `xidx`.
/// dst `[n_max, nz]` — proj z's output lives in column z (`dst + z*n_max`).
unsafe extern "C" fn exl3_mz_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<Exl3MzArgs>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let y = unsafe { inkwell_ggml::ink_tensor_data(dst) } as *mut f32;
    let nz = a.nz as usize;
    let mut trel = [std::ptr::null::<std::ffi::c_void>(); 8];
    let mut suh = [std::ptr::null::<std::ffi::c_void>(); 8];
    let mut svh = [std::ptr::null::<std::ffi::c_void>(); 8];
    let mut x = [std::ptr::null::<f32>(); 8];
    let mut scr = [std::ptr::null_mut::<f32>(); 8];
    let mut kk = [0i64; 8];
    let mut nn = [0i64; 8];
    let mut ka = [0i64; 8];
    let mut hk = [0i64; 8];
    for z in 0..nz {
        let w = unsafe { &*a.ws[z] };
        trel[z] = w.trel.dev_ptr();
        suh[z] = w.suh.dev_ptr();
        svh[z] = w.svh.dev_ptr();
        x[z] = unsafe { inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, a.xidx[z])) as *const f32 };
        scr[z] = w.scratch.dev_ptr().cast();
        kk[z] = w.k as i64;
        nn[z] = w.n as i64;
        ka[z] = w.ka as i64;
        hk[z] = w.halfk as i64;
    }
    let n_max = unsafe { inkwell_ggml::ink_tensor_ne(dst, 0) };
    unsafe {
        inkwell_cuda::exl3_gemv_mz_async(&trel[..nz], &suh[..nz], &svh[..nz], &x[..nz],
            y, n_max, &scr[..nz], &kk[..nz], &nn[..nz], &ka[..nz], &hk[..nz])
    };
}

unsafe extern "C" fn conv_step_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<ConvStepArgs>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let src = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let dyn_p = if a.n_heads > 0 { src(2) } else { std::ptr::null() };
    unsafe {
        inkwell_cuda::conv_step_async(src(0), a.state, a.region_off, src(1), dyn_p,
            a.ch, a.n_heads, a.scale,
            inkwell_ggml::ink_tensor_data(dst).cast(), a.snap);
    }
}

unsafe extern "C" fn conv_step_add_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<ConvStepArgs>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let src = |i: i32| unsafe {
        inkwell_ggml::ink_tensor_data(inkwell_ggml::ink_tensor_src(dst, i)) as *const f32
    };
    let dyn_p = if a.n_heads > 0 { src(2) } else { std::ptr::null() };
    let residual = src(if a.n_heads > 0 { 3 } else { 2 });
    unsafe {
        inkwell_cuda::conv_step_add_async(src(0), a.state, a.region_off, src(1), dyn_p, residual,
            a.ch, a.n_heads, a.scale,
            inkwell_ggml::ink_tensor_data(dst).cast(), a.snap);
    }
}

/// One memcpy segment: src/dst as usize so copy plans are `Send`.
#[derive(Debug)]
struct CopySeg {
    src: usize,
    dst: usize,
    len: usize,
}

/// Copy disjoint segments with `nth` workers and drop source PTEs as in the
/// original loader. `INKWELL_NO_PARCOPY=1` forces serial copying.
fn par_memcpy_abs(segs: &[CopySeg], nth: usize) {
    let one = |seg: &CopySeg| {
        unsafe {
            std::ptr::copy_nonoverlapping(seg.src as *const u8, seg.dst as *mut u8, seg.len);
            madv::dontneed(seg.src as *const u8, seg.len);
        }
    };
    if std::env::var_os("INKWELL_NO_PARCOPY").is_some() || nth < 2 || segs.len() < 8 {
        for seg in segs { one(seg); }
        return;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..nth {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= segs.len() { break; }
                one(&segs[i]);
            });
        }
    });
}

struct PinJob {
    rx: std::sync::mpsc::Receiver<(usize, usize)>,
    handle: std::thread::JoinHandle<inkwell_cuda::PinnedBuf>,
}

struct ExpertLoad {
    il: usize,
    descs: Vec<u64>,
    on_dev: bool,
    max_slot: usize,
    total: usize,
    host_base: usize,
    dev_base: usize,
    pin: Option<PinJob>,
}

/// Allocate a pinned expert layer and copy it in the background. Allocation
/// coordinates are sent back as soon as `cudaHostAlloc` returns; the copy keeps
/// overlapping subsequent device-resident expert uploads and CUDA setup.
fn spawn_pin_job(mut segs: Vec<CopySeg>, total: usize) -> PinJob {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::Builder::new()
        .name(format!("inkwell-pinload-{total}"))
        .spawn(move || {
            let pb = inkwell_cuda::host_alloc_mapped(total);
            tx.send((pb.host as usize, pb.dev as usize)).expect("pinload channel");
            let base = pb.host as usize;
            for seg in &mut segs { seg.dst += base; }
            let nth = std::env::var("INKWELL_PIN_COPY_THREADS").ok()
                .and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
            par_memcpy_abs(&segs, nth);
            pb
        })
        .expect("spawn pinned expert loader");
    PinJob { rx, handle }
}

/// bf16 bytes -> f32 bytes; `T`: transpose first (HF `[in,out]` -> ggml `[in,out]` col-major).
fn bf16_to_f32_t(d: &[u8], shape: &[u64]) -> Vec<u8> {
    let (r, c) = (shape[0] as usize, shape[1] as usize);
    let mut f = vec![0f32; r * c];
    for (i, ch) in d.chunks_exact(2).enumerate() {
        let bits = u32::from(ch[0]) | (u32::from(ch[1]) << 8);
        f[i] = f32::from_bits(bits << 16);
    }
    // A[r][c] row-major -> B[c][r] row-major = ggml dims [r,c] col-major
    let mut out = vec![0f32; r * c];
    for a in 0..r {
        for b in 0..c {
            out[b * r + a] = f[a * c + b];
        }
    }
    unsafe { std::slice::from_raw_parts(out.as_ptr().cast::<u8>(), r * c * 4) }.to_vec()
}

/// BF16 bytes transposed as raw 2-byte elements, same layout as `bf16_to_f32_t`.
fn bf16_t(d: &[u8], shape: &[u64]) -> Vec<u8> {
    let (r, c) = (shape[0] as usize, shape[1] as usize);
    let mut out = vec![0u8; d.len()];
    for a in 0..r {
        for b in 0..c {
            let src = (a * c + b) * 2;
            let dst = (b * r + a) * 2;
            out[dst..dst + 2].copy_from_slice(&d[src..src + 2]);
        }
    }
    out
}

/// bf16 bytes -> f32 bytes.
fn bf16_to_f32(d: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; d.len() * 2];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { bf16_to_f32_avx2(d, &mut out) };
        return out;
    }
    bf16_to_f32_scalar(d, &mut out);
    out
}

fn bf16_to_f32_scalar(d: &[u8], out: &mut [u8]) {
    for (i, ch) in d.chunks_exact(2).enumerate() {
        let bits = u32::from(ch[0]) | (u32::from(ch[1]) << 8);
        out[i * 4..i * 4 + 4].copy_from_slice(&(bits << 16).to_le_bytes());
    }
}

/// Widening is an exact bit shift, so SIMD conversion is bit-identical to the
/// scalar loop while doing eight elements per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bf16_to_f32_avx2(d: &[u8], out: &mut [u8]) {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm256_cvtepu16_epi32, _mm256_slli_epi32, _mm256_storeu_si256,
    };
    let n = d.len() / 2;
    let mut i = 0usize;
    while i + 8 <= n {
        let x = unsafe { _mm_loadu_si128(d.as_ptr().add(i * 2).cast()) };
        let wide = unsafe { _mm256_cvtepu16_epi32(x) };
        let shifted = unsafe { _mm256_slli_epi32(wide, 16) };
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().add(i * 4).cast(), shifted) };
        i += 8;
    }
    while i < n {
        let bits = u32::from(d[i * 2]) | (u32::from(d[i * 2 + 1]) << 8);
        out[i * 4..i * 4 + 4].copy_from_slice(&(bits << 16).to_le_bytes());
        i += 1;
    }
}

/// Tensor layout and bytes for a safetensors mul_mat weight, using the same
/// BF16-storage rule as the `densew!` loader macro.
fn st_densew_parts(repo: &inkwell_st::StRepo, name: &str, bf16w: bool) -> (i32, Vec<i64>, Vec<u8>) {
    let ti = repo.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
    let mut dims: Vec<i64> = ti.shape.iter().rev().map(|&d| d as i64).collect();
    dims.retain(|&d| d != 1);
    let keep_bf16 = bf16w && ti.dtype == "BF16" && (name.contains(".controllers.")
        || name.ends_with("mlp.gate.weight") || name.ends_with("attn.attn_sink_proj.weight")
        || name.ends_with("sconv.weight_proj"));
    let bytes = if keep_bf16 {
        repo.data(name).unwrap_or_else(|| panic!("missing data {name}")).to_vec()
    } else if ti.dtype == "BF16" {
        bf16_to_f32(repo.data(name).unwrap_or_else(|| panic!("missing data {name}")))
    } else {
        assert_eq!(ti.dtype, "F32");
        repo.data(name).unwrap_or_else(|| panic!("missing data {name}")).to_vec()
    };
    ((if keep_bf16 { BF16 } else { F32 }), dims, bytes)
}

/// Process device-memory estimate. NVML is exact; the fallback uses the
/// cudaMemGetInfo baseline captured just after context creation plus a
/// conservative fixed estimate for the context/ggml runtime itself.
fn process_vram_bytes(baseline_free: usize) -> (usize, &'static str) {
    if let Some(b) = inkwell_cuda::process_memory_bytes() {
        return (b, "nvml");
    }
    let base: usize = std::env::var("INKWELL_VRAM_BASELINE_MB").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(300) << 20;
    let (free_now, _) = inkwell_cuda::mem_info();
    (base + baseline_free.saturating_sub(free_now), "meminfo-baseline")
}

impl Runner {
    /// Load the model: tensors → device buffer (dense) / cpu buffer (experts).
    ///
    /// # Panics
    /// On missing tensors or alloc failure.
    #[must_use]
    pub fn load(gg: &GgufFile, cfg: &InklingConfig, max_ctx: i64) -> Self {
        let rt = Rt::cuda();
        let cpu = Ptr(unsafe { inkwell_ggml::ink_backend_cpu(4) });
        let cuda = unsafe { inkwell_ggml::ink_backend_cuda(0) };
        let buft_dev = rt.device_buft();
        let buft_cpu = Ptr(unsafe { inkwell_ggml::ink_cpu_buft() });
        let sched = Ptr(unsafe {
            let mut bs = [cuda, cpu.0];
            let mut bufts = [buft_dev.0, buft_cpu.0];
            inkwell_ggml::ink_sched_new(bs.as_mut_ptr(), bufts.as_mut_ptr(), 2, 32768)
        });

        // classify + create tensors
        let mut dev_bytes = 0usize;
        let mut cpu_bytes = 2304 * 4 + 256; // embd_row tensor
        let mut plan: Vec<(&String, &inkwell_gguf::TensorInfo, bool, bool, bool)> = Vec::new();
        for (name, ti) in &gg.tensors {
            let expert = name.contains("exps") && !name.contains("probs") && !name.contains("scale");
            // keep quantized (IQ2_XXS): embd handled host-side, output via mmvq
            let to_f32 = false;
            let host_only = name == "token_embd.weight";
            let n_el: u64 = ti.dims.iter().product();
            let nbytes = if to_f32 { n_el * 4 } else { ti.nbytes };
            if !expert && !host_only { dev_bytes += nbytes as usize + 256; }
            plan.push((name, ti, expert, to_f32, host_only));
        }
        dev_bytes += 240 << 20; // headroom for materialized views (padded rel_bias)
        let dev_buf = rt.buf(buft_dev, dev_bytes);
        let cpu_buf = rt.buf(buft_cpu, cpu_bytes);
        // mark as weight buffers: the scheduler runs matmuls on the backend
        // where the weights live (experts -> CPU, like upstream -ncmoe)
        for b in [dev_buf.0, cpu_buf.0] {
            unsafe { inkwell_ggml::ink_buf_set_usage(b.0, 1) }; // USAGE_WEIGHTS
        }
        let mut off_dev = 0usize;
        let mut off_cpu = 0usize;
        let mut w = HashMap::new();
        let mut pinned: Vec<inkwell_cuda::PinnedBuf> = Vec::new();
        // layer → (gu pinned idx, dn pinned idx, gu stride, dn stride)
        let mut moe_w: HashMap<usize, (usize, usize, usize, usize)> = HashMap::new();
        let mut embd_src: (*const u8, i32, usize) = (std::ptr::null(), 0, 0);
        let embd_dev = std::ptr::null_mut();
        for (name, ti, expert, to_f32, host_only) in plan {
            if host_only {
                let src = unsafe { gg.mmap.as_ptr().add(gg.data_start as usize + ti.offset as usize) };
                embd_src = (src, ti.ggml_type as i32, ti.nbytes as usize / 201024usize);
                continue;
            }
            if expert {
                
                // copy into cudaHostAlloc'd pinned+mapped memory (file-backed
                // mmap cannot be registered — verified)
                let pb = inkwell_cuda::host_alloc_mapped(ti.nbytes as usize);
                let src = unsafe { gg.mmap.as_ptr().add(gg.data_start as usize + ti.offset as usize) };
                unsafe { std::ptr::copy_nonoverlapping(src, pb.host, ti.nbytes as usize) };
                // free the duplicated range: the pinned copy is the only one needed
                unsafe { madv::dontneed(src, ti.nbytes as usize) };
                let layer: usize = name.split('.').nth(1).expect("blk.N").parse().expect("layer idx");
                let idx = pinned.len();
                pinned.push(pb);
                let e = moe_w.entry(layer).or_insert((usize::MAX, usize::MAX, 0, 0));
                let stride = ti.nbytes as usize / 128;
                if name.contains("gate_up") { e.0 = idx; e.2 = stride; } else { e.1 = idx; e.3 = stride; }
                continue;
            }
            let ty = if to_f32 { F32 } else { ti.ggml_type as i32 };
            let tensor = t::new(rt.ctx(), ty, &gguf_dims(ti));
            let (buf, off) = if expert { (&cpu_buf, &mut off_cpu) } else { (&dev_buf, &mut off_dev) };
            let o = (*off + 127) & !127;
            buf.assign(tensor, o);
            let src = unsafe { gg.mmap.as_ptr().add(gg.data_start as usize + ti.offset as usize) };
            if to_f32 {
                let n_el: u64 = ti.dims.iter().product();
                let mut v = vec![0f32; n_el as usize];
                let rc = unsafe { inkwell_ggml::ink_dequant(ti.ggml_type as i32, src.cast(), v.as_mut_ptr(), n_el as i64) };
                assert_eq!(rc, 0, "dequant {name}");
                unsafe { t::set(tensor, v.as_ptr().cast(), 0, (n_el * 4) as usize) };
            } else {
                unsafe { t::set(tensor, src, 0, ti.nbytes as usize) };
            }
            *off = o + t::nbytes(tensor);
            w.insert(name.clone(), tensor);
        }

        let (f1,_) = inkwell_cuda::mem_info();
        vlog!("MEM after weights: {f1}");
        // the dequantized embedding row lives on the device: a CPU tensor here
        // forced a backend split per step, which blocks CUDA-graph capture.
        let embd_row = t::new(rt.ctx(), F32, &[2304, 1, 1, 1]);
        { let o = (off_dev + 127) & !127; dev_buf.assign(embd_row, o); }
        unsafe { inkwell_ggml::ink_sched_set_tensor_backend(sched.0, embd_row.0, cuda) };

        let get = |n: &str| -> Ptr {
            *w.get(n).unwrap_or_else(|| panic!("missing {n}"))
        };
        let opt = |n: &str| -> Option<Ptr> { w.get(n).copied() };

        let mut moe_args: Vec<Box<crate::moe::MoeArgs>> = Vec::new();
        let mut moe_pos: HashMap<usize, usize> = HashMap::new();
        // VRAM LRU expert cache: size from free VRAM after dense weights.
        let (vfree, _vtotal) = inkwell_cuda::mem_info();
        let budget = vfree.saturating_sub(1500 << 20); // fill free VRAM; headroom covers dev_buf (~840 MiB) + margin
        let budget_per_layer = budget / 24;
        for i in 0..24usize {
            let Some(&(gi, di, gs, ds)) = moe_w.get(&i) else { continue };
            let gu_ti = gg.tensors.get(&format!("blk.{i}.ffn_gate_up_exps.weight")).expect("gu ti");
            let dn_ti = gg.tensors.get(&format!("blk.{i}.ffn_down_exps.weight")).expect("dn ti");
            moe_pos.insert(i, moe_args.len());
            let k = (budget_per_layer / (gs + ds)).min(128);
            let caches = (k > 0).then(|| {
                std::sync::Mutex::new(Some((
                    crate::moe::Cache::new(k, gs),
                    crate::moe::Cache::new(k, ds),
                )))
            }).unwrap_or_else(|| std::sync::Mutex::new(None));
            if i == 0 { vlog!("VRAM cache: vfree={vfree} k={k} experts/layer ({:.1} MiB/layer)", k as f64 * (gs + ds) as f64 / 1e6); }
            moe_args.push(Box::new(crate::moe::MoeArgs {
                gu_w: pinned[gi].dev, gu_stride: gs, gu_type: gu_ti.ggml_type as i32,
                dn_w: pinned[di].dev, dn_stride: ds, dn_type: dn_ti.ggml_type as i32,
                has_sgu: gg.tensors.contains_key(&format!("blk.{i}.ffn_gate_up_exps.scale")),
                has_sdn: gg.tensors.contains_key(&format!("blk.{i}.ffn_down_exps.scale")),
                gu_host: pinned[gi].host, dn_host: pinned[di].host,
                caches,
            }));
        }

        let mut layers = Vec::with_capacity(24);
        for i in 0..24usize {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let pb = |s: &str| format!("blk.{i}.{s}.bias");
            let loop_p = |s: &str| format!("blk.{i}.loop_{s}.weight");
            let mut l = L {
                attn_norm: opt(&p("attn_norm")),
                attn_q_inp_norm: opt(&p("attn_q_inp_norm")),
                attn_k_inp_norm: opt(&p("attn_k_inp_norm")),
                attn_v_inp_norm: opt(&p("attn_v_inp_norm")),
                attn_res_query: opt(&p("attn_res_query")),
                ffn_res_query: opt(&p("ffn_res_query")),
                wq: opt(&p("attn_q")),
                wk: opt(&p("attn_k")),
                wv: opt(&p("attn_v")),
                wo: opt(&p("attn_output")),
                attn_rel: opt(&p("attn_rel")),
                attn_rel_bias: opt(&p("attn_rel_bias")),
                attn_rel_bias_pad: opt(&p("attn_rel_bias")).map(|b| {
                    // pad(b, 0,1,0,0) materialized once: appends a zero column [n0, n1+1]
                    let n0 = t::ne(b, 0) as usize;
                    let n1 = t::ne(b, 1) as usize;
                    let esz = t::nbytes(b) / (n0 * n1);
                    assert_eq!(esz, 4, "rel_bias not f32");
                    let np = t::new(rt.ctx(), F32, &[n0 as i64, n1 as i64 + 1, 1, 1]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(np, o);
                    let mut src = vec![0u8; n0 * n1 * esz];
                    let mut dst = vec![0u8; n0 * (n1 + 1) * esz];
                    unsafe { t::get(b, src.as_mut_ptr(), 0, src.len()) };
                    dst[..n0 * n1 * esz].copy_from_slice(&src);
                    unsafe { t::set(np, dst.as_ptr(), 0, dst.len()) };
                    off_dev = o + t::nbytes(np);
                    np
                }),
                attn_rel_clip: opt(&p("attn_rel_clip")),
                attn_q_clip: opt(&p("attn_q_clip")),
                attn_sink_proj: opt(&p("attn_sink_proj")),
                wqkv_gate: opt(&p("attn_gate")),
                attn_k_conv: opt(&p("attn_k_conv")),
                attn_v_conv: opt(&p("attn_v_conv")),
                attn_post_conv: opt(&p("attn_post_conv")),
                ffn_post_conv: opt(&p("ffn_post_conv")),
                attn_k_conv_proj: opt(&p("attn_k_conv_proj")),
                attn_v_conv_proj: opt(&p("attn_v_conv_proj")),
                ffn_norm: opt(&p("ffn_norm")),
                ffn_gate_inp: opt(&p("ffn_gate_inp")),
                ffn_exp_probs_b: opt(&pb("exp_probs_b")),
                moe: moe_pos.get(&i).copied(),
                gate_up_scale: opt(&format!("blk.{i}.ffn_gate_up_exps.scale")),
                down_scale: opt(&format!("blk.{i}.ffn_down_exps.scale")),
                msg_query: opt(&p("loop_msg_query")),
                ..Default::default()
            };
            for (c, cn) in CTRL_NAMES.iter().enumerate() {
                l.ctrl_down[c] = opt(&loop_p(&format!("{cn}_ctrl_down")));
                l.ctrl_gain[c] = opt(&loop_p(&format!("{cn}_ctrl_gain")));
                l.ctrl_forget[c] = opt(&loop_p(&format!("{cn}_ctrl_forget")));
            }
            // concat same-type {attn_q, attn_gate, attn_rel} -> one GEMV on inputs[0]
            {
                const MEMBERS: [(&str, i64); 3] =
                    [("attn_q", 3072), ("attn_gate", 3072), ("attn_rel", 384)];
                let tys: Vec<u32> = MEMBERS
                    .iter()
                    .map(|(n, _)| gg.tensors.get(&p(n)).map(|ti| ti.ggml_type).unwrap_or(u32::MAX))
                    .collect();
                let mut by_ty: HashMap<u32, Vec<usize>> = HashMap::new();
                for (mi, &ty) in tys.iter().enumerate() {
                    by_ty.entry(ty).or_default().push(mi);
                }
                l.proj0_off = [-1; 3];
                let group = by_ty
                    .values()
                    .filter(|g| g.len() >= 2)
                    .max_by_key(|g| g.iter().map(|&mi| MEMBERS[mi].1).sum::<i64>())
                    .cloned();
                if let Some(g) = group.filter(|_| std::env::var("INKWELL_NOCAT").is_err()) {
                    let total: i64 = g.iter().map(|&mi| MEMBERS[mi].1).sum();
                    let cat = t::new(rt.ctx(), tys[g[0]] as i32, &[2304, total]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(cat, o);
                    let mut boff = 0usize;
                    for &mi in &g {
                        let src = opt(&p(MEMBERS[mi].0)).expect("proj member");
                        let n = t::nbytes(src);
                        let mut host = vec![0u8; n];
                        unsafe { t::get(src, host.as_mut_ptr(), 0, n) };
                        unsafe { t::set(cat, host.as_ptr(), boff, n) };
                        l.proj0_off[mi] = (boff / (n / MEMBERS[mi].1 as usize)) as i64;
                        boff += n;
                    }
                    l.proj0_cat = Some(cat);
                    off_dev = o + t::nbytes(cat);
                }
            }
            layers.push(l);
        }

        let tok_embd = Ptr(std::ptr::null_mut()); // embd stays host-side
        let tok_norm = opt("token_embd_norm.weight").unwrap_or_else(|| opt("output_norm.weight").expect("tok_norm"));
        let output = get("output.weight");
        let out_head = {
            let tinfo = gg.tensors.get("output.weight");
            if tinfo.map(|t| t.ggml_type as i32) == Some(23) {
                Some(unsafe {
                    (
                        {
                            // Wp array lives on device: kernel derefs Wp[slot] on GPU
                            let mut wp_dev = inkwell_cuda::DevBuf::new_bytes(8);
                            wp_dev.write_bytes(&(inkwell_ggml::ink_tensor_data(output.0) as usize).to_ne_bytes());
                            wp_dev
                        },
                        inkwell_cuda::DevBuf::new(2304),
                        inkwell_cuda::DevBuf::new(201024),
                    )
                })
            } else {
                None
            }
        };
        let output_norm = get("output_norm.weight");
        let output_res_query = get("output_res_query.weight");

        // visits: (layer, slot) in execution order — slot like upstream:
        // pass0 → ip; pass p>0 → n_phys + (p-1)*12 + (ip-6)
        let (start, end, np, nphys) = (6usize, 18usize, 3usize, 24usize);
        let mut visits = Vec::new();
        for il in 0..start { visits.push((il, il)); }
        for pass in 0..np {
            for ip in start..end {
                let slot = if pass == 0 { ip } else { nphys + (pass - 1) * (end - start) + (ip - start) };
                visits.push((ip, slot));
            }
        }
        for il in end..nphys { visits.push((il, il)); }
        let n_full = visits.iter().filter(|(il, _)| il % 6 == 5 || il % 6 == 5).count();

        // KV arenas + conv states per slot
        let (hd, hkv, swa) = (128i64, 8i64, 1024i64);
        let swa_oracle = std::env::var_os("INKWELL_SWA_ORACLE").is_some();
        let swa_legacy_min = std::env::var_os("INKWELL_SWA_LEGACY_MIN").is_some();
        let swa_cap = if swa_oracle { max_ctx } else { swa };
        let n_slots = nphys + (np - 1) * (end - start);
        let mut k_arena = Vec::new();
        let mut v_arena = Vec::new();
        let mut conv_state = Vec::new();
        let n_sconv = (2 * hd * hkv + 2 * 2304) as usize;
        for (il, _slot) in visits.iter() {
            let _ = il;
        }
        for slot in 0..n_slots {
            let il_phys = if slot < nphys { slot } else { start + (slot - nphys) % (end - start) };
            let full = il_phys % 6 == 5;
            let cap = if full { max_ctx } else { swa_cap };
            k_arena.push(t::new(rt.ctx(), F32, &[hd, cap, hkv]));
            v_arena.push(t::new(rt.ctx(), F32, &[hd, cap, hkv]));
            conv_state.push(t::new(rt.ctx(), F32, &[3, n_sconv as i64]));
        }
        // allocate KV + conv in one extra device buffer
        let kv_bytes: usize = k_arena.iter().chain(v_arena.iter()).map(|x| t::nbytes(*x) + 256).sum::<usize>()
            + conv_state.iter().map(|x| t::nbytes(*x) + 256).sum::<usize>();
        let kv_buf = rt.buf(buft_dev, kv_bytes);
        let mut o = 0usize;
        for x in k_arena.iter().chain(v_arena.iter()).chain(conv_state.iter()) {
            o = (o + 127) & !127;
            kv_buf.assign(*x, o);
            o += t::nbytes(*x);
        }
        std::mem::forget(kv_buf); // keep alive — TODO: store in struct

        Self {
            sched_cpu: Some((sched, cpu)),
            conv_args: std::cell::UnsafeCell::new(Vec::new()),
            conv_arg_idx: std::cell::Cell::new(0),
            cur_ctx: Ptr(std::ptr::null_mut()),
            mix_prefix: std::cell::UnsafeCell::new(None),
            dump_list: std::cell::UnsafeCell::new(Vec::new()),
            in_buf: None,
            rt, sched, dev_buf, cpu_buf, w, pinned, expert_dev: Vec::new(), moe_args,
            exl3_w: Vec::new(), exl3_blk: Vec::new(), moe_args_exl3: Vec::new(), output_exl3: None,
            cfg: cfg.clone(), layers,
            tok_embd, embd_src, embd_dev, embd_row, tok_norm, output, output_norm, output_res_query,
            msg_query_packed: Ptr(std::ptr::null_mut()), out_head,
            k_arena, v_arena, conv_state, conv_snap: Vec::new(), block_feats: None, feat_ud: std::cell::RefCell::new(Vec::new()), feat_ops: std::cell::RefCell::new(Vec::new()), max_ctx, swa, swa_oracle, swa_legacy_min, visits,
            n_full, step_gcache: None, sid: None,
        }
    }

    /// Load a SAGE-REXL3 safetensors repo (dir with `model.safetensors.index.json`).
    /// Dense tensors -> F32 ggml; attention/expert projections -> EXL3 descriptors.
    ///
    /// # Panics
    /// On missing tensors or alloc failure.
    #[must_use]
    #[expect(clippy::too_many_lines)]
    pub fn load_st(repo: &inkwell_st::StRepo, cfg: &InklingConfig, max_ctx: i64) -> Self {
        crate::prof::init();
        let _g_rt = crate::prof::guard("rt_cuda_init");
        let rt = Rt::cuda();
        let (vram_baseline_free, _) = inkwell_cuda::mem_info();
        let cpu = Ptr(unsafe { inkwell_ggml::ink_backend_cpu(4) });
        let cuda = unsafe { inkwell_ggml::ink_backend_cuda(0) };
        let buft_dev = rt.device_buft();
        let buft_cpu = Ptr(unsafe { inkwell_ggml::ink_cpu_buft() });
        let sched = Ptr(unsafe {
            let mut bs = [cuda, cpu.0];
            let mut bufts = [buft_dev.0, buft_cpu.0];
            inkwell_ggml::ink_sched_new(bs.as_mut_ptr(), bufts.as_mut_ptr(), 2, 32768)
        });

        drop(_g_rt);
        let lp = |il: usize, name: &str| format!("model.llm.layers.{il}.{name}");
        let tp = |name: &str| repo.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
        let dtp = |name: &str| repo.data(name).unwrap_or_else(|| panic!("missing data {name}"));

        // ---- device byte budget: dense F32 (mul_mat weights BF16) + headroom ----
        // INKWELL_NO_BF16CTRL=1 restores F32 storage for the mul_mat-consumed
        // weights (controllers, router gate, sinks, conv projections).
        let bf16w = std::env::var_os("INKWELL_NO_BF16CTRL").is_none();
        let bf16_ok = |name: &str| bf16w && (name.contains(".controllers.")
            || name.ends_with("mlp.gate.weight") || name.ends_with("attn.attn_sink_proj.weight")
            || name.ends_with("sconv.weight_proj"));
        // The old loader reserved 160 MiB of slack. Measured occupancy after
        // the gain/forget dedup is within a few MiB of the exact tensor sum;
        // keep 8 MiB for alignment/derived tensors. INKWELL_NO_DEVBUF_TRIM=1
        // restores the old reserve.
        let devbuf_trim = std::env::var_os("INKWELL_NO_DEVBUF_TRIM").is_none();
        let dev_headroom: usize = if devbuf_trim {
            std::env::var("INKWELL_DEVBUF_HEADROOM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
        } else { 160 } << 20;
        let count_dup_gfcat = devbuf_trim
            && std::env::var_os("INKWELL_NO_GFCAT_TRIM").is_some()
            && std::env::var_os("INKWELL_NOCTRLCAT").is_none();
        let relbias_bf16 = bf16w && std::env::var_os("INKWELL_NO_SINKRELBIAS").is_none();
        let mut dev_bytes = dev_headroom;
        for (n, t) in &repo.tensors {
            let (name, ti) = (n.as_str(), &t.1);
            if name.contains(".experts.") || name.contains(".trellis") || name.contains(".suh")
                || name.contains(".svh") || name.contains(".mul1") || name == "model.llm.embed.weight"
                || name.contains("visual") { continue; }
            let rb2 = relbias_bf16 && name.ends_with("attn.rel_logits_proj.proj");
            let w = if (bf16_ok(name) || rb2) && ti.dtype == "BF16" { 2 } else { 4 };
            let bytes = (ti.elems() * w) as usize;
            dev_bytes += bytes + 256;
            if count_dup_gfcat && name.contains(".controllers.")
                && (name.ends_with("gain_weight") || name.ends_with("forget_weight")) {
                dev_bytes += bytes;
            }
        }
        let dev_buf = rt.buf(buft_dev, dev_bytes);
        let cpu_buf = rt.buf(buft_cpu, 2304 * 4 + 256);
        for b in [dev_buf.0, cpu_buf.0] {
            unsafe { inkwell_ggml::ink_buf_set_usage(b.0, 1) };
        }
        let mut off_dev = 0usize;

        /// upload a dense tensor as F32 ggml tensor, dims = reversed HF shape (drop ==1 middles)
        macro_rules! dense {
            ($name:expr) => {{
                let _pg = crate::prof::guard("dense_upload");
                let ti = tp($name);
                let mut dims: Vec<i64> = ti.shape.iter().rev().map(|&d| d as i64).collect();
                dims.retain(|&d| d != 1);
                let ten = t::new(rt.ctx(), F32, &dims);
                let o = (off_dev + 127) & !127;
                dev_buf.assign(ten, o);
                let src = dtp($name);
                if ti.dtype == "BF16" {
                    let bytes = bf16_to_f32(src);
                    unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                } else {
                    assert_eq!(ti.dtype, "F32");
                    // F32 safetensors already have the exact upload bytes;
                    // copying them into a temporary Vec first is pure startup cost.
                    unsafe { t::set(ten, src.as_ptr(), 0, src.len()) };
                }
                off_dev = o + t::nbytes(ten);
                ten
            }};
        }

        /// upload a mul_mat weight: keep the file's BF16 bytes when enabled
        /// (ggml-cuda mmvf reads BF16 src0 with identical F32 accumulation
        /// order on sm_61); widen to F32 otherwise.
        macro_rules! densew {
            ($name:expr) => {{
                let _pg = crate::prof::guard("densew_upload");
                let ti = tp($name);
                let mut dims: Vec<i64> = ti.shape.iter().rev().map(|&d| d as i64).collect();
                dims.retain(|&d| d != 1);
                let src = dtp($name);
                let keep_bf16 = bf16_ok($name) && ti.dtype == "BF16";
                let ty = if keep_bf16 { BF16 } else { F32 };
                let ten = t::new(rt.ctx(), ty, &dims);
                let o = (off_dev + 127) & !127;
                dev_buf.assign(ten, o);
                if ti.dtype == "BF16" && !keep_bf16 {
                    let bytes = bf16_to_f32(src);
                    unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                } else {
                    if !keep_bf16 { assert_eq!(ti.dtype, "F32"); }
                    // Keep-BF16 and F32 inputs upload straight from the mmap
                    // without a host-side copy.
                    unsafe { t::set(ten, src.as_ptr(), 0, src.len()) };
                }
                off_dev = o + t::nbytes(ten);
                ten
            }};
        }

        /// build an Exl3W from `<base>.{trellis,suh,svh}` on device
        fn exl3_of(repo: &inkwell_st::StRepo, base: &str) -> Box<Exl3W> {
            let _pg = crate::prof::guard("exl3proj_upload");
            let tr = repo.tensor(&format!("{base}.trellis")).unwrap_or_else(|| panic!("{base}.trellis"));
            let (k, n) = ((tr.shape[0] * 16) as usize, (tr.shape[1] * 16) as usize);
            let k16 = tr.shape[2] as usize;
            let (ka, halfk) = (k16 / 16, k16 % 16 != 0);
            let mut trel = inkwell_cuda::DevBuf::new_bytes(tr.nbytes() as usize);
            trel.write_bytes(repo.data(&format!("{base}.trellis")).unwrap());
            let su = repo.tensor(&format!("{base}.suh")).unwrap();
            let mut suh = inkwell_cuda::DevBuf::new_bytes(su.nbytes() as usize);
            suh.write_bytes(repo.data(&format!("{base}.suh")).unwrap());
            let sv = repo.tensor(&format!("{base}.svh")).unwrap();
            let mut svh = inkwell_cuda::DevBuf::new_bytes(sv.nbytes() as usize);
            svh.write_bytes(repo.data(&format!("{base}.svh")).unwrap());
            // fused sgemv uses scratch[k + 8n .. +512) as atomic strip
            // counters: must start zeroed (kernel self-resets per launch).
            let scratch = inkwell_cuda::DevBuf::new(k + 8 * n + 512);
            scratch.zero();
            Box::new(Exl3W { trel, suh, svh, ka, halfk, k, n, scratch })
        }

        // ---- globals ----
        let mut pinned: Vec<inkwell_cuda::PinnedBuf> = Vec::new();
        let embd_bytes = dtp("model.llm.embed.weight");
        let (embd_src, embd_dev): ((*const u8, i32, usize), *mut std::ffi::c_void) =
            if std::env::var_os("INKWELL_NO_PIPESTEP").is_none() {
                let pb = inkwell_cuda::host_alloc_mapped(embd_bytes.len());
                unsafe {
                    std::ptr::copy_nonoverlapping(embd_bytes.as_ptr(), pb.host, embd_bytes.len());
                    madv::dontneed(embd_bytes.as_ptr(), embd_bytes.len());
                }
                let r = ((pb.host.cast_const(), -2, 2304 * 2), pb.dev);
                pinned.push(pb);
                r
            } else {
                ((embd_bytes.as_ptr(), -2, 2304 * 2), std::ptr::null_mut())
            };
        let tok_embd = Ptr(std::ptr::null_mut());
        let tok_norm = dense!("model.llm.embed_norm.weight");
        let output_norm = dense!("model.llm.norm.weight");
        let output_res_query = dense!("model.llm.attn_res.output_query");
        let mut exl3_w: Vec<Box<Exl3W>> = Vec::new();
        let output_exl3 = Some(exl3_w.len());
        exl3_w.push(exl3_of(repo, "model.llm.unembed"));
        let output = Ptr(std::ptr::null_mut());
        let out_head = None;
        let embd_row = t::new(rt.ctx(), F32, &[2304, 1, 1, 1]);
        { let o = (off_dev + 127) & !127; dev_buf.assign(embd_row, o); off_dev = o + t::nbytes(embd_row); }
        unsafe { inkwell_ggml::ink_sched_set_tensor_backend(sched.0, embd_row.0, cuda) };

        // ---- middle_cycle queries: loop_msg_query(il) cols = [q[2i], kv[2i], kv[2i+1], q[2i+1]] ----
        let mcq = dtp("model.llm.middle_cycle_cross_pass_queries");
        let mckv = dtp("model.llm.middle_cycle_cross_pass_kv_queries");
        let row: fn(&[u8], usize) -> &[u8] = |b, r| &b[r * 2304 * 4..(r + 1) * 2304 * 4];

        // ---- on_dev expert placement plan ----
        // Loop layers (6..18) run 3 passes/token, the rest 1: plan which layers'
        // expert blobs sit in VRAM by visits/byte desc within an expert budget,
        // instead of greedy index order (which put all of 0-11 on_dev while loop
        // layers 12-17 paid PCIe misses). INKWELL_PLACE_INDEX=1 restores the old
        // index-order policy; INKWELL_DEV_LAYERS="6-17,..." forces an explicit set;
        // INKWELL_DEV_EXPERT_MB overrides the budget. A planned layer still falls
        // back to pinned at load time if it does not fit (same margin check).
        let place_plan: Option<Vec<bool>> = if std::env::var_os("INKWELL_PLACE_INDEX").is_some() {
            None
        } else {
            // sizing pre-pass: header metadata only, no data reads
            let mut totals = [0usize; 24];
            for il in 0..24usize {
                let pre = lp(il, "mlp.experts");
                for e in 0..128usize {
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        for part in ["trellis", "suh", "svh"] {
                            totals[il] += tp(&format!("{pre}.{e}.{proj}.{part}")).nbytes() as usize;
                        }
                    }
                }
            }
            let mut plan = vec![false; 24];
            let mut budget = None;
            let joint_default = std::env::var_os("INKWELL_NO_JOINTPLACE").is_none()
                && std::env::var("INKWELL_DEV_LAYERS").is_err()
                && std::env::var("INKWELL_DEV_EXPERT_MB").is_err();
            if joint_default {
                // Joint placement/cache optimum replayed on the short-context
                // routing trace: keeping these layers resident frees more cache
                // bytes than it costs, and lowers miss traffic by ~13% without reducing resident expert bytes.
                for il in [3usize, 6, 7, 8, 10, 11, 12, 14, 16, 17] { plan[il] = true; }
            } else if let Ok(spec) = std::env::var("INKWELL_DEV_LAYERS") {
                for tok in spec.split(',') {
                    let t = tok.trim();
                    if t.is_empty() { continue }
                    let (a, b) = t.split_once('-').unwrap_or((t, t));
                    if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                        for il in a..=b.min(23) { plan[il] = true; }
                    }
                }
            } else {
                // default = what index-order placed with the BEST env (layers 0-11)
                let bud = std::env::var("INKWELL_DEV_EXPERT_MB").ok()
                    .and_then(|v| v.parse::<usize>().ok()).map(|m| m << 20)
                    .unwrap_or(3_121_348_608);
                budget = Some(bud);
                let mut ord: Vec<usize> = (0..24).collect();
                ord.sort_by(|&a, &b| {
                    let va = if (6..18).contains(&a) { 3usize } else { 1 };
                    let vb = if (6..18).contains(&b) { 3usize } else { 1 };
                    (vb * totals[a]).cmp(&(va * totals[b]))
                });
                let mut used = 0usize;
                for &il in &ord {
                    if used + totals[il] <= bud { plan[il] = true; used += totals[il]; }
                }
            }
            let on: Vec<usize> = (0..24).filter(|&i| plan[i]).collect();
            let mib: usize = on.iter().map(|&i| totals[i] >> 20).sum();
            vlog!("place: on_dev={on:?} experts={mib}MiB budget={}MiB",
                budget.map_or("inf".to_string(), |b| (b >> 20).to_string()));
            Some(plan)
        };

        let joint_place_active = place_plan.as_ref().map_or(false, |p| {
            [3usize, 6, 7, 8, 10, 11, 12, 14, 16, 17].iter().all(|&i| p[i])
                && std::env::var_os("INKWELL_NO_JOINTPLACE").is_none()
                && std::env::var("INKWELL_DEV_LAYERS").is_err()
                && std::env::var("INKWELL_DEV_EXPERT_MB").is_err()
        });

        // ---- per-layer ----
        let mut layers = Vec::with_capacity(24);
        // One packed [2304, 48] query tensor backs all 12 middle layers'
        // zero-copy [2304, 4] views. The packed tensor also lets passes 1/2
        // evaluate all 48 queries in one mix launch group.
        let msg_query_packed = t::new(rt.ctx(), F32, &[2304, 48]);
        {
            let o = (off_dev + 127) & !127;
            dev_buf.assign(msg_query_packed, o);
            off_dev = o + t::nbytes(msg_query_packed);
        }
        let mut pinned_bytes = 0usize;
        let mut expert_dev: Vec<inkwell_cuda::DevBuf> = Vec::new();
        let mut moe_args_exl3: Vec<Box<crate::moe::MoeArgsExl3>> = Vec::new();
        let mut moe_gw: Vec<[u32; 384]> = Vec::new();
        let mut expert_loads: Vec<ExpertLoad> = Vec::with_capacity(24);
        let conv = |il: usize| il % 6 == 5;
        for il in 0..24usize {
            let mut l = L {
                attn_norm: (!conv(il)).then(|| dense!(&lp(il, "attn_norm.weight"))),
                attn_q_inp_norm: conv(il).then(|| dense!(&lp(il, "q_input_norm.weight"))),
                attn_k_inp_norm: conv(il).then(|| dense!(&lp(il, "k_input_norm.weight"))),
                attn_v_inp_norm: conv(il).then(|| dense!(&lp(il, "v_input_norm.weight"))),
                attn_rel_bias: Some({
                    // HF stores [in,out]=[16,1024]; ggml wants dims [16,1024] col-major → transpose
                    let ti = tp(&lp(il, "attn.rel_logits_proj.proj"));
                    let bytes = bf16_to_f32_t(dtp(&lp(il, "attn.rel_logits_proj.proj")), &ti.shape);
                    let ten = t::new(rt.ctx(), F32, &[16, 1024]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(ten, o);
                    unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                    off_dev = o + t::nbytes(ten);
                    ten
                }),
                attn_rel_bias_pad: Some({
                    let n0 = 16usize; let n1 = 1024usize;
                    let raw = dtp(&lp(il, "attn.rel_logits_proj.proj"));
                    let fused = bf16w && std::env::var_os("INKWELL_NO_SINKRELBIAS").is_none();
                    let (ty, esz, bytes) = if fused {
                        (BF16, 2usize, bf16_t(raw, &[16, 1024]))
                    } else {
                        (F32, 4usize, bf16_to_f32_t(raw, &[16, 1024]))
                    };
                    let np = t::new(rt.ctx(), ty, &[n0 as i64, n1 as i64 + 1, 1, 1]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(np, o);
                    let mut dst = vec![0u8; n0 * (n1 + 1) * esz];
                    dst[..n0 * n1 * esz].copy_from_slice(&bytes);
                    unsafe { t::set(np, dst.as_ptr(), 0, dst.len()) };
                    off_dev = o + t::nbytes(np);
                    np
                }),
                attn_rel_clip: Some(dense!(&lp(il, "attn.rel_l2_clip.scale"))),
                attn_q_clip: Some(dense!(&lp(il, "attn.q_l2_clip.scale"))),
                attn_sink_proj: Some(densew!(&lp(il, "attn.attn_sink_proj.weight"))),
                attn_k_conv: if conv(il) {
                    // HF static_w is [4,1024] [tap,ch] — ggml [4,1024] col-major wants taps contiguous per ch -> transpose
                    let bytes = bf16_to_f32_t(dtp(&lp(il, "attn.k_sconv.static_w")), &[4, 1024]);
                    let ten = t::new(rt.ctx(), F32, &[4, 1024]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(ten, o);
                    unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                    off_dev = o + t::nbytes(ten);
                    Some(ten)
                } else { Some(dense!(&lp(il, "attn.k_sconv.weight"))) },
                attn_v_conv: if conv(il) {
                    let bytes = bf16_to_f32_t(dtp(&lp(il, "attn.v_sconv.static_w")), &[4, 1024]);
                    let ten = t::new(rt.ctx(), F32, &[4, 1024]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(ten, o);
                    unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                    off_dev = o + t::nbytes(ten);
                    Some(ten)
                } else { Some(dense!(&lp(il, "attn.v_sconv.weight"))) },
                attn_k_conv_proj: conv(il).then(|| densew!(&lp(il, "attn.k_sconv.weight_proj"))),
                attn_v_conv_proj: conv(il).then(|| densew!(&lp(il, "attn.v_sconv.weight_proj"))),
                attn_post_conv: Some(dense!(&lp(il, "attn_sconv.weight"))),
                ffn_post_conv: Some(dense!(&lp(il, "mlp_sconv.weight"))),
                ffn_norm: Some(dense!(&lp(il, "mlp_norm.weight"))),
                ffn_gate_inp: Some(densew!(&lp(il, "mlp.gate.weight"))),
                ffn_exp_probs_b: Some(dense!(&lp(il, "mlp.gate.bias"))),
                attn_res_query: (il >= 1).then(|| {
                    let q = tp(&format!("model.llm.attn_res.attn_queries.{}", il - 1));
                    let dims: Vec<i64> = q.shape.iter().rev().map(|&d| d as i64).collect();
                    let ten = t::new(rt.ctx(), F32, &dims);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(ten, o);
                    let b = dtp(&format!("model.llm.attn_res.attn_queries.{}", il - 1));
                    unsafe { t::set(ten, b.as_ptr(), 0, b.len()) };
                    off_dev = o + t::nbytes(ten);
                    ten
                }),
                ffn_res_query: Some(dense!(&format!("model.llm.attn_res.mlp_queries.{il}"))),
                msg_query: (il >= 6 && il < 18).then(|| {
                    let i = il - 6;
                    // cols [q[2i], kv[2i], kv[2i+1], q[2i+1]] contiguous col-major
                    let mut buf = Vec::with_capacity(4 * 2304 * 4);
                    buf.extend_from_slice(row(mcq, 2 * i));
                    buf.extend_from_slice(row(mckv, 2 * i));
                    buf.extend_from_slice(row(mckv, 2 * i + 1));
                    buf.extend_from_slice(row(mcq, 2 * i + 1));
                    let off = i * 4 * 2304 * 4;
                    unsafe { t::set(msg_query_packed, buf.as_ptr(), off, buf.len()) };
                    t::view2(rt.ctx(), msg_query_packed, 2304, 4, t::nb(msg_query_packed, 1), off as i64)
                }),
                ..Default::default()
            };
            // exl3 attn projections
            for (key, hf) in [("wq", "attn.wq_du"), ("gate", "attn.gate_attn"), ("wk", "attn.wk_dv"),
                ("wv", "attn.wv_dv"), ("rel", "attn.wr_du"), ("wo", "attn.wo_ud")] {
                l.exl3.insert(key, exl3_w.len());
                exl3_w.push(exl3_of(repo, &lp(il, hf)));
            }
            // loop controllers
            if il >= 6 && il < 18 {
                for (ci, cn) in CTRL_NAMES.iter().enumerate() {
                    let hn = match *cn { "q" => "attn", "ffn" => "mlp", x => x };
                    l.ctrl_down[ci] = Some(densew!(&lp(il, &format!("controllers.{hn}.down_weight"))));
                    let gain_name = lp(il, &format!("controllers.{hn}.gain_weight"));
                    let forget_name = lp(il, &format!("controllers.{hn}.forget_weight"));
                    // Concatenate directly from host bytes instead of uploading
                    // gain/forget and then a second concatenated copy.
                    // INKWELL_NO_GFCAT_TRIM=1 restores the duplicate copies.
                    if std::env::var_os("INKWELL_NO_GFCAT_TRIM").is_none()
                        && std::env::var_os("INKWELL_NOCTRLCAT").is_none() {
                        let (tg, dg, bg) = st_densew_parts(repo, &gain_name, bf16w);
                        let (tf, df, bf) = st_densew_parts(repo, &forget_name, bf16w);
                        if tg == tf && dg.len() == 2 && df.len() == 2 && dg[0] == df[0] {
                            let cat = t::new(rt.ctx(), tg, &[dg[0], dg[1] + df[1]]);
                            let o = (off_dev + 127) & !127;
                            dev_buf.assign(cat, o);
                            let mut bytes = Vec::with_capacity(bg.len() + bf.len());
                            bytes.extend_from_slice(&bg);
                            bytes.extend_from_slice(&bf);
                            unsafe { t::set(cat, bytes.as_ptr(), 0, bytes.len()) };
                            l.ctrl_gf_cat[ci] = Some(cat);
                            l.ctrl_gf_off[ci] = [0, dg[1]];
                            off_dev = o + t::nbytes(cat);
                        }
                    }
                    if l.ctrl_gf_cat[ci].is_none() {
                        l.ctrl_gain[ci] = Some(densew!(&gain_name));
                        l.ctrl_forget[ci] = Some(densew!(&forget_name));
                    }
                }
            }
            // concat same-type {ctrl_gain, ctrl_forget} -> one GEMV on shared input
            for ci in 0..4 {
                let (Some(gw), Some(fw)) = (l.ctrl_gain[ci], l.ctrl_forget[ci]) else { continue };
                let (tg, tf) = unsafe { (inkwell_ggml::ink_tensor_type(gw.0), inkwell_ggml::ink_tensor_type(fw.0)) };
                if tg != tf || std::env::var("INKWELL_NOCTRLCAT").is_ok() { continue }
                let ng = unsafe { inkwell_ggml::ink_tensor_ne(gw.0, 1) };
                let cat = t::new(rt.ctx(), tg as i32, &[unsafe { inkwell_ggml::ink_tensor_ne(gw.0, 0) }, ng + unsafe { inkwell_ggml::ink_tensor_ne(fw.0, 1) }]);
                let o = (off_dev + 127) & !127;
                dev_buf.assign(cat, o);
                let mut boff = 0usize;
                for src in [gw, fw] {
                    let n = t::nbytes(src);
                    let mut host = vec![0u8; n];
                    unsafe { t::get(src, host.as_mut_ptr(), 0, n) };
                    unsafe { t::set(cat, host.as_ptr(), boff, n) };
                    boff += n;
                }
                l.ctrl_gf_cat[ci] = Some(cat);
                l.ctrl_gf_off[ci] = [0, ng];
                off_dev = o + t::nbytes(cat);
            }
            // ---- experts: VRAM blob + 3 desc tables (pinned host staging) ----
            {
                let pre = lp(il, "mlp.experts");
                let mut total = 0usize;
                for e in 0..128usize {
                    for proj in ["gate_proj", "up_proj", "down_proj"] {
                        for part in ["trellis", "suh", "svh"] {
                            total += tp(&format!("{pre}.{e}.{proj}.{part}")).nbytes() as usize;
                        }
                    }
                }
                if il == 6 { let (f0, _) = inkwell_cuda::mem_info(); vlog!("MEM before experts: {f0}"); }
                let mut descs: Vec<u64> = vec![0; 3 * 128 * 4];
                let mut off = 0usize;
                let mut max_slot = 0usize;
                let mut gxp = [0u32; 384];
                let mut segs: Vec<CopySeg> = Vec::with_capacity(3 * 128 * 3);
                for e in 0..128usize {
                    for (pi, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                        let base = format!("{pre}.{e}.{proj}");
                        for (ai, part) in ["trellis", "suh", "svh"].iter().enumerate() {
                            let b = dtp(&format!("{base}.{part}"));
                            segs.push(CopySeg { src: b.as_ptr() as usize, dst: off, len: b.len() });
                            descs[pi * 512 + e * 4 + ai] = off as u64; // patched after upload
                            off += b.len();
                        }
                        let k16 = tp(&format!("{base}.trellis")).shape[2] as usize;
                        descs[pi * 512 + e * 4 + 3] = (k16 / 16) as u64 | (((k16 % 16 != 0) as u64) << 8);
                        // per-layer cache slot = max expert blob (trellis+suh+svh)
                        let w = (k16 / 16) * 8 + usize::from(k16 % 16 != 0) * 4;
                        let blob = 10368 * w * 4 + 6912;
                        gxp[e * 3 + pi] = (blob / 6912) as u32; // = 6*w + 1 granules
                        max_slot = max_slot.max(blob);
                    }
                }
                moe_gw.push(gxp);
                let (f_free, _) = inkwell_cuda::mem_info();
                vlog!("experts il={il}: free={}MiB total={}MiB", f_free >> 20, total >> 20);
                let dev_margin: usize = std::env::var("INKWELL_DEV_MARGIN_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1300);
                let on_dev = place_plan.as_ref().map_or(true, |p| p[il])
                    && (f_free as usize) > total + (dev_margin << 20);
                let mut host_base = 0usize;
                let mut dev_base = 0usize;
                let mut pin = None;
                if std::env::var_os("INKWELL_NO_PINDIR").is_some() {
                    // Compatibility path: gather into pageable memory first, then
                    // copy once more into the final pinned/device destination.
                    let mut host = vec![0u8; total];
                    let mut off2 = 0usize;
                    for e in 0..128usize {
                        for proj in ["gate_proj", "up_proj", "down_proj"] {
                            let base = format!("{pre}.{e}.{proj}");
                            for part in ["trellis", "suh", "svh"] {
                                let b = dtp(&format!("{base}.{part}"));
                                host[off2..off2 + b.len()].copy_from_slice(b);
                                unsafe { madv::dontneed(b.as_ptr(), b.len()) };
                                off2 += b.len();
                            }
                        }
                    }
                    if on_dev {
                        let mut blob = inkwell_cuda::DevBuf::new_bytes(total);
                        blob.write_bytes(&host);
                        dev_base = blob.dev_ptr() as usize;
                        expert_dev.push(blob);
                    } else {
                        let pb = inkwell_cuda::host_alloc_mapped(total);
                        unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), pb.host, total) };
                        pinned_bytes += total;
                        host_base = pb.host as usize;
                        dev_base = pb.dev as usize;
                        pinned.push(pb);
                    }
                } else if on_dev {
                    // Parallel-gather the layer once, then keep the existing
                    // synchronous contiguous H2D upload. Pinned layers copy in
                    // background threads underneath these device uploads.
                    let mut blob = inkwell_cuda::DevBuf::new_bytes(total);
                    let mut host = vec![0u8; total];
                    let base_addr = host.as_mut_ptr() as usize;
                    for seg in &mut segs { seg.dst += base_addr; }
                    let nth = std::env::var("INKWELL_LOAD_THREADS").ok()
                        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(4);
                    {
                        let _pg = crate::prof::guard("expdev_gather");
                        par_memcpy_abs(&segs, nth);
                    }
                    {
                        let _pg = crate::prof::guard("expdev_h2d");
                        blob.write_bytes(&host);
                    }
                    dev_base = blob.dev_ptr() as usize;
                    expert_dev.push(blob);
                } else if std::env::var_os("INKWELL_NO_PINBG").is_some() {
                    // Direct mmap -> pinned copy, but synchronous on the loader
                    // thread (killswitch for the default overlapped loader).
                    let pb = inkwell_cuda::host_alloc_mapped(total);
                    let base_addr = pb.host as usize;
                    for seg in &mut segs { seg.dst += base_addr; }
                    let nth = std::env::var("INKWELL_LOAD_THREADS").ok()
                        .and_then(|v| v.parse::<usize>().ok()).unwrap_or(4);
                    par_memcpy_abs(&segs, nth);
                    pinned_bytes += total;
                    host_base = pb.host as usize;
                    dev_base = pb.dev as usize;
                    pinned.push(pb);
                } else {
                    {
                        let _pg = crate::prof::guard("exppin_spawn");
                        pin = Some(spawn_pin_job(segs, total));
                    }
                }
                expert_loads.push(ExpertLoad {
                    il, descs, on_dev, max_slot, total, host_base, dev_base, pin,
                });
            }
            l.proj0_off = [-1; 3];
            layers.push(l);
        }
        // Materialize expert descriptors after the layer loop. Background
        // pinned jobs report their cudaHostAlloc coordinates as soon as the
        // allocation exists; their byte copies keep running while dense setup,
        // device-resident expert uploads, KV allocation, and cache setup proceed.
        let mut pin_handles = Vec::new();
        for mut x in expert_loads {
            if let Some(job) = x.pin {
                let (h, d) = job.rx.recv().expect("pinned expert allocation channel");
                x.host_base = h;
                x.dev_base = d;
                pinned_bytes += x.total;
                pin_handles.push(job.handle);
            }
            let base = x.dev_base as u64;
            for d in x.descs.chunks_exact_mut(4) {
                d[0] += base; d[1] += base; d[2] += base;
            }
            let v2_descs_ok = crate::moe::exl3_descs_v2_ok(&x.descs);
            let mk = |pi: usize| {
                crate::moe::DescTable::from_bytes(unsafe {
                    std::slice::from_raw_parts(x.descs[pi * 512..].as_ptr().cast::<u8>(), 128 * 32)
                })
            };
            layers[x.il].moe_exl3 = Some(moe_args_exl3.len());
            moe_args_exl3.push(Box::new(crate::moe::MoeArgsExl3 {
                descs_g: mk(0), descs_u: mk(1), descs_d: mk(2),
                v2_descs_ok, on_dev: x.on_dev, cache: None, max_slot: x.max_slot,
                layer: x.il as i32,
                descs_host: x.descs.clone(), expert_host_base: x.host_base,
                expert_dev_base: x.dev_base, expert_total: x.total,
            }));
        }
        assert!(off_dev <= dev_bytes, "dev_buf overflow: use={off_dev} alloc={dev_bytes}");
        vlog!("dev_buf: use={:.1}MiB alloc={:.1}MiB slack={:.1}MiB",
            off_dev as f64 / 1048576.0, dev_bytes as f64 / 1048576.0,
            (dev_bytes - off_dev) as f64 / 1048576.0);
        vlog!("load_st: dev={}MiB exl3_w={} pinned={}MiB", dev_bytes >> 20, exl3_w.len(),
            pinned_bytes >> 20);
        let exl3_blk: Vec<Box<Exl3Blk>> = exl3_w.iter().map(|e| {
            let b = Box::new(Exl3Blk {
                trel: e.trel.dev_ptr(), suh: e.suh.dev_ptr(), svh: e.svh.dev_ptr(),
                k: e.k, n: e.n, ka: e.ka, halfk: e.halfk,
                scratch_len: 8 * e.k + 64 * e.n,
                scratch: std::sync::OnceLock::new(),
            });
            // INKWELL_NO_LAZY_BLKSCRATCH=1 restores eager allocation at load.
            if std::env::var_os("INKWELL_NO_LAZY_BLKSCRATCH").is_some() {
                b.ensure_scratch();
            }
            b
        }).collect();

        // visits + KV arenas (same as GGUF path)
        let (start, end, np, nphys) = (6usize, 18usize, 3usize, 24usize);
        let mut visits = Vec::new();
        for il in 0..start { visits.push((il, il)); }
        for pass in 0..np {
            for ip in start..end {
                let slot = if pass == 0 { ip } else { nphys + (pass - 1) * (end - start) + (ip - start) };
                visits.push((ip, slot));
            }
        }
        for il in end..nphys { visits.push((il, il)); }
        let n_full = visits.iter().filter(|(il, _)| il % 6 == 5).count();

        let (hd, hkv, swa) = (128i64, 8i64, 1024i64);
        let swa_oracle = std::env::var_os("INKWELL_SWA_ORACLE").is_some();
        let swa_legacy_min = std::env::var_os("INKWELL_SWA_LEGACY_MIN").is_some();
        let swa_cap = if swa_oracle { max_ctx } else { swa };
        let n_slots = nphys + (np - 1) * (end - start);
        let mut k_arena = Vec::new();
        let mut v_arena = Vec::new();
        let mut conv_state = Vec::new();
        let n_sconv = (2 * hd * hkv + 2 * 2304) as usize;
        for slot in 0..n_slots {
            let il_phys = if slot < nphys { slot } else { start + (slot - nphys) % (end - start) };
            let full = il_phys % 6 == 5;
            let cap = if full { max_ctx } else { swa_cap };
            k_arena.push(t::new(rt.ctx(), F16, &[hd, cap, hkv]));
            v_arena.push(t::new(rt.ctx(), F16, &[hd, cap, hkv]));
            conv_state.push(t::new(rt.ctx(), F32, &[3, n_sconv as i64]));
        }
        let _g_kv = crate::prof::guard("kv_arena");
        let kv_bytes: usize = k_arena.iter().chain(v_arena.iter()).map(|x| t::nbytes(*x) + 256).sum::<usize>()
            + conv_state.iter().map(|x| t::nbytes(*x) + 256).sum::<usize>();
        let kv_buf = rt.buf(buft_dev, kv_bytes);
        let mut o = 0usize;
        for x in k_arena.iter().chain(v_arena.iter()).chain(conv_state.iter()) {
            o = (o + 127) & !127;
            kv_buf.assign(*x, o);
            o += t::nbytes(*x);
        }
        vlog!("bufs: dev={}MiB kv={}MiB hd={hd} hkv={hkv} swa={swa} n_slots={n_slots}",
            dev_bytes >> 20, kv_bytes >> 20);
        std::mem::forget(kv_buf);
        drop(_g_kv);

        {
            let _g = crate::prof::guard("warm_staging");
            crate::moe::warm_staging(); // staging buffers must exist before CUDA-graph capture
        }
        // Pinned layers stage expert blobs over PCIe every visit (~25-40 ms/tok
        // total). Cache k experts/layer in VRAM; k must be ≥ N_SLOTS(8) so a
        // single visit never evicts its own slots. Runs post-load so f_free is
        // the real remainder: dev_buf, kv_buf and staging already exist — only
        // CUDA-graph capture + desktop fluctuation still need room.
        let ck: usize = std::env::var("INKWELL_EXL3_CACHE_K")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(25);
        let kslot = std::env::var_os("INKWELL_NO_KSLOT").is_none();
        let _g_cache = crate::prof::guard("cache_setup");
        if ck >= 8 {
            // Per-layer arena MiB, measured by replaying the real routing
            // trace through the kslot policy (opt/ledger.md). Loop layers
            // L16/L17 hold ~half the miss bytes at uniform sizing.
            // INKWELL_NO_CACHEL=1 restores uniform ck*3*max_slot;
            // INKWELL_CACHE_MB_L="16:150,17:200" overrides per layer.
            static CACHEL_MB: [usize; 24] = [
                56, 56, 56, 48, 56, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                156, 156, 48, 44, 48, 52, 48, 52];
            // Greedy marginal-value tables regenerated with
            // opt/tools/kslot_alloc.py on the 68-token routing trace.
            static T0: [usize; 24] = [0; 24];
            static T192: [usize; 24] = [
                0, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                56, 12, 24, 24, 24, 20, 12, 8];
            static T256: [usize; 24] = [
                0, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                72, 48, 24, 24, 24, 28, 12, 12];
            static T300: [usize; 24] = [
                0, 8, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                72, 88, 24, 24, 24, 28, 12, 12];
            static T384: [usize; 24] = [
                0, 8, 0, 8, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                72, 88, 24, 24, 28, 40, 32, 40];
            static T600: [usize; 24] = [
                8, 48, 40, 8, 32, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                108, 120, 32, 36, 32, 40, 44, 40];
            static T940: [usize; 24] = [
                72, 68, 56, 40, 68, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                136, 120, 52, 36, 52, 52, 48, 64];
            static T1200: [usize; 24] = [
                100, 80, 68, 80, 84, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                156, 164, 52, 48, 64, 68, 64, 76];
            static T1400: [usize; 24] = [
                104, 100, 84, 100, 108, 108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                176, 164, 68, 80, 80, 72, 68, 80];
            // Joint-placement-aware tables for resident set
            // [3,6,7,8,10,11,12,14,16,17], replayed through the exact kslot
            // policy. They are one nested greedy allocation ladder truncated at
            // each size, so every layer's bytes are monotonic in the budget and
            // streamed loop layers L9/L13/L15 are never starved.
            static T192J: [usize; 24] = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 0, 0, 0, 48, 0, 48,
                0, 0, 12, 12, 12, 8, 12, 4];
            static T256J: [usize; 24] = [
                0, 0, 0, 0, 4, 0, 0, 0, 0, 48, 0, 0, 0, 52, 0, 72,
                0, 0, 12, 24, 16, 8, 12, 8];
            static T300J: [usize; 24] = [
                0, 4, 0, 0, 4, 0, 0, 0, 0, 48, 0, 0, 0, 56, 0, 72,
                0, 0, 24, 24, 24, 24, 12, 8];
            static T384J: [usize; 24] = [
                0, 8, 0, 0, 8, 12, 0, 0, 0, 48, 0, 0, 0, 56, 0, 72,
                0, 0, 24, 24, 24, 36, 32, 40];
            static T400J: [usize; 24] = [
                0, 8, 0, 0, 8, 12, 0, 0, 0, 48, 0, 0, 0, 56, 0, 80,
                0, 0, 24, 24, 28, 40, 32, 40];
            static T432J: [usize; 24] = [
                0, 8, 32, 0, 8, 12, 0, 0, 0, 48, 0, 0, 0, 56, 0, 80,
                0, 0, 24, 24, 28, 40, 32, 40];
            static T464J: [usize; 24] = [
                8, 8, 32, 0, 28, 12, 0, 0, 0, 48, 0, 0, 0, 56, 0, 80,
                0, 0, 24, 24, 32, 40, 32, 40];
            static T512J: [usize; 24] = [
                8, 32, 40, 0, 32, 12, 0, 0, 0, 48, 0, 0, 0, 56, 0, 80,
                0, 0, 24, 36, 32, 40, 32, 40];
            static T532J: [usize; 24] = [
                8, 48, 40, 0, 32, 12, 0, 0, 0, 48, 0, 0, 0, 56, 0, 80,
                0, 0, 24, 36, 32, 40, 36, 40];
            // Real-text 64K routing replay, exact per-layer knapsack over the
            // kslot policy (opt/kvq64/front4b/kslot_dp_alloc.py). Unlike the
            // 68-token ctx512 greedy table, every small arena holds at least
            // one full 8-expert visit and loop layer L9 is weighted correctly.
            static T568J: [usize; 24] = [
                36, 24, 20, 0, 24, 40, 0, 0, 0, 112, 0, 0, 0, 80, 0, 72,
                0, 0, 28, 24, 20, 36, 28, 24];
            static T600J: [usize; 24] = [
                8, 48, 40, 0, 32, 12, 0, 0, 0, 64, 0, 0, 0, 72, 0, 100,
                0, 0, 32, 36, 32, 40, 44, 40];
            static T768J: [usize; 24] = [
                64, 64, 40, 0, 36, 60, 0, 0, 0, 64, 0, 0, 0, 84, 0, 100,
                0, 0, 44, 36, 36, 40, 48, 52];
            static T940J: [usize; 24] = [
                72, 68, 64, 0, 68, 76, 0, 0, 0, 88, 0, 0, 0, 100, 0, 100,
                0, 0, 52, 36, 52, 52, 48, 64];
            static T1024J: [usize; 24] = [
                76, 76, 68, 0, 68, 88, 0, 0, 0, 104, 0, 0, 0, 100, 0, 100,
                0, 0, 52, 44, 52, 64, 64, 68];
            static T1164J: [usize; 24] = [
                100, 88, 68, 0, 84, 96, 0, 0, 0, 112, 0, 0, 0, 108, 0, 128,
                0, 0, 52, 48, 64, 72, 68, 76];
            static T1200J: [usize; 24] = [
                104, 96, 68, 0, 84, 96, 0, 0, 0, 112, 0, 0, 0, 112, 0, 128,
                0, 0, 64, 48, 72, 72, 68, 76];
            static T1232J: [usize; 24] = [
                104, 96, 84, 0, 84, 100, 0, 0, 0, 112, 0, 0, 0, 124, 0, 128,
                0, 0, 64, 48, 72, 72, 68, 76];
            static T1264J: [usize; 24] = [
                104, 100, 84, 0, 92, 108, 0, 0, 0, 112, 0, 0, 0, 124, 0, 128,
                0, 0, 68, 48, 80, 72, 68, 76];

            let vramcap = std::env::var_os("INKWELL_NO_VRAMCAP").is_none();
            let budgeted = std::env::var_os("INKWELL_NO_CACHE_BUDGET").is_none();
            let target_mib: usize = std::env::var("INKWELL_VRAM_CAP_MB").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(5080);
            let reserve_mib: usize = std::env::var("INKWELL_RUNTIME_RESERVE_MB").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(if target_mib > 5080 { 48 } else { 96 });
            let (pre_cache_used, vram_src) = process_vram_bytes(vram_baseline_free);
            let target_bytes = target_mib << 20;
            let reserve_bytes = reserve_mib << 20;
            let budget_mib = target_bytes
                .saturating_sub(pre_cache_used)
                .saturating_sub(reserve_bytes) >> 20;
            let (table_mib, table): (usize, &[usize; 24]) = if vramcap && budgeted {
                let mut chosen: (usize, &'static [usize; 24]) = (0, &T0);
                if joint_place_active {
                    let tables: [(usize, &'static [usize; 24]); 18] = [
                        (192, &T192J), (256, &T256J), (300, &T300J), (384, &T384J),
                        (400, &T400J), (432, &T432J), (464, &T464J), (512, &T512J),
                        (532, &T532J), (568, &T568J), (600, &T600J), (768, &T768J), (940, &T940J),
                        (1024, &T1024J), (1164, &T1164J), (1200, &T1200J), (1232, &T1232J),
                        (1264, &T1264J),
                    ];
                    for (total, candidate) in tables {
                        if total <= budget_mib { chosen = (total, candidate); }
                    }
                } else {
                    let tables: [(usize, &'static [usize; 24]); 8] = [
                        (192, &T192), (256, &T256), (300, &T300), (384, &T384),
                        (600, &T600), (940, &T940), (1200, &T1200), (1392, &T1400),
                    ];
                    for (total, candidate) in tables {
                        if total <= budget_mib { chosen = (total, candidate); }
                    }
                }
                chosen
            } else {
                (940, &CACHEL_MB)
            };
            vlog!("vramcap: target={target_mib}MiB reserve={reserve_mib}MiB used_pre_cache={}MiB budget={budget_mib}MiB table={table_mib}MiB source={vram_src} enabled={vramcap}",
                pre_cache_used >> 20);
            let cachel = std::env::var_os("INKWELL_NO_CACHEL").is_none();
            let over = std::env::var("INKWELL_CACHE_MB_L").unwrap_or_default();
            let per_l = |il: i32| -> Option<usize> {
                if !cachel { return None; }
                for t in over.split(',') {
                    if let Some((a, b)) = t.split_once(':') {
                        if a.trim().parse::<i32>() == Ok(il) {
                            if let Ok(mb) = b.trim().parse::<usize>() { return Some(mb); }
                        }
                    }
                }
                table.get(il as usize).copied()
            };
            let mut cache_allocated = 0usize;
            for (i, a) in moe_args_exl3.iter_mut().enumerate() {
                if a.on_dev || a.max_slot == 0 { continue; }
                let mut cbytes = per_l(a.layer).map_or(ck * 3 * a.max_slot, |mb| mb << 20);
                if cbytes == 0 { continue; }
                if !kslot && cbytes / (3 * a.max_slot) > 64 {
                    eprintln!("uniform EXL3 cache L{}: clamping k={} to 64 (eviction mask is u64)", a.layer, cbytes / (3 * a.max_slot));
                    cbytes = 64 * 3 * a.max_slot;
                }
                let (f, _) = inkwell_cuda::mem_info();
                if vramcap {
                    let (used_now, _) = process_vram_bytes(vram_baseline_free);
                    if used_now.saturating_add(cbytes).saturating_add(reserve_bytes) > target_bytes {
                        vlog!("vramcap: skip L{} cache={}MiB used={}MiB budget_left={}MiB",
                            a.layer, cbytes >> 20, used_now >> 20,
                            target_bytes.saturating_sub(used_now).saturating_sub(reserve_bytes) >> 20);
                        continue;
                    }
                } else {
                    let margin: usize = std::env::var("INKWELL_CACHE_MARGIN_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
                    if f <= cbytes + (margin << 20) { continue; }
                }
                let Some(buf) = inkwell_cuda::DevBuf::try_new_bytes(cbytes) else { continue };
                cache_allocated += cbytes;
                // kslot: same byte budget as ck uniform max-size slots, but as
                // one granule arena with a device-side free list — slots are
                // packed to each expert's real blob, so ~1.4x more experts stay
                // resident under the same global-LFU policy. soff[v*3+p] is the
                // (slot,proj) byte offset maintained by the resolve kernel;
                // gexp = per-(e,proj) granule counts; fl = free-extent list.
                // INKWELL_NO_KSLOT=1 = today's uniform ck x max_slot.
                let mut k_tot = cbytes / (3 * a.max_slot); // = ck at uniform sizing
                let mut gexp = None;
                let mut soff = None;
                let mut flb = None;
                if kslot {
                    let gxp = &moe_gw[i];
                    let arena = cbytes / 6912;
                    // max residents = arena / smallest 3-proj expert footprint
                    let gmin3 = (0..128)
                        .map(|e| (gxp[3 * e] + gxp[3 * e + 1] + gxp[3 * e + 2]) as usize)
                        .min().unwrap_or(1).max(1);
                    let kmax = (arena / gmin3).clamp(1, 64); // claimed mask is u64
                    let gd = inkwell_cuda::DevBuf::new_bytes(384 * 4);
                    gd.write_bytes_async(unsafe {
                        std::slice::from_raw_parts(gxp.as_ptr().cast::<u8>(), 384 * 4)
                    });
                    let sd = inkwell_cuda::DevBuf::new_bytes(3 * kmax * 8);
                    sd.write_bytes_async(&vec![0u8; 3 * kmax * 8]);
                    let mut fl_h = vec![0i32; 513];
                    fl_h[0] = 1;
                    fl_h[257] = arena as i32; // one free extent [0, arena)
                    let fd = inkwell_cuda::DevBuf::new_bytes(513 * 4);
                    fd.write_bytes_async(unsafe {
                        std::slice::from_raw_parts(fl_h.as_ptr().cast::<u8>(), 513 * 4)
                    });
                    k_tot = kmax;
                    gexp = Some(gd);
                    soff = Some(sd);
                    flb = Some(fd);
                    vlog!("kslot L{}: arena={} gran kmax={} (cap {}/{}MiB)",
                        a.layer, arena, kmax,
                        arena * 6912 / 1048576, cbytes / 1048576);
                }
                let tag = inkwell_cuda::DevBuf::new_bytes(k_tot * 4);
                tag.write_bytes_async(&vec![0xFFu8; k_tot * 4]);
                let cnt = inkwell_cuda::DevBuf::new_bytes(128 * 4);
                cnt.write_bytes_async(&[0u8; 128 * 4]);
                a.cache = Some(crate::moe::Exl3Cache {
                    buf, tag, cnt, k: k_tot as i32, slot: a.max_slot,
                    hm: crate::moe::new_hm(),
                    gexp, soff, fl: flb,
                    prefetch: std::env::var_os("INKWELL_MOE_PREFETCH").is_some()
                        .then(|| crate::moe::Exl3Prefetch::new(a.max_slot)).flatten(),
                });
                if std::env::var_os("INKWELL_KSIZES").is_some() {
                    let gxp = &moe_gw[i];
                    let esz: Vec<u32> = (0..128)
                        .map(|e| 6912 * gxp[3 * e..3 * e + 3].iter().max().copied().unwrap_or(0))
                        .collect();
                    let mut sizes: Vec<u32> = esz.clone();
                    sizes.sort_unstable();
                    sizes.dedup();
                    let dist: Vec<String> = sizes.iter()
                        .map(|&s| format!("{}:{}", s >> 10, esz.iter().filter(|&&e| e == s).count()))
                        .collect();
                    let sum: u64 = esz.iter().map(|&e| u64::from(e)).sum();
                    vlog!("KSIZES L{} max={}KiB sum_max={:.1}MiB sum_esz={:.1}MiB k_eff={:.1} [{}]",
                        a.layer, a.max_slot >> 10,
                        128.0 * a.max_slot as f64 / 1048576.0, sum as f64 / 1048576.0,
                        ck as f64 * a.max_slot as f64 * 128.0 / sum as f64,
                        dist.join(" "));
                }
                vlog!("exl3 cache moe={i}: k={ck} slot={}B ({:.0} MiB) CACHED free={}MiB",
                    a.max_slot, cbytes as f64 / 1e6, f >> 20);
            }
            if vramcap {
                let (post_cache_used, _) = process_vram_bytes(vram_baseline_free);
                vlog!("vramcap: cache_allocated={}MiB used_post_cache={}MiB target={target_mib}MiB reserve={reserve_mib}MiB",
                    cache_allocated >> 20, post_cache_used >> 20);
            }
        }
        // Pinned expert bytes must be complete before the first decode can
        // zero-copy them. Joining here keeps the copies overlapped with all of
        // the device upload/cache setup above rather than serializing per layer.
        if !pin_handles.is_empty() {
            let _g = crate::prof::guard("pinbg_wait");
            for h in pin_handles {
                pinned.push(h.join().expect("pinned expert loader panicked"));
            }
        }
        drop(_g_cache);
        crate::prof::mark("load_st_done");
        crate::prof::dump();
        if embd_dev.is_null() && std::env::var_os("INKWELL_NO_EMBED_WARM").is_none() {
            warm_file_pages(embd_src.0, embd_bytes.len());
        }
        Self {
            sched_cpu: Some((sched, cpu)),
            conv_args: std::cell::UnsafeCell::new(Vec::new()),
            conv_arg_idx: std::cell::Cell::new(0),
            cur_ctx: Ptr(std::ptr::null_mut()),
            mix_prefix: std::cell::UnsafeCell::new(None),
            dump_list: std::cell::UnsafeCell::new(Vec::new()),
            in_buf: None,
            rt, sched, dev_buf, cpu_buf,
            w: HashMap::new(), pinned, expert_dev,
            moe_args: Vec::new(), exl3_w, exl3_blk, moe_args_exl3, output_exl3,
            cfg: cfg.clone(), layers,
            tok_embd, embd_src, embd_dev, embd_row, tok_norm, output, output_norm, output_res_query,
            msg_query_packed, out_head,
            k_arena, v_arena, conv_state, conv_snap: Vec::new(), block_feats: None, feat_ud: std::cell::RefCell::new(Vec::new()), feat_ops: std::cell::RefCell::new(Vec::new()), max_ctx, swa, swa_oracle, swa_legacy_min, visits, n_full,
            step_gcache: None, sid: None,
        }
    }
}

// ============ forward ============

impl Runner {
    /// Reject batch positions that would index outside the allocated KV arenas.
    fn check_ctx_span(&self, pos: i64, n: i64, what: &str) {
        let end = pos.checked_add(n).expect("context position overflow");
        assert!(pos >= 0 && n >= 0 && end <= self.max_ctx,
            "{what} positions [{pos}, {end}) exceed max_ctx {}; encode/decode must stop at the configured context cap",
            self.max_ctx);
    }

    /// layer helpers
    fn is_swa_layer(&self, il: usize) -> bool { il % 6 != 5 }

    fn swa_capacity(&self) -> i64 { if self.swa_oracle { self.max_ctx } else { self.swa } }

    fn swa_positions(&self, n: i64) -> i64 { if self.swa_oracle { n } else { n.min(self.swa) } }

    fn swa_kv_idx(&self, pos: i64) -> i64 {
        if self.swa_oracle { pos }
        else if self.swa_legacy_min { pos.min(self.swa - 1) }
        else { pos.rem_euclid(self.swa) }
    }

    fn swa_filled_count(&self, pos: i64) -> i64 { self.swa_positions(pos + 1) }

    fn swa_filled(&self, pos: i64, slot: i64) -> bool {
        if self.swa_oracle { 0 <= slot && slot <= pos }
        else { 0 <= slot && slot < self.swa_filled_count(pos) }
    }

    fn swa_visible(&self, pos: i64, slot: i64) -> bool {
        if self.swa_oracle { self.swa_filled(pos, slot) && slot + self.swa - 1 >= pos }
        else { self.swa_filled(pos, slot) }
    }

    fn swa_rel_pos(&self, pos: i64, slot: i64) -> i32 {
        let d = if self.swa_oracle { pos - slot }
        else if self.swa_legacy_min { self.swa_filled_count(pos) - 1 - slot }
        else { (pos - slot).rem_euclid(self.swa) };
        d.min(self.cfg.rel_extent as i64) as i32
    }

    fn swa_wrap_hazard(&self, pos: i64, n: i64) -> bool {
        // Once the ring is full, every later write evicts a row near the edge of
        // every earlier token's window in the same batch.
        !self.swa_oracle && !self.swa_legacy_min && n > 1 && pos + n - 1 >= self.swa
    }

    /// GEMV dispatch: EXL3 descriptor or ggml mmq.
    fn proj(&self, c: Ptr, l: &L, key: &'static str, w: Option<Ptr>, x: Ptr) -> Ptr {
        match l.exl3.get(key) {
            Some(&i) => {
                let e = &*self.exl3_w[i];
                let mut args = [x.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [e.n as i64, 1, 1, 1], &mut args,
                    exl3_cb_addr(), (e as *const Exl3W).cast_mut().cast()) })
            }
            None => t::mmq(c, w.expect(key), x),
        }
    }

    fn norm(&self, x: Ptr, w: Ptr) -> Ptr {
        let c = self.cur_ctx;
        t::mul(c, t::rms_norm(c, x, 1e-6), w)
    }

    /// register a tensor for post-compute dump (INKWELL_DUMP)
    fn dump(&self, name: &str, t0: Ptr) {
        if std::env::var("INKWELL_DUMP").is_err() && std::env::var_os("INKWELL_DUMP_IDS").is_none() { return; }
        t::output(t0); // keep alive: gallocr won't reuse output memory
        unsafe { (*self.dump_list.get()).push((name.to_string(), t0)); }
    }

    fn sr_dump(&self, il: usize, pass: usize, pos: i64, name: &str, t0: Ptr) {
        if std::env::var_os("INKWELL_SR_DUMP_DIR").is_none() { return; }
        let want_l = std::env::var("INKWELL_SR_DUMP_LAYER").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
        let want_pos = std::env::var("INKWELL_SR_DUMP_POS").ok().and_then(|v| v.parse::<i64>().ok()).unwrap_or(64);
        if il != want_l || pass != 0 || pos + 1 != want_pos { return; }
        t::output(t0);
        unsafe { (*self.dump_list.get()).push((format!("sr_pos{}_l{il}p{pass}_{name}", pos + 1), t0)); }
    }
    fn dump_list_get(&self) -> &Vec<(String, Ptr)> { unsafe { &*self.dump_list.get() } }
    fn attn_dump_enabled(il: usize, pass: usize, nkv: i64) -> bool {
        static FILTER: std::sync::OnceLock<Vec<(Option<usize>, Option<usize>, Option<i64>)>> = std::sync::OnceLock::new();
        let filter = FILTER.get_or_init(|| {
            let parse_opt_u = |s: &str| if s == "*" { None } else { s.parse::<usize>().ok() };
            let parse_opt_i = |s: &str| if s == "*" { None } else { s.parse::<i64>().ok() };
            let Ok(spec) = std::env::var("INKWELL_ATTN_DUMP_FILTER") else {
                return vec![(Some(5), Some(0), None)];
            };
            spec.split(',').filter_map(|ent| {
                let mut p = ent.trim().split(':');
                let l = parse_opt_u(p.next()?);
                let pass = parse_opt_u(p.next()?);
                let nkv = parse_opt_i(p.next()?);
                Some((l, pass, nkv))
            }).collect()
        });
        filter.iter().any(|(l, p, n)| {
            l.is_none_or(|x| x == il) && p.is_none_or(|x| x == pass) && n.is_none_or(|x| x == nkv)
        })
    }
    fn dump_attn_bin(&self, name: String, t0: Ptr) {
        if std::env::var_os("INKWELL_ATTN_DUMP_DIR").is_none() { return; }
        t::output(t0);
        unsafe { (*self.dump_list.get()).push((name, t0)); }
    }
    fn dump_all(&self) {
        for (name, t0) in unsafe { &*self.dump_list.get() } {
            if name.starts_with("sr_") {
                let Some(dir) = std::env::var_os("INKWELL_SR_DUMP_DIR") else { continue };
                let n = t::ne(*t0,0)*t::ne(*t0,1)*t::ne(*t0,2)*t::ne(*t0,3);
                let mut v = vec![0f32; n as usize];
                unsafe { t::get(*t0, v.as_mut_ptr().cast(), 0, (n*4) as usize) };
                let _ = std::fs::create_dir_all(&dir);
                if let Ok(mut f) = std::fs::File::create(std::path::Path::new(&dir).join(format!("{name}.f32"))) {
                    use std::io::Write as _;
                    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
                    let _ = f.write_all(bytes);
                }
                continue;
            }
            if name.starts_with("attn_pos") {
                let Some(dir) = std::env::var_os("INKWELL_ATTN_DUMP_DIR") else { continue };
                let nbytes = t::nbytes(*t0);
                let mut v = vec![0u8; nbytes];
                unsafe { t::get(*t0, v.as_mut_ptr().cast(), 0, nbytes) };
                let mut it = name.split('_');
                let _ = it.next();
                let pos = it.next().and_then(|x| x.strip_prefix("pos")?.parse::<i64>().ok()).unwrap_or(0);
                let lp = it.next().unwrap_or("l5p0");
                let kind = it.next().unwrap_or("x");
                let nkv = pos + 1;
                let ext = if kind == "k" || kind == "v" { "f16" } else { "f32" };
                let outdir = std::path::Path::new(&dir).join(nkv.to_string()).join(lp);
                let _ = std::fs::create_dir_all(&outdir);
                if let Ok(mut f) = std::fs::File::create(outdir.join(format!("{kind}.{ext}"))) {
                    use std::io::Write as _;
                    let _ = f.write_all(&v);
                }
                continue;
            }
            let n = t::ne(*t0,0)*t::ne(*t0,1)*t::ne(*t0,2)*t::ne(*t0,3);
            let mut v = vec![0f32; n as usize];
            unsafe { t::get(*t0, v.as_mut_ptr().cast(), 0, (n*4) as usize) };
            let sum: f32 = v.iter().sum();
            eprintln!("DBG {name} ne={n} sum={sum:.6} first3={:?}", &v[..8]);
        }
        // INKWELL_DUMP_IDS=<path>: append every `b_ids<il>` router tensor of
        // this block as [il i32][64 i32] — real expert ids for kernel benches.
        if let Ok(path) = std::env::var("INKWELL_DUMP_IDS") {
            use std::io::Write as _;
            let rec: Vec<(i32, Vec<i32>)> = self.dump_list_get().iter()
                .filter_map(|(name, t0)| {
                    let il = name.strip_prefix("b_ids")?.parse::<i32>().ok()?;
                    let n = t::ne(*t0, 0) * t::ne(*t0, 1);
                    let mut v = vec![0f32; n as usize];
                    unsafe { t::get(*t0, v.as_mut_ptr().cast(), 0, (n * 4) as usize) };
                    Some((il, v.iter().map(|x| x.to_bits() as i32).collect::<Vec<i32>>()))
                })
                .collect();
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                for (il, ids) in &rec {
                    let _ = f.write_all(&il.to_le_bytes());
                    for x in ids { let _ = f.write_all(&x.to_le_bytes()); }
                }
            }
        }
        unsafe { (*self.dump_list.get()).clear(); }
    }

    fn clip(&self, x: Ptr, scale: Option<Ptr>, thr: f32) -> Ptr {
        let c = self.cur_ctx;
        if std::env::var_os("INKWELL_NO_CLIPFUSE").is_none() {
            let a: &'static ClipL2Args = Box::leak(Box::new(ClipL2Args {
                thr, has_w: scale.is_some() as i32 }));
            let mut args = vec![x.0];
            if let Some(s) = scale { args.push(s.0); }
            let (d, h, nt) = (t::ne(x, 0), t::ne(x, 1), t::ne(x, 2));
            return Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [d, h, nt, 1], &mut args,
                clip_l2_cb as usize, (a as *const ClipL2Args).cast_mut().cast()) });
        }
        let mut cur = x;
        if let Some(s) = scale { cur = t::mul(c, cur, s); }
        let norm = t::scale_bias(c, t::sqrt(c, t::sum_rows(c, t::sqr(c, cur))), 1.0, 1e-6);
        let factor = t::div(c, t::fill(c, norm, thr), norm);
        t::mul(c, cur, t::clamp(c, factor, 0.0, 1.0))
    }

    /// attention over depth stack: mix(blocks, cur, query)
    fn mix(&self, blocks: &[Ptr], cur: Ptr, query: Ptr) -> Ptr {
        let c = self.cur_ctx;
        let n_embd = self.cfg.n_embd as i64;
        let q_cols = t::ne(query, 1);
        let n_rows = blocks.len() as i64 + 1;
        let zero_cat = std::env::var_os("INKWELL_NO_MIX_ZEROCAT").is_none()
            && std::env::var_os("INKWELL_NO_MIXFUSE").is_none()
            && n_embd == 2304 && n_rows <= 9 && q_cols <= 8
            && unsafe { inkwell_ggml::ink_is_contiguous(cur.0) != 0 }
            && unsafe { inkwell_ggml::ink_is_contiguous(query.0) != 0 }
            && blocks.iter().all(|b| unsafe { inkwell_ggml::ink_is_contiguous(b.0) != 0 });
        if zero_cat {
            let mut margs: Vec<*mut std::ffi::c_void> = blocks.iter().rev().map(|b| b.0).collect();
            margs.push(cur.0);
            margs.push(query.0);
            let ud = Box::leak(Box::new(MixPtrsArgs {
                embd: n_embd as i32, rows: n_rows as i32, qcols: q_cols as i32,
                scale: 1.0 / (n_embd as f32).sqrt() }));
            return Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_embd, q_cols, 1, 1], &mut margs,
                mix_ptrs_cb as usize, (ud as *mut MixPtrsArgs).cast()) });
        }
        // same block stack is re-mixed ~4x per visit (res/ffn/msg/output):
        // cache the [b_n..b_1] prefix so only one concat adds `cur` per call
        let key = blocks.iter().fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b.0 as usize));
        let prefix = match unsafe { *self.mix_prefix.get() } {
            Some((k, p)) if k == key => p,
            _ => {
                let mut p = t::reshape3(c, if blocks.is_empty() { cur } else { blocks[0] }, n_embd, 1, 1);
                for b in &blocks[1..] {
                    p = t::concat(c, t::reshape3(c, *b, n_embd, 1, 1), p, 1);
                }
                unsafe { *self.mix_prefix.get() = Some((key, p)) };
                p
            }
        };
        let states = if blocks.is_empty() { prefix } else { t::concat(c, prefix, t::reshape3(c, cur, n_embd, 1, 1), 1) };
        let n_rows = t::ne(states, 1);
        let wide_ok = n_rows <= 8 && q_cols <= 64 && std::env::var_os("INKWELL_NO_MIX_V2").is_none();
        if ((n_rows <= 64 && q_cols <= 8) || wide_ok) && std::env::var_os("INKWELL_NO_MIXFUSE").is_none() {
            let mut margs = [states.0, query.0];
            let ud = Box::leak(Box::new(MixArgs {
                embd: n_embd as i32, rows: n_rows as i32, qcols: q_cols as i32,
                scale: 1.0 / (n_embd as f32).sqrt() }));
            Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_embd, q_cols, 1, 1], &mut margs,
                mix_cb as usize, (ud as *mut MixArgs).cast()) })
        } else {
            let states = t::rms_norm(c, states, 1e-6);
            let n_rows = t::ne(query, 1);
            let queries = t::repeat(c, query, t::new(c, F32, &[n_embd, n_rows, 1]));
            let scores = t::soft_max(c, t::mmq(c, states, queries), Ptr(std::ptr::null_mut()), 1.0 / (n_embd as f32).sqrt(), 0.0);
            t::mmq(c, t::cont(c, t::transpose(c, states)), scores)
        }
    }

    /// conv state: x rows [4, ch] = [past(3), cur]; shift-store last 3 into region.
    /// `region` = channel offset inside the per-visit conv state [3, n_sconv].
    /// pin a tensor (and its producer ops) to the CPU backend, like
    /// upstream's offload of host-resident expert matmuls (-ncmoe)
    fn set_cpu(&self, t0: Ptr) {
        if let Some((sched, cpu)) = self.sched_cpu {
            unsafe { inkwell_ggml::ink_sched_set_tensor_backend(sched.0, t0.0, cpu.0) };
        }
    }

    /// `dynamic` = per-head taps tensor (conv-gated layers).
    fn convolve(&self, x_in: Ptr, w: Ptr, dynamic: Option<Ptr>, slot: usize, region_off: i64, _stores: &mut Vec<Ptr>, snap: *mut f32) -> Ptr {
        let c = self.cur_ctx;
        let n_conv = 4i64;
        let ch = t::ne(x_in, 0); // x_in: [ch, 1]
        let n_heads = dynamic.map_or(0, |d| t::ne(d, 0) / n_conv) as i32;
        let scale = 1.0 / ((if dynamic.is_some() { 2.0 } else { 1.0 }) * n_conv as f32 + 1.0).sqrt();
        let idx = self.conv_arg_idx.get();
        let pool = unsafe { &mut *self.conv_args.get() };
        let fresh = ConvStepArgs {
            state: unsafe { inkwell_ggml::ink_tensor_data(self.conv_state[slot].0) }.cast(),
            region_off: region_off as i32,
            ch: ch as i32,
            n_heads,
            scale,
            snap,
        };
        if idx >= pool.len() {
            pool.push(Box::new(fresh));
        } else {
            // step_block calls sites in a different order than step(): rebind
            // every call so a pooled arg never carries another site's fields.
            *pool[idx] = fresh;
        }
        let arg = &*pool[idx];
        self.conv_arg_idx.set(idx + 1);
        if std::env::var_os("INKWELL_VDBG").is_some() {
            eprintln!("convdbg idx={idx} slot={slot} roff={region_off} ch={ch} nh={n_heads} state={:?}", arg.state);
        }
        let mut args: Vec<*mut std::ffi::c_void> = vec![x_in.0, w.0];
        if let Some(d) = dynamic { args.push(d.0); }
        Ptr(unsafe {
            inkwell_ggml::custom_4d(c.0, F32, [ch, 1, 1, 1], &mut args,
                conv_step_cb as usize, (arg as *const ConvStepArgs).cast_mut().cast())
        })
    }

    fn convolve_add(&self, x_in: Ptr, w: Ptr, dynamic: Option<Ptr>, slot: usize, region_off: i64, stores: &mut Vec<Ptr>, snap: *mut f32, residual: Ptr) -> Ptr {
        let c = self.cur_ctx;
        let contiguous = unsafe { inkwell_ggml::ink_is_contiguous(x_in.0) != 0 }
            && unsafe { inkwell_ggml::ink_is_contiguous(residual.0) != 0 };
        if contiguous && std::env::var_os("INKWELL_NO_CONV_ADD").is_none() {
            let n_conv = 4i64;
            let ch = t::ne(x_in, 0);
            let n_heads = dynamic.map_or(0, |d| t::ne(d, 0) / n_conv) as i32;
            let scale = 1.0 / ((if dynamic.is_some() { 2.0 } else { 1.0 }) * n_conv as f32 + 1.0).sqrt();
            let idx = self.conv_arg_idx.get();
            let pool = unsafe { &mut *self.conv_args.get() };
            let fresh = ConvStepArgs {
                state: unsafe { inkwell_ggml::ink_tensor_data(self.conv_state[slot].0) }.cast(),
                region_off: region_off as i32,
                ch: ch as i32,
                n_heads,
                scale,
                snap,
            };
            if idx >= pool.len() { pool.push(Box::new(fresh)); } else { *pool[idx] = fresh; }
            let arg = &*pool[idx];
            self.conv_arg_idx.set(idx + 1);
            let mut args: Vec<*mut std::ffi::c_void> = vec![x_in.0, w.0];
            if let Some(d) = dynamic { args.push(d.0); }
            args.push(residual.0);
            return Ptr(unsafe {
                inkwell_ggml::custom_4d(c.0, F32, [ch, 1, 1, 1], &mut args,
                    conv_step_add_cb as usize, (arg as *const ConvStepArgs).cast_mut().cast())
            });
        }
        t::add(c, residual, self.convolve(x_in, w, dynamic, slot, region_off, stores, snap))
    }

    /// device ptr into `conv_snap[slot]` for verify position `j`.
    fn snap_ptr(&self, slot: usize, j: usize) -> *mut f32 {
        if self.conv_snap.len() <= slot { return std::ptr::null_mut(); }
        let n = (t::ne(self.conv_state[slot], 0) * t::ne(self.conv_state[slot], 1)) as usize;
        unsafe { self.conv_snap[slot].dev_ptr().cast::<f32>().add(j * n) }
    }

    /// Spec partial-accept: rewind conv state to after verify input index
    /// `emitted - 1` (`snap` copies written during `step_block`). No-op on
    /// full acceptance.
    pub fn rollback_conv(&self, emitted: usize) {
        if emitted == 0 || emitted >= NT { return; }
        let m = emitted - 1;
        for (s, snap) in self.conv_state.iter().zip(self.conv_snap.iter()) {
            let n = (t::ne(*s, 0) * t::ne(*s, 1)) as usize * 4;
            unsafe {
                inkwell_cuda::memcpy_d2d_async(
                    inkwell_ggml::ink_tensor_data(s.0),
                    snap.dev_ptr().cast::<u8>().add(m * n).cast(), n);
            }
        }
    }

    /// DSpark ctx tap: copy `cur` [2304, NT] into `block_feats` tap slab
    /// (layout [4][NT][2304]) when `vi` is a tap visit.
    fn feat_tap(&self, c: &Ptr, vi: usize, cur: Ptr) {
        let slab = match vi { 5 | 23 | 41 => (vi - 5) / 18, 48 => 3, _ => return };
        let fb = self.block_feats.as_ref().expect("block_feats");
        let d = unsafe { fb.dev_ptr().cast::<u8>().add(slab * NT * 2304 * 4) };
        let mut args = [t::cont(*c, cur).0];
        let ud = Box::into_raw(Box::new(d.cast::<std::ffi::c_void>()));
        self.feat_ud.borrow_mut().push(unsafe { Box::from_raw(ud) });
        let op = Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [1, 1, 1, 1], &mut args,
            feat_copy_cb as usize, ud.cast()) });
        self.feat_ops.borrow_mut().push(op);
    }

    /// The target unembed EXL3 desc `(trel, suh, svh, k, n, ka, halfk,
    /// scratch)` — borrowed device pointers for the DSpark drafter.
    #[must_use]
    pub fn unembed_desc(&self) -> Option<(*const std::ffi::c_void, *const std::ffi::c_void,
            *const std::ffi::c_void, usize, usize, usize, bool, *mut f32)> {
        self.output_exl3.map(|i| {
            let w = &*self.exl3_w[i];
            (w.trel.dev_ptr().cast_const(), w.suh.dev_ptr().cast_const(), w.svh.dev_ptr().cast_const(),
                w.k, w.n, w.ka, w.halfk, w.scratch.dev_ptr().cast())
        })
    }

    /// device ptr to the `[4][NT][n_embd]` tap slabs written by the last
    /// `step_block` (visits {5,23,41} + final norm).
    #[must_use]
    pub fn block_feats_ptr(&self) -> *const f32 {
        match self.block_feats.as_ref() {
            Some(b) => b.dev_ptr().cast(),
            None => std::ptr::null(),
        }
    }

    /// Batch the three attention controllers into one custom op when their
    /// BF16 concatenated gain/forget layout exactly matches gf_apply.
    fn control_triad(&self, inputs: [Ptr; 3], msgs: [Ptr; 3], l: &L) -> Option<[Ptr; 3]> {
        if std::env::var_os("INKWELL_NO_CTRLBATCH").is_some() { return None; }
        let c = self.cur_ctx;
        let n = t::ne(inputs[0], 0);
        if n != 2304 { return None; }
        let mut down = [std::ptr::null(); 3];
        let mut gf = [std::ptr::null(); 3];
        let mut fo = 0i32;
        for j in 0..3 {
            let (dn, cat) = (l.ctrl_down[j]?, l.ctrl_gf_cat[j]?);
            let [go, foj] = l.ctrl_gf_off[j];
            unsafe {
                if inkwell_ggml::ink_tensor_type(dn.0) != BF16
                    || inkwell_ggml::ink_tensor_type(cat.0) != BF16
                    || inkwell_ggml::ink_tensor_ne(dn.0, 0) != 2 * n
                    || inkwell_ggml::ink_tensor_ne(dn.0, 1) != 64
                    || inkwell_ggml::ink_tensor_ne(cat.0, 0) != 64
                    || go != 0 || foj != n || inkwell_ggml::ink_tensor_ne(cat.0, 1) - foj != n {
                    return None;
                }
                down[j] = inkwell_ggml::ink_tensor_data(dn.0);
                gf[j] = inkwell_ggml::ink_tensor_data(cat.0);
            }
            fo = foj as i32;
        }
        let ud = Box::leak(Box::new(CtrlTriadArgs { down, gf, fo }));
        let mut args = [inputs[0].0, msgs[0].0, inputs[1].0, msgs[1].0, inputs[2].0, msgs[2].0];
        let out = Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n, 3, 1, 1], &mut args,
            ctrl_triad_cb as *const () as usize, (ud as *mut CtrlTriadArgs).cast()) });
        let stride = t::nb(out, 1);
        Some([
            t::view2(c, out, n, 1, stride, 0),
            t::view2(c, out, n, 1, stride, stride),
            t::view2(c, out, n, 1, stride, 2 * stride),
        ])
    }

    /// loop control gate: (1-forget)*cur + gain*msg
    fn control(&self, cur: Ptr, msg: Ptr, l: &L, ctrl: usize) -> Ptr {
        let c = self.cur_ctx;
        // glue fusion (INKWELL_NO_CTRLFUSE=1 restores the unfused chain):
        // rms2cat packs the two rms_norms + concat into one kernel, and
        // ctrl_apply does tanh x2 + the (1-forget)*cur + gain*msg tail.
        let fuse = std::env::var_os("INKWELL_NO_CTRLFUSE").is_none();
        let n = t::ne(cur, 0);
        let ntok = t::ne(cur, 1);
        if ntok > 1 {
            // The fused control kernels are single-token. Preserve their exact
            // serial numerics in NT batches by applying them per token.
            let mut xs = Vec::with_capacity(ntok as usize);
            for j in 0..ntok {
                let cj = t::cont(c, self.tokslice(c, cur, n, j));
                let mj = t::cont(c, self.tokslice(c, msg, n, j));
                xs.push(self.control(cj, mj, l, ctrl));
            }
            let mut acc = t::reshape3(c, xs[0], n, 1, 1);
            for x in &xs[1..] {
                acc = t::concat(c, acc, t::reshape3(c, *x, n, 1, 1), 1);
            }
            return t::reshape2(c, acc, n, ntok);
        }
        let dn = l.ctrl_down[ctrl].expect("ctrl_down");
        // rms2down fuses rms2cat + the down GEMV (INKWELL_NO_RMSDOWN=1
        // restores the pair). Bit-identical to it.
        let gate = if fuse && n == 2304 && std::env::var_os("INKWELL_NO_RMSDOWN").is_none() {
            let mut args = [cur.0, msg.0, dn.0];
            Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [t::ne(dn, 1), 1, 1, 1], &mut args,
                rms2down_cb as *const () as usize, std::ptr::null_mut()) })
        } else {
            let gate = if fuse && n % 4 == 0 && n / 4 <= 1024 {
                let mut args = [cur.0, msg.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [2 * n, 1, 1, 1], &mut args,
                    rms2cat_cb as *const () as usize, std::ptr::null_mut()) })
            } else {
                t::concat(c, t::rms_norm(c, cur, 1e-6), t::rms_norm(c, msg, 1e-6), 0)
            };
            t::mmq(c, dn, gate)
        };
        let (gain, forget);
        if let Some(cat) = l.ctrl_gf_cat[ctrl] {
            let [go, fo] = l.ctrl_gf_off[ctrl];
            let ok = fuse && n % 4 == 0 && n / 4 <= 1024 && go == 0 && fo == n && t::ne(cat, 1) - fo == n;
            // gf_apply fuses the g3 GEMV + apply tail (INKWELL_NO_GFAPPLY=1
            // restores the mmq + ctrl_apply chain). Bit-identical to it.
            if ok && std::env::var_os("INKWELL_NO_GFAPPLY").is_none() {
                let mut args = [cat.0, gate.0, cur.0, msg.0];
                return Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n, 1, 1, 1], &mut args,
                    gf_apply_cb as *const () as usize, fo as usize as *mut std::ffi::c_void) });
            }
            let g3 = t::mmq(c, cat, gate);
            // fused kernel reads gain = g3[0..n), forget = g3[fo..fo+n):
            // only valid for that exact layout. ng travels in the userdata
            // pointer value (no per-build allocation).
            if ok {
                let mut args = [g3.0, cur.0, msg.0];
                return Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n, 1, 1, 1], &mut args,
                    ctrl_apply_cb as *const () as usize, fo as usize as *mut std::ffi::c_void) });
            }
            gain = t::tanh(c, t::view1(c, g3, fo - go, go * 4));
            forget = t::tanh(c, t::view1(c, g3, t::ne(cat, 1) - fo, fo * 4));
        } else {
            gain = t::tanh(c, t::mmq(c, l.ctrl_gain[ctrl].expect("ctrl_gain"), gate));
            forget = t::tanh(c, t::mmq(c, l.ctrl_forget[ctrl].expect("ctrl_forget"), gate));
        }
        t::add(c, t::sub(c, cur, t::mul(c, forget, cur)), t::mul(c, gain, msg))
    }

    /// one layer; returns (new cur, attn_out push or add). messages present on loop passes>0.
    #[expect(clippy::too_many_arguments)]
    fn layer(&self, il: usize, ip: usize, slot: usize, pass: usize, mut cur: Ptr, blocks: &mut Vec<Ptr>,
             messages: Option<Ptr>, l: &L, step: &Step, stores: &mut Vec<Ptr>) -> Ptr {
        let c = self.cur_ctx;
        let is_swa = self.is_swa_layer(il);
        let (n_embd, hd, nh, hkv, n_rel) =
            (self.cfg.n_embd as i64, 128i64, 24i64, 8i64, self.cfg.n_rel as i64);
        let n_rep = nh / hkv;
        let inp_sa = cur;

        // attention inputs
        let mut inputs = [cur, cur, cur];
        if let Some(rq) = l.attn_res_query {
            let mixed = self.mix(blocks, cur, rq);
            for (j, inp) in inputs.iter_mut().enumerate() {
                *inp = t::view2(c, mixed, n_embd, 1, t::nb(mixed, 2), (if is_swa { 0 } else { j as i64 }) * t::nb(mixed, 1));
            }
        }
        if let Some(msgs) = messages {
            let ms = [
                t::view2(c, msgs, n_embd, 1, t::nb(msgs, 2), 0),
                t::view2(c, msgs, n_embd, 1, t::nb(msgs, 2), t::nb(msgs, 1)),
                t::view2(c, msgs, n_embd, 1, t::nb(msgs, 2), 2 * t::nb(msgs, 1)),
            ];
            if let Some(batched) = self.control_triad(inputs, ms, l) {
                inputs = batched;
            } else {
                for (j, inp) in inputs.iter_mut().enumerate() {
                    *inp = self.control(*inp, ms[j], l, j);
                }
            }
        }
        if il == 0 && std::env::var("INKWELL_DUMP").is_ok() {
            let nw = l.attn_norm.expect("attn_norm");
            let mut nv = vec![0f32; 8];
            unsafe { t::get(nw, nv.as_mut_ptr().cast(), 0, 32) };
            eprintln!("DBG attn_norm_w first8={nv:?} n0={}", t::ne(nw, 0));
        }
        if is_swa && messages.is_some() {
            for inp in inputs.iter_mut() { *inp = self.norm(*inp, l.attn_norm.expect("attn_norm")); }
        } else if is_swa {
            inputs[0] = self.norm(inputs[0], l.attn_norm.expect("attn_norm"));
            inputs[1] = inputs[0];
            inputs[2] = inputs[0];
        } else {
            inputs[0] = self.norm(inputs[0], l.attn_q_inp_norm.expect("q_inp"));
            inputs[1] = self.norm(inputs[1], l.attn_k_inp_norm.expect("k_inp"));
            inputs[2] = self.norm(inputs[2], l.attn_v_inp_norm.expect("v_inp"));
        }

        // projections: on all-EXL3 layers fuse the 5 independent projections
        // (wq/gate/rel on inputs[0..2]) into one multi-stream custom op —
        // each ~100us gemv runs as a parallel branch of the captured graph.
        let mz_keys: [&'static str; 5] = ["wq", "gate", "rel", "wk", "wv"];
        let mz_ok = std::env::var_os("INKWELL_NO_MZ").is_none()
            && mz_keys.iter().all(|k| l.exl3.contains_key(*k));
        let mut mz_nmax = 0i64;
        let (q, gate_lin, r_pre, wk_p, wv_p) = if mz_ok {
            let mut ws = [std::ptr::null::<Exl3W>(); 6];
            let mut n_max = 0i64;
            for (z, k) in mz_keys.iter().enumerate() {
                let e: &Exl3W = &*self.exl3_w[l.exl3[k]];
                ws[z] = e;
                n_max = n_max.max(e.n as i64);
            }
            mz_nmax = n_max;
            let a: &'static mut Exl3MzArgs = Box::leak(Box::new(Exl3MzArgs {
                ws, xidx: [0, 0, 0, 1, 2, 0], nz: 5 }));
            let mut args = [inputs[0].0, inputs[1].0, inputs[2].0];
            let out = Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32,
                [n_max, 5, 1, 1], &mut args,
                exl3_mz_cb as usize, (a as *mut Exl3MzArgs).cast()) });
            let n = |k: &'static str| self.exl3_w[l.exl3[k]].n as i64;
            (
                t::view1(c, out, n("wq"), 0),
                t::view1(c, out, n("gate"), n_max * 4),
                t::view1(c, out, n("rel"), 2 * n_max * 4),
                t::view1(c, out, n("wk"), 3 * n_max * 4),
                t::view1(c, out, n("wv"), 4 * n_max * 4),
            )
        } else {
            let (q, gate_lin, r_pre) = if let Some(cat) = l.proj0_cat {
                let out = t::mmq(c, cat, inputs[0]);
                // ggml_view_1d offset is in BYTES in this fork — out is F32 (row idx * 4)
                let slice = |off: i64, n: i64, key: &'static str, w: Option<Ptr>| {
                    if off >= 0 { t::view1(c, out, n, off * 4) } else { self.proj(c, l, key, w, inputs[0]) }
                };
                (
                    slice(l.proj0_off[0], 3072, "wq", l.wq),
                    slice(l.proj0_off[1], 3072, "gate", l.wqkv_gate),
                    slice(l.proj0_off[2], 384, "rel", l.attn_rel),
                )
            } else {
                (
                    self.proj(c, l, "wq", l.wq, inputs[0]),
                    self.proj(c, l, "gate", l.wqkv_gate, inputs[0]),
                    self.proj(c, l, "rel", l.attn_rel, inputs[0]),
                )
            };
            (q, gate_lin, r_pre,
                self.proj(c, l, "wk", l.wk, inputs[1]),
                self.proj(c, l, "wv", l.wv, inputs[2]))
        };
        let mut kv_dyn_batch = false;
        let (k_dyn, v_dyn) = if is_swa {
            (None, None)
        } else {
            let (wk, wv) = (l.attn_k_conv_proj.expect("kcp"), l.attn_v_conv_proj.expect("vcp"));
            let batch = std::env::var_os("INKWELL_NO_KVBATCH").is_none() && unsafe {
                inkwell_ggml::ink_tensor_type(wk.0) == BF16 && inkwell_ggml::ink_tensor_type(wv.0) == BF16
                    && inkwell_ggml::ink_tensor_ne(wk.0, 0) == 2304 && inkwell_ggml::ink_tensor_ne(wk.0, 1) == 32
                    && inkwell_ggml::ink_tensor_ne(wv.0, 0) == 2304 && inkwell_ggml::ink_tensor_ne(wv.0, 1) == 32
            };
            if batch {
                kv_dyn_batch = true;
                let mut args = [wk.0, inputs[1].0, wv.0, inputs[2].0];
                let out = Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [32, 2, 1, 1], &mut args,
                    kv_sconv_cb as *const () as usize, std::ptr::null_mut()) });
                let stride = t::nb(out, 1);
                (Some(t::view2(c, out, 32, 1, stride, 0)), Some(t::view2(c, out, 32, 1, stride, stride)))
            } else {
                (Some(t::mmq(c, wk, inputs[1])), Some(t::mmq(c, wv, inputs[2])))
            }
        };
        let gqa = (hd * hkv) as i64;
        let qkvr_active = std::env::var_os("INKWELL_NO_QKVR_FUSE").is_none()
            && mz_ok && hd == 128 && nh == 24 && hkv == 8 && n_rel == 16 && mz_nmax > 0
            && (is_swa || kv_dyn_batch)
            && l.attn_q_clip.is_none_or(|x| unsafe { inkwell_ggml::ink_is_contiguous(x.0) != 0 })
            && l.attn_rel_clip.is_none_or(|x| unsafe { inkwell_ggml::ink_is_contiguous(x.0) != 0 });
        let (q, k, v, r, qkvr_fused) = if qkvr_active {
            let total = hd * nh + 2 * hd * hkv + n_rel * nh;
            let mut args: Vec<*mut std::ffi::c_void> = vec![q.0];
            let mut next = 1i32;
            let mut take = |x: Ptr| { let v = next; args.push(x.0); next += 1; v };
            let dyn_src = if is_swa { -1 } else { take(k_dyn.expect("k dyn")) };
            let kw_src = take(l.attn_k_conv.expect("kconv"));
            let vw_src = take(l.attn_v_conv.expect("vconv"));
            let qw_src = l.attn_q_clip.map_or(-1, &mut take);
            let rw_src = l.attn_rel_clip.map_or(-1, &mut take);
            let tau_src = if is_swa { -1 } else { take(step_tau(step)) };
            let conv_scale = 1.0 / ((if is_swa { 1.0 } else { 2.0 }) * 4.0f32 + 1.0).sqrt();
            let ud = Box::leak(Box::new(QkvrPrepArgs {
                state: unsafe { inkwell_ggml::ink_tensor_data(self.conv_state[slot].0) }.cast(),
                k_region_off: 0,
                v_region_off: gqa as i32,
                nmax: mz_nmax as i32,
                hd: hd as i32,
                nh: nh as i32,
                hkv: hkv as i32,
                nrel: n_rel as i32,
                conv_scale,
                qk_thr: self.cfg.qk_clip,
                rel_thr: self.cfg.rel_clip,
                dyn_src,
                kw_src,
                vw_src,
                qw_src,
                rw_src,
                tau_src,
            }));
            let packed = Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [total, 1, 1, 1], &mut args,
                qkvr_prep_cb as usize, (ud as *mut QkvrPrepArgs).cast()) });
            let koff = hd * nh * 4;
            let voff = koff + hd * hkv * 4;
            let roff = voff + hd * hkv * 4;
            (
                t::reshape3(c, t::view1(c, packed, hd * nh, 0), hd, nh, 1),
                t::reshape3(c, t::view1(c, packed, hd * hkv, koff), hd, hkv, 1),
                t::reshape3(c, t::view1(c, packed, hd * hkv, voff), hd, hkv, 1),
                t::reshape3(c, t::view1(c, packed, n_rel * nh, roff), n_rel, nh, 1),
                true,
            )
        } else {
            let k = self.convolve(wk_p, l.attn_k_conv.expect("kconv"), k_dyn, slot, 0, stores, std::ptr::null_mut());
            let v = self.convolve(wv_p, l.attn_v_conv.expect("vconv"), v_dyn, slot, gqa, stores, std::ptr::null_mut());
            let r = r_pre;
            (
                self.clip(t::reshape3(c, q, hd, nh, 1), l.attn_q_clip, self.cfg.qk_clip),
                self.clip(t::reshape3(c, k, hd, hkv, 1), None, self.cfg.qk_clip),
                t::rms_norm(c, t::reshape3(c, v, hd, hkv, 1), 1e-6),
                self.clip(t::reshape3(c, r, n_rel, nh, 1), l.attn_rel_clip, self.cfg.rel_clip),
                false,
            )
        };
        if il == 0 {
            self.dump("rel_raw", r); self.dump("inputs0", inputs[0]);
            // weight raw bytes check (GGUF path only)
            if let Some(wp) = l.attn_rel {
                let nbytes = t::nbytes(wp);
                let mut vb = vec![0u8; nbytes.min(64) as usize];
                unsafe { t::get(wp, vb.as_mut_ptr().cast(), 0, vb.len()) };
                eprintln!("DBG attn_rel nbytes={nbytes} head={:?}", &vb[..16]);
            }
        }

        if il == 0 {
            self.dump("Qcur-0", q); self.dump("Kcur-0", k); self.dump("Vcur-0", v); self.dump("Rcur-0", r);
        }

        // KV write via set_rows: in-place write into arena + dep edge so the
        // attention reads below see this token's k/v (upstream pattern).
        // Dynamic mode uses fixed-capacity views and reads exact n_kv from a
        // device scalar; the fallback keeps master's 64-token padded shapes.
        let n_kv = if step.dyn_nkv {
            if is_swa { step.cap_swa } else { step.cap_full }
        } else if is_swa {
            step.n_kv_swa_pad
        } else {
            step.n_kv_full_pad
        };
        let kv_idx = if is_swa { step.kv_idx_swa } else { step.kv_idx_full };
        // set_rows wants b [hd, n_idx, hkv]: permute [hd,hkv,1] -> [hd,1,hkv]
        let k_rows = t::permute(c, k, 0, 2, 1, 3);
        let v_rows = t::permute(c, v, 0, 2, 1, 3);
        let k_upd = t::set_rows(c, self.k_arena[slot], k_rows, kv_idx);
        let v_upd = t::set_rows(c, self.v_arena[slot], v_rows, kv_idx);

        // tau scale on full layers
        let mut q = q;
        let mut r = r;
        if !is_swa && !qkvr_fused {
            q = t::mul(c, q, step_tau(step));
            r = t::mul(c, r, step_tau(step));
        }

        // rel bias (upstream): table = mm(pad(rel_bias), r) - sinks; gather by distance
        let sink_w = l.attn_sink_proj.expect("sink");
        let rel_w = l.attn_rel_bias_pad.or(l.attn_rel_bias).expect("relbias");
        let rel_pos = if is_swa { step.rel_pos_swa } else { step.rel_pos_full };
        let fuse_sink_relbias = std::env::var_os("INKWELL_NO_SINKRELBIAS").is_none() && unsafe {
            inkwell_ggml::ink_tensor_type(sink_w.0) == BF16
                && inkwell_ggml::ink_tensor_type(rel_w.0) == BF16
                && inkwell_ggml::ink_tensor_ne(rel_w.0, 0) == n_rel
        };
        let (kq_b, table_dbg) = if fuse_sink_relbias {
            let (mode, pos_src, nkv_src, dynamic) = if step.dyn_nkv {
                let nkv_dev = if is_swa { step.n_kv_swa_dev } else { step.n_kv_full_dev }.expect("dyn n_kv scalar");
                if step.direct_relbias {
                    let mode = if !is_swa { 0 } else if self.swa_legacy_min { 2 } else { 1 };
                    (mode, step.pos, nkv_dev, 1)
                } else {
                    (-1, rel_pos, nkv_dev, 1)
                }
            } else {
                (-1, rel_pos, rel_pos, 0)
            };
            let ud = Box::leak(Box::new(SinkRelbiasArgs {
                mode, swa: self.swa as i32, rel_extent: self.cfg.rel_extent as i32, dynamic }));
            let out = if dynamic != 0 {
                let mut args = [sink_w.0, inputs[0].0, rel_w.0, r.0, pos_src.0, nkv_src.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_kv, 1, nh, 1], &mut args,
                    sink_relbias_cb as *const () as usize, (ud as *mut SinkRelbiasArgs).cast()) })
            } else {
                let mut args = [sink_w.0, inputs[0].0, rel_w.0, r.0, pos_src.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_kv, 1, nh, 1], &mut args,
                    sink_relbias_cb as *const () as usize, (ud as *mut SinkRelbiasArgs).cast()) })
            };
            (out, None)
        } else {
            let sinks = t::mmq(c, sink_w, inputs[0]); // [nh, 1]
            let table = t::mmq(c, rel_w, r); // [ext+1, nh, 1]
            // fused gather+sub (INKWELL_NO_RELGATHER=1 restores the chain):
            // kq_b[j,0,h] = table[rel_pos[j], h] - sinks[h] in one kernel.
            let kq_b = if step.dyn_nkv {
                let nkv_dev = if is_swa { step.n_kv_swa_dev } else { step.n_kv_full_dev }.expect("dyn n_kv scalar");
                if step.direct_relbias {
                    let mode = if !is_swa { 0 } else if self.swa_legacy_min { 2 } else { 1 };
                    let ud = Box::leak(Box::new(RelbiasDynPosArgs {
                        mode, swa: self.swa as i32, rel_extent: self.cfg.rel_extent as i32 }));
                    let mut args = [table.0, sinks.0, step.pos.0, nkv_dev.0];
                    Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_kv, 1, nh, 1], &mut args,
                        relbias_dynpos_cb as *const () as usize, (ud as *mut RelbiasDynPosArgs).cast()) })
                } else {
                    let mut args = [table.0, sinks.0, rel_pos.0, nkv_dev.0];
                    Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_kv, 1, nh, 1], &mut args,
                        relbias_dyn_cb as *const () as usize, std::ptr::null_mut()) })
                }
            } else if std::env::var_os("INKWELL_NO_RELGATHER").is_none() {
                let mut args = [table.0, sinks.0, rel_pos.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [n_kv, 1, nh, 1], &mut args,
                    relbias_cb as *const () as usize, std::ptr::null_mut()) })
            } else {
                let table = t::sub(c, table, t::reshape3(c, sinks, 1, nh, 1));
                let table = t::cont(c, t::permute(c, table, 1, 0, 2, 3)); // [nh, ext+1, 1]
                let kq_b = t::get_rows(c, table, rel_pos); // [nh, n_kv, 1]
                // permute(2,0,1,3): src dim i goes to dst dim arg_i -> [n_kv, 1, nh, 1]
                t::cont(c, t::permute(c, t::reshape4(c, kq_b, nh, n_kv, 1, 1), 2, 0, 1, 3))
            };
            (kq_b, Some(table))
        };
        if il == 0 {
            if let Some(table) = table_dbg { self.dump("table-0", table); }
            self.dump("kqb-0", kq_b);
        }
        self.sr_dump(il, pass, step.pos_val, "inp0", inputs[0]);
        self.sr_dump(il, pass, step.pos_val, "r", r);
        self.sr_dump(il, pass, step.pos_val, "kqb", kq_b);

        // Fused decode attention (replaces kq GEMV + softmax + V GEMV chain)
        let k_view = t::view3(c, k_upd, hd, n_kv, hkv, t::nb(k_upd, 1), t::nb(k_upd, 2), 0);
        let q3 = t::permute(c, q, 0, 2, 1, 3); // [hd, 1, nh]
        let kq_mask = if is_swa { step.mask_swa } else { step.mask_full };
        let v_view = t::view3(c, v_upd, hd, n_kv, hkv, t::nb(v_upd, 1), t::nb(v_upd, 2), 0);
        let pow2_ctx = (step.pos_val + 1) & step.pos_val == 0;
        if !is_swa && pow2_ctx && (1024..=32768).contains(&(step.pos_val + 1))
            && Self::attn_dump_enabled(il, pass, step.pos_val + 1) {
            self.dump_attn_bin(format!("attn_pos{}_l{il}p{pass}_k", step.pos_val), t::cont(c, k_view));
            self.dump_attn_bin(format!("attn_pos{}_l{il}p{pass}_v", step.pos_val), t::cont(c, v_view));
            self.dump_attn_bin(format!("attn_pos{}_l{il}p{pass}_q", step.pos_val), q);
            self.dump_attn_bin(format!("attn_pos{}_l{il}p{pass}_bias", step.pos_val), kq_b);
            self.dump_attn_bin(format!("attn_pos{}_l{il}p{pass}_mask", step.pos_val), kq_mask);
        }
        let out = if step.dyn_nkv {
            let nkv_dev = if is_swa { step.n_kv_swa_dev } else { step.n_kv_full_dev }.expect("dyn n_kv scalar");
            let (regime, allow_gqa) = if is_swa {
                (step.attn_regime_swa, step.attn_allow_gqa_swa)
            } else {
                (step.attn_regime_full, step.attn_allow_gqa_full)
            };
            let ud = Box::leak(Box::new(AttnDynArgs {
                use_split: i32::from(regime != 0), allow_gqa: i32::from(allow_gqa),
                chunk_len: attn_chunk_len(regime), split_min: attn_split_cfg().1,
            }));
            t::attn_fused_dyn(c, k_view, v_view, q3, kq_b, kq_mask, nkv_dev, hd, nh,
                (ud as *mut AttnDynArgs).cast())
        } else {
            t::attn_fused(c, k_view, v_view, q3, kq_b, kq_mask, hd, nh)
        }; // [hd*nh, 1]

        let mut cur = out;
        cur = t::mul(c, cur, t::sigmoid(c, gate_lin));
        if il == 0 { self.dump("kqv_out-0", out); }
        let wo_in = self.proj(c, l, "wo", l.wo, cur);
        if ip % (self.cfg.attn_res_block_size as usize) == 0 {
            cur = self.convolve(wo_in, l.attn_post_conv.expect("postconv"), None, slot, 2 * gqa, stores, std::ptr::null_mut());
            blocks.push(inp_sa);
        } else {
            cur = self.convolve_add(wo_in, l.attn_post_conv.expect("postconv"), None, slot, 2 * gqa, stores, std::ptr::null_mut(), inp_sa);
        }
        let ffn_inp = cur;

        // ffn mix + control + norm
        let mut fcur = t::reshape2(c, self.mix(blocks, cur, l.ffn_res_query.expect("ffn_res")), n_embd, 1);
        if let Some(msgs) = messages {
            let m = t::view2(c, msgs, n_embd, 1, t::nb(msgs, 2), 3 * t::nb(msgs, 1));
            fcur = self.control(fcur, m, l, 3);
        }
        fcur = self.norm(fcur, l.ffn_norm.expect("ffn_norm"));

        // MoE — sigmoid gating + bias select, fused gate_up, SITU
        let gate_w = l.ffn_gate_inp.expect("g");
        let (ids, weights3);
        if std::env::var_os("INKWELL_NO_ROUTERFUSE").is_none() {
            // fused: sigmoid+bias+top8+normalize in one kernel; dst[0..8]=w, dst[8..16]=ids bits
            let pb = l.ffn_exp_probs_b.expect("pb");
            let ud = Box::leak(Box::new(RouterArgs {
                wscale: self.cfg.expert_weights_scale, wfloor: 6.103515625e-5 }));
            let fuse_gate = unsafe {
                inkwell_ggml::ink_tensor_type(gate_w.0) == BF16
                    && inkwell_ggml::ink_tensor_ne(gate_w.0, 0) == 2304
                    && inkwell_ggml::ink_tensor_ne(gate_w.0, 1) == 128
            };
            let r_out = if fuse_gate {
                let mut rargs = [gate_w.0, fcur.0, pb.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [16, 1, 1, 1], &mut rargs,
                    gate_router_cb as *const () as usize, (ud as *mut RouterArgs).cast()) })
            } else {
                // Non-BF16 fallback: keep the ggml GEMV but use the parallel router.
                let logits = t::mmq(c, gate_w, fcur); // [128, 1]
                let mut rargs = [logits.0, pb.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [16, 1, 1, 1], &mut rargs,
                    router_cb as *const () as usize, (ud as *mut RouterArgs).cast()) })
            };
            ids = t::view1(c, r_out, 8, 32);
            weights3 = t::reshape3(c, t::view1(c, r_out, 8, 0), 1, 8, 1);
        } else {
            let logits = t::mmq(c, gate_w, fcur); // [128, 1]
            let probs = t::sigmoid(c, logits);
            let sel = t::add(c, probs, l.ffn_exp_probs_b.expect("pb"));
            ids = t::argsort_top_k(c, sel, 8); // [8, ntok]
            let probs3 = t::reshape3(c, probs, 1, 128, 1); // [1, n_expert, ntok]
            let mut weights = t::get_rows(c, probs3, ids); // [1, 8, ntok]
            let weights2 = t::reshape2(c, weights, 8, 1);
            let wsum = t::clamp(c, t::sum_rows(c, weights2), 6.103515625e-5, f32::INFINITY);
            weights = t::reshape3(c, t::div(c, weights2, wsum), 1, 8, 1);
            weights = t::scale(c, weights, self.cfg.expert_weights_scale);
            weights3 = weights;
        }

        // custom op: 8 expert GEMVs on GPU over zero-copy pinned weights +
        // fused SITU + per-expert scales + weighted sum — all in one callback
        if il == 0 { self.dump("moe_in-0", fcur); self.dump("moe_ids-0", ids); self.dump("moe_w-0", weights3); }
        if let Some(mi) = l.moe_exl3 {
            let m = &self.moe_args_exl3[mi];
            let mut args: Vec<*mut std::ffi::c_void> = vec![fcur.0, ids.0, weights3.0];
            let ffn_out = Ptr(unsafe {
                inkwell_ggml::custom_4d(c.0, F32, [n_embd, 1, 1, 1], &mut args,
                    crate::moe::moe_cb_exl3_addr(), (&**m as *const crate::moe::MoeArgsExl3).cast_mut().cast())
            });
            // EXL3 experts are device-side (VRAM staging + pinned DMA): do NOT pin
            // ffn_out to CPU — that forced a backend split + host copies per layer.
            let fcur = self.convolve_add(ffn_out, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, stores, std::ptr::null_mut(), ffn_inp);
            if il == 0 { self.dump("ffn_out-0", fcur); }
            let out = fcur;
            self.dump(&format!("s_lout{il}"), out);
            return out;
        }
        let m = &self.moe_args[l.moe.expect("moe")];
        let mut args: Vec<*mut std::ffi::c_void> = vec![fcur.0, ids.0, weights3.0];
        if m.has_sgu {
            args.push(t::get_rows(c, t::reshape3(c, l.gate_up_scale.expect("s"), 1, 128, 1), ids).0);
        }
        if m.has_sdn {
            args.push(t::get_rows(c, t::reshape3(c, l.down_scale.expect("s"), 1, 128, 1), ids).0);
        }
        let ffn_out = Ptr(unsafe {
            inkwell_ggml::custom_4d(c.0, F32, [n_embd, 1, 1, 1], &mut args,
                crate::moe::moe_cb_addr(), (&**m as *const crate::moe::MoeArgs).cast_mut().cast())
        });
        self.set_cpu(ffn_out);

        let fcur = self.convolve_add(ffn_out, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, stores, std::ptr::null_mut(), ffn_inp);
        if il == 0 { self.dump("ffn_out-0", fcur); }
        let out2 = fcur;
        self.dump(&format!("l_out-{slot}"), out2);
        out2
    }
}

/// tau scale tensor (built per step in `step()`)
fn step_tau(s: &Step) -> Ptr { s.tau }

#[derive(Copy, Clone, Debug)]
struct Step {
    ids: Ptr,
    kv_idx_full: Ptr,
    kv_idx_swa: Ptr,
    pos: Ptr,
    rel_pos_full: Ptr,
    rel_pos_swa: Ptr,
    sink_zero: Ptr,
    tau: Ptr,
    pos_val: i64,
    mask_full: Ptr,
    mask_swa: Ptr,
    n_kv_full_pad: i64,
    n_kv_swa_pad: i64,
    n_kv_full_dev: Option<Ptr>,
    n_kv_swa_dev: Option<Ptr>,
    cap_full: i64,
    cap_swa: i64,
    attn_regime_full: u8,
    attn_regime_swa: u8,
    attn_allow_gqa_full: bool,
    attn_allow_gqa_swa: bool,
    dyn_nkv: bool,
    direct_relbias: bool,
}

fn step_shape_match(step: &Step, pos: i64, max_ctx: i64, swa: i64) -> bool {
    let n_kv_full = kv_bucket(pos + 1, max_ctx);
    let n_kv_swa = kv_bucket((pos + 1).min(swa), swa);
    if step.dyn_nkv {
        let (_, split_min, _) = attn_split_cfg();
        step.attn_regime_full == attn_regime_for_pad(n_kv_full)
            && step.attn_regime_swa == attn_regime_for_pad(n_kv_swa)
            && step.attn_allow_gqa_full == (n_kv_full < split_min)
            && step.attn_allow_gqa_swa == (n_kv_swa < split_min)
    } else {
        step.n_kv_full_pad == n_kv_full && step.n_kv_swa_pad == n_kv_swa
    }
}

impl Runner {
    /// Decode one step: `id` at absolute `pos`. Returns logits [vocab_unpadded].
    ///
    /// # Panics
    /// On graph/alloc/compute failure.
    pub fn step(&mut self, id: u32, pos: i64) -> Vec<f32> {
        self.step_impl(id, pos, false)
    }

    fn step_impl(&mut self, id: u32, pos: i64, dump_selection: bool) -> Vec<f32> {
        self.step_impl_mode(id, pos, dump_selection, false, true)
    }

    fn step_impl_mode(&mut self, id: u32, pos: i64, dump_selection: bool,
        skip_head: bool, need_logits: bool) -> Vec<f32> {
        self.check_ctx_span(pos, 1, "decode");
        let tb0 = std::time::Instant::now();
        // per-visit max n_kv this step, padded to a bucket multiple so tensor
        // shapes stay constant across steps (CUDA-graph replayable). The
        // -inf softmax masks cover the padding columns.
        let n_kv_full_raw = pos + 1;
        let n_kv_swa_raw = self.swa_positions(pos + 1);
        let n_kv_full = kv_bucket(n_kv_full_raw, self.max_ctx);
        let n_kv_swa = kv_bucket(n_kv_swa_raw, self.swa_capacity());

        // Reuse the built ggml graph + sched splits + instantiated CUDA graph
        // while the shape signature is unchanged: only input data varies.
        // Killswitch: INKWELL_NO_GCACHE=1 forces the per-step rebuild.
        static CACHING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static ATTN_DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let caching = *CACHING.get_or_init(|| std::env::var_os("INKWELL_NO_GCACHE").is_none());
        let dump_pow2 = *ATTN_DUMP.get_or_init(|| std::env::var_os("INKWELL_ATTN_DUMP_DIR").is_some())
            && ((pos + 1) & pos) == 0 && (1024..=32768).contains(&(pos + 1));
        if caching && !dump_pow2 {
            if let Some(gc) = self.step_gcache.take() {
                if step_shape_match(&gc.step, pos, self.max_ctx, self.swa_capacity())
                    && gc.has_head == !skip_head {
                    let logits = self.step_cached(id, pos, &gc);
                    self.step_gcache = Some(gc);
                    if dump_selection && crate::head_dump::enabled() {
                        let token = host_argmax(&logits);
                        self.dump_head_record(pos, token, gc.head_in, gc.out, Some(&logits));
                    }
                    return logits;
                }
            }
        }

        // drop the live split before ctx_reset frees the tensor objects it
        // references (is_alloc survives across steps when caching is on)
        unsafe { inkwell_ggml::ink_sched_reset(self.sched.0) };
        // persistent ctx + reset: tensor addresses repeat every token so the
        // CUDA graph cache key (nodes[0]) matches and capture can warm up
        if self.cur_ctx.0.is_null() {
            self.cur_ctx = Ptr(unsafe { inkwell_ggml::ink_ctx_new() });
        } else {
            unsafe { inkwell_ggml::ink_ctx_reset(self.cur_ctx.0) };
        }
        unsafe { *self.mix_prefix.get() = None };
        self.conv_arg_idx.set(0);
        let c = self.cur_ctx;
        let n_embd = self.cfg.n_embd as i64;
        let (hd, nh, hkv) = (128i64, 24i64, 8i64);
        let n_rep = nh / hkv;

        // --- inputs ---
        let dyn_nkv = dyn_nkv_enabled();
        let cap_full = if dyn_nkv { self.max_ctx } else { n_kv_full };
        let cap_swa = if dyn_nkv { self.max_ctx.min(self.swa_capacity()) } else { n_kv_swa };
        let ids = t::new(c, I32, &[1]);
        let pos_t = t::new(c, I32, &[1]);
        let kv_idx_full = t::new(c, I32, &[1]);
        let kv_idx_swa = t::new(c, I32, &[1]);
        let rel_pos_full = t::new(c, I32, &[cap_full, 1]);
        let rel_pos_swa = t::new(c, I32, &[cap_swa, 1]);
        let mask_full = t::new(c, F32, &[cap_full + 1, 1, 1, 1]);
        let mask_swa = t::new(c, F32, &[cap_swa + 1, 1, 1, 1]);
        let sink_zero = t::new(c, F32, &[1, 1, hkv, n_rep]);
        let (n_kv_full_dev, n_kv_swa_dev) = if dyn_nkv {
            (Some(t::new(c, I32, &[1])), Some(t::new(c, I32, &[1])))
        } else {
            (None, None)
        };
        // upload input data into cpu-side buffers? These tensors are graph
        // inputs — allocate them in the device buffer and set.
        // (they're tiny; alloc in dev buf of the step)
        if self.in_buf.is_none() {
            self.in_buf = Some(self.rt.buf(self.rt.device_buft(), 16 << 20));
        }
        let buf = self.in_buf.as_ref().unwrap().clone();
        let mut o = 0usize;
        let mut in_offs = [usize::MAX; 11];
        let mut inputs: Vec<(usize, Ptr)> = vec![
            (0, ids), (1, pos_t), (2, kv_idx_full), (3, kv_idx_swa)];
        if let (Some(nf), Some(ns)) = (n_kv_full_dev, n_kv_swa_dev) {
            inputs.push((9, nf));
            inputs.push((10, ns));
        }
        let n_small_inputs = inputs.len();
        inputs.extend([
            (4, rel_pos_full), (5, rel_pos_swa), (6, mask_full), (7, mask_swa), (8, sink_zero)]);
        let mut small_end = 0usize;
        for (ord, (i, x)) in inputs.iter().enumerate() {
            unsafe { inkwell_ggml::ink_set_input(x.0) };
            let off = (o + 255) & !255;
            buf.assign(*x, off);
            in_offs[*i] = off;
            o = off + t::nbytes(*x);
            if ord + 1 == n_small_inputs { small_end = o; }
        }
        let direct_relbias = dyn_nkv && !self.swa_oracle;
        // Pack the embedding row after the small inputs so the cached fast path
        // needs one HtoD copy per token. INKWELL_NO_ONEH2D keeps the old copy pair.
        let embd_off = if std::env::var_os("INKWELL_NO_ONEH2D").is_none() {
            let off = (o + 255) & !255;
            buf.assign(self.embd_row, off);
            o = off + t::nbytes(self.embd_row);
            off
        } else {
            usize::MAX
        };
        let input_copy_len = if direct_relbias { small_end } else { embd_off };
        // keep buf alive for the graph
        let _ = buf;

        // tau: log-scaled position (full layers)
        let posf = t::cast(c, pos_t, F32);
        let posf = t::scale_bias(c, posf, 1.0, 1.0);
        let tau = t::log(c, t::clamp(c, t::scale(c, posf, 1.0 / self.cfg.log_scale_floor as f32), 1.0, f32::INFINITY));
        let tau = t::reshape3(c, t::scale_bias(c, tau, self.cfg.log_scale_alpha, 1.0), 1, 1, 1);

        let step = Step { ids, kv_idx_full, kv_idx_swa, pos: pos_t, rel_pos_full, rel_pos_swa, sink_zero, tau, pos_val: pos,
                          mask_full, mask_swa, n_kv_full_pad: n_kv_full, n_kv_swa_pad: n_kv_swa,
                          n_kv_full_dev, n_kv_swa_dev, cap_full, cap_swa,
                          attn_regime_full: attn_regime_for_pad(n_kv_full),
                          attn_regime_swa: attn_regime_for_pad(n_kv_swa),
                          attn_allow_gqa_full: n_kv_full < attn_split_cfg().1,
                          attn_allow_gqa_swa: n_kv_swa < attn_split_cfg().1, dyn_nkv, direct_relbias };
        self.upload_step_inputs(&step, id, pos);

        // embed: dequantize the row host-side (embd stays quantized, ~230 MiB not 1.85 GiB)
        self.upload_embd_row(id);
        let mut cur = self.embd_row;
        let _ = ids;
        self.dump("embd_raw", cur);
        cur = self.norm(cur, self.tok_norm);
        self.dump("embd0", cur);

        // driver: prelude, n_passes loop, tail
        let (start, end, np) = (6usize, 18usize, 3usize);
        let mut blocks: Vec<Ptr> = Vec::new();
        let mut stores: Vec<Ptr> = Vec::new();
        let mut vi = 0usize;
        for il in 0..start {
            cur = self.layer(il, il, self.visits[vi].1, 0, cur, &mut blocks, None, &self.layers[il], &step, &mut stores);
            vi += 1;
        }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let msg_batch = std::env::var_os("INKWELL_NO_MSG_BATCH").is_none()
            && std::env::var_os("INKWELL_NO_MIX_V2").is_none()
            && !self.msg_query_packed.is_null();
        for pass in 0..np {
            let mut bank: Vec<Ptr> = Vec::new();
            let mut bank_out = Ptr(std::ptr::null_mut());
            let mut pass_msgs = Ptr(std::ptr::null_mut());
            if pass > 0 {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks.clone();
                if msg_batch {
                    pass_msgs = self.mix(&bank, bank_out, self.msg_query_packed);
                }
            }
            for ip in start..end {
                let (il_v, slot) = self.visits[vi];
                debug_assert_eq!(il_v, ip);
                if pass == 0 {
                    cur = self.layer(ip, ip, slot, pass, cur, &mut blocks, None, &self.layers[ip], &step, &mut stores);
                } else {
                    let l = &self.layers[ip];
                    let msgs = if pass_msgs.is_null() {
                        self.mix(&bank, bank_out, l.msg_query.expect("msg_query"))
                    } else {
                        let off = (ip - start) as i64 * 4 * t::nb(pass_msgs, 1);
                        t::view2(c, pass_msgs, n_embd, 4, t::nb(pass_msgs, 1), off)
                    };
                    cur = self.layer(ip, ip, slot, pass, cur, &mut blocks, Some(msgs), l, &step, &mut stores);
                }
                vi += 1;
            }
        }
        for il in end..24 {
            let (_il_v, slot) = self.visits[vi];
            cur = self.layer(il, il, slot, 0, cur, &mut blocks, None, &self.layers[il], &step, &mut stores);
            vi += 1;
        }
        debug_assert_eq!(vi, 48);

        // output head: mix → norm → lm_head → softcap → unpadded
        cur = t::reshape2(c, self.mix(&blocks, cur, self.output_res_query), n_embd, 1);
        cur = self.norm(cur, self.output_norm);
        self.dump("result_norm", cur);
        let head_in = cur;
        let own_head = self.out_head.is_some();
        if !skip_head && !own_head {
            cur = if let Some(oi) = self.output_exl3 {
                let e = &*self.exl3_w[oi];
                let mut args = [cur.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [e.n as i64, 1, 1, 1], &mut args,
                    exl3_cb_addr(), (e as *const Exl3W).cast_mut().cast()) })
            } else {
                t::mmq(c, self.output, cur)
            };
            if self.cfg.softcap != 0.0 {
                let cap = self.cfg.softcap;
                cur = t::scale(c, t::tanh(c, t::scale(c, cur, 1.0 / cap)), cap);
            }
        }
        t::output(cur);

        // graph
        let g = Ptr(unsafe { inkwell_ggml::ink_graph_new(c.0, 32768) });
        t::build_forward(g, cur);
        for s in &stores {
            t::build_forward(g, *s);
        }
        for (_, ptr) in unsafe { &*self.dump_list.get() } {
            t::build_forward(g, *ptr);
        }
        if std::env::var_os("INKWELL_OPCNT").is_some() {
            unsafe { inkwell_ggml::ink_graph_op_counts(g.0) };
        }
        if std::env::var_os("INKWELL_GRAPHDUMP").is_some() && pos == 0 {
            unsafe { inkwell_ggml::ink_graph_dump(g.0) };
        }
        // issue this token's expert prefetch on the side stream: predicted
        // ids are hist[] from the last token (already drained). Overlaps the
        // graph build + early compute below.
        self.issue_moe_prefetch(pos);
        let prof = std::env::var_os("INKWELL_PROF").is_some();
        if prof { eprintln!("PROF build={:.1}ms", tb0.elapsed().as_secs_f64()*1e3); }
        let t0 = std::time::Instant::now();
        unsafe {
            assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, g.0), 0, "sched compute");
            self.dump_all();
        }
        if prof { eprintln!("PROF compute={:.1}ms", t0.elapsed().as_secs_f64() * 1e3); }
        let logits = if need_logits { self.read_logits(cur, prof) } else { Vec::new() };
        if dump_selection && crate::head_dump::enabled() {
            let token = host_argmax(&logits);
            self.dump_head_record(pos, token, head_in, cur, Some(&logits));
        }
        // (unpadded vocab is a contiguous prefix slice of the padded logits)
        if caching {
            // keep ctx + cgraph + sched split + instantiated CUDA graph alive
            self.step_gcache = Some(StepCache { step, graph: g, out: cur, head_in, in_offs, in_span: o, embd_off, input_copy_len, has_head: !skip_head });
        } else {
            // release the cached split plan so the next step's graph gets re-allocated
            unsafe { inkwell_ggml::ink_sched_reset(self.sched.0) };
        }
        logits
    }

    /// Cache-hit step: same cgraph, same sched splits, same instantiated CUDA
    /// graph — refresh input data and replay. Bit-identical to a rebuild:
    /// every graph node, kernel argument and device address is unchanged.
    fn step_cached(&mut self, id: u32, pos: i64, gc: &StepCache) -> Vec<f32> {
        let tb0 = std::time::Instant::now();
        self.upload_step_inputs(&gc.step, id, pos);
        self.upload_embd_row(id);
        self.issue_moe_prefetch(pos);
        let prof = std::env::var_os("INKWELL_PROF").is_some();
        if prof { eprintln!("PROF build={:.1}ms", tb0.elapsed().as_secs_f64()*1e3); }
        let t0 = std::time::Instant::now();
        unsafe {
            assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, gc.graph.0), 0, "sched compute");
            self.dump_all();
        }
        if prof { eprintln!("PROF compute={:.1}ms", t0.elapsed().as_secs_f64() * 1e3); }
        self.read_logits(gc.out, prof)
    }

    /// Upload this step's input values into already buffer-assigned tensors.
    fn upload_step_inputs(&self, s: &Step, id: u32, pos: i64) {
        unsafe {
            t::set(s.ids, [id as i32].as_ptr().cast(), 0, 4);
            t::set(s.pos, [pos as i32].as_ptr().cast(), 0, 4);
            t::set(s.kv_idx_full, [pos as i32].as_ptr().cast(), 0, 4);
            t::set(s.kv_idx_swa, [self.swa_kv_idx(pos) as i32].as_ptr().cast(), 0, 4);
            // rel_pos[j] = distance (pos - key_pos) clamped to extent. Ring mode
            // recovers the key position modulo swa; padded slots get a harmless
            // index because their logits are masked to -inf before softmax.
            let n_kv_full_raw = pos + 1;
            let rp = |nk: i64, raw: i64| -> Vec<i32> {
                (0..nk).map(|j| if j < raw { (raw - 1 - j).min(self.cfg.rel_extent as i64) as i32 } else { 0 }).collect()
            };
            let n_kv_swa_raw = self.swa_positions(pos + 1);
            if let Some(nf) = s.n_kv_full_dev {
                t::set(nf, [n_kv_full_raw as i32].as_ptr().cast(), 0, 4);
            }
            if let Some(ns) = s.n_kv_swa_dev {
                t::set(ns, [n_kv_swa_raw as i32].as_ptr().cast(), 0, 4);
            }
            if !s.direct_relbias {
                let rp_f: Vec<i32> = if s.dyn_nkv {
                    (0..n_kv_full_raw).map(|j| (n_kv_full_raw - 1 - j).min(self.cfg.rel_extent as i64) as i32).collect()
                } else {
                    rp(s.n_kv_full_pad, n_kv_full_raw)
                };
                let rp_s: Vec<i32> = if s.dyn_nkv {
                    (0..n_kv_swa_raw).map(|j| self.swa_rel_pos(pos, j)).collect()
                } else {
                    (0..s.n_kv_swa_pad).map(|j| if self.swa_filled(pos, j) { self.swa_rel_pos(pos, j) } else { 0 }).collect()
                };
                t::set(s.rel_pos_full, rp_f.as_ptr().cast(), 0, rp_f.len() * 4);
                t::set(s.rel_pos_swa, rp_s.as_ptr().cast(), 0, rp_s.len() * 4);
            }
            // Dynamic mode never reads padding, so its mask is all zero.  The
            // fallback preserves master's -inf padded columns (plus the sink 0).
            let mk = |nk: i64, raw: i64| -> Vec<f32> {
                (0..=nk).map(|j| if s.dyn_nkv || j < raw || j == nk { 0.0 } else { f32::NEG_INFINITY }).collect()
            };
            let mk_f: Vec<f32> = if s.direct_relbias {
                vec![0.0; (s.cap_full + 1) as usize]
            } else if s.dyn_nkv {
                vec![0.0; (n_kv_full_raw + 1) as usize]
            } else {
                mk(s.n_kv_full_pad, n_kv_full_raw)
            };
            let mk_s: Vec<f32> = if s.direct_relbias {
                vec![0.0; (s.cap_swa + 1) as usize]
            } else if s.dyn_nkv {
                (0..=n_kv_swa_raw).map(|j| if j == n_kv_swa_raw || self.swa_visible(pos, j) { 0.0 } else { f32::NEG_INFINITY }).collect()
            } else {
                (0..=s.n_kv_swa_pad).map(|j| if j == s.n_kv_swa_pad || self.swa_visible(pos, j) { 0.0 } else { f32::NEG_INFINITY }).collect()
            };
            t::set(s.mask_full, mk_f.as_ptr().cast(), 0, mk_f.len() * 4);
            t::set(s.mask_swa, mk_s.as_ptr().cast(), 0, mk_s.len() * 4);
            let z = vec![0f32; (t::ne(s.sink_zero, 2) * t::ne(s.sink_zero, 3)) as usize];
            t::set(s.sink_zero, z.as_ptr().cast(), 0, z.len() * 4);
        }
    }

    /// Host-dequantize embd row `id` and upload it (embd stays quantized,
    /// ~230 MiB not 1.85 GiB).
    fn upload_embd_row(&self, id: u32) {
        let (src, ty, stride) = self.embd_src;
        let mut row = vec![0f32; 2304];
        if ty == -2 {
            // bf16 rows (safetensors path)
            let s = unsafe { src.add(id as usize * stride) } as *const u16;
            for (i, r) in row.iter_mut().enumerate() {
                *r = f32::from_bits(u32::from(unsafe { *s.add(i) }) << 16);
            }
        } else {
            let rc = unsafe { inkwell_ggml::ink_dequant(ty, src.add(id as usize * stride).cast(), row.as_mut_ptr(), 2304) };
            assert_eq!(rc, 0, "dequant embd row");
        }
        unsafe { t::set(self.embd_row, row.as_ptr().cast(), 0, 2304 * 4) };
    }

    /// Issue this token's expert prefetch on the side stream.
    fn issue_moe_prefetch(&mut self, pos: i64) {
        let pfstat = pos % 7 == 6 && std::env::var_os("INKWELL_PF_STAT").is_some();
        let active = self.moe_args_exl3.iter().any(|a|
            a.cache.as_ref().is_some_and(|c| c.prefetch.is_some()));
        let mut reuse_event = std::ptr::null_mut();
        if active {
            static REUSE_EV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            static CLEAR_EV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            reuse_event = *REUSE_EV.get_or_init(|| inkwell_cuda::new_event() as usize)
                as *mut std::ffi::c_void;
            let clear_event = *CLEAR_EV.get_or_init(|| inkwell_cuda::new_event() as usize)
                as *mut std::ffi::c_void;
            unsafe {
                inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
                inkwell_cuda::event_record_current(reuse_event);
                for a in &mut self.moe_args_exl3 {
                    let Some(c) = &a.cache else { continue };
                    let Some(pf) = &c.prefetch else { continue };
                    inkwell_cuda::exl3_prefetch_clear_async(pf.pready.dev_ptr().cast());
                }
                inkwell_cuda::event_record(clear_event, inkwell_cuda::side_stream_raw());
                inkwell_cuda::stream_wait_current(clear_event);
            }
        }
        for a in &mut self.moe_args_exl3 {
            let Some(c) = &mut a.cache else { continue };
            if pfstat {
                let mut b = [0u8; 8];
                c.hm.read_bytes(&mut b);
                let hits = i32::from_le_bytes(b[0..4].try_into().unwrap());
                let misses = i32::from_le_bytes(b[4..8].try_into().unwrap());
                eprintln!("PFSTAT L{} hits={} misses={}", a.layer, hits, misses);
            }
            let Some(pf) = &mut c.prefetch else { continue };
            unsafe {
                inkwell_cuda::exl3_prefetch2_async(
                    a.descs_g.dev(), a.descs_u.dev(), a.descs_d.dev(),
                    pf.hist.dev_ptr().cast(), c.tag.dev_ptr().cast(),
                    pf.bank.dev_ptr(), pf.ptag.dev_ptr().cast(),
                    pf.pready.dev_ptr().cast(), pf.pf_map.dev_ptr().cast(),
                    pf.pf_miss.dev_ptr().cast(), 8, c.k, c.slot as i64,
                    reuse_event);
            }
        }
    }

    /// Read back the [vocab_unpadded] logits from graph output tensor `out`
    /// (or run the own-head IQ4_XS lm_head when the model has no EXL3 head).
    fn read_logits(&mut self, out: Ptr, prof: bool) -> Vec<f32> {
        let tl0 = std::time::Instant::now();
        let mut logits = vec![0f32; self.cfg.vocab_unpadded as usize];
        if let Some((w, x_dev, y_dev)) = self.out_head.as_mut() {
            let mut normed = vec![0f32; self.cfg.n_embd as usize];
            unsafe { t::get(out, normed.as_mut_ptr().cast(), 0, normed.len() * 4) };
            x_dev.write(&normed);
            unsafe {
                inkwell_cuda::mmvq_iq4_xs_moe(
                    w.dev_ptr(), x_dev.dev_ptr(), y_dev, 201024, 2304, 2304, 1);
            }
            inkwell_cuda::softcap(y_dev, self.cfg.softcap);
            let all = y_dev.read();
            logits.copy_from_slice(&all[..self.cfg.vocab_unpadded as usize]);
        } else {
            unsafe { t::get(out, logits.as_mut_ptr().cast(), 0, logits.len() * 4) };
        }
        if prof { eprintln!("PROF logits={:.1}ms", tl0.elapsed().as_secs_f64()*1e3); }
        logits
    }

    fn read_dump_logits(&self, out: Ptr) -> Vec<f32> {
        let nv = self.cfg.vocab_unpadded as usize;
        if let Some((_, _, y_dev)) = self.out_head.as_ref() {
            let all = y_dev.read();
            all[..nv].to_vec()
        } else {
            let mut logits = vec![0f32; nv];
            unsafe { t::get(out, logits.as_mut_ptr().cast(), 0, nv * 4) };
            logits
        }
    }

    fn dump_head_record(&self, pos: i64, token: u32, head_in: Ptr, out: Ptr, logits: Option<&[f32]>) {
        if !crate::head_dump::enabled() { return; }
        let ne = self.cfg.n_embd as usize;
        let mut x = vec![0f32; ne];
        unsafe { t::get(head_in, x.as_mut_ptr().cast(), 0, ne * 4) };
        let owned_logits;
        let logits = if let Some(logits) = logits {
            logits
        } else {
            owned_logits = self.read_dump_logits(out);
            &owned_logits
        };
        crate::head_dump::write_record(
            u32::try_from(pos).expect("dump step fits u32"), token, &x, logits);
    }

    /// Lazily create the pinned/device side buffers of the `step_id` path.
    fn sid_bufs(&mut self) -> &mut SidBufs {
        if self.sid.is_none() {
            let am = inkwell_cuda::DevBuf::new(8 + 2 * 128 + 8);
            am.zero(); // argmax ticket ctr must start at 0
            let in_pin = [
                inkwell_cuda::host_alloc_mapped(1 << 20),
                inkwell_cuda::host_alloc_mapped(1 << 20),
            ];
            // zero once: the sink_zero mirror region stays zero forever
            unsafe {
                std::ptr::write_bytes(in_pin[0].host, 0, 1 << 20);
                std::ptr::write_bytes(in_pin[1].host, 0, 1 << 20);
            }
            self.sid = Some(SidBufs {
                am,
                out_pin: inkwell_cuda::host_alloc_mapped(64),
                in_pin,
                emb_pin: inkwell_cuda::host_alloc_mapped(2304 * 4 + 256),
                ev: inkwell_cuda::event_new(),
                marshaled: i64::MIN,
                queued: i64::MIN,
            });
        }
        self.sid.as_mut().expect("sid")
    }

    /// Raw pointers into `sid` (argmax buf, readback, mirrors, event) —
    /// copied out so no `&mut self` borrow persists across kernel launches.
    fn sid_parts(&mut self) -> SidParts {
        let s = self.sid_bufs();
        SidParts {
            am: s.am.dev_ptr(),
            ctr: unsafe { (s.am.dev_ptr() as *mut u32).add(8 + 2 * 128) },
            out: s.out_pin.host,
            inp: [s.in_pin[0].host, s.in_pin[1].host],
            emb: s.emb_pin.host,
            ev: s.ev,
        }
    }
    /// Pos whose pos-dep inputs are marshaled in `in_pin` (`i64::MIN` = none).
    fn marshaled(&self) -> i64 {
        self.sid.as_ref().map_or(i64::MIN, |s| s.marshaled)
    }
    /// Record the pos the `in_pin` mirror now holds (or `i64::MIN`).
    fn set_marshaled(&mut self, v: i64) {
        if let Some(s) = self.sid.as_mut() { s.marshaled = v; }
    }

    fn queued(&mut self) -> i64 {
        self.sid.as_mut().map_or(i64::MIN, |s| s.queued)
    }

    fn set_queued(&mut self, v: i64) {
        if let Some(s) = self.sid.as_mut() { s.queued = v; }
    }

    /// Marshal the pos-dependent inputs into the pinned `in_buf` mirror.
    /// `ids` is written by the caller once the token id is known; the
    /// sink_zero region stays zeroed since buffer init.
    fn marshal_inputs(&self, s: &Step, offs: &[usize; 11], pos: i64, dst: *mut u8) {
        unsafe {
            (dst.add(offs[1]) as *mut i32).write(pos as i32);
            (dst.add(offs[2]) as *mut i32).write(pos as i32);
            (dst.add(offs[3]) as *mut i32).write(self.swa_kv_idx(pos) as i32);
            let nfr = pos + 1;
            let nsr = self.swa_positions(pos + 1);
            if s.dyn_nkv {
                (dst.add(offs[9]) as *mut i32).write(nfr as i32);
                (dst.add(offs[10]) as *mut i32).write(nsr as i32);
            }
            if !s.direct_relbias {
                let ext = self.cfg.rel_extent as i64;
                let rf = dst.add(offs[4]) as *mut i32;
                if s.dyn_nkv {
                    for j in 0..nfr {
                        rf.add(j as usize).write((nfr - 1 - j).min(ext) as i32);
                    }
                } else {
                    for j in 0..s.cap_full {
                        rf.add(j as usize).write(if j < nfr { (nfr - 1 - j).min(ext) as i32 } else { 0 });
                    }
                }
                let rs = dst.add(offs[5]) as *mut i32;
                if s.dyn_nkv {
                    for j in 0..nsr {
                        rs.add(j as usize).write(self.swa_rel_pos(pos, j));
                    }
                } else {
                    for j in 0..s.cap_swa {
                        rs.add(j as usize).write(if self.swa_filled(pos, j) { self.swa_rel_pos(pos, j) } else { 0 });
                    }
                }
            }
            if s.dyn_nkv {
                if self.swa_oracle {
                    let ms = dst.add(offs[7]) as *mut f32;
                    for j in 0..=nsr {
                        ms.add(j as usize).write(if j == nsr || self.swa_visible(pos, j) { 0.0 } else { f32::NEG_INFINITY });
                    }
                }
            } else {
                let mf = dst.add(offs[6]) as *mut f32;
                for j in 0..=s.n_kv_full_pad {
                    mf.add(j as usize).write(if j < nfr || j == s.n_kv_full_pad { 0.0 } else { f32::NEG_INFINITY });
                }
                let ms = dst.add(offs[7]) as *mut f32;
                for j in 0..=s.n_kv_swa_pad {
                    ms.add(j as usize).write(if j == s.n_kv_swa_pad || self.swa_visible(pos, j) { 0.0 } else { f32::NEG_INFINITY });
                }
            }
        }
    }

    /// Device argmax of the step logits: 4 B readback instead of the full
    /// [vocab] DtoH. Bit-identical to host `max_by(total_cmp)` (same key
    /// order, same last-max tiebreak).
    fn read_argmax(&mut self, gc: &StepCache) -> u32 {
        let sp = self.sid_parts();
        let (nv, cap, ne) = (self.cfg.vocab_unpadded as usize, self.cfg.softcap, self.cfg.n_embd as usize);
        unsafe {
            inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
            if let Some((w, x_dev, y_dev)) = self.out_head.as_mut() {
                // own-head: normed activations -> host-launched IQ4_XS mmvq
                let mut normed = vec![0f32; ne];
                t::get(gc.out, normed.as_mut_ptr().cast(), 0, ne * 4);
                x_dev.write(&normed);
                inkwell_cuda::mmvq_iq4_xs_moe(
                    w.dev_ptr(), x_dev.dev_ptr(), y_dev, 201024, 2304, 2304, 1);
                inkwell_cuda::softcap(y_dev, cap);
                inkwell_cuda::argmax_async(y_dev.dev_ptr(), nv, sp.am.cast(), sp.ctr);
            } else {
                inkwell_cuda::argmax_async(inkwell_ggml::tensor_data(gc.out.0),
                    nv, sp.am.cast(), sp.ctr);
            }
            inkwell_cuda::memcpy_d2h_async(sp.out.cast(), sp.am, 4);
            inkwell_cuda::event_record(sp.ev, inkwell_ggml::cuda_stream());
            inkwell_cuda::event_sync(sp.ev);
            *(sp.out as *const u32)
        }
    }

    /// Decode one step and return the greedy token id.
    /// Fast path: packed pinned input upload + async sched compute + device
    /// argmax (4 B readback) + pos-dep marshal of the next step overlapped
    /// with the GPU. `INKWELL_NO_GARGMAX` restores the Vec logits read-back
    /// + host argmax; `INKWELL_NO_ASYNCSTEP` restores the per-tensor synced
    /// uploads + blocking sched compute (device argmax is kept).
    ///
    /// # Panics
    /// On sched compute failure.
    pub fn step_id(&mut self, id: u32, pos: i64) -> u32 {
        self.check_ctx_span(pos, 1, "decode");
        static NO_GA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *NO_GA.get_or_init(|| std::env::var_os("INKWELL_NO_GARGMAX").is_some()) {
            return host_argmax(&self.step_impl(id, pos, true));
        }
        static CACHING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CACHING.get_or_init(|| std::env::var_os("INKWELL_NO_GCACHE").is_none()) {
            if let Some(gc) = self.step_gcache.take() {
                if step_shape_match(&gc.step, pos, self.max_ctx, self.swa_capacity()) && gc.has_head {
                    let r = self.step_id_cached(id, pos, &gc);
                    self.step_gcache = Some(gc);
                    return r;
                }
                self.step_gcache = Some(gc);
            }
        }
        self.set_queued(i64::MIN);
        host_argmax(&self.step_impl(id, pos, true))
    }

    /// Cached prompt-state step. The graph ends at the normed hidden state, so
    /// neither the lm_head nor any logit readback runs. The blocking compute is
    /// deliberate: it makes reuse of the pinned input mirror safe without the
    /// speculative generation pipeline.
    fn step_prompt_cached(&mut self, id: u32, pos: i64, gc: &StepCache) {
        let sp = self.sid_parts();
        let cur_pin = sp.inp[(pos & 1) as usize];
        if self.marshaled() != pos {
            self.marshal_inputs(&gc.step, &gc.in_offs, pos, cur_pin);
        }
        unsafe { (cur_pin.add(gc.in_offs[0]) as *mut i32).write(id as i32) };
        let (src, ty, stride) = self.embd_src;
        let row = if gc.embd_off == usize::MAX {
            sp.emb as *mut f32
        } else {
            unsafe { cur_pin.add(gc.embd_off) as *mut f32 }
        };
        if ty == -2 {
            let src = unsafe { src.add(id as usize * stride) } as *const u16;
            for i in 0..2304 {
                unsafe { *row.add(i) = f32::from_bits(u32::from(*src.add(i)) << 16) };
            }
        } else {
            let rc = unsafe {
                inkwell_ggml::ink_dequant(ty, src.add(id as usize * stride).cast(), row, 2304)
            };
            assert_eq!(rc, 0, "dequant embd row");
        }
        unsafe {
            inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
            let base = inkwell_ggml::ink_buf_base(self.in_buf.as_ref().expect("in_buf").0 .0);
            inkwell_cuda::memcpy_h2d_async(base, cur_pin.cast_const().cast(), gc.in_span);
            if gc.embd_off == usize::MAX {
                inkwell_cuda::memcpy_h2d_async(
                    inkwell_ggml::tensor_data(self.embd_row.0),
                    sp.emb.cast_const().cast(), 2304 * 4);
            }
        }
        self.issue_moe_prefetch(pos);
        unsafe {
            assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, gc.graph.0), 0,
                "prompt sched compute");
            self.dump_all();
        }
        self.set_marshaled(pos);
        self.set_queued(i64::MIN);
    }

    /// Teacher-forced prompt/state step with the lm_head omitted. This is used
    /// only for rows whose logits are never consumed; state updates are the
    /// same graph nodes in the same order.
    pub fn step_prompt_state(&mut self, id: u32, pos: i64) {
        self.check_ctx_span(pos, 1, "prompt decode");
        static CACHING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let caching = *CACHING.get_or_init(|| std::env::var_os("INKWELL_NO_GCACHE").is_none());
        if caching {
            if let Some(gc) = self.step_gcache.take() {
                if step_shape_match(&gc.step, pos, self.max_ctx, self.swa_capacity()) && !gc.has_head {
                    self.step_prompt_cached(id, pos, &gc);
                    self.step_gcache = Some(gc);
                    return;
                }
                self.step_gcache = Some(gc);
            }
        }
        self.set_queued(i64::MIN);
        let _ = self.step_impl_mode(id, pos, false, true, false);
    }

    /// Drop the prompt-shaped cache before generation builds/uses the headed graph.
    pub fn clear_step_cache(&mut self) {
        self.step_gcache = None;
        self.set_queued(i64::MIN);
        self.set_marshaled(i64::MIN);
    }


    /// Gcache-hit step returning the greedy id (see [`Self::step_id`]).
    fn step_id_cached(&mut self, id: u32, pos: i64, gc: &StepCache) -> u32 {
        static NO_AS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let prof = *PROF.get_or_init(|| std::env::var_os("INKWELL_PROF").is_some());
        if *NO_AS.get_or_init(|| std::env::var_os("INKWELL_NO_ASYNCSTEP").is_some()) {
            // killswitch: per-tensor synced uploads + blocking sched compute
            self.upload_step_inputs(&gc.step, id, pos);
            self.upload_embd_row(id);
            self.issue_moe_prefetch(pos);
            unsafe {
                assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, gc.graph.0), 0, "sched compute");
                self.dump_all();
            }
            let token = self.read_argmax(gc);
            self.dump_head_record(pos, token, gc.head_in, gc.out, None);
            return token;
        }
        let tb0 = std::time::Instant::now();
        let detailed = stepprof_enabled();
        let sp = self.sid_parts();
        let already = self.queued() == pos;
        let mut tp1 = tb0;
        let mut tp2 = tb0;
        let mut tp3 = tb0;
        let mut tp4 = tb0;
        if !already {
            let cur_pin = sp.inp[(pos & 1) as usize];
            if self.marshaled() != pos {
                self.marshal_inputs(&gc.step, &gc.in_offs, pos, cur_pin);
            }
            unsafe { (cur_pin.add(gc.in_offs[0]) as *mut i32).write(id as i32) };
            tp1 = std::time::Instant::now();
            // embd row dequant into the pinned input mirror (or its separate staging
            // buffer under INKWELL_NO_ONEH2D), then async copy.
            let (src, ty, stride) = self.embd_src;
            let row = if gc.embd_off == usize::MAX {
                sp.emb as *mut f32
            } else {
                unsafe { cur_pin.add(gc.embd_off) as *mut f32 }
            };
            if ty == -2 {
                let s = unsafe { src.add(id as usize * stride) } as *const u16;
                for i in 0..2304 {
                    unsafe { *row.add(i) = f32::from_bits(u32::from(*s.add(i)) << 16) };
                }
            } else {
                let rc = unsafe {
                    inkwell_ggml::ink_dequant(ty, src.add(id as usize * stride).cast(), row, 2304)
                };
                assert_eq!(rc, 0, "dequant embd row");
            }
            tp2 = std::time::Instant::now();
            unsafe {
                inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
                let base = inkwell_ggml::ink_buf_base(self.in_buf.as_ref().expect("in_buf").0 .0);
                inkwell_cuda::memcpy_h2d_async(base, cur_pin.cast_const().cast(), gc.in_span);
                if gc.embd_off == usize::MAX {
                    inkwell_cuda::memcpy_h2d_async(
                        inkwell_ggml::tensor_data(self.embd_row.0),
                        sp.emb.cast_const().cast(), 2304 * 4);
                }
            }
            tp3 = std::time::Instant::now();
            if prof { eprintln!("PROF build={:.1}ms", tb0.elapsed().as_secs_f64() * 1e3); }
            let t0 = std::time::Instant::now();
            self.issue_moe_prefetch(pos);
            tp4 = std::time::Instant::now();
            unsafe {
                assert_eq!(
                    inkwell_ggml::ink_sched_compute_async(self.sched.0, gc.graph.0), 0,
                    "sched compute async");
                self.dump_all();
            }
            if prof { eprintln!("PROF compute={:.1}ms", t0.elapsed().as_secs_f64() * 1e3); }
        }
        let tp5 = std::time::Instant::now();
        // queue argmax + 4B readback after the replay, then marshal the next
        // step's pos-dep inputs while the GPU is still busy (single pinned
        // buf: this step's copy already ran before the graph).
        let next = pos + 1;
        let premarshal = step_shape_match(&gc.step, next, self.max_ctx, self.swa_capacity());
        unsafe {
            inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
            if self.out_head.is_none() {
                inkwell_cuda::argmax_async(inkwell_ggml::tensor_data(gc.out.0),
                    self.cfg.vocab_unpadded as usize, sp.am.cast(), sp.ctr);
                inkwell_cuda::memcpy_d2h_async(sp.out.cast(), sp.am, 4);
                inkwell_cuda::event_record(sp.ev, inkwell_ggml::cuda_stream());
            }
        }
        let tp6 = std::time::Instant::now();
        static NO_PIPE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_pipe = *NO_PIPE.get_or_init(|| std::env::var_os("INKWELL_NO_PIPESTEP").is_some());
        let prefetch_active = self.moe_args_exl3.iter().any(|a|
            a.cache.as_ref().map_or(false, |c| c.prefetch.is_some()));
        let (_, ty, stride) = self.embd_src;
        let dev_resolve = ty == -2 && !self.embd_dev.is_null();
        let pipeline = premarshal && !no_pipe && gc.embd_off != usize::MAX
            && self.out_head.is_none() && !prefetch_active && dev_resolve;
        if pipeline {
            let next_pin = sp.inp[(next & 1) as usize];
            self.marshal_inputs(&gc.step, &gc.in_offs, next, next_pin);
            unsafe {
                inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
                let base = inkwell_ggml::ink_buf_base(self.in_buf.as_ref().expect("in_buf").0 .0);
                inkwell_cuda::memcpy_h2d_async(base, next_pin.cast_const().cast(), gc.input_copy_len);
                let b = base as *mut u8;
                inkwell_cuda::pipe_resolve_async(
                    sp.am.cast(), self.embd_dev.cast_const(), stride,
                    b.add(gc.in_offs[0]).cast(), b.add(gc.embd_off).cast(), 1);
                assert_eq!(
                    inkwell_ggml::ink_sched_compute_async(self.sched.0, gc.graph.0), 0,
                    "pipelined sched compute async");
            }
            self.set_queued(next);
            self.set_marshaled(next);
        } else {
            self.set_queued(i64::MIN);
            if premarshal {
                self.marshal_inputs(&gc.step, &gc.in_offs, next, sp.inp[(next & 1) as usize]);
                self.set_marshaled(next);
            } else {
                self.set_marshaled(i64::MIN);
            }
        }
        let tp7 = std::time::Instant::now();
        if self.out_head.is_some() {
            // own-head needs host launches: sync then take the slow read path
            unsafe { inkwell_cuda::DevBuf::sync() };
            let token = self.read_argmax(gc);
            self.dump_head_record(pos, token, gc.head_in, gc.out, None);
            return token;
        }
        unsafe { inkwell_cuda::event_sync(sp.ev) };
        let tp8 = std::time::Instant::now();
        if detailed {
            stepprof_push(StepProfRow {
                marshal: stepprof_dur(tb0, tp1),
                embd: stepprof_dur(tp1, tp2),
                h2d: stepprof_dur(tp2, tp3),
                prefetch: stepprof_dur(tp3, tp4),
                sched: stepprof_dur(tp4, tp5),
                tailq: stepprof_dur(tp5, tp6),
                premarshal: stepprof_dur(tp6, tp7),
                wait: stepprof_dur(tp7, tp8),
                total: stepprof_dur(tb0, tp8),
            });
        }
        let token = unsafe { *(sp.out as *const u32) };
        self.dump_head_record(pos, token, gc.head_in, gc.out, None);
        token
    }
}

/// Raw pointers into [`SidBufs`], copied out of the `&mut self` borrow.
struct SidParts {
    /// argmax device buf base
    am: *mut std::ffi::c_void,
    /// ticket counter (u32 index 8+2*128)
    ctr: *mut u32,
    /// 4 B readback pinned buf
    out: *mut u8,
    /// in_buf mirror pinned bufs (parity by pos)
    inp: [*mut u8; 2],
    /// embd row staging pinned buf
    emb: *mut u8,
    /// readback event
    ev: *mut std::ffi::c_void,
}

/// Pinned/device side buffers for `Runner::step_id`.
#[derive(Debug)]
struct SidBufs {
    /// argmax device buf: [0..2] final {idx,key}, [8..264] block partials,
    /// ticket ctr at u32 index 8+2*128
    am: inkwell_cuda::DevBuf,
    /// 4 B argmax readback
    out_pin: inkwell_cuda::PinnedBuf,
    /// mirror of the `in_buf` input block (packed single HtoD), parity by
    /// pos so next-step marshaling can't race this step's async copy
    in_pin: [inkwell_cuda::PinnedBuf; 2],
    /// dequantized embd row staging (one async HtoD)
    emb_pin: inkwell_cuda::PinnedBuf,
    /// argmax readback completion event
    ev: *mut std::ffi::c_void,
    /// pos whose pos-dep inputs are already marshaled in `in_pin`
    marshaled: i64,
    /// pos whose graph is already enqueued by the decode pipeline
    queued: i64,
}

/// Host-side greedy argmax (the pre-`step_id` selection rule).
fn host_argmax(l: &[f32]) -> u32 {
    l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

// ============ spec-decode verify: ntok=8 block forward ============

const NT: usize = 8;

/// per-token step inputs for the block path
struct StepBlk {
    kv_idx_full: [Ptr; NT],
    kv_idx_swa: [Ptr; NT],
    rel_pos_full: [Ptr; NT],
    rel_pos_swa: [Ptr; NT],
    mask_full: [Ptr; NT],
    mask_swa: [Ptr; NT],
    tau8: Ptr, // [8] f32
    n_kv_full_pad: i64,
    n_kv_swa_pad: i64,
    swa_wrap_hazard: bool,
}

impl Runner {
    /// slice token `j` out of a contiguous tensor whose last dim is `NT`
    /// (`n` = per-token element count: product of the leading dims).
    fn tokslice(&self, c: Ptr, x: Ptr, n: i64, j: i64) -> Ptr {
        t::view1(c, x, n, j * n * 4)
    }

    fn cat8(&self, c: Ptr, xs: &[Ptr]) -> Ptr {
        let mut acc = t::reshape3(c, xs[0], t::ne(xs[0], 0), 1, 1);
        for x in &xs[1..] {
            acc = t::concat(c, acc, t::reshape3(c, *x, t::ne(*x, 0), 1, 1), 1);
        }
        t::reshape2(c, acc, t::ne(acc, 0), NT as i64)
    }

    /// GEMV dispatch for `[k, NT]` inputs → `[n, NT]` (batched EXL3 or mmq).
    fn proj8(&self, c: Ptr, l: &L, key: &'static str, w: Option<Ptr>, x: Ptr) -> Ptr {
        match l.exl3.get(key) {
            Some(&i) => {
                let e = &*self.exl3_blk[i];
                e.ensure_scratch();
                let mut args = [x.0];
                Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [e.n as i64, NT as i64, 1, 1], &mut args,
                    exl3_ntok_cb_addr(), (e as *const Exl3Blk).cast_mut().cast()) })
            }
            None => t::mmq(c, w.expect(key), x),
        }
    }

    /// `mix` over `[n, NT]` residual checkpoints → `[n, rows, NT]`
    fn mix8(&self, blocks: &[Ptr], cur: Ptr, query: Ptr) -> Ptr {
        let c = self.cur_ctx;
        let n_embd = self.cfg.n_embd as i64;
        let mut states = if blocks.is_empty() {
            t::reshape3(c, cur, n_embd, 1, NT as i64)
        } else {
            let mut p = t::reshape3(c, blocks[0], n_embd, 1, NT as i64);
            for b in &blocks[1..] {
                p = t::concat(c, p, t::reshape3(c, *b, n_embd, 1, NT as i64), 1);
            }
            t::concat(c, p, t::reshape3(c, cur, n_embd, 1, NT as i64), 1)
        };
        states = t::rms_norm(c, states, 1e-6);
        let n_rows = t::ne(query, 1);
        let queries = t::repeat(c, query, t::new(c, F32, &[n_embd, n_rows, NT as i64]));
        let scores = t::soft_max(c, t::mmq(c, states, queries), Ptr(std::ptr::null_mut()), 1.0 / (n_embd as f32).sqrt(), 0.0);
        t::mmq(c, t::cont(c, t::transpose(c, states)), scores)
    }

    /// block forward of one layer-visit: `cur`/`blocks`/`messages` are `[n, NT]` tensors;
    /// stateful ops (conv state, KV write, attention, MoE) run per token on slices.
    #[expect(clippy::too_many_arguments)]
    fn layer_blk(&self, il: usize, ip: usize, slot: usize, pass: usize, mut cur: Ptr, blocks: &mut Vec<Ptr>,
             messages: Option<Ptr>, l: &L, step: &StepBlk) -> Ptr {
        let c = self.cur_ctx;
        let is_swa = self.is_swa_layer(il);
        let (n_embd, hd, nh, hkv, n_rel) =
            (self.cfg.n_embd as i64, 128i64, 24i64, 8i64, self.cfg.n_rel as i64);
        let n_rep = nh / hkv;
        let gqa = (hd * hkv) as i64;
        let inp_sa = cur;

        let mut inputs = [cur, cur, cur];
        if let Some(rq) = l.attn_res_query {
            let mixed = self.mix8(blocks, cur, rq); // [n, 3, NT]
            for (j, inp) in inputs.iter_mut().enumerate() {
                let (a, n, rs, off) = (mixed, n_embd, t::nb(mixed, 2), (if is_swa { 0 } else { j as i64 }) * t::nb(mixed, 1));
                if std::env::var_os("INKWELL_VDBG").is_some() { eprintln!("vdbg il={il} p{pass} mixed[{},{},{}] nb[{},{},{}] n0={n} rs={rs} off={off}", t::ne(a,0),t::ne(a,1),t::ne(a,2),t::nb(a,0),t::nb(a,1),t::nb(a,2)); }
                *inp = t::cont(c, t::view2(c, a, n, NT as i64, rs, off));
            }
        }
        if let Some(msgs) = messages {
            for (j, inp) in inputs.iter_mut().enumerate() {
                let (a, n, rs, off) = (msgs, n_embd, t::nb(msgs, 2), j as i64 * t::nb(msgs, 1));
                if std::env::var_os("INKWELL_VDBG").is_some() { eprintln!("vdbg il={il} p{pass} msgs[{},{},{}] nb[{},{},{}] n0={n} rs={rs} off={off}", t::ne(a,0),t::ne(a,1),t::ne(a,2),t::nb(a,0),t::nb(a,1),t::nb(a,2)); }
                let m = t::cont(c, t::view2(c, a, n, NT as i64, rs, off));
                *inp = self.control(*inp, m, l, j);
            }
        }
        if is_swa && messages.is_some() {
            for inp in inputs.iter_mut() { *inp = self.norm(*inp, l.attn_norm.expect("attn_norm")); }
        } else if is_swa {
            inputs[0] = self.norm(inputs[0], l.attn_norm.expect("attn_norm"));
            inputs[1] = inputs[0];
            inputs[2] = inputs[0];
        } else {
            inputs[0] = self.norm(inputs[0], l.attn_q_inp_norm.expect("q_inp"));
            inputs[1] = self.norm(inputs[1], l.attn_k_inp_norm.expect("k_inp"));
            inputs[2] = self.norm(inputs[2], l.attn_v_inp_norm.expect("v_inp"));
        }

        if il == 0 {
            self.dump("b_cur_in", cur);
            self.dump("b_inputs0", inputs[0]);
            self.dump("b_inputs1", inputs[1]);
        }
        // projections on [k, NT]
        let (q8, gate8, r8) = if let Some(cat) = l.proj0_cat {
            let out = t::mmq(c, cat, inputs[0]); // [n_cat, NT]
            let slice = |off: i64, n: i64, key: &'static str, w: Option<Ptr>| {
                if off >= 0 {
                    if std::env::var_os("INKWELL_VDBG").is_some() { eprintln!("vdbg il={il} p{pass} catout[{},{},{}] nb[{},{},{}] n0={n} rs={} off={}", t::ne(out,0),t::ne(out,1),t::ne(out,2),t::nb(out,0),t::nb(out,1),t::nb(out,2), t::nb(out,1), off*4); }
                    t::view2(c, out, n, NT as i64, t::nb(out, 1), off * 4)
                } else { self.proj8(c, l, key, w, inputs[0]) }
            };
            (
                slice(l.proj0_off[0], 3072, "wq", l.wq),
                slice(l.proj0_off[1], 3072, "gate", l.wqkv_gate),
                slice(l.proj0_off[2], 384, "rel", l.attn_rel),
            )
        } else {
            (
                self.proj8(c, l, "wq", l.wq, inputs[0]),
                self.proj8(c, l, "gate", l.wqkv_gate, inputs[0]),
                self.proj8(c, l, "rel", l.attn_rel, inputs[0]),
            )
        };
        let k_dyn8 = if is_swa { None } else { Some(t::mmq(c, l.attn_k_conv_proj.expect("kcp"), inputs[1])) };
        let v_dyn8 = if is_swa { None } else { Some(t::mmq(c, l.attn_v_conv_proj.expect("vcp"), inputs[2])) };
        // batched exl3 GEMV, then per-token conv state updates
        let wk8 = self.proj8(c, l, "wk", l.wk, inputs[1]);
        let wv8 = self.proj8(c, l, "wv", l.wv, inputs[2]);
        let mut ks: Vec<Ptr> = Vec::with_capacity(NT);
        let mut vs: Vec<Ptr> = Vec::with_capacity(NT);
        for j in 0..NT {
            let kd = k_dyn8.map(|d| t::cont(c, self.tokslice(c, d, t::ne(d, 0), j as i64)));
            let vd = v_dyn8.map(|d| t::cont(c, self.tokslice(c, d, t::ne(d, 0), j as i64)));
            ks.push(self.convolve(t::cont(c, self.tokslice(c, wk8, t::ne(wk8, 0), j as i64)), l.attn_k_conv.expect("kconv"), kd, slot, 0, &mut Vec::new(), self.snap_ptr(slot, j)));
            vs.push(self.convolve(t::cont(c, self.tokslice(c, wv8, t::ne(wv8, 0), j as i64)), l.attn_v_conv.expect("vconv"), vd, slot, gqa, &mut Vec::new(), self.snap_ptr(slot, j)));
        }
        let k8 = self.cat8(c, &ks);
        let v8 = self.cat8(c, &vs);
        if il == 0 {
            self.dump("b_q8", q8); self.dump("b_k8", k8);
            self.dump("b_v8", v8); self.dump("b_r8", r8);
        }

        let q8 = self.clip(t::reshape3(c, q8, hd, nh, NT as i64), l.attn_q_clip, self.cfg.qk_clip);
        let k8 = self.clip(t::reshape3(c, k8, hd, hkv, NT as i64), None, self.cfg.qk_clip);
        let v8 = t::rms_norm(c, t::reshape3(c, v8, hd, hkv, NT as i64), 1e-6);
        let r8 = self.clip(t::reshape3(c, r8, n_rel, nh, NT as i64), l.attn_rel_clip, self.cfg.rel_clip);

        let n_kv = if is_swa { step.n_kv_swa_pad } else { step.n_kv_full_pad };
        let mut k_cur = self.k_arena[slot];
        let mut v_cur = self.v_arena[slot];
        let mut outs: Vec<Ptr> = Vec::with_capacity(NT);
        for j in 0..NT {
            let k_j = t::reshape3(c, self.tokslice(c, k8, hd * hkv, j as i64), hd, hkv, 1);
            let v_j = t::reshape3(c, self.tokslice(c, v8, hd * hkv, j as i64), hd, hkv, 1);
            let mut k_rows = t::permute(c, k_j, 0, 2, 1, 3);
            let mut v_rows = t::permute(c, v_j, 0, 2, 1, 3);
            if is_swa && step.swa_wrap_hazard && j > 0 {
                // A wrap overwrite must not pass the attention that still reads the old row.
                let one = t::scale_bias(c, t::view1(c, outs[j - 1], 1, 0), 0.0, 1.0);
                k_rows = t::mul(c, k_rows, one);
                v_rows = t::mul(c, v_rows, one);
            }
            let kv_idx = if is_swa { step.kv_idx_swa[j] } else { step.kv_idx_full[j] };
            k_cur = t::set_rows(c, k_cur, k_rows, kv_idx);
            v_cur = t::set_rows(c, v_cur, v_rows, kv_idx);

            let mut q_j = t::reshape3(c, self.tokslice(c, q8, hd * nh, j as i64), hd, nh, 1);
            let mut r_j = t::reshape3(c, self.tokslice(c, r8, n_rel * nh, j as i64), n_rel, nh, 1);
            if !is_swa {
                let tau_j = t::view1(c, step.tau8, 1, (j * 4) as i64);
                q_j = t::mul(c, q_j, t::reshape3(c, tau_j, 1, 1, 1));
                r_j = t::mul(c, r_j, t::reshape3(c, tau_j, 1, 1, 1));
            }
            let inputs0_j = self.tokslice(c, inputs[0], n_embd, j as i64);
            let sinks = t::mmq(c, l.attn_sink_proj.expect("sink"), inputs0_j);
            let table = t::mmq(c, l.attn_rel_bias_pad.or(l.attn_rel_bias).expect("relbias"), r_j);
            let table = t::sub(c, table, t::reshape3(c, sinks, 1, nh, 1));
            let table = t::cont(c, t::permute(c, table, 1, 0, 2, 3));
            let rel_pos = if is_swa { step.rel_pos_swa[j] } else { step.rel_pos_full[j] };
            let kq_b = t::get_rows(c, table, rel_pos);
            let kq_b = t::cont(c, t::permute(c, t::reshape4(c, kq_b, nh, n_kv, 1, 1), 2, 0, 1, 3));
            let k_view = t::view3(c, k_cur, hd, n_kv, hkv, t::nb(k_cur, 1), t::nb(k_cur, 2), 0);
            let q3 = t::permute(c, q_j, 0, 2, 1, 3);
            let kq_mask = if is_swa { step.mask_swa[j] } else { step.mask_full[j] };
            let v_view = t::view3(c, v_cur, hd, n_kv, hkv, t::nb(v_cur, 1), t::nb(v_cur, 2), 0);
            if il == 0 && j == 0 {
                self.dump("b_qj0", q_j); self.dump("b_rj0", r_j);
                self.dump("b_table0", table); self.dump("b_kqb0", kq_b);
            }
            let out = t::attn_fused(c, k_view, v_view, q3, kq_b, kq_mask, hd, nh);
            if il == 0 && j == 0 { self.dump("b_attnout0", out); }
            outs.push(t::mul(c, out, t::sigmoid(c, self.tokslice(c, gate8, t::ne(gate8, 0), j as i64))));
        }
        let mut cur = self.cat8(c, &outs);
        cur = self.proj8(c, l, "wo", l.wo, cur);
        // post conv per token
        let mut pcs: Vec<Ptr> = Vec::with_capacity(NT);
        for j in 0..NT {
            pcs.push(self.convolve(t::cont(c, self.tokslice(c, cur, n_embd, j as i64)), l.attn_post_conv.expect("postconv"), None, slot, 2 * gqa, &mut Vec::new(), self.snap_ptr(slot, j)));
        }
        cur = self.cat8(c, &pcs);

        if ip % (self.cfg.attn_res_block_size as usize) == 0 {
            blocks.push(inp_sa);
        } else {
            cur = t::add(c, inp_sa, cur);
        }
        let ffn_inp = cur;

        let mut fcur = t::reshape2(c, self.mix8(blocks, cur, l.ffn_res_query.expect("ffn_res")), n_embd, NT as i64);
        if let Some(msgs) = messages {
            let m = t::cont(c, t::view2(c, msgs, n_embd, NT as i64, t::nb(msgs, 2), 3 * t::nb(msgs, 1)));
            fcur = self.control(fcur, m, l, 3);
        }
        fcur = self.norm(fcur, l.ffn_norm.expect("ffn_norm"));

        // MoE: batched router for 8 tokens + union-batched expert chain
        let logits8 = t::mmq(c, l.ffn_gate_inp.expect("g"), fcur); // [128, 8]
        let probs8 = t::sigmoid(c, logits8);
        let sel8 = t::add(c, probs8, l.ffn_exp_probs_b.expect("pb"));
        let ids8v = t::argsort_top_k(c, sel8, 8); // view — make contiguous
        let ids8 = t::cont(c, ids8v); // [8, 8] — linear dst[t*8+j]
        self.dump(&format!("b_ids{il}"), ids8);
        let probs3 = t::reshape3(c, probs8, 1, 128, 8);
        let ids3 = t::reshape3(c, ids8, 8, 8, 1); // get_rows: dst[1,8,8] = probs3[0, ids[j,t], t]
        let weights = t::get_rows(c, probs3, ids3); // [1, 8, 8]
        let weights2 = t::reshape2(c, weights, 8, 8);
        let wsum = t::clamp(c, t::sum_rows(c, weights2), 6.103515625e-5, f32::INFINITY);
        let w8 = t::scale(c, t::div(c, weights2, wsum), self.cfg.expert_weights_scale); // [8, 8]
        let mi = l.moe_exl3.expect("moe_exl3");
        let m = &self.moe_args_exl3[mi];
        let mut args: Vec<*mut std::ffi::c_void> = vec![fcur.0, ids8.0, w8.0];
        let ffn8 = Ptr(unsafe {
            inkwell_ggml::custom_4d(c.0, F32, [n_embd, NT as i64, 1, 1], &mut args,
                crate::moe::moe_cb_exl3_ntok_addr(), (&**m as *const crate::moe::MoeArgsExl3).cast_mut().cast())
        });
        let mut fouts: Vec<Ptr> = Vec::with_capacity(NT);
        for j in 0..NT {
            let fj = self.tokslice(c, ffn8, n_embd, j as i64);
            fouts.push(self.convolve(fj, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, &mut Vec::new(), self.snap_ptr(slot, j)));
        }
        let fcur = self.cat8(c, &fouts);
        let out = t::add(c, ffn_inp, fcur);
        self.dump(&format!("b_lout{il}"), out);
        out
    }

    /// Host snapshot of every conv state tensor (`[3][n_sconv]` f32 each) —
    /// restore with [`Runner::restore_conv`] to rewind stateful conv layers.
    pub fn snapshot_conv(&self) -> Vec<Vec<f32>> {
        self.conv_state.iter().map(|s| unsafe {
            let n = t::ne(*s, 0) as usize * t::ne(*s, 1) as usize;
            let mut v = vec![0f32; n];
            t::get(*s, v.as_mut_ptr().cast(), 0, n * 4);
            v
        }).collect()
    }

    /// Rewind conv states to a `snapshot_conv` copy.
    pub fn restore_conv(&self, snap: &[Vec<f32>]) {
        for (s, v) in self.conv_state.iter().zip(snap.iter()) {
            unsafe { t::set(*s, v.as_ptr().cast(), 0, v.len() * 4) };
        }
    }

    /// Reset per-sequence state (zero conv states and clear cached step graph)
    /// so the runner can serve a fresh prompt from position 0 without reloading weights.
    pub fn reset(&mut self) {
        inkwell_cuda::DevBuf::sync();
        self.clear_step_cache();
        for s in &self.conv_state {
            unsafe {
                let n = t::ne(*s, 0) as usize * t::ne(*s, 1) as usize;
                let zeros = vec![0f32; n];
                t::set(*s, zeros.as_ptr().cast(), 0, n * 4);
            }
        }
    }

    /// Configured maximum context length in tokens.
    #[must_use]
    pub fn max_ctx(&self) -> i64 {
        self.max_ctx
    }

    /// Save all persistent decode state tensors (KV arenas + conv states).
    pub fn save_state(&self, path: &str) {
        let tensors: Vec<Ptr> = self.k_arena.iter().chain(self.v_arena.iter()).chain(self.conv_state.iter()).copied().collect();
        let mut f = std::fs::File::create(path).expect("create state snapshot");
        use std::io::Write as _;
        f.write_all(b"INKWST02").expect("write state magic");
        f.write_all(&(tensors.len() as u64).to_le_bytes()).expect("write state count");
        for x in tensors {
            let n = t::nbytes(x);
            let mut v = vec![0u8; n];
            unsafe { t::get(x, v.as_mut_ptr(), 0, n) };
            f.write_all(&(n as u64).to_le_bytes()).expect("write state length");
            f.write_all(&v).expect("write state tensor");
        }
    }

    /// Load state written by [`Runner::save_state`].
    pub fn load_state(&self, path: &str) {
        let tensors: Vec<Ptr> = self.k_arena.iter().chain(self.v_arena.iter()).chain(self.conv_state.iter()).copied().collect();
        let mut f = std::fs::File::open(path).expect("open state snapshot");
        use std::io::Read as _;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).expect("read state magic");
        assert_eq!(&magic, b"INKWST02", "state snapshot magic");
        let mut u64b = [0u8; 8];
        f.read_exact(&mut u64b).expect("read state count");
        assert_eq!(u64::from_le_bytes(u64b) as usize, tensors.len(), "state tensor count");
        for x in tensors {
            f.read_exact(&mut u64b).expect("read state length");
            let n = u64::from_le_bytes(u64b) as usize;
            assert_eq!(n, t::nbytes(x), "state tensor byte length");
            let mut v = vec![0u8; n];
            f.read_exact(&mut v).expect("read state tensor");
            unsafe { t::set(x, v.as_ptr(), 0, n) };
        }
    }

    /// Verify forward: run `ids[NT]` at positions `pos..pos+NT` through the full
    /// model in one graph. Returns logits `[NT][vocab_unpadded]` (row j = token j).
    /// KV writes for all 8 tokens land in the arenas; conv state ends at pos+7
    /// (caller snapshots/restores on partial acceptance).
    ///
    /// # Panics
    /// On graph/alloc/compute failure.
    pub fn step_block(&mut self, ids: &[u32; NT], pos: i64) -> Vec<f32> {
        self.check_ctx_span(pos, NT as i64, "speculative decode");
        if self.block_feats.is_none() {
            self.block_feats = Some(inkwell_cuda::DevBuf::new_bytes(4 * NT * 2304 * 4));
        }
        self.feat_ud.borrow_mut().clear();
        self.feat_ops.borrow_mut().clear();
        if self.conv_snap.len() < self.conv_state.len() {
            for s in &self.conv_state {
                let n = (t::ne(*s, 0) * t::ne(*s, 1)) as usize;
                self.conv_snap.push(inkwell_cuda::DevBuf::new_bytes(n * NT * 4));
            }
        }
        // invalidates any cached step graph: ctx_reset frees its tensors
        self.step_gcache = None;
        unsafe { inkwell_ggml::ink_sched_reset(self.sched.0) };
        if self.cur_ctx.0.is_null() {
            self.cur_ctx = Ptr(unsafe { inkwell_ggml::ink_ctx_new() });
        } else {
            unsafe { inkwell_ggml::ink_ctx_reset(self.cur_ctx.0) };
        }
        unsafe { *self.mix_prefix.get() = None };
        self.conv_arg_idx.set(0);
        let c = self.cur_ctx;
        let n_embd = self.cfg.n_embd as i64;

        let n_kv_full_raw = pos + NT as i64;
        let n_kv_swa_raw = self.swa_positions(n_kv_full_raw);
        let n_kv_full = kv_bucket(n_kv_full_raw, self.max_ctx);
        let n_kv_swa = kv_bucket(n_kv_swa_raw, self.swa_capacity());

        // per-token input tensors
        let mk_i32 = |v: &[i32]| -> Ptr {
            let t0 = t::new(c, I32, &[1]);
            unsafe { inkwell_ggml::ink_set_input(t0.0) };
            t0
        };
        let _ = mk_i32; // silence
        let mut step = StepBlk {
            kv_idx_full: [Ptr(std::ptr::null_mut()); NT],
            kv_idx_swa: [Ptr(std::ptr::null_mut()); NT],
            rel_pos_full: [Ptr(std::ptr::null_mut()); NT],
            rel_pos_swa: [Ptr(std::ptr::null_mut()); NT],
            mask_full: [Ptr(std::ptr::null_mut()); NT],
            mask_swa: [Ptr(std::ptr::null_mut()); NT],
            tau8: t::new(c, F32, &[NT as i64]),
            n_kv_full_pad: n_kv_full,
            n_kv_swa_pad: n_kv_swa,
            swa_wrap_hazard: self.swa_wrap_hazard(pos, NT as i64),
        };
        let embd8 = t::new(c, F32, &[n_embd, NT as i64]);
        unsafe { inkwell_ggml::ink_set_input(embd8.0) };
        unsafe { inkwell_ggml::ink_set_input(step.tau8.0) };
        let mut small: Vec<Ptr> = vec![embd8, step.tau8];
        let mut sets: Vec<(Ptr, Vec<i32>)> = Vec::new();
        let mut setf: Vec<(Ptr, Vec<f32>)> = Vec::new();
        for j in 0..NT {
            for (arr, v) in [(&mut step.kv_idx_full, (pos + j as i64) as i32),
                             (&mut step.kv_idx_swa, self.swa_kv_idx(pos + j as i64) as i32)] {
                let t0 = t::new(c, I32, &[1]);
                unsafe { inkwell_ggml::ink_set_input(t0.0) };
                sets.push((t0, vec![v]));
                arr[j] = t0;
                small.push(t0);
            }
            for is_swa in [true, false] {
                let ppos = pos + j as i64;
                let (nk, raw) = if is_swa { (n_kv_swa, self.swa_positions(ppos + 1)) }
                                else { (n_kv_full, ppos + 1) };
                // rel_pos[i] = distance to the key in slot i, clamped to extent
                let rp: Vec<i32> = (0..nk).map(|i| {
                    if i >= raw { 0 }
                    else if is_swa { self.swa_rel_pos(ppos, i) }
                    else { (raw - 1 - i).min(self.cfg.rel_extent as i64) as i32 }
                }).collect();
                let t_rp = t::new(c, I32, &[nk, 1]);
                let mk: Vec<f32> = (0..=nk).map(|i| {
                    let visible = if is_swa { self.swa_visible(ppos, i) } else { i < raw };
                    if i == nk || visible { 0.0 } else { f32::NEG_INFINITY }
                }).collect();
                let t_mk = t::new(c, F32, &[nk + 1, 1, 1, 1]);
                unsafe {
                    inkwell_ggml::ink_set_input(t_rp.0);
                    inkwell_ggml::ink_set_input(t_mk.0);
                }
                sets.push((t_rp, rp));
                setf.push((t_mk, mk));
                if is_swa { step.rel_pos_swa[j] = t_rp; step.mask_swa[j] = t_mk; }
                else { step.rel_pos_full[j] = t_rp; step.mask_full[j] = t_mk; }
                small.push(t_rp);
                small.push(t_mk);
            }
        }
        // assign all inputs into the device in_buf
        if self.in_buf.is_none() {
            self.in_buf = Some(self.rt.buf(self.rt.device_buft(), 8 << 20));
        }
        let buf = self.in_buf.as_ref().unwrap().clone();
        let mut o = 0usize;
        for x in &small {
            let off = (o + 255) & !255;
            buf.assign(*x, off);
            o = off + t::nbytes(*x);
        }
        let _ = buf;
        // now that every input is buffer-assigned, upload values
        for (t0, v) in &sets {
            unsafe { t::set(*t0, v.as_ptr().cast(), 0, v.len() * 4) };
        }
        for (t0, v) in &setf {
            unsafe { t::set(*t0, v.as_ptr().cast(), 0, v.len() * 4) };
        }
        // tau values + embd rows
        let tau: Vec<f32> = (0..NT).map(|j| {
            let p = (pos + j as i64 + 1) as f32;
            // serial: scale_bias(ln, alpha, 1) = alpha*ln + 1
            self.cfg.log_scale_alpha * (p / self.cfg.log_scale_floor as f32).max(1.0).ln() + 1.0
        }).collect();
        unsafe { t::set(step.tau8, tau.as_ptr().cast(), 0, tau.len() * 4) };
        {
            let (esrc, ty, stride) = self.embd_src;
            let mut rows = vec![0f32; 2304 * NT];
            for (j, id) in ids.iter().enumerate() {
                if ty == -2 {
                    let s = unsafe { esrc.add(*id as usize * stride) } as *const u16;
                    for (i, r) in rows[j * 2304..(j + 1) * 2304].iter_mut().enumerate() {
                        *r = f32::from_bits(u32::from(unsafe { *s.add(i) }) << 16);
                    }
                } else {
                    let rc = unsafe { inkwell_ggml::ink_dequant(ty, esrc.add(*id as usize * stride).cast(),
                        rows[j * 2304..].as_mut_ptr(), 2304) };
                    assert_eq!(rc, 0, "dequant embd row");
                }
            }
            unsafe { t::set(embd8, rows.as_ptr().cast(), 0, rows.len() * 4) };
        }
        let mut cur = self.norm(embd8, self.tok_norm);

        // driver
        let (start, end, np) = (6usize, 18usize, 3usize);
        let mut blocks: Vec<Ptr> = Vec::new();
        let mut vi = 0usize;
        for il in 0..start {
            self.feat_tap(&c, vi, cur);
            cur = self.layer_blk(il, il, self.visits[vi].1, 0, cur, &mut blocks, None, &self.layers[il], &step);
            vi += 1;
        }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        for pass in 0..np {
            let mut bank: Vec<Ptr> = Vec::new();
            let mut bank_out = Ptr(std::ptr::null_mut());
            if pass > 0 {
                bank = blocks[prelude_blocks.len() + 1..].to_vec();
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks.clone();
            }
            for ip in start..end {
                let (_il_v, slot) = self.visits[vi];
                self.feat_tap(&c, vi, cur);
                if pass == 0 {
                    cur = self.layer_blk(ip, ip, slot, pass, cur, &mut blocks, None, &self.layers[ip], &step);
                } else {
                    let l = &self.layers[ip];
                    let msgs = self.mix8(&bank, bank_out, l.msg_query.expect("msg_query"));
                    cur = self.layer_blk(ip, ip, slot, pass, cur, &mut blocks, Some(msgs), l, &step);
                }
                vi += 1;
            }
        }
        for il in end..24 {
            let (_il_v, slot) = self.visits[vi];
            self.feat_tap(&c, vi, cur);
            cur = self.layer_blk(il, il, slot, 0, cur, &mut blocks, None, &self.layers[il], &step);
            vi += 1;
        }
        debug_assert_eq!(vi, 48);

        cur = t::reshape2(c, self.mix8(&blocks, cur, self.output_res_query), n_embd, NT as i64);
        cur = self.norm(cur, self.output_norm);
        self.feat_tap(&c, 48, cur);
        let oi = self.output_exl3.expect("output exl3");
        let e = &*self.exl3_blk[oi];
        e.ensure_scratch();
        let mut args = [cur.0];
        let logits_t = Ptr(unsafe { inkwell_ggml::custom_4d(c.0, F32, [e.n as i64, NT as i64, 1, 1], &mut args,
            exl3_ntok_cb_addr(), (e as *const Exl3Blk).cast_mut().cast()) });
        let logits_t = t::cont(c, logits_t);
        if self.cfg.softcap != 0.0 {
            let cap = self.cfg.softcap;
            cur = t::scale(c, t::tanh(c, t::scale(c, logits_t, 1.0 / cap)), cap);
        } else {
            cur = logits_t;
        }
        t::output(cur);

        let g = Ptr(unsafe { inkwell_ggml::ink_graph_new(c.0, 32768) });
        t::build_forward(g, cur);
        for op in self.feat_ops.borrow().iter() { t::build_forward(g, *op); }
        let prof = std::env::var_os("INKWELL_PROF").is_some();
        let t0 = std::time::Instant::now();
        unsafe {
            assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, g.0), 0, "sched compute");
            self.dump_all();
        }
        if prof { eprintln!("PROF blk_compute={:.1}ms", t0.elapsed().as_secs_f64() * 1e3); }
        let mut logits = vec![0f32; self.cfg.vocab_unpadded as usize * NT];
        // logits tensor is [n_vocab_padded, NT]; read per token prefix
        let nv = t::ne(logits_t, 0) as usize;
        for j in 0..NT {
            unsafe {
                t::get(logits_t, logits[j * self.cfg.vocab_unpadded as usize..].as_mut_ptr().cast(),
                    j * nv * 4, self.cfg.vocab_unpadded as usize * 4);
            }
        }
        unsafe { inkwell_ggml::ink_sched_reset(self.sched.0) };
        logits
    }
}
/// `ggml_custom_op_t` for fused decode attention: src[0]=K view, src[1]=V view,
/// src[2]=q3 [hd,1,nh], src[3]=bias [n_kv,1,nh], src[4]=mask [n_kv+1]; dst=[hd*nh].
pub fn attn_cb_addr() -> usize { attn_cb as *const () as usize }

unsafe extern "C" fn attn_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe {
        inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
        let k_t = inkwell_ggml::ink_tensor_src(dst, 0);
        let v_t = inkwell_ggml::ink_tensor_src(dst, 1);
        let q_t = inkwell_ggml::ink_tensor_src(dst, 2);
        let b_t = inkwell_ggml::ink_tensor_src(dst, 3);
        let m_t = inkwell_ggml::ink_tensor_src(dst, 4);
        let hd = inkwell_ggml::ink_tensor_ne(k_t, 0);
        let n_kv = inkwell_ggml::ink_tensor_ne(k_t, 1);
        let hkv = inkwell_ggml::ink_tensor_ne(k_t, 2);
        let nh = inkwell_ggml::ink_tensor_ne(q_t, 2);
        // k/v strides in BYTES (kernel picks elem size by kv16); q/b stay f32 elems
        let kv16 = inkwell_ggml::ink_tensor_type(k_t) == 1;
        inkwell_cuda::attn_decode_async(
            inkwell_ggml::ink_tensor_data(k_t).cast_const(),
            inkwell_ggml::ink_tensor_data(v_t).cast_const(),
            inkwell_ggml::ink_tensor_data(q_t).cast_const(),
            inkwell_ggml::ink_tensor_data(b_t).cast_const(),
            inkwell_ggml::ink_tensor_data(m_t).cast_const(),
            inkwell_ggml::ink_tensor_data(dst).cast::<f32>(),
            nh, hkv, n_kv, hd,
            inkwell_ggml::ink_tensor_nb(k_t, 1), inkwell_ggml::ink_tensor_nb(k_t, 2),
            inkwell_ggml::ink_tensor_nb(v_t, 1), inkwell_ggml::ink_tensor_nb(v_t, 2),
            inkwell_ggml::ink_tensor_nb(q_t, 2) / 4, inkwell_ggml::ink_tensor_nb(b_t, 2) / 4,
            1.0 / (hd as f32).sqrt(),
            kv16,
        );
    }
}

struct AttnDynArgs {
    use_split: i32,
    allow_gqa: i32,
    chunk_len: i32,
    split_min: i64,
}

/// `ggml_custom_op_t` for shape-invariant fused decode attention.
/// src[0..4] mirror `attn_cb`; src[5] is the device exact-n_kv scalar.
pub fn attn_dyn_cb_addr() -> usize { attn_dyn_cb as *const () as usize }

unsafe extern "C" fn attn_dyn_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let a = unsafe { &*ud.cast::<AttnDynArgs>() };
    unsafe {
        inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
        let k_t = inkwell_ggml::ink_tensor_src(dst, 0);
        let v_t = inkwell_ggml::ink_tensor_src(dst, 1);
        let q_t = inkwell_ggml::ink_tensor_src(dst, 2);
        let b_t = inkwell_ggml::ink_tensor_src(dst, 3);
        let m_t = inkwell_ggml::ink_tensor_src(dst, 4);
        let nkv_t = inkwell_ggml::ink_tensor_src(dst, 5);
        let hd = inkwell_ggml::ink_tensor_ne(k_t, 0);
        let cap = inkwell_ggml::ink_tensor_ne(k_t, 1);
        let hkv = inkwell_ggml::ink_tensor_ne(k_t, 2);
        let nh = inkwell_ggml::ink_tensor_ne(q_t, 2);
        let kv16 = inkwell_ggml::ink_tensor_type(k_t) == 1;
        inkwell_cuda::attn_decode_dyn_async(
            inkwell_ggml::ink_tensor_data(k_t).cast_const(),
            inkwell_ggml::ink_tensor_data(v_t).cast_const(),
            inkwell_ggml::ink_tensor_data(q_t).cast_const(),
            inkwell_ggml::ink_tensor_data(b_t).cast_const(),
            inkwell_ggml::ink_tensor_data(m_t).cast_const(),
            inkwell_ggml::ink_tensor_data(dst).cast::<f32>(),
            nh, hkv, inkwell_ggml::ink_tensor_data(nkv_t).cast_const().cast(), hd, cap,
            inkwell_ggml::ink_tensor_nb(k_t, 1), inkwell_ggml::ink_tensor_nb(k_t, 2),
            inkwell_ggml::ink_tensor_nb(v_t, 1), inkwell_ggml::ink_tensor_nb(v_t, 2),
            inkwell_ggml::ink_tensor_nb(q_t, 2) / 4, inkwell_ggml::ink_tensor_nb(b_t, 2) / 4,
            1.0 / (hd as f32).sqrt(), kv16,
            a.use_split != 0, a.allow_gqa != 0, a.chunk_len, a.split_min,
        );
    }
}

/// `ggml_custom_op_t` for dense quantized GEMVs (Q8_0 → dp4a i8 path).
/// src[0] = weight (quantized, device), src[1] = x (f32, device), dst = y.
#[expect(clippy::cast_ptr_alignment)]
pub fn dense_cb_addr() -> usize { dense_cb as *const () as usize }

unsafe extern "C" fn dense_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, _ud: *mut std::ffi::c_void) {
    unsafe {
        inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
        let w_t = inkwell_ggml::ink_tensor_src(dst, 0);
        let x_t = inkwell_ggml::ink_tensor_src(dst, 1);
        let w = inkwell_ggml::ink_tensor_data(w_t);
        let x = inkwell_ggml::ink_tensor_data(x_t);
        let y = inkwell_ggml::ink_tensor_data(dst).cast::<f32>();
        let ncols = inkwell_ggml::ink_tensor_ne(w_t, 0) as usize;
        let nrows = inkwell_ggml::ink_tensor_ne(w_t, 1) as usize;
        match inkwell_ggml::ink_tensor_type(w_t) {
            13 => inkwell_cuda::mmvq_q5_k_i8_async(w.cast_const(), x.cast_const(), y, nrows, ncols),
            14 => inkwell_cuda::mmvq_q6_k_i8_async(w.cast_const(), x.cast_const(), y, nrows, ncols),
            _ => inkwell_cuda::mmvq_q8_0_i8_async(w.cast_const(), x.cast_const(), y, nrows, ncols),
        }
    }
}

// ============ chunked prefill driver (phase 2, direct row-major CUDA path) ============

#[derive(Debug)]
struct PfArena {
    buf: Option<inkwell_cuda::DevBuf>,
    exact_attn_ws: Option<inkwell_cuda::DevBuf>,
    base: *mut u8,
    c: usize,
    off: HashMap<&'static str, usize>,
    bytes: usize,
}

impl PfArena {
    fn layout(c: usize) -> (HashMap<&'static str, usize>, usize) {
        fn al(x: usize) -> usize { (x + 255) & !255 }
        let mut off = HashMap::new();
        let mut o = 0usize;
        macro_rules! put_seq {
            ($name:expr, $bytes:expr) => {{
                o = al(o);
                off.insert($name, o);
                o += $bytes;
            }};
        }
        macro_rules! put_abs {
            ($name:expr, $pos:expr) => {{ off.insert($name, $pos); }};
        }

        let h = c * 2304 * 4;
        put_seq!("cur", h);
        put_seq!("prelude", h);
        let prelude_pos = off["prelude"];
        put_abs!("head", prelude_pos);
        for name in ["blk0","blk1","blk2","blk3","bank","bankout"] {
            put_seq!(name, h);
        }
        put_seq!("router", c * 128 * 4);
        put_seq!("ids", c * 8 * 4);
        put_seq!("wts", c * 8 * 4);
        put_seq!("tokids", c * 4);
        put_seq!("sort", c * 8 * 8 * 4 + 8192);

        let union = al(o);
        macro_rules! put_rel {
            ($name:expr, $pos:expr) => {{ off.insert($name, union + $pos); }};
        }

        put_rel!("tmp2", 0);
        let mix_pos = h;
        put_rel!("mixbuf", mix_pos);
        let kb = c * 1024 * 4;
        put_rel!("k", mix_pos);
        put_rel!("v", mix_pos + kb);
        put_rel!("msgbuf", 4 * h);
        put_rel!("x0", 8 * h);
        put_rel!("x1", 9 * h);
        put_rel!("x2", 10 * h);

        let ctrl = 11 * h;
        put_rel!("cn", ctrl);
        put_rel!("cm", ctrl + h);
        let cat_pos = ctrl + 2 * h;
        let scratch_bytes = c * (3072 + 3072) * 4;
        let cat_scratch = 2 * h;
        put_rel!("cat", cat_pos);
        put_rel!("ctrlg", cat_pos);
        let cg_pos = cat_pos + cat_scratch;
        put_rel!("cg", cg_pos);
        let ctrl_end = cg_pos + h;

        let qb = c * 3072 * 4;
        let mut t = ctrl;
        macro_rules! rel_alloc {
            ($name:expr, $bytes:expr) => {{
                t = al(t);
                off.insert($name, union + t);
                t += $bytes;
            }};
        }
        let q_pos = t;
        rel_alloc!("q", qb);
        let gate_pos = t;
        rel_alloc!("gate", qb);
        rel_alloc!("attn", qb);
        rel_alloc!("r", c * 384 * 4);
        rel_alloc!("kout", kb);
        rel_alloc!("vout", kb);
        rel_alloc!("ksnap", c * 1024 * 2);
        rel_alloc!("vsnap", c * 1024 * 2);
        rel_alloc!("kdyn", c * 32 * 4);
        rel_alloc!("vdyn", c * 32 * 4);
        rel_alloc!("sink", c * 24 * 4);
        rel_alloc!("rel", 24 * 1025 * 16 * 4);
        let transient_end = al(t);
        put_rel!("scratch", transient_end);
        let scratch_end = transient_end + scratch_bytes;
        put_abs!("wo", union + q_pos);
        put_abs!("woout", union + gate_pos);

        let p = c * 8;
        let moews_bytes = p * (if pf_micro_enabled() { 9216usize } else { 6912usize }) * 4 + p * 4 + 256;
        put_rel!("moews", 0);
        let fcur_pos = al(moews_bytes);
        put_rel!("fcur", fcur_pos);
        let moeout_pos = fcur_pos + h;
        put_rel!("moe_out", moeout_pos);
        put_rel!("ffnout", 0);
        put_rel!("logits", 0);

        let union_need = std::cmp::max(moeout_pos + h, std::cmp::max(scratch_end, ctrl_end));
        o = union + union_need;
        (off, o)
    }

    fn new(c: usize) -> Self {
        let (off, bytes) = Self::layout(c);
        let buf = inkwell_cuda::DevBuf::new_bytes(bytes);
        let base = buf.dev_ptr().cast();
        Self { buf: Some(buf), exact_attn_ws: None, base, c, off, bytes }
    }
    /// Borrow an already allocated device buffer. The caller guarantees that
    /// it remains alive for the whole encode and is not otherwise in use.
    fn borrowed(c: usize, base: *mut u8) -> Self {
        let (off, bytes) = Self::layout(c);
        Self { buf: None, exact_attn_ws: None, base, c, off, bytes }
    }
    fn ptr(&self, name: &'static str) -> *mut f32 {
        unsafe { self.base.add(self.off[name]).cast() }
    }
}

struct PfExpertStageBuf {
    buf: inkwell_cuda::DevBuf,
    dg: inkwell_cuda::DevBuf,
    du: inkwell_cuda::DevBuf,
    dd: inkwell_cuda::DevBuf,
    reuse_event: *mut std::ffi::c_void,
    done_event: *mut std::ffi::c_void,
    layer: i32,
}

struct PfExpertStage {
    bufs: Vec<PfExpertStageBuf>,
    next: usize,
    table_keepalive: Vec<Vec<u64>>,
}

type PfMoeDescs = (*const std::ffi::c_void, *const std::ffi::c_void, *const std::ffi::c_void, *mut std::ffi::c_void);

impl PfExpertStage {
    fn new(bytes: usize) -> Option<Self> {
        let mut bufs = Vec::new();
        for _ in 0..2 {
            bufs.push(PfExpertStageBuf {
                buf: inkwell_cuda::DevBuf::try_new_bytes(bytes)?,
                dg: inkwell_cuda::DevBuf::new_bytes(4096),
                du: inkwell_cuda::DevBuf::new_bytes(4096),
                dd: inkwell_cuda::DevBuf::new_bytes(4096),
                reuse_event: inkwell_cuda::new_event(),
                done_event: inkwell_cuda::new_event(),
                layer: -1,
            });
        }
        Some(Self { bufs, next: 0, table_keepalive: Vec::new() })
    }
}

#[derive(Debug)]
struct PfMicroSlot {
    buf: inkwell_cuda::DevBuf,
    dg: inkwell_cuda::DevBuf,
    du: inkwell_cuda::DevBuf,
    dd: inkwell_cuda::DevBuf,
    reuse_event: *mut std::ffi::c_void,
    done_event: *mut std::ffi::c_void,
    layer: i32,
    batch: i32,
}

#[derive(Debug)]
struct PfMicroStage {
    bufs: Vec<PfMicroSlot>,
    batch_experts: usize,
    meta_event: *mut std::ffi::c_void,
    table_keepalive: Vec<Vec<u64>>,
}

impl PfMicroStage {
    fn new(bytes: usize, batch_experts: usize) -> Option<Self> {
        let mut bufs = Vec::new();
        for _ in 0..2 {
            bufs.push(PfMicroSlot {
                buf: inkwell_cuda::DevBuf::try_new_bytes(bytes)?,
                dg: inkwell_cuda::DevBuf::new_bytes(4096),
                du: inkwell_cuda::DevBuf::new_bytes(4096),
                dd: inkwell_cuda::DevBuf::new_bytes(4096),
                reuse_event: inkwell_cuda::new_event(),
                done_event: inkwell_cuda::new_event(),
                layer: -1,
                batch: -1,
            });
        }
        Some(Self { bufs, batch_experts, meta_event: inkwell_cuda::new_event(), table_keepalive: Vec::new() })
    }
}

fn pf_micro_enabled() -> bool {
    std::env::var("INKWELL_PF_MICRO_STAGING").ok().as_deref() != Some("0")
}

fn pf_expert_byte_range(m: &crate::moe::MoeArgsExl3, e0: usize, e1: usize) -> (usize, usize) {
    let base = m.expert_dev_base as u64;
    let begin_at = |e: usize| -> usize {
        let mut v = u64::MAX;
        for pi in 0..3 {
            for j in 0..3 {
                v = v.min(m.descs_host[pi * 512 + e * 4 + j]);
            }
        }
        (v - base) as usize
    };
    let start = if e0 == 0 { 0 } else { begin_at(e0) };
    let end = if e1 >= 128 { m.expert_total } else { begin_at(e1) };
    (start, end)
}

fn pf_micro_slot_bytes(r: &Runner, batch_experts: usize) -> usize {
    let mut out = 0usize;
    for m in &r.moe_args_exl3 {
        if m.on_dev || m.expert_host_base == 0 { continue; }
        for e0 in (0..128).step_by(batch_experts) {
            let e1 = (e0 + batch_experts).min(128);
            let (s0, s1) = pf_expert_byte_range(m, e0, e1);
            out = out.max(s1 - s0);
        }
    }
    out
}

fn pf_prepare_micro(r: &Runner, stage: &mut PfMicroStage, il: usize, batch: usize) -> Option<usize> {
    let m = pf_host_experts(r, il)?;
    let nb = 128usize.div_ceil(stage.batch_experts);
    if batch >= nb { return None; }
    let bi = batch & 1;
    if stage.bufs[bi].layer == il as i32 && stage.bufs[bi].batch == batch as i32 { return Some(bi); }
    let e0 = batch * stage.batch_experts;
    let e1 = (e0 + stage.batch_experts).min(128);
    let (start, end) = pf_expert_byte_range(m, e0, e1);
    let bytes = end - start;
    let b = &mut stage.bufs[bi];
    if bytes > b.buf.len * 4 { return None; }
    unsafe {
        if b.layer >= 0 { inkwell_cuda::side_wait_event(b.reuse_event); }
        inkwell_cuda::memcpy_h2d_async_side(b.buf.dev_ptr(),
            (m.expert_host_base + start) as *const std::ffi::c_void, bytes);
    }
    let stage_base = b.buf.dev_ptr() as u64;
    let copy_base = m.expert_dev_base as u64 + start as u64;
    let mut table = Vec::with_capacity(3 * 512);
    for pi in 0..3 {
        for e in 0..128 {
            for j in 0..4 {
                let v = m.descs_host[pi * 512 + e * 4 + j];
                table.push(if j < 3 && e >= e0 && e < e1 { v - copy_base + stage_base } else { v });
            }
        }
    }
    unsafe {
        for (pi, dst) in [&b.dg, &b.du, &b.dd].into_iter().enumerate() {
            let src = table.as_ptr().add(pi * 512).cast::<std::ffi::c_void>();
            inkwell_cuda::memcpy_h2d_async_side(dst.dev_ptr(), src, 4096);
        }
        inkwell_cuda::event_record(b.done_event, inkwell_cuda::side_stream_raw());
    }
    stage.table_keepalive.push(table);
    b.layer = il as i32;
    b.batch = batch as i32;
    Some(bi)
}

fn pf_stream_experts_enabled() -> bool {
    std::env::var("INKWELL_PF_STREAM_EXPERTS").ok().as_deref() != Some("0")
}

/// Releasing a decode expert cache wipes it, and the experts it held then have
/// to come back over PCIe during decode. Keep every cache the prefill arena
/// and expert stage do not need (killswitch: INKWELL_PF_KEEP_CACHE=0).
fn pf_keep_cache_enabled() -> bool {
    std::env::var("INKWELL_PF_KEEP_CACHE").ok().as_deref() != Some("0")
}

fn pf_host_experts(r: &Runner, il: usize) -> Option<&crate::moe::MoeArgsExl3> {
    let mi = r.layers[il].moe_exl3?;
    let m = &r.moe_args_exl3[mi];
    (!m.on_dev && m.expert_host_base != 0 && m.expert_total != 0).then_some(m)
}

/// Start an asynchronous side-stream copy of one host-backed layer into the
/// next double-buffer slot.  The reuse event is recorded before the caller
/// enqueues the current layer's compute, so the copy overlaps that compute but
/// cannot overwrite the slot before all previously enqueued consumers retire.
fn pf_prepare_experts(r: &Runner, stage: &mut PfExpertStage, il: usize) -> Option<usize> {
    let m = pf_host_experts(r, il)?;
    // Repeated loop layers revisit identical weights with only the other loop
    // layer interleaved.  If either double-buffer slot already holds this
    // layer, reuse it instead of streaming hundreds of MiB again.
    if let Some(hit) = stage.bufs.iter().position(|b| b.layer == il as i32) {
        return Some(hit);
    }
    let bi = stage.next;
    stage.next ^= 1;
    let b = &mut stage.bufs[bi];
    if m.expert_total > b.buf.len * 4 { return None; }
    unsafe {
        inkwell_cuda::event_record_current(b.reuse_event);
        inkwell_cuda::side_wait_event(b.reuse_event);
        inkwell_cuda::memcpy_h2d_async_side(b.buf.dev_ptr(), m.expert_host_base as *const std::ffi::c_void, m.expert_total);
    }
    let stage_base = b.buf.dev_ptr() as u64;
    let old_base = m.expert_dev_base as u64;
    let mut table = Vec::with_capacity(3 * 512);
    for pi in 0..3 {
        for e in 0..128 {
            for j in 0..4 {
                let v = m.descs_host[pi * 512 + e * 4 + j];
                table.push(if j < 3 { v - old_base + stage_base } else { v });
            }
        }
    }
    unsafe {
        for (pi, dst) in [&b.dg, &b.du, &b.dd].into_iter().enumerate() {
            let src = table.as_ptr().add(pi * 512).cast::<std::ffi::c_void>();
            inkwell_cuda::memcpy_h2d_async_side(dst.dev_ptr(), src, 4096);
        }
        inkwell_cuda::event_record(b.done_event, inkwell_cuda::side_stream_raw());
    }
    stage.table_keepalive.push(table);
    b.layer = il as i32;
    Some(bi)
}

fn pf_stage_ready(stage: &PfExpertStage, map: &[Option<usize>], pos: usize) -> Option<PfMoeDescs> {
    let bi = (*map.get(pos)?)?;
    let b = &stage.bufs[bi];
    Some((b.dg.dev_ptr() as *const std::ffi::c_void,
        b.du.dev_ptr() as *const std::ffi::c_void,
        b.dd.dev_ptr() as *const std::ffi::c_void,
        b.done_event))
}

fn tdata(x: Ptr) -> *mut f32 { unsafe { inkwell_ggml::ink_tensor_data(x.0).cast() } }
fn ttype(x: Ptr) -> i32 { unsafe { inkwell_ggml::ink_tensor_type(x.0) } }

fn pf_dense_exact_layout(w: Ptr, n: usize, k: usize) -> bool {
    if k & 1 != 0 { return false; }
    let ts: i64 = match ttype(w) {
        BF16 => 2,
        F32 => 4,
        _ => return false,
    };
    let pair = 2 * ts;
    unsafe {
        inkwell_ggml::ink_tensor_ne(w.0, 0) == k as i64
            && inkwell_ggml::ink_tensor_ne(w.0, 1) == n as i64
            && inkwell_ggml::ink_tensor_nb(w.0, 0) == ts
            && inkwell_ggml::ink_tensor_nb(w.0, 1) == k as i64 * ts
            && (inkwell_ggml::ink_tensor_data(w.0) as usize) % (pair as usize) == 0
            && inkwell_ggml::ink_tensor_nb(w.0, 2) % pair == 0
            && inkwell_ggml::ink_tensor_nb(w.0, 3) % pair == 0
    }
}

fn pf_copy(dst: *mut f32, src: *const f32, bytes: usize) {
    unsafe { inkwell_cuda::memcpy_d2d_async(dst.cast(), src.cast(), bytes) };
}
fn pf_exl3(e: &Exl3W, x: *const f32, y: *mut f32, c: usize, scratch: *mut f32) {
    unsafe {
        inkwell_cuda::exl3_gemm_smem_async(e.trel.dev_ptr().cast_const(), e.suh.dev_ptr().cast_const(),
            e.svh.dev_ptr().cast_const(), x, y, c as i32, e.k, e.n, e.ka, e.halfk, scratch);
    }
}

fn pf_exact_attn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("INKWELL_PF_EXACT_ATTN").is_ok_and(|v| v == "1"))
}

fn pf_dbg(name: &str, x: *const f32, n: usize) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("INKWELL_PFDBG").is_some()) { return; }
    let mut v = vec![0f32; n];
    unsafe {
        inkwell_cuda::memcpy_d2h_async(v.as_mut_ptr().cast(), x.cast(), n * 4);
        inkwell_cuda::DevBuf::sync();
    }
    eprintln!("PFDBG {name} sum={:.6} first8={:?}", v.iter().sum::<f32>(), &v[..8.min(n)]);
}

impl Runner {
    fn pf_norm(&self, x: *mut f32, w: Ptr, rows: usize, n: usize) {
        unsafe { inkwell_cuda::pf_rmsnorm_async(x, tdata(w).cast_const(), rows as i32, n as i32, 1e-6) };
    }
    fn pf_dense(&self, x: *const f32, w: Ptr, y: *mut f32, c: usize, n: usize, k: usize) {
        let bf = i32::from(ttype(w) == BF16);
        unsafe {
            if pf_dense_exact_layout(w, n, k) && (x as usize) % 8 == 0 {
                inkwell_cuda::pf_dense_gemm_exact_async(x, tdata(w).cast(), y, c as i32, n as i32, k as i32, bf);
            } else {
                inkwell_cuda::pf_dense_gemm_tiled_async(x, tdata(w).cast(), y, c as i32, n as i32, k as i32, bf);
            }
        }
    }
    fn pf_mix_run(&self, states: &[*const f32], query: Ptr, out: *mut f32, c: usize, qcols: usize) {
        let mut ps = [std::ptr::null::<std::ffi::c_void>(); 8];
        for (i, s) in states.iter().enumerate() { ps[i] = (*s).cast(); }
        unsafe {
            inkwell_cuda::pf_mix_async(ps.as_ptr(), states.len() as i32, tdata(query).cast_const(),
                out, c as i32, 2304, qcols as i32, 1.0 / (2304f32).sqrt(), 1e-6);
        }
    }
    #[expect(clippy::too_many_arguments)]
    fn pf_control(&self, a: &PfArena, cur: *mut f32, msg: *const f32, l: &L, ctrl: usize) {
        let c = a.c;
        let (n, h) = (2304usize, c * 2304);
        let (x0, x1, cat, gate, gain) =
            (a.ptr("cn"), a.ptr("cm"), a.ptr("cat"), a.ptr("cg"), a.ptr("ctrlg"));
        let dn = l.ctrl_down[ctrl].expect("ctrl_down");
        if let Some(gfcat) = l.ctrl_gf_cat[ctrl] {
            let [go, fo] = l.ctrl_gf_off[ctrl];
            let exact = unsafe {
                inkwell_ggml::ink_tensor_type(dn.0) == BF16
                    && inkwell_ggml::ink_tensor_type(gfcat.0) == BF16
                    && inkwell_ggml::ink_tensor_ne(dn.0, 0) == 2 * n as i64
                    && inkwell_ggml::ink_tensor_ne(dn.0, 1) == 64
                    && inkwell_ggml::ink_tensor_ne(gfcat.0, 0) == 64
                    && go == 0 && fo == n as i64
                    && inkwell_ggml::ink_tensor_ne(gfcat.0, 1) - fo == n as i64
            };
            if exact {
                unsafe {
                    inkwell_cuda::pf_ctrl_exact_async(cur.cast_const(), msg, tdata(dn).cast(),
                        tdata(gfcat).cast(), gate, cur, c as i32, n as i32, 64, 1e-6, fo as i32);
                }
                return;
            }
        }
        pf_copy(x0, cur, h * 4);
        pf_copy(x1, msg, h * 4);
        unsafe {
            inkwell_cuda::pf_rmsnorm_async(x0, std::ptr::null(), c as i32, n as i32, 1e-6);
            inkwell_cuda::pf_rmsnorm_async(x1, std::ptr::null(), c as i32, n as i32, 1e-6);
            for j in 0..c {
                inkwell_cuda::memcpy_d2d_async(cat.add(j * 2 * n).cast(), x0.add(j * n).cast_const().cast(), n * 4);
                inkwell_cuda::memcpy_d2d_async(cat.add(j * 2 * n + n).cast(), x1.add(j * n).cast_const().cast(), n * 4);
            }
        }
        self.pf_dense(cat.cast_const(), dn, gate, c, 64, 2 * n);
        if let Some(gfcat) = l.ctrl_gf_cat[ctrl] {
            self.pf_dense(gate.cast_const(), gfcat, gain, c, 2 * n, 64);
            unsafe { inkwell_cuda::pf_ctrl_apply_cat_async(gain.cast_const(), cur, msg, c as i32, n as i32) };
        } else {
            self.pf_dense(gate.cast_const(), l.ctrl_gain[ctrl].expect("ctrl_gain"), gain, c, n, 64);
            self.pf_dense(gate.cast_const(), l.ctrl_forget[ctrl].expect("ctrl_forget"), unsafe{gain.add(c*n)}, c, n, 64);
            unsafe { inkwell_cuda::pf_ctrl_apply_async(gain.cast_const(), unsafe{gain.add(c*n)}.cast_const(), cur, msg, (h) as i64) };
        }
    }

    #[expect(clippy::too_many_arguments)]
    fn pf_layer(&self, a: &PfArena, il: usize, ip: usize, slot: usize, pass: usize,
        blocks: &mut Vec<usize>, cur_name: &'static str, messages: bool, pos0: i64,
        moe_descs: Option<PfMoeDescs>, mut micro_stage: Option<&mut PfMicroStage>, next_host: Option<usize>) {
        let c = a.c;
        let l = &self.layers[il];
        let is_swa = self.is_swa_layer(il);
        let n = 2304usize;
        let hbytes = c * n * 4;
        let cur = a.ptr(cur_name);
        let inp_sa = a.ptr("tmp2");
        pf_copy(inp_sa, cur, hbytes);

        // Residual-stack attention queries (q/k/v input selectors).
        let mut inputs = [cur, cur, cur];
        if let Some(rq) = l.attn_res_query {
            let mut states = [std::ptr::null::<f32>(); 8];
            let mut ns = 0usize;
            for &b in blocks.iter().rev() {
                states[ns] = a.ptr(match b {0=>"blk0",1=>"blk1",2=>"blk2",_=>"blk3"}).cast_const();
                ns += 1;
            }
            states[ns] = cur.cast_const();
            ns += 1;
            self.pf_mix_run(&states[..ns], rq, a.ptr("mixbuf"), c, if is_swa {1} else {3});
            inputs = if is_swa { [a.ptr("mixbuf"), a.ptr("mixbuf"), a.ptr("mixbuf")] } else { [a.ptr("mixbuf"), unsafe{a.ptr("mixbuf").add(c*n)}, unsafe{a.ptr("mixbuf").add(2*c*n)}] };
        }
        // Loop-pass controller messages. The message tensor rows are materialized by mixbuf.
        if messages {
            let states = [a.ptr("bank").cast_const(), a.ptr("bankout").cast_const()];
            self.pf_mix_run(&states, l.msg_query.expect("msg_query"), a.ptr("msgbuf"), c, 4);
            for j in 0..3 {
                let dst = a.ptr(match j {0=>"x0",1=>"x1",_=>"x2"});
                pf_copy(dst, inputs[j], hbytes);
                self.pf_control(a, dst, unsafe{a.ptr("msgbuf").add(j*c*n)}.cast_const(), l, j);
                inputs[j] = dst;
            }
        }
        if il == 0 { pf_dbg("inputs0", inputs[0], 8.min(c*n).max(8)); }
        let (x0,x1,x2) = (a.ptr("x0"),a.ptr("x1"),a.ptr("x2"));
        if is_swa && messages {
            for (z,x) in [x0,x1,x2].into_iter().enumerate() { let _=z; self.pf_norm(x, l.attn_norm.expect("attn_norm"), c, n); inputs = match z {0=>[x,x1,x2],1=>[x0,x,x2],_=>[x0,x1,x]}; }
        } else if is_swa {
            pf_copy(x0, inputs[0], hbytes);
            self.pf_norm(x0, l.attn_norm.expect("attn_norm"), c, n);
            inputs = [x0,x0,x0];
        } else {
            for (j,x) in [x0,x1,x2].into_iter().enumerate() {
                pf_copy(x, inputs[j], hbytes);
                self.pf_norm(x, [l.attn_q_inp_norm,l.attn_k_inp_norm,l.attn_v_inp_norm][j].expect("full norm"), c, n);
                inputs[j] = x;
            }
        }

        let (q,gate,r,kbuf,vbuf,scratch) = (a.ptr("q"),a.ptr("gate"),a.ptr("r"),a.ptr("k"),a.ptr("v"),a.ptr("scratch"));
        for (key,x,y) in [("wq",inputs[0],q),("gate",inputs[0],gate),("rel",inputs[0],r),("wk",inputs[1],kbuf),("wv",inputs[2],vbuf)] {
            let e = &*self.exl3_w[l.exl3[key]];
            pf_exl3(e, x.cast_const(), y, c, scratch);
        }
        if !is_swa {
            self.pf_dense(inputs[1].cast_const(), l.attn_k_conv_proj.expect("kcp"), a.ptr("kdyn"), c, 32, n);
            self.pf_dense(inputs[2].cast_const(), l.attn_v_conv_proj.expect("vcp"), a.ptr("vdyn"), c, 32, n);
        }
        let scale_static = 1.0 / 5f32.sqrt();
        let scale_dyn = 1.0 / 3.0f32;
        let conv_state = unsafe { inkwell_ggml::ink_tensor_data(self.conv_state[slot].0).cast::<f32>() };
        unsafe {
            inkwell_cuda::conv_prefill_async(kbuf.cast_const(), conv_state, 0, tdata(l.attn_k_conv.expect("kconv")).cast_const(),
                if is_swa {std::ptr::null()} else {a.ptr("kdyn").cast_const()}, c as i32, 1024, if is_swa {0} else {8}, if is_swa {scale_static} else {scale_dyn}, a.ptr("kout"));
            inkwell_cuda::conv_prefill_async(vbuf.cast_const(), conv_state, 1024, tdata(l.attn_v_conv.expect("vconv")).cast_const(),
                if is_swa {std::ptr::null()} else {a.ptr("vdyn").cast_const()}, c as i32, 1024, if is_swa {0} else {8}, if is_swa {scale_static} else {scale_dyn}, a.ptr("vout"));
            inkwell_cuda::pf_clip_async(q, tdata(l.attn_q_clip.expect("qclip")).cast_const(), c as i32, 24, 128, self.cfg.qk_clip);
            inkwell_cuda::pf_clip_async(a.ptr("kout"), std::ptr::null(), c as i32, 8, 128, self.cfg.qk_clip);
            inkwell_cuda::pf_rmsnorm_async(a.ptr("vout"), std::ptr::null(), c as i32 * 8, 128, 1e-6);
            inkwell_cuda::pf_clip_async(r, tdata(l.attn_rel_clip.expect("rclip")).cast_const(), c as i32, 24, 16, self.cfg.rel_clip);
        }
        if il == 0 { pf_dbg("Qcur-0", q, 8); pf_dbg("Kcur-0", a.ptr("kout"), 8); pf_dbg("Vcur-0", a.ptr("vout"), 8); pf_dbg("Rcur-0", r, 8); }
        if !is_swa {
            unsafe { inkwell_cuda::pf_tau_qr_async(q, r, c as i32, self.cfg.log_scale_alpha, self.cfg.log_scale_floor as f32, pos0 as i32) };
        }
        self.pf_dense(inputs[0].cast_const(), l.attn_sink_proj.expect("sink"), a.ptr("sink"), c, 24, n);
        unsafe {
            // `attn_rel_bias_pad` is BF16 while the fused decode sink/relbias
            // path is active (m3a, default on). Reinterpreting those bytes as
            // FP32 silently corrupts every prefill attention bias, so dispatch
            // on the real tensor type.
            let rb = l.attn_rel_bias_pad.or(l.attn_rel_bias).expect("relbias");
            if inkwell_ggml::ink_tensor_type(rb.0) == BF16 {
                inkwell_cuda::pf_expand_rel_bf16_async(tdata(rb).cast(), a.ptr("rel"));
            } else {
                inkwell_cuda::pf_expand_rel_async(tdata(rb).cast_const(), a.ptr("rel"));
            }
            let kd = inkwell_ggml::ink_tensor_data(self.k_arena[slot].0);
            let vd = inkwell_ggml::ink_tensor_data(self.v_arena[slot].0);
            let cap = if is_swa { self.swa_capacity() } else { self.max_ctx };
            let swa = if is_swa { self.swa as i32 } else { 0 };
            let (snap_start, snap_len) = if is_swa {
                let start = (pos0 - self.swa).max(0) as i32;
                let end = pos0 as i32 + c as i32 - swa;
                (start, (end - start).max(0))
            } else { (0, 0) };
            if snap_len > 0 {
                inkwell_cuda::pf_snapshot_kv_async(kd.cast_const(), vd.cast_const(), a.ptr("ksnap").cast(), a.ptr("vsnap").cast(),
                    snap_start, snap_len, cap as i32, swa);
            }
            inkwell_cuda::pf_write_kv_async(a.ptr("kout").cast_const(), a.ptr("vout").cast_const(), kd, vd, c as i32, pos0 as i32, cap as i32, swa);
            if let Some(ws) = a.exact_attn_ws.as_ref().filter(|_| pf_exact_attn_enabled()) {
                inkwell_cuda::pf_exact_attn_gqa3_snap_async(kd.cast_const(), vd.cast_const(), q.cast_const(), r.cast_const(),
                    a.ptr("rel").cast_const(), a.ptr("sink").cast_const(), a.ptr("attn"), ws.dev_ptr().cast::<f32>(),
                    self.max_ctx as i32, c as i32, pos0 as i32, cap as i32, swa, 1.0 / (128f32).sqrt(),
                    a.ptr("ksnap").cast::<std::ffi::c_void>().cast_const(), a.ptr("vsnap").cast::<std::ffi::c_void>().cast_const(), snap_start, snap_len);
            } else {
                inkwell_cuda::flash_prefill_gqa3_snap_async(kd.cast_const(), vd.cast_const(), q.cast_const(), r.cast_const(),
                    a.ptr("rel").cast_const(), a.ptr("sink").cast_const(), a.ptr("attn"), c as i32, pos0 as i32,
                    cap as i32, swa, 1.0 / (128f32).sqrt(), a.ptr("ksnap").cast::<std::ffi::c_void>().cast_const(), a.ptr("vsnap").cast::<std::ffi::c_void>().cast_const(), snap_start, snap_len);
            }
            if il == 0 { pf_dbg("kqv_out-0", a.ptr("attn"), 8); }
            inkwell_cuda::pf_mul_sigmoid_async(a.ptr("attn"), gate.cast_const(), (c * 3072) as i64);
        }
        let ewo = &*self.exl3_w[l.exl3["wo"]];
        pf_exl3(ewo, a.ptr("attn").cast_const(), a.ptr("wo"), c, scratch);
        unsafe {
            inkwell_cuda::conv_prefill_async(a.ptr("wo").cast_const(), conv_state, 2048, tdata(l.attn_post_conv.expect("postconv")).cast_const(),
                std::ptr::null(), c as i32, 2304, 0, scale_static, a.ptr("woout"));
            if il == 0 { pf_dbg("attn_post-0", a.ptr("woout"), 8); }
        }
        if ip % (self.cfg.attn_res_block_size as usize) == 0 {
            let bi = blocks.len().min(3);
            pf_copy(a.ptr(match bi {0=>"blk0",1=>"blk1",2=>"blk2",_=>"blk3"}), inp_sa, hbytes);
            blocks.push(bi);
            pf_copy(cur, a.ptr("woout"), hbytes);
        } else {
            unsafe { inkwell_cuda::pf_add_async(a.ptr("woout"), inp_sa, (c*n) as i64) };
            pf_copy(cur, a.ptr("woout"), hbytes);
        }
        // FFN input mix/control/norm.
        let fcur = a.ptr("fcur");
        if let Some(fq) = l.ffn_res_query {
            let mut states = [std::ptr::null::<f32>(); 8];
            let mut ns = 0usize;
            for &b in blocks.iter().rev() {
                states[ns] = a.ptr(match b {0=>"blk0",1=>"blk1",2=>"blk2",_=>"blk3"}).cast_const();
                ns += 1;
            }
            states[ns] = cur.cast_const();
            ns += 1;
            self.pf_mix_run(&states[..ns], fq, fcur, c, 1);
        } else { pf_copy(fcur, cur, hbytes); }
        if messages {
            self.pf_control(a, fcur, unsafe{a.ptr("msgbuf").add(3*c*n)}.cast_const(), l, 3);
        }
        self.pf_norm(fcur, l.ffn_norm.expect("ffn_norm"), c, n);
        if il == 0 { pf_dbg("ffn_inp-0", cur, 8); pf_dbg("moe_in-0", fcur, 8); }
        self.pf_dense(fcur.cast_const(), l.ffn_gate_inp.expect("gate"), a.ptr("router"), c, 128, n);
        unsafe {
            inkwell_cuda::pf_router8_async(a.ptr("router").cast_const(), tdata(l.ffn_exp_probs_b.expect("pb")).cast_const(), c as i32,
                self.cfg.expert_weights_scale, 6.103515625e-5, a.ptr("ids").cast(), a.ptr("wts"));
        }
        let p = c * 8;
        let so = a.ptr("sort");
        let (eo, stok, sslot, sw, inv, te, ts, tc, tn) = unsafe {
            (so.cast::<i32>(), so.add(129).cast::<i32>(), so.add(129+p).cast::<i32>(), so.add(129+2*p).cast::<f32>().cast::<i32>(),
             so.add(129+3*p).cast::<i32>(), so.add(129+4*p).cast::<i32>(), so.add(129+5*p).cast::<i32>(), so.add(129+6*p).cast::<i32>(), so.add(129+7*p).cast::<i32>())
        };
        let mi = l.moe_exl3.expect("moe exl3");
        let m = &self.moe_args_exl3[mi];
        let (dg, du, dd, copy_done) = moe_descs.unwrap_or((m.descs_g.dev(), m.descs_u.dev(), m.descs_d.dev(), std::ptr::null_mut()));
        unsafe {
            if il == 0 { pf_dbg("router-0", a.ptr("router"), 8); }
            inkwell_cuda::exl3_moe_group_sort_async(a.ptr("ids").cast(), a.ptr("wts").cast(), c as i32,
                eo, stok, sslot, sw.cast(), inv, te, ts, tc, tn);
            let mut micro_used = false;
            if micro_stage.is_some() && pf_host_experts(self, il).is_some() {
                if let Some(st) = micro_stage.as_deref_mut() {
                    micro_used = true;
                    let max_tiles = p / 64 + 128;
                    let mut eo_h = [0i32; 129];
                    let mut te_h = vec![0i32; max_tiles];
                    let mut ts_h = vec![0i32; max_tiles];
                    let mut tc_h = vec![0i32; max_tiles];
                    let mut tn_h = [0i32; 1];
                    inkwell_cuda::memcpy_d2h_async(eo_h.as_mut_ptr().cast(), eo.cast_const().cast(), 129 * 4);
                    inkwell_cuda::memcpy_d2h_async(te_h.as_mut_ptr().cast(), te.cast_const().cast(), max_tiles * 4);
                    inkwell_cuda::memcpy_d2h_async(ts_h.as_mut_ptr().cast(), ts.cast_const().cast(), max_tiles * 4);
                    inkwell_cuda::memcpy_d2h_async(tc_h.as_mut_ptr().cast(), tc.cast_const().cast(), max_tiles * 4);
                    inkwell_cuda::memcpy_d2h_async(tn_h.as_mut_ptr().cast(), tn.cast_const().cast(), 4);
                    inkwell_cuda::event_record_current(st.meta_event);
                    inkwell_cuda::event_sync(st.meta_event);
                    let nt = tn_h[0] as usize;
                    let nb = 128usize.div_ceil(st.batch_experts);
                    // Seed both double-buffer slots before the loop.  Preparing
                    // batch 2 before batch 1 would otherwise put copy 2 ahead of
                    // copy 1 on the side stream and break the overlap pipeline.
                    for batch in 0..nb.min(2) {
                        let _ = pf_prepare_micro(self, st, il, batch);
                    }
                    for batch in 0..nb {
                        let e0 = batch * st.batch_experts;
                        let e1 = (e0 + st.batch_experts).min(128);
                        let row0 = eo_h[e0];
                        let rows = eo_h[e1] - row0;
                        let tile_base = (0..nt).position(|i| te_h[i] >= e0 as i32).unwrap_or(nt);
                        let mut tile_end = tile_base;
                        while tile_end < nt && te_h[tile_end] < e1 as i32 { tile_end += 1; }
                        let tile_batches = tile_end - tile_base;
                        let bi = pf_prepare_micro(self, st, il, batch).expect("micro expert batch stage");
                        let (bdg, bdu, bdd, done, reuse) = {
                            let sl = &st.bufs[bi];
                            (sl.dg.dev_ptr(), sl.du.dev_ptr(), sl.dd.dev_ptr(), sl.done_event, sl.reuse_event)
                        };
                        inkwell_cuda::stream_wait_current(done);
                        inkwell_cuda::exl3_moe_grouped_gemm_batch_async(bdg.cast(), bdu.cast(), bdd.cast(),
                            fcur.cast_const(), c as i32, row0, rows, tile_base as i32, tile_batches as i32,
                            i32::from(batch + 1 == nb), 1, stok, te, ts, tc, tn, sw.cast(), inv,
                            a.ptr("moews"), a.ptr("moe_out"));
                        inkwell_cuda::event_record_current(reuse);
                        if batch + 2 < nb {
                            let _ = pf_prepare_micro(self, st, il, batch + 2);
                        } else if let Some(nl) = next_host {
                            let _ = pf_prepare_micro(self, st, nl, batch + 2 - nb);
                        }
                    }
                }
            }
            if !micro_used {
                if !copy_done.is_null() { inkwell_cuda::stream_wait_current(copy_done); }
                inkwell_cuda::exl3_moe_grouped_gemm_async(dg, du, dd,
                    fcur.cast_const(), c as i32, stok, te, ts, tc, tn, sw.cast(), inv,
                    a.ptr("moews"), a.ptr("moe_out"));
            }
            inkwell_cuda::conv_prefill_async(a.ptr("moe_out").cast_const(), conv_state, 2048 + 2304,
                tdata(l.ffn_post_conv.expect("fpc")).cast_const(), std::ptr::null(), c as i32, 2304, 0, scale_static, a.ptr("ffnout"));
            if il == 0 { pf_dbg("ffn_out-0", a.ptr("ffnout"), 8); }
            inkwell_cuda::pf_add_async(a.ptr("ffnout"), cur, (c*n) as i64);
        }
        pf_copy(cur, a.ptr("ffnout"), hbytes);
        if matches!(il, 0|1|2|5|6|11|12|17|18|23) {
            let name = format!("l_out-{il}");
            pf_dbg(&name, cur, 8);
            if c > 1 {
                let last = format!("llast_out-{il}");
                pf_dbg(&last, unsafe { cur.add((c - 1) * n) }, 8);
            }
        }
    }

    /// Encode token ids in chunks through the batched prefill path. Returns the
    /// final row's unpadded logits. State writes match serial teacher forcing.
    pub fn prefill_chunked(&mut self, ids: &[u32], pos0: i64) -> Vec<f32> {
        self.check_ctx_span(pos0, ids.len() as i64, "prefill");
        let cmax = std::env::var("INKWELL_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(64).clamp(1, 1024);
        let need = PfArena::layout(cmax).1;
        let stream_experts = pf_stream_experts_enabled();
        let micro = pf_micro_enabled();
        let micro_batch = std::env::var("INKWELL_PF_MICRO_BATCH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(32).clamp(1, 128);
        let stage_max = if stream_experts {
            if micro {
                pf_micro_slot_bytes(self, micro_batch)
            } else {
                self.moe_args_exl3.iter()
                    .filter(|m| !m.on_dev && m.expert_host_base != 0)
                    .map(|m| m.expert_total).max().unwrap_or(0)
            }
        } else { 0 };
        // Reuse the largest decode expert-cache arena while encoding. The
        // direct driver reads expert weights from their resident/pinned source
        // instead of those caches; cache metadata is invalidated before decode.
        let borrow = if stage_max == 0 {
            self.moe_args_exl3.iter()
                .filter_map(|m| m.cache.as_ref().map(|c| (m.layer, c.buf.dev_ptr().cast::<u8>(), c.buf.len * 4)))
                .filter(|&(_, _, bytes)| bytes >= need)
                .max_by_key(|&(_, _, bytes)| bytes)
        } else { None };
        let mut released_caches: Vec<(i32, usize)> = Vec::new();
        let (mut a, borrowed_layer) = if let Some((layer, base, _)) = borrow {
            (PfArena::borrowed(cmax, base), Some(layer))
        } else {
            let mut candidates: Vec<(i32, usize)> = self.moe_args_exl3.iter()
                .filter_map(|m| m.cache.as_ref().map(|c| (m.layer, c.buf.len * 4)))
                .collect();
            candidates.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
            let n_cand = candidates.len();
            let total: usize = candidates.iter().map(|&(_, bytes)| bytes).sum();
            if stage_max > 0 || total >= need {
                // Old behaviour released every cache whenever the expert stage
                // was in play; that wiped the whole decode cache for a few
                // hundred MiB. Free only what the arena and stage need.
                let target = if pf_keep_cache_enabled() {
                    need.saturating_add(2 * stage_max)
                } else if stage_max == 0 {
                    need
                } else {
                    usize::MAX
                };
                let mut have = 0usize;
                for (layer, bytes) in candidates {
                    if have >= target { break; }
                    if let Some(m) = self.moe_args_exl3.iter_mut().find(|m| m.layer == layer) {
                        if let Some(c) = &mut m.cache {
                            let old = std::mem::replace(&mut c.buf, inkwell_cuda::DevBuf::new_bytes(8));
                            drop(old);
                            released_caches.push((layer, bytes));
                            have += bytes;
                        }
                    }
                }
                vlog!("prefill arena: released {} of {} caches, {} MiB (need {} MiB)",
                    released_caches.len(), n_cand, have >> 20, target >> 20);
            }
            (PfArena::new(cmax), None)
        };
        let mut expert_stage = if !micro && stage_max > 0 { PfExpertStage::new(stage_max) } else { None };
        let mut micro_stage = if micro && stage_max > 0 { PfMicroStage::new(stage_max, micro_batch) } else { None };
        if stage_max > 0 && expert_stage.is_none() && micro_stage.is_none() {
            vlog!("prefill experts: stage allocation {} MiB failed; using direct expert memory", stage_max >> 20);
        }
        if pf_exact_attn_enabled() {
            a.exact_attn_ws = Some(inkwell_cuda::DevBuf::new(24 * self.max_ctx as usize));
        }
        vlog!("prefill arena: {} MiB chunk={cmax}{}", a.bytes >> 20,
            borrowed_layer.map_or(String::new(), |l| format!(" borrowed=L{l}")));
        let mut pos = pos0;
        let mut out_logits = Vec::new();
        let seq: Vec<usize> = (0..6)
            .chain((0..3).flat_map(|_| 6..18))
            .chain(18..24)
            .collect();
        let host_positions: Vec<usize> = seq.iter().enumerate()
            .filter_map(|(pos, &il)| pf_host_experts(self, il).map(|_| pos))
            .collect();
        for chunk in ids.chunks(cmax) {
            let c = chunk.len();
            a.c = c;
            let host_ids: Vec<i32> = chunk.iter().map(|&x| x as i32).collect();
            unsafe {
                inkwell_cuda::memcpy_h2d_async(a.ptr("tokids").cast(), host_ids.as_ptr().cast(), c*4);
                let embd_dev: *const u8 = if self.embd_dev.is_null() { self.embd_src.0 } else { self.embd_dev.cast() };
                inkwell_cuda::pf_embed_async(embd_dev.cast(), a.ptr("tokids").cast(), a.ptr("cur"), c as i32, 2304, self.embd_src.2 as i64);
                inkwell_cuda::pf_rmsnorm_async(a.ptr("cur"), tdata(self.tok_norm).cast_const(), c as i32, 2304, 1e-6);
            }
            let mut blocks: Vec<usize> = Vec::new();
            let mut vi = 0usize;
            let mut staged_pos = vec![None; seq.len()];
            let mut seq_pos = 0usize;
            let mut prep_idx = 0usize;
            if !micro {
                if let Some(st) = expert_stage.as_mut() {
                    if let Some(&hp) = host_positions.first() {
                        if let Some(bi) = pf_prepare_experts(self, st, seq[hp]) {
                            staged_pos[hp] = Some(bi);
                        }
                        prep_idx = 1;
                    }
                }
            }
            for il in 0..6 {
                let md = expert_stage.as_ref().and_then(|st| pf_stage_ready(st, &staged_pos, seq_pos));
                if host_positions.binary_search(&seq_pos).is_ok() && prep_idx < host_positions.len() {
                    let hp = host_positions[prep_idx];
                    prep_idx += 1;
                    if !micro {
                        if let Some(st) = expert_stage.as_mut() {
                            if let Some(bi) = pf_prepare_experts(self, st, seq[hp]) {
                                staged_pos[hp] = Some(bi);
                            }
                        }
                    }
                }
                let next_host = host_positions.iter().copied().find(|&hp| hp > seq_pos).map(|hp| seq[hp]);
                self.pf_layer(&a, il, il, self.visits[vi].1, 0, &mut blocks, "cur", false, pos, md, micro_stage.as_mut(), next_host);
                vi += 1;
                seq_pos += 1;
            }
            pf_copy(a.ptr("prelude"), a.ptr("cur"), c*2304*4);
            let prelude_blocks = blocks.clone();
            for pass in 0..3 {
                if pass > 0 {
                    // The loop message bank is the layer-12 residual checkpoint of the
                    // previous pass; bank_out is that pass's final loop output.
                    pf_copy(a.ptr("bank"), a.ptr("blk2"), c*2304*4);
                    pf_copy(a.ptr("bankout"), a.ptr("cur"), c*2304*4);
                    pf_copy(a.ptr("cur"), a.ptr("prelude"), c*2304*4);
                    blocks = prelude_blocks.clone();
                }
                for ip in 6..18 {
                    let md = expert_stage.as_ref().and_then(|st| pf_stage_ready(st, &staged_pos, seq_pos));
                    if host_positions.binary_search(&seq_pos).is_ok() && prep_idx < host_positions.len() {
                        let hp = host_positions[prep_idx];
                        prep_idx += 1;
                        if !micro {
                            if let Some(st) = expert_stage.as_mut() {
                                if let Some(bi) = pf_prepare_experts(self, st, seq[hp]) {
                                    staged_pos[hp] = Some(bi);
                                }
                            }
                        }
                    }
                    let next_host = host_positions.iter().copied().find(|&hp| hp > seq_pos).map(|hp| seq[hp]);
                    self.pf_layer(&a, ip, ip, self.visits[vi].1, pass, &mut blocks, "cur", pass > 0, pos, md, micro_stage.as_mut(), next_host);
                    vi += 1;
                    seq_pos += 1;
                }
            }
            for il in 18..24 {
                let md = expert_stage.as_ref().and_then(|st| pf_stage_ready(st, &staged_pos, seq_pos));
                if host_positions.binary_search(&seq_pos).is_ok() && prep_idx < host_positions.len() {
                    let hp = host_positions[prep_idx];
                    prep_idx += 1;
                    if !micro {
                        if let Some(st) = expert_stage.as_mut() {
                            if let Some(bi) = pf_prepare_experts(self, st, seq[hp]) {
                                staged_pos[hp] = Some(bi);
                            }
                        }
                    }
                }
                let next_host = host_positions.iter().copied().find(|&hp| hp > seq_pos).map(|hp| seq[hp]);
                self.pf_layer(&a, il, il, self.visits[vi].1, 0, &mut blocks, "cur", false, pos, md, micro_stage.as_mut(), next_host);
                vi += 1;
                seq_pos += 1;
            }
            // final mix + norm; compute lm_head only for the last row of the final chunk.
            let is_last_chunk = pos + c as i64 == pos0 + ids.len() as i64;
            if is_last_chunk {
            let mut states = [std::ptr::null::<f32>(); 8];
            let mut ns = 0usize;
            for &b in blocks.iter().rev() {
                states[ns] = a.ptr(match b {0=>"blk0",1=>"blk1",2=>"blk2",_=>"blk3"}).cast_const();
                ns += 1;
            }
            states[ns] = a.ptr("cur").cast_const();
            ns += 1;
            self.pf_mix_run(&states[..ns], self.output_res_query, a.ptr("head"), c, 1);
            self.pf_norm(a.ptr("head"), self.output_norm, c, 2304);
            let last = unsafe { a.ptr("head").add((c-1)*2304) };
            if let Some(oi) = self.output_exl3 {
                let e = &*self.exl3_w[oi];
                pf_exl3(e, last.cast_const(), a.ptr("logits"), 1, a.ptr("scratch"));
                let mut logits = vec![0f32; self.cfg.vocab_unpadded as usize];
                if self.cfg.softcap != 0.0 {
                    unsafe { inkwell_cuda::pf_softcap_async(a.ptr("logits"), logits.len() as i32, self.cfg.softcap) };
                }
                unsafe { inkwell_cuda::memcpy_d2h_sync(logits.as_mut_ptr().cast(), a.ptr("logits").cast_const().cast(), logits.len()*4) };
                out_logits = logits;
            }
            }
            pos += c as i64;
        }
        inkwell_cuda::DevBuf::sync();
        if let Some(layer) = borrowed_layer {
            if let Some(m) = self.moe_args_exl3.iter_mut().find(|m| m.layer == layer) {
                if let Some(c) = &mut m.cache {
                    c.tag.write_bytes(&vec![0xFFu8; c.k as usize * 4]);
                    c.cnt.write_bytes(&vec![0u8; 128 * 4]);
                    c.hm.write_bytes(&[0u8; 8]);
                    if let Some(soff) = &mut c.soff {
                        soff.write_bytes(&vec![0u8; 3 * c.k as usize * 8]);
                    }
                    if let Some(fl) = &mut c.fl {
                        let mut fl_h = vec![0i32; 513];
                        fl_h[0] = 1;
                        fl_h[257] = (c.buf.len * 4 / 6912) as i32;
                        fl.write_bytes(unsafe { std::slice::from_raw_parts(fl_h.as_ptr().cast::<u8>(), 513 * 4) });
                    }
                }
            }
            vlog!("prefill arena: invalidated borrowed cache L{layer}");
        }
        if !released_caches.is_empty() {
            drop(a);
            drop(expert_stage);
            for (layer, bytes) in released_caches {
                if let Some(m) = self.moe_args_exl3.iter_mut().find(|m| m.layer == layer) {
                    if let Some(c) = &mut m.cache {
                        let old = std::mem::replace(&mut c.buf, inkwell_cuda::DevBuf::new_bytes(bytes));
                        drop(old);
                        c.tag.write_bytes(&vec![0xFFu8; c.k as usize * 4]);
                        c.cnt.write_bytes(&vec![0u8; 128 * 4]);
                        c.hm.write_bytes(&[0u8; 8]);
                        if let Some(soff) = &mut c.soff {
                            soff.write_bytes(&vec![0u8; 3 * c.k as usize * 8]);
                        }
                        if let Some(fl) = &mut c.fl {
                            let mut fl_h = vec![0i32; 513];
                            fl_h[0] = 1;
                            fl_h[257] = (bytes / 6912) as i32;
                            fl.write_bytes(unsafe { std::slice::from_raw_parts(fl_h.as_ptr().cast::<u8>(), 513 * 4) });
                        }
                    }
                }
                vlog!("prefill arena: restored released cache L{layer}");
            }
        }
        out_logits
    }
}
