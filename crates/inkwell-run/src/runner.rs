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
use inkwell_st as _st;
use inkwell_core::InklingConfig;
use std::collections::HashMap;

const F32: i32 = 0;
const F16: i32 = 1;
const I32: i32 = 26;
const BF16: i32 = 30;

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
    tok_embd: Ptr, embd_src: (*const u8, i32, usize), embd_row: Ptr, tok_norm: Ptr, output: Ptr, output_norm: Ptr, output_res_query: Ptr,
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
    /// device offsets of the 9 input tensors inside `in_buf` + span bytes
    in_offs: [usize; 9],
    in_span: usize,
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
    scratch: inkwell_cuda::DevBuf,
}

/// `ggml_custom_op_t`: batched EXL3 GEMV — `dst->src[0]` = x `[k, 8]` f32 device.
unsafe extern "C" fn exl3_ntok_cb(dst: *mut std::ffi::c_void, _ith: i32, _nth: i32, ud: *mut std::ffi::c_void) {
    let w = unsafe { &*ud.cast::<Exl3Blk>() };
    unsafe { inkwell_cuda::set_stream(inkwell_ggml::cuda_stream()) };
    let x = unsafe { inkwell_ggml::ink_tensor_src(dst, 0) };
    let x_dev = unsafe { inkwell_ggml::ink_tensor_data(x) } as *const f32;
    let y = unsafe { inkwell_ggml::ink_tensor_data(dst) } as *mut f32;
    unsafe {
        inkwell_cuda::exl3_gemv_ntok_async(w.trel, w.suh, w.svh,
            x_dev, y, w.k, w.n, w.ka, w.halfk, w.scratch.dev_ptr().cast());
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

/// bf16 bytes -> f32 bytes.
fn bf16_to_f32(d: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; d.len() * 2];
    for (i, ch) in d.chunks_exact(2).enumerate() {
        let bits = u32::from(ch[0]) | (u32::from(ch[1]) << 8);
        out[i * 4..i * 4 + 4].copy_from_slice(&(bits << 16).to_le_bytes());
    }
    out
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
        eprintln!("MEM after weights: {f1}");
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
            if i == 0 { eprintln!("VRAM cache: vfree={vfree} k={k} experts/layer ({:.1} MiB/layer)", k as f64 * (gs + ds) as f64 / 1e6); }
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
            let cap = if full { max_ctx } else { swa };
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
            tok_embd, embd_src, embd_row, tok_norm, output, output_norm, output_res_query, out_head,
            k_arena, v_arena, conv_state, conv_snap: Vec::new(), block_feats: None, feat_ud: std::cell::RefCell::new(Vec::new()), feat_ops: std::cell::RefCell::new(Vec::new()), max_ctx, swa, visits,
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
        let mut dev_bytes = 160usize << 20;
        for (n, t) in &repo.tensors {
            let (name, ti) = (n.as_str(), &t.1);
            if name.contains(".experts.") || name.contains(".trellis") || name.contains(".suh")
                || name.contains(".svh") || name.contains(".mul1") || name == "model.llm.embed.weight"
                || name.contains("visual") { continue; }
            let w = if bf16_ok(name) && ti.dtype == "BF16" { 2 } else { 4 };
            dev_bytes += (ti.elems() * w) as usize + 256;
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
                let ti = tp($name);
                let mut dims: Vec<i64> = ti.shape.iter().rev().map(|&d| d as i64).collect();
                dims.retain(|&d| d != 1);
                let bytes = if ti.dtype == "BF16" { bf16_to_f32(dtp($name)) }
                    else { assert_eq!(ti.dtype, "F32"); dtp($name).to_vec() };
                let ten = t::new(rt.ctx(), F32, &dims);
                let o = (off_dev + 127) & !127;
                dev_buf.assign(ten, o);
                unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                off_dev = o + t::nbytes(ten);
                ten
            }};
        }

        /// upload a mul_mat weight: keep the file's BF16 bytes when enabled
        /// (ggml-cuda mmvf reads BF16 src0 with identical F32 accumulation
        /// order on sm_61); widen to F32 otherwise.
        macro_rules! densew {
            ($name:expr) => {{
                let ti = tp($name);
                let mut dims: Vec<i64> = ti.shape.iter().rev().map(|&d| d as i64).collect();
                dims.retain(|&d| d != 1);
                let (ty, bytes) = if bf16_ok($name) && ti.dtype == "BF16" {
                    (BF16, dtp($name).to_vec())
                } else if ti.dtype == "BF16" {
                    (F32, bf16_to_f32(dtp($name)))
                } else { assert_eq!(ti.dtype, "F32"); (F32, dtp($name).to_vec()) };
                let ten = t::new(rt.ctx(), ty, &dims);
                let o = (off_dev + 127) & !127;
                dev_buf.assign(ten, o);
                unsafe { t::set(ten, bytes.as_ptr(), 0, bytes.len()) };
                off_dev = o + t::nbytes(ten);
                ten
            }};
        }

        /// build an Exl3W from `<base>.{trellis,suh,svh}` on device
        fn exl3_of(repo: &inkwell_st::StRepo, base: &str) -> Box<Exl3W> {
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
        let embd_src: (*const u8, i32, usize) = (dtp("model.llm.embed.weight").as_ptr(), -2, 2304 * 2);
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
            if let Ok(spec) = std::env::var("INKWELL_DEV_LAYERS") {
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
            eprintln!("place: on_dev={on:?} experts={mib}MiB budget={}MiB",
                budget.map_or("inf".to_string(), |b| (b >> 20).to_string()));
            Some(plan)
        };

        // ---- per-layer ----
        let mut layers = Vec::with_capacity(24);
        let mut pinned: Vec<inkwell_cuda::PinnedBuf> = Vec::new();
        let mut pinned_bytes = 0usize;
        let mut expert_dev: Vec<inkwell_cuda::DevBuf> = Vec::new();
        let mut moe_args_exl3: Vec<Box<crate::moe::MoeArgsExl3>> = Vec::new();
        let mut moe_gw: Vec<[u32; 384]> = Vec::new();
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
                    let np = t::new(rt.ctx(), F32, &[n0 as i64, n1 as i64 + 1, 1, 1]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(np, o);
                    let bytes = bf16_to_f32_t(dtp(&lp(il, "attn.rel_logits_proj.proj")), &[16, 1024]);
                    let mut dst = vec![0u8; n0 * (n1 + 1) * 4];
                    dst[..n0 * n1 * 4].copy_from_slice(&bytes);
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
                    let ten = t::new(rt.ctx(), F32, &[2304, 4]);
                    let o = (off_dev + 127) & !127;
                    dev_buf.assign(ten, o);
                    // cols [q[2i], kv[2i], kv[2i+1], q[2i+1]] contiguous col-major
                    let mut buf = Vec::with_capacity(4 * 2304 * 4);
                    buf.extend_from_slice(row(mcq, 2 * i));
                    buf.extend_from_slice(row(mckv, 2 * i));
                    buf.extend_from_slice(row(mckv, 2 * i + 1));
                    buf.extend_from_slice(row(mcq, 2 * i + 1));
                    unsafe { t::set(ten, buf.as_ptr(), 0, buf.len()) };
                    off_dev = o + t::nbytes(ten);
                    ten
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
                    l.ctrl_gain[ci] = Some(densew!(&lp(il, &format!("controllers.{hn}.gain_weight"))));
                    l.ctrl_forget[ci] = Some(densew!(&lp(il, &format!("controllers.{hn}.forget_weight"))));
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
                if il == 6 { let (f0, _) = inkwell_cuda::mem_info(); eprintln!("MEM before experts: {f0}"); }
                let mut host = vec![0u8; total];
                let mut descs: Vec<u64> = vec![0; 3 * 128 * 4];
                let mut off = 0usize;
                let mut max_slot = 0usize;
                let mut gxp = [0u32; 384];
                for e in 0..128usize {
                    for (pi, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                        let base = format!("{pre}.{e}.{proj}");
                        for (ai, part) in ["trellis", "suh", "svh"].iter().enumerate() {
                            let b = dtp(&format!("{base}.{part}"));
                            host[off..off + b.len()].copy_from_slice(b);
                            unsafe { madv::dontneed(b.as_ptr(), b.len()) };
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
                eprintln!("experts il={il}: free={}MiB total={}MiB", f_free >> 20, total >> 20);
                let dev_margin: usize = std::env::var("INKWELL_DEV_MARGIN_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(1300);
                let on_dev = place_plan.as_ref().map_or(true, |p| p[il])
                    && (f_free as usize) > total + (dev_margin << 20);
                let base = if on_dev {
                    let mut blob = inkwell_cuda::DevBuf::new_bytes(total);
                    blob.write_bytes(&host);
                    drop(host);
                    let b = blob.dev_ptr() as u64;
                    expert_dev.push(blob);
                    b
                } else {
                    let pb = inkwell_cuda::host_alloc_mapped(total);
                    unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), pb.host, total) };
                    pinned_bytes += total;
                    let b = pb.dev as u64;
                    pinned.push(pb);
                    b
                };
                for d in descs.chunks_exact_mut(4) {
                    d[0] += base; d[1] += base; d[2] += base;
                }
                let v2_descs_ok = crate::moe::exl3_descs_v2_ok(&descs);
                let mk = |pi: usize| {
                    crate::moe::DescTable::from_bytes(unsafe {
                        std::slice::from_raw_parts(descs[pi * 512..].as_ptr().cast::<u8>(), 128 * 32)
                    })
                };
                l.moe_exl3 = Some(moe_args_exl3.len());
                moe_args_exl3.push(Box::new(crate::moe::MoeArgsExl3 {
                    descs_g: mk(0), descs_u: mk(1), descs_d: mk(2),
                    v2_descs_ok, on_dev, cache: None, max_slot,
                    layer: moe_args_exl3.len() as i32,
                }));
            }
            l.proj0_off = [-1; 3];
            layers.push(l);
        }
        eprintln!("load_st: dev={}MiB exl3_w={} pinned={}MiB", dev_bytes >> 20, exl3_w.len(),
            pinned_bytes >> 20);
        let exl3_blk: Vec<Box<Exl3Blk>> = exl3_w.iter().map(|e| Box::new(Exl3Blk {
            trel: e.trel.dev_ptr(), suh: e.suh.dev_ptr(), svh: e.svh.dev_ptr(),
            k: e.k, n: e.n, ka: e.ka, halfk: e.halfk,
            scratch: inkwell_cuda::DevBuf::new(8 * e.k + 64 * e.n),
        })).collect();

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
        let n_slots = nphys + (np - 1) * (end - start);
        let mut k_arena = Vec::new();
        let mut v_arena = Vec::new();
        let mut conv_state = Vec::new();
        let n_sconv = (2 * hd * hkv + 2 * 2304) as usize;
        for slot in 0..n_slots {
            let il_phys = if slot < nphys { slot } else { start + (slot - nphys) % (end - start) };
            let full = il_phys % 6 == 5;
            let cap = if full { max_ctx } else { swa };
            k_arena.push(t::new(rt.ctx(), F16, &[hd, cap, hkv]));
            v_arena.push(t::new(rt.ctx(), F16, &[hd, cap, hkv]));
            conv_state.push(t::new(rt.ctx(), F32, &[3, n_sconv as i64]));
        }
        let kv_bytes: usize = k_arena.iter().chain(v_arena.iter()).map(|x| t::nbytes(*x) + 256).sum::<usize>()
            + conv_state.iter().map(|x| t::nbytes(*x) + 256).sum::<usize>();
        let kv_buf = rt.buf(buft_dev, kv_bytes);
        let mut o = 0usize;
        for x in k_arena.iter().chain(v_arena.iter()).chain(conv_state.iter()) {
            o = (o + 127) & !127;
            kv_buf.assign(*x, o);
            o += t::nbytes(*x);
        }
        eprintln!("bufs: dev={}MiB kv={}MiB hd={hd} hkv={hkv} swa={swa} n_slots={n_slots}",
            dev_bytes >> 20, kv_bytes >> 20);
        std::mem::forget(kv_buf);

        crate::moe::warm_staging(); // staging buffers must exist before CUDA-graph capture
        // Pinned layers stage expert blobs over PCIe every visit (~25-40 ms/tok
        // total). Cache k experts/layer in VRAM; k must be ≥ N_SLOTS(8) so a
        // single visit never evicts its own slots. Runs post-load so f_free is
        // the real remainder: dev_buf, kv_buf and staging already exist — only
        // CUDA-graph capture + desktop fluctuation still need room.
        let ck: usize = std::env::var("INKWELL_EXL3_CACHE_K")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(25);
        let kslot = std::env::var_os("INKWELL_NO_KSLOT").is_none();
        if ck >= 8 {
            // Per-layer arena MiB, measured by replaying the real routing
            // trace through the kslot policy (opt/ledger.md). Loop layers
            // L16/L17 hold ~half the miss bytes at uniform sizing.
            // INKWELL_NO_CACHEL=1 restores uniform ck*3*max_slot;
            // INKWELL_CACHE_MB_L="16:150,17:200" overrides per layer.
            static CACHEL_MB: [usize; 24] = [
                56, 56, 56, 48, 56, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                156, 156, 48, 44, 48, 52, 48, 52];
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
                CACHEL_MB.get(il as usize).copied().filter(|&m| m > 0)
            };
            for (i, a) in moe_args_exl3.iter_mut().enumerate() {
                if a.on_dev || a.max_slot == 0 { continue; }
                let cbytes = per_l(a.layer).map_or(ck * 3 * a.max_slot, |mb| mb << 20);
                let (f, _) = inkwell_cuda::mem_info();
                let margin: usize = std::env::var("INKWELL_CACHE_MARGIN_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
                if f <= cbytes + (margin << 20) { continue; }
                let Some(buf) = inkwell_cuda::DevBuf::try_new_bytes(cbytes) else { continue };
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
                    eprintln!("kslot L{}: arena={} gran kmax={} (cap {}/{}MiB)",
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
                    eprintln!("KSIZES L{} max={}KiB sum_max={:.1}MiB sum_esz={:.1}MiB k_eff={:.1} [{}]",
                        a.layer, a.max_slot >> 10,
                        128.0 * a.max_slot as f64 / 1048576.0, sum as f64 / 1048576.0,
                        ck as f64 * a.max_slot as f64 * 128.0 / sum as f64,
                        dist.join(" "));
                }
                eprintln!("exl3 cache moe={i}: k={ck} slot={}B ({:.0} MiB) CACHED free={}MiB",
                    a.max_slot, cbytes as f64 / 1e6, f >> 20);
            }
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
            tok_embd, embd_src, embd_row, tok_norm, output, output_norm, output_res_query, out_head,
            k_arena, v_arena, conv_state, conv_snap: Vec::new(), block_feats: None, feat_ud: std::cell::RefCell::new(Vec::new()), feat_ops: std::cell::RefCell::new(Vec::new()), max_ctx, swa, visits, n_full,
            step_gcache: None, sid: None,
        }
    }
}

// ============ forward ============

impl Runner {
    /// layer helpers
    fn is_swa_layer(&self, il: usize) -> bool { il % 6 != 5 }

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
    fn dump_list_get(&self) -> &Vec<(String, Ptr)> { unsafe { &*self.dump_list.get() } }
    fn dump_all(&self) {
        for (name, t0) in unsafe { &*self.dump_list.get() } {
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
        let q_cols = t::ne(query, 1);
        if n_rows <= 64 && q_cols <= 8 && std::env::var_os("INKWELL_NO_MIXFUSE").is_none() {
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
        let (q, gate_lin, r_pre, wk_p, wv_p) = if mz_ok {
            let mut ws = [std::ptr::null::<Exl3W>(); 6];
            let mut n_max = 0i64;
            for (z, k) in mz_keys.iter().enumerate() {
                let e: &Exl3W = &*self.exl3_w[l.exl3[k]];
                ws[z] = e;
                n_max = n_max.max(e.n as i64);
            }
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
        let k_dyn = if is_swa { None } else { Some(t::mmq(c, l.attn_k_conv_proj.expect("kcp"), inputs[1])) };
        let v_dyn = if is_swa { None } else { Some(t::mmq(c, l.attn_v_conv_proj.expect("vcp"), inputs[2])) };
        let k = self.convolve(wk_p, l.attn_k_conv.expect("kconv"), k_dyn, slot, 0, stores, std::ptr::null_mut());
        let gqa = (hd * hkv) as i64;
        let v = self.convolve(wv_p, l.attn_v_conv.expect("vconv"), v_dyn, slot, gqa, stores, std::ptr::null_mut());
        let r = r_pre;
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

        let q = self.clip(t::reshape3(c, q, hd, nh, 1), l.attn_q_clip, self.cfg.qk_clip);
        let k = self.clip(t::reshape3(c, k, hd, hkv, 1), None, self.cfg.qk_clip);
        let v = t::rms_norm(c, t::reshape3(c, v, hd, hkv, 1), 1e-6);
        let r = self.clip(t::reshape3(c, r, n_rel, nh, 1), l.attn_rel_clip, self.cfg.rel_clip);
        if il == 0 {
            self.dump("Qcur-0", q); self.dump("Kcur-0", k); self.dump("Vcur-0", v); self.dump("Rcur-0", r);
        }

        // KV write via set_rows: in-place write into arena + dep edge so the
        // attention reads below see this token's k/v (upstream pattern).
        let pos_k = step_pos_key(step, is_swa, self.swa);
        // n_kv padded to a multiple of 64: constant shapes across steps keep
        // the CUDA graph reusable (recapture only on 64-token boundaries).
        let n_kv = if is_swa { step.n_kv_swa_pad } else { step.n_kv_full_pad };
        let kv_idx = if is_swa { step.kv_idx_swa } else { step.kv_idx_full };
        // set_rows wants b [hd, n_idx, hkv]: permute [hd,hkv,1] -> [hd,1,hkv]
        let k_rows = t::permute(c, k, 0, 2, 1, 3);
        let v_rows = t::permute(c, v, 0, 2, 1, 3);
        let k_upd = t::set_rows(c, self.k_arena[slot], k_rows, kv_idx);
        let v_upd = t::set_rows(c, self.v_arena[slot], v_rows, kv_idx);

        // tau scale on full layers
        let mut q = q;
        let mut r = r;
        if !is_swa {
            q = t::mul(c, q, step_tau(step));
            r = t::mul(c, r, step_tau(step));
        }

        // rel bias (upstream): table = mm(pad(rel_bias), r) - sinks; gather by distance
        let sinks = t::mmq(c, l.attn_sink_proj.expect("sink"), inputs[0]); // [nh, 1]
        let table = t::mmq(c, l.attn_rel_bias_pad.or(l.attn_rel_bias).expect("relbias"), r); // [ext+1, nh, 1]
        let rel_pos = if is_swa { step.rel_pos_swa } else { step.rel_pos_full };
        // fused gather+sub (INKWELL_NO_RELGATHER=1 restores the chain):
        // kq_b[j,0,h] = table[rel_pos[j], h] - sinks[h] in one kernel.
        let kq_b = if std::env::var_os("INKWELL_NO_RELGATHER").is_none() {
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
        if il == 0 { self.dump("table-0", table); self.dump("kqb-0", kq_b); }

        // Fused decode attention (replaces kq GEMV + softmax + V GEMV chain)
        let k_view = t::view3(c, k_upd, hd, n_kv, hkv, t::nb(k_upd, 1), t::nb(k_upd, 2), 0);
        let q3 = t::permute(c, q, 0, 2, 1, 3); // [hd, 1, nh]
        let kq_mask = if is_swa { step.mask_swa } else { step.mask_full };
        let v_view = t::view3(c, v_upd, hd, n_kv, hkv, t::nb(v_upd, 1), t::nb(v_upd, 2), 0);
        let out = t::attn_fused(c, k_view, v_view, q3, kq_b, kq_mask, hd, nh); // [hd*nh, 1]

        let mut cur = out;
        cur = t::mul(c, cur, t::sigmoid(c, gate_lin));
        if il == 0 { self.dump("kqv_out-0", out); }
        cur = self.convolve(self.proj(c, l, "wo", l.wo, cur), l.attn_post_conv.expect("postconv"), None, slot, 2 * gqa, stores, std::ptr::null_mut());

        // residual / push block
        if ip % (self.cfg.attn_res_block_size as usize) == 0 {
            blocks.push(inp_sa);
        } else {
            cur = t::add(c, inp_sa, cur);
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
            let fcur = self.convolve(ffn_out, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, stores, std::ptr::null_mut());
            if il == 0 { self.dump("ffn_out-0", fcur); }
            let out = t::add(c, ffn_inp, fcur);
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

        let fcur = self.convolve(ffn_out, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, stores, std::ptr::null_mut());
        if il == 0 { self.dump("ffn_out-0", fcur); }
        let out2 = t::add(c, ffn_inp, fcur);
        self.dump(&format!("l_out-{slot}"), out2);
        out2
    }
}

/// absolute key position for this step's KV write
fn step_pos_key(s: &Step, is_swa: bool, swa: i64) -> i64 {
    if is_swa { s.pos_val % swa } else { s.pos_val }
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
}

impl Runner {
    /// Decode one step: `id` at absolute `pos`. Returns logits [vocab_unpadded].
    ///
    /// # Panics
    /// On graph/alloc/compute failure.
    pub fn step(&mut self, id: u32, pos: i64) -> Vec<f32> {
        let tb0 = std::time::Instant::now();
        // per-visit max n_kv this step, padded to a multiple of 64 so tensor
        // shapes stay constant across steps (CUDA-graph replayable). The
        // -inf softmax masks cover the padding columns.
        let n_kv_full_raw = pos + 1;
        let n_kv_swa_raw = (pos + 1).min(self.swa);
        let n_kv_full = ((n_kv_full_raw + 63) & !63).min(self.max_ctx);
        let n_kv_swa = ((n_kv_swa_raw + 63) & !63).min(self.swa);

        // Reuse the built ggml graph + sched splits + instantiated CUDA graph
        // while the shape signature is unchanged: only input data varies.
        // Killswitch: INKWELL_NO_GCACHE=1 forces the per-step rebuild.
        let caching = std::env::var_os("INKWELL_NO_GCACHE").is_none();
        if caching {
            if let Some(gc) = self.step_gcache.take() {
                if gc.step.n_kv_full_pad == n_kv_full && gc.step.n_kv_swa_pad == n_kv_swa {
                    let logits = self.step_cached(id, pos, &gc);
                    self.step_gcache = Some(gc);
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
        let ids = t::new(c, I32, &[1]);
        let pos_t = t::new(c, I32, &[1]);
        let kv_idx_full = t::new(c, I32, &[1]);
        let kv_idx_swa = t::new(c, I32, &[1]);
        let rel_pos_full = t::new(c, I32, &[n_kv_full, 1]);
        let rel_pos_swa = t::new(c, I32, &[n_kv_swa, 1]);
        let mask_full = t::new(c, F32, &[n_kv_full + 1, 1, 1, 1]);
        let mask_swa = t::new(c, F32, &[n_kv_swa + 1, 1, 1, 1]);
        let sink_zero = t::new(c, F32, &[1, 1, hkv, n_rep]);
        // upload input data into cpu-side buffers? These tensors are graph
        // inputs — allocate them in the device buffer and set.
        // (they're tiny; alloc in dev buf of the step)
        if self.in_buf.is_none() {
            self.in_buf = Some(self.rt.buf(self.rt.device_buft(), 16 << 20));
        }
        let buf = self.in_buf.as_ref().unwrap().clone();
        let mut o = 0usize;
        let mut in_offs = [0usize; 9];
        for (i, x) in [ids, pos_t, kv_idx_full, kv_idx_swa, rel_pos_full, rel_pos_swa, mask_full, mask_swa, sink_zero].iter().enumerate() {
            unsafe { inkwell_ggml::ink_set_input(x.0) };
            let off = (o + 255) & !255;
            buf.assign(*x, off);
            in_offs[i] = off;
            o = off + t::nbytes(*x);
        }
        // keep buf alive for the graph
        std::mem::forget(buf);

        // tau: log-scaled position (full layers)
        let posf = t::cast(c, pos_t, F32);
        let posf = t::scale_bias(c, posf, 1.0, 1.0);
        let tau = t::log(c, t::clamp(c, t::scale(c, posf, 1.0 / self.cfg.log_scale_floor as f32), 1.0, f32::INFINITY));
        let tau = t::reshape3(c, t::scale_bias(c, tau, self.cfg.log_scale_alpha, 1.0), 1, 1, 1);

        let step = Step { ids, kv_idx_full, kv_idx_swa, pos: pos_t, rel_pos_full, rel_pos_swa, sink_zero, tau, pos_val: pos,
                          mask_full, mask_swa, n_kv_full_pad: n_kv_full, n_kv_swa_pad: n_kv_swa };
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
                let (il_v, slot) = self.visits[vi];
                debug_assert_eq!(il_v, ip);
                if pass == 0 {
                    cur = self.layer(ip, ip, slot, pass, cur, &mut blocks, None, &self.layers[ip], &step, &mut stores);
                } else {
                    let l = &self.layers[ip];
                    let msgs = self.mix(&bank, bank_out, l.msg_query.expect("msg_query"));
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
        let own_head = self.out_head.is_some();
        if !own_head {
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
        let logits = self.read_logits(cur, prof);
        // (unpadded vocab is a contiguous prefix slice of the padded logits)
        if caching {
            // keep ctx + cgraph + sched split + instantiated CUDA graph alive
            self.step_gcache = Some(StepCache { step, graph: g, out: cur, in_offs, in_span: o });
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
            t::set(s.kv_idx_swa, [(pos.min(self.swa - 1)) as i32].as_ptr().cast(), 0, 4);
            // rel_pos[j] = distance (pos - key_pos) clamped to extent; slot j of
            // the SWA window holds position pos-(n_kv-1-j). Padded slots get a
            // harmless index — their logits are masked to -inf before softmax.
            let n_kv_full_raw = pos + 1;
            let n_kv_swa_raw = (pos + 1).min(self.swa);
            let rp = |nk: i64, raw: i64| -> Vec<i32> {
                (0..nk).map(|j| if j < raw { (raw - 1 - j).min(self.cfg.rel_extent as i64) as i32 } else { 0 }).collect()
            };
            let rp_f = rp(s.n_kv_full_pad, n_kv_full_raw);
            let rp_s = rp(s.n_kv_swa_pad, n_kv_swa_raw);
            t::set(s.rel_pos_full, rp_f.as_ptr().cast(), 0, rp_f.len() * 4);
            t::set(s.rel_pos_swa, rp_s.as_ptr().cast(), 0, rp_s.len() * 4);
            // -inf on padded columns (plus a 0 for the appended sink column)
            let mk = |nk: i64, raw: i64| -> Vec<f32> {
                (0..=nk).map(|j| if j < raw || j == nk { 0.0 } else { f32::NEG_INFINITY }).collect()
            };
            let mk_f = mk(s.n_kv_full_pad, n_kv_full_raw);
            let mk_s = mk(s.n_kv_swa_pad, n_kv_swa_raw);
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
                    pf.pf_miss.dev_ptr().cast(), 8, c.k, c.slot as i64);
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

    /// Marshal the pos-dependent inputs into the pinned `in_buf` mirror.
    /// `ids` is written by the caller once the token id is known; the
    /// sink_zero region stays zeroed since buffer init.
    fn marshal_inputs(&self, s: &Step, offs: &[usize; 9], pos: i64, dst: *mut u8) {
        unsafe {
            (dst.add(offs[1]) as *mut i32).write(pos as i32);
            (dst.add(offs[2]) as *mut i32).write(pos as i32);
            (dst.add(offs[3]) as *mut i32).write(pos.min(self.swa - 1) as i32);
            let nfr = pos + 1;
            let nsr = (pos + 1).min(self.swa);
            let ext = self.cfg.rel_extent as i64;
            let rf = dst.add(offs[4]) as *mut i32;
            for j in 0..s.n_kv_full_pad {
                rf.add(j as usize).write(if j < nfr { (nfr - 1 - j).min(ext) as i32 } else { 0 });
            }
            let rs = dst.add(offs[5]) as *mut i32;
            for j in 0..s.n_kv_swa_pad {
                rs.add(j as usize).write(if j < nsr { (nsr - 1 - j).min(ext) as i32 } else { 0 });
            }
            let mf = dst.add(offs[6]) as *mut f32;
            for j in 0..=s.n_kv_full_pad {
                mf.add(j as usize).write(if j < nfr || j == s.n_kv_full_pad { 0.0 } else { f32::NEG_INFINITY });
            }
            let ms = dst.add(offs[7]) as *mut f32;
            for j in 0..=s.n_kv_swa_pad {
                ms.add(j as usize).write(if j < nsr || j == s.n_kv_swa_pad { 0.0 } else { f32::NEG_INFINITY });
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
        static NO_GA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *NO_GA.get_or_init(|| std::env::var_os("INKWELL_NO_GARGMAX").is_some()) {
            return host_argmax(&self.step(id, pos));
        }
        static CACHING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CACHING.get_or_init(|| std::env::var_os("INKWELL_NO_GCACHE").is_none()) {
            if let Some(gc) = self.step_gcache.take() {
                let n_kv_full = ((pos + 1 + 63) & !63).min(self.max_ctx);
                let n_kv_swa = (((pos + 1).min(self.swa) + 63) & !63).min(self.swa);
                if gc.step.n_kv_full_pad == n_kv_full && gc.step.n_kv_swa_pad == n_kv_swa {
                    let r = self.step_id_cached(id, pos, &gc);
                    self.step_gcache = Some(gc);
                    return r;
                }
                self.step_gcache = Some(gc);
            }
        }
        host_argmax(&self.step(id, pos))
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
            return self.read_argmax(gc);
        }
        let tb0 = std::time::Instant::now();
        let sp = self.sid_parts();
        let cur_pin = sp.inp[(pos & 1) as usize];
        if self.marshaled() != pos {
            self.marshal_inputs(&gc.step, &gc.in_offs, pos, cur_pin);
        }
        unsafe { (cur_pin.add(gc.in_offs[0]) as *mut i32).write(id as i32) };
        // embd row dequant into pinned staging, then one async copy
        let (src, ty, stride) = self.embd_src;
        let row = sp.emb as *mut f32;
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
        unsafe {
            inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
            let base = inkwell_ggml::ink_buf_base(self.in_buf.as_ref().expect("in_buf").0 .0);
            inkwell_cuda::memcpy_h2d_async(base, cur_pin.cast_const().cast(), gc.in_span);
            inkwell_cuda::memcpy_h2d_async(
                inkwell_ggml::tensor_data(self.embd_row.0),
                sp.emb.cast_const().cast(), 2304 * 4);
        }
        if prof { eprintln!("PROF build={:.1}ms", tb0.elapsed().as_secs_f64() * 1e3); }
        let t0 = std::time::Instant::now();
        self.issue_moe_prefetch(pos);
        unsafe {
            assert_eq!(
                inkwell_ggml::ink_sched_compute_async(self.sched.0, gc.graph.0), 0,
                "sched compute async");
            self.dump_all();
        }
        if prof { eprintln!("PROF compute={:.1}ms", t0.elapsed().as_secs_f64() * 1e3); }
        // queue argmax + 4B readback after the replay, then marshal the next
        // step's pos-dep inputs while the GPU is still busy (single pinned
        // buf: this step's copy already ran before the graph).
        let next = pos + 1;
        let n_kv_full = ((next + 1 + 63) & !63).min(self.max_ctx);
        let n_kv_swa = (((next + 1).min(self.swa) + 63) & !63).min(self.swa);
        let premarshal = n_kv_full == gc.step.n_kv_full_pad && n_kv_swa == gc.step.n_kv_swa_pad;
        unsafe {
            inkwell_cuda::set_stream(inkwell_ggml::cuda_stream());
            if self.out_head.is_none() {
                inkwell_cuda::argmax_async(inkwell_ggml::tensor_data(gc.out.0),
                    self.cfg.vocab_unpadded as usize, sp.am.cast(), sp.ctr);
                inkwell_cuda::memcpy_d2h_async(sp.out.cast(), sp.am, 4);
                inkwell_cuda::event_record(sp.ev, inkwell_ggml::cuda_stream());
            }
        }
        if premarshal {
            self.marshal_inputs(&gc.step, &gc.in_offs, next, sp.inp[(next & 1) as usize]);
            self.set_marshaled(next);
        } else {
            self.set_marshaled(i64::MIN);
        }
        if self.out_head.is_some() {
            // own-head needs host launches: sync then take the slow read path
            unsafe { inkwell_cuda::DevBuf::sync() };
            return self.read_argmax(gc);
        }
        unsafe { inkwell_cuda::event_sync(sp.ev) };
        unsafe { *(sp.out as *const u32) }
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
            let k_rows = t::permute(c, k_j, 0, 2, 1, 3);
            let v_rows = t::permute(c, v_j, 0, 2, 1, 3);
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

    /// Verify forward: run `ids[NT]` at positions `pos..pos+NT` through the full
    /// model in one graph. Returns logits `[NT][vocab_unpadded]` (row j = token j).
    /// KV writes for all 8 tokens land in the arenas; conv state ends at pos+7
    /// (caller snapshots/restores on partial acceptance).
    ///
    /// # Panics
    /// On graph/alloc/compute failure.
    pub fn step_block(&mut self, ids: &[u32; NT], pos: i64) -> Vec<f32> {
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
        let n_kv_swa_raw = n_kv_full_raw.min(self.swa);
        let n_kv_full = ((n_kv_full_raw + 63) & !63).min(self.max_ctx);
        let n_kv_swa = ((n_kv_swa_raw + 63) & !63).min(self.swa);

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
        };
        let embd8 = t::new(c, F32, &[n_embd, NT as i64]);
        unsafe { inkwell_ggml::ink_set_input(embd8.0) };
        unsafe { inkwell_ggml::ink_set_input(step.tau8.0) };
        let mut small: Vec<Ptr> = vec![embd8, step.tau8];
        let mut sets: Vec<(Ptr, Vec<i32>)> = Vec::new();
        let mut setf: Vec<(Ptr, Vec<f32>)> = Vec::new();
        for j in 0..NT {
            for (arr, v) in [(&mut step.kv_idx_full, (pos + j as i64) as i32),
                             (&mut step.kv_idx_swa, ((pos + j as i64).min(self.swa - 1)) as i32)] {
                let t0 = t::new(c, I32, &[1]);
                unsafe { inkwell_ggml::ink_set_input(t0.0) };
                sets.push((t0, vec![v]));
                arr[j] = t0;
                small.push(t0);
            }
            for is_swa in [true, false] {
                let (nk, raw) = if is_swa { (n_kv_swa, n_kv_swa_raw.min(pos + j as i64 + 1)) }
                                else { (n_kv_full, pos + j as i64 + 1) };
                // rel_pos[j] = distance to key at slot j (0..raw-1), clamp extent
                let rp: Vec<i32> = (0..nk).map(|i| if i < raw { (raw - 1 - i).min(self.cfg.rel_extent as i64) as i32 } else { 0 }).collect();
                let t_rp = t::new(c, I32, &[nk, 1]);
                let mk: Vec<f32> = (0..=nk).map(|i| if i < raw || i == nk { 0.0 } else { f32::NEG_INFINITY }).collect();
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
        std::mem::forget(buf);
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
pub fn attn_cb_addr() -> usize { attn_cb as usize }

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

/// `ggml_custom_op_t` for dense quantized GEMVs (Q8_0 → dp4a i8 path).
/// src[0] = weight (quantized, device), src[1] = x (f32, device), dst = y.
#[expect(clippy::cast_ptr_alignment)]
pub fn dense_cb_addr() -> usize { dense_cb as usize }

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
