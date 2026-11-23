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
    max_ctx: i64,
    swa: i64,
    visits: Vec<(usize /*layer*/, usize /*slot*/)>,
    n_full: usize,
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

/// userdata for the fused conv-step custom op.
#[derive(Debug)]
struct ConvStepArgs {
    state: *mut f32,      // device ptr, conv_state[slot] base
    region_off: i32,      // channel offset inside [3, n_sconv]
    ch: i32,
    n_heads: i32,         // 0 = no dynamic taps
    scale: f32,
}

/// `ggml_custom_op_t`: fused conv shift + ssm_conv + taps + residual + scale.
/// src[0]=x_in [ch], src[1]=w [4,ch], src[2]=dyn [4,nh] (optional).
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
            inkwell_ggml::ink_tensor_data(dst).cast());
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
            exl3_w: Vec::new(), moe_args_exl3: Vec::new(), output_exl3: None,
            cfg: cfg.clone(), layers,
            tok_embd, embd_src, embd_row, tok_norm, output, output_norm, output_res_query, out_head,
            k_arena, v_arena, conv_state, max_ctx, swa, visits,
            n_full,
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

        // ---- device byte budget: all dense F32 + headroom ----
        let mut dev_bytes = 160usize << 20;
        for (n, t) in &repo.tensors {
            let (name, ti) = (n.as_str(), &t.1);
            if name.contains(".experts.") || name.contains(".trellis") || name.contains(".suh")
                || name.contains(".svh") || name.contains(".mul1") || name == "model.llm.embed.weight"
                || name.contains("visual") { continue; }
            dev_bytes += (ti.elems() * 4) as usize + 256;
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
            Box::new(Exl3W { trel, suh, svh, ka, halfk, k, n,
                scratch: inkwell_cuda::DevBuf::new(k + 8 * n) })
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

        // ---- per-layer ----
        let mut layers = Vec::with_capacity(24);
        let mut pinned: Vec<inkwell_cuda::PinnedBuf> = Vec::new();
        let mut pinned_bytes = 0usize;
        let mut expert_dev: Vec<inkwell_cuda::DevBuf> = Vec::new();
        let mut moe_args_exl3: Vec<Box<crate::moe::MoeArgsExl3>> = Vec::new();
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
                attn_sink_proj: Some(dense!(&lp(il, "attn.attn_sink_proj.weight"))),
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
                attn_k_conv_proj: conv(il).then(|| dense!(&lp(il, "attn.k_sconv.weight_proj"))),
                attn_v_conv_proj: conv(il).then(|| dense!(&lp(il, "attn.v_sconv.weight_proj"))),
                attn_post_conv: Some(dense!(&lp(il, "attn_sconv.weight"))),
                ffn_post_conv: Some(dense!(&lp(il, "mlp_sconv.weight"))),
                ffn_norm: Some(dense!(&lp(il, "mlp_norm.weight"))),
                ffn_gate_inp: Some(dense!(&lp(il, "mlp.gate.weight"))),
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
                    l.ctrl_down[ci] = Some(dense!(&lp(il, &format!("controllers.{hn}.down_weight"))));
                    l.ctrl_gain[ci] = Some(dense!(&lp(il, &format!("controllers.{hn}.gain_weight"))));
                    l.ctrl_forget[ci] = Some(dense!(&lp(il, &format!("controllers.{hn}.forget_weight"))));
                }
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
                        max_slot = max_slot.max(10368 * w * 4 + 6912);
                    }
                }
                let (f_free, _) = inkwell_cuda::mem_info();
                eprintln!("experts il={il}: free={}MiB total={}MiB", f_free >> 20, total >> 20);
                let base = if (f_free as usize) > total + (1300usize << 20) {
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
                let mk = |pi: usize| {
                    let pb = inkwell_cuda::host_alloc_mapped(128 * 32);
                    unsafe { std::ptr::copy_nonoverlapping(descs[pi * 512..].as_ptr().cast::<u8>(), pb.host, 128 * 32) };
                    pb
                };
                let on_dev = f_free as usize > total + (1300usize << 20);
                l.moe_exl3 = Some(moe_args_exl3.len());
                moe_args_exl3.push(Box::new(crate::moe::MoeArgsExl3 {
                    descs_g: mk(0), descs_u: mk(1), descs_d: mk(2),
                    on_dev, cache: None, max_slot,
                }));
            }
            l.proj0_off = [-1; 3];
            layers.push(l);
        }
        eprintln!("load_st: dev={}MiB exl3_w={} pinned={}MiB", dev_bytes >> 20, exl3_w.len(),
            pinned_bytes >> 20);

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
            .ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        if ck >= 8 {
            for (i, a) in moe_args_exl3.iter_mut().enumerate() {
                if a.on_dev || a.max_slot == 0 { continue; }
                let cbytes = ck * 3 * a.max_slot;
                let (f, _) = inkwell_cuda::mem_info();
                if f <= cbytes + (190usize << 20) { continue; }
                let Some(buf) = inkwell_cuda::DevBuf::try_new_bytes(cbytes) else { continue };
                let tag = inkwell_cuda::DevBuf::new_bytes(ck * 4);
                tag.write_bytes_async(&vec![0xFFu8; ck * 4]);
                let rr = inkwell_cuda::DevBuf::new_bytes(4);
                rr.write_bytes_async(&[0u8; 4]);
                a.cache = Some(crate::moe::Exl3Cache {
                    buf, tag, rr, k: ck as i32, slot: a.max_slot,
                });
                eprintln!("exl3 cache moe={i}: k={ck} slot={}B ({:.0} MiB) CACHED",
                    a.max_slot, cbytes as f64 / 1e6);
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
            moe_args: Vec::new(), exl3_w, moe_args_exl3, output_exl3,
            cfg: cfg.clone(), layers,
            tok_embd, embd_src, embd_row, tok_norm, output, output_norm, output_res_query, out_head,
            k_arena, v_arena, conv_state, max_ctx, swa, visits, n_full,
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
        if std::env::var("INKWELL_DUMP").is_err() { return; }
        t::output(t0); // keep alive: gallocr won't reuse output memory
        unsafe { (*self.dump_list.get()).push((name.to_string(), t0)); }
    }
    fn dump_all(&self) {
        for (name, t0) in unsafe { &*self.dump_list.get() } {
            let n = t::ne(*t0,0)*t::ne(*t0,1)*t::ne(*t0,2)*t::ne(*t0,3);
            let mut v = vec![0f32; n as usize];
            unsafe { t::get(*t0, v.as_mut_ptr().cast(), 0, (n*4) as usize) };
            let sum: f32 = v.iter().sum();
            eprintln!("DBG {name} ne={n} sum={sum:.6} first3={:?}", &v[..8]);
        }
        unsafe { (*self.dump_list.get()).clear(); }
    }

    fn clip(&self, x: Ptr, scale: Option<Ptr>, thr: f32) -> Ptr {
        let c = self.cur_ctx;
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
        let mut states = if blocks.is_empty() { prefix } else { t::concat(c, prefix, t::reshape3(c, cur, n_embd, 1, 1), 1) };
        states = t::rms_norm(c, states, 1e-6);
        let n_rows = t::ne(query, 1);
        let queries = t::repeat(c, query, t::new(c, F32, &[n_embd, n_rows, 1]));
        let scores = t::soft_max(c, t::mmq(c, states, queries), Ptr(std::ptr::null_mut()), 1.0 / (n_embd as f32).sqrt(), 0.0);
        t::mmq(c, t::cont(c, t::transpose(c, states)), scores)
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
    fn convolve(&self, x_in: Ptr, w: Ptr, dynamic: Option<Ptr>, slot: usize, region_off: i64, _stores: &mut Vec<Ptr>) -> Ptr {
        let c = self.cur_ctx;
        let n_conv = 4i64;
        let ch = t::ne(x_in, 0); // x_in: [ch, 1]
        let n_heads = dynamic.map_or(0, |d| t::ne(d, 0) / n_conv) as i32;
        let scale = 1.0 / ((if dynamic.is_some() { 2.0 } else { 1.0 }) * n_conv as f32 + 1.0).sqrt();
        let idx = self.conv_arg_idx.get();
        let pool = unsafe { &mut *self.conv_args.get() };
        if idx >= pool.len() {
            pool.push(Box::new(ConvStepArgs {
                state: unsafe { inkwell_ggml::ink_tensor_data(self.conv_state[slot].0) }.cast(),
                region_off: region_off as i32,
                ch: ch as i32,
                n_heads,
                scale,
            }));
        }
        let arg = &*pool[idx];
        self.conv_arg_idx.set(idx + 1);
        let mut args: Vec<*mut std::ffi::c_void> = vec![x_in.0, w.0];
        if let Some(d) = dynamic { args.push(d.0); }
        Ptr(unsafe {
            inkwell_ggml::custom_4d(c.0, F32, [ch, 1, 1, 1], &mut args,
                conv_step_cb as usize, (arg as *const ConvStepArgs).cast_mut().cast())
        })
    }

    /// loop control gate: (1-forget)*cur + gain*msg
    fn control(&self, cur: Ptr, msg: Ptr, l: &L, ctrl: usize) -> Ptr {
        let c = self.cur_ctx;
        let gate = t::concat(c, t::rms_norm(c, cur, 1e-6), t::rms_norm(c, msg, 1e-6), 0);
        let gate = t::mmq(c, l.ctrl_down[ctrl].expect("ctrl_down"), gate);
        let gain = t::tanh(c, t::mmq(c, l.ctrl_gain[ctrl].expect("ctrl_gain"), gate));
        let forget = t::tanh(c, t::mmq(c, l.ctrl_forget[ctrl].expect("ctrl_forget"), gate));
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
                *inp = t::cont(c, t::view2(c, mixed, n_embd, 1, t::nb(mixed, 2), (if is_swa { 0 } else { j as i64 }) * t::nb(mixed, 1)));
            }
        }
        if let Some(msgs) = messages {
            for (j, inp) in inputs.iter_mut().enumerate() {
                let m = t::cont(c, t::view2(c, msgs, n_embd, 1, t::nb(msgs, 2), j as i64 * t::nb(msgs, 1)));
                *inp = self.control(*inp, m, l, j);
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

        // projections: concat {q,gate,rel} when types allowed it at load
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
        let k_dyn = if is_swa { None } else { Some(t::mmq(c, l.attn_k_conv_proj.expect("kcp"), inputs[1])) };
        let v_dyn = if is_swa { None } else { Some(t::mmq(c, l.attn_v_conv_proj.expect("vcp"), inputs[2])) };
        let k = self.convolve(self.proj(c, l, "wk", l.wk, inputs[1]), l.attn_k_conv.expect("kconv"), k_dyn, slot, 0, stores);
        let gqa = (hd * hkv) as i64;
        let v = self.convolve(self.proj(c, l, "wv", l.wv, inputs[2]), l.attn_v_conv.expect("vconv"), v_dyn, slot, gqa, stores);
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
        let k_rows = t::cont(c, t::permute(c, k, 0, 2, 1, 3));
        let v_rows = t::cont(c, t::permute(c, v, 0, 2, 1, 3));
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
        let table = t::sub(c, table, t::reshape3(c, sinks, 1, nh, 1));
        let table = t::cont(c, t::permute(c, table, 1, 0, 2, 3)); // [nh, ext+1, 1]
        let rel_pos = if is_swa { step.rel_pos_swa } else { step.rel_pos_full };
        let kq_b = t::get_rows(c, table, rel_pos); // [nh, n_kv, 1]
        // permute(2,0,1,3): src dim i goes to dst dim arg_i -> [n_kv, 1, nh, 1]
        let kq_b = t::cont(c, t::permute(c, t::reshape4(c, kq_b, nh, n_kv, 1, 1), 2, 0, 1, 3));
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
        cur = self.convolve(self.proj(c, l, "wo", l.wo, cur), l.attn_post_conv.expect("postconv"), None, slot, 2 * gqa, stores);

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
            let m = t::cont(c, t::view2(c, msgs, n_embd, 1, t::nb(msgs, 2), 3 * t::nb(msgs, 1)));
            fcur = self.control(fcur, m, l, 3);
        }
        fcur = self.norm(fcur, l.ffn_norm.expect("ffn_norm"));

        // MoE — sigmoid gating + bias select, fused gate_up, SITU
        let logits = t::mmq(c, l.ffn_gate_inp.expect("g"), fcur); // [128, 1]
        let probs = t::sigmoid(c, logits);
        let sel = t::add(c, probs, l.ffn_exp_probs_b.expect("pb"));
        let ids = t::argsort_top_k(c, sel, 8); // [8, ntok]
        let probs3 = t::reshape3(c, probs, 1, 128, 1); // [1, n_expert, ntok]
        let mut weights = t::get_rows(c, probs3, ids); // [1, 8, ntok]
        let weights2 = t::reshape2(c, weights, 8, 1);
        let wsum = t::clamp(c, t::sum_rows(c, weights2), 6.103515625e-5, f32::INFINITY);
        weights = t::reshape3(c, t::div(c, weights2, wsum), 1, 8, 1);
        weights = t::scale(c, weights, self.cfg.expert_weights_scale);
        let weights3 = weights;

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
            let fcur = self.convolve(ffn_out, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, stores);
            if il == 0 { self.dump("ffn_out-0", fcur); }
            return t::add(c, ffn_inp, fcur);
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

        let fcur = self.convolve(ffn_out, l.ffn_post_conv.expect("fpc"), None, slot, 2 * gqa + n_embd, stores);
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
        let extent = (self.cfg.rel_extent + 1) as i64;

        // per-visit max n_kv this step, padded to a multiple of 64 so tensor
        // shapes stay constant across steps (CUDA-graph replayable). The
        // -inf softmax masks cover the padding columns.
        let n_kv_full_raw = pos + 1;
        let n_kv_swa_raw = (pos + 1).min(self.swa);
        let n_kv_full = ((n_kv_full_raw + 63) & !63).min(self.max_ctx);
        let n_kv_swa = ((n_kv_swa_raw + 63) & !63).min(self.swa);

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
        for x in [ids, pos_t, kv_idx_full, kv_idx_swa, rel_pos_full, rel_pos_swa, mask_full, mask_swa, sink_zero] {
            unsafe { inkwell_ggml::ink_set_input(x.0) };
            let off = (o + 255) & !255;
            buf.assign(x, off);
            o = off + t::nbytes(x);
        }
        unsafe {
            t::set(ids, [id as i32].as_ptr().cast(), 0, 4);
            t::set(pos_t, [pos as i32].as_ptr().cast(), 0, 4);
            t::set(kv_idx_full, [pos as i32].as_ptr().cast(), 0, 4);
            t::set(kv_idx_swa, [(pos.min(self.swa - 1)) as i32].as_ptr().cast(), 0, 4);
            // rel_pos[j] = distance (pos - key_pos) clamped to extent; slot j of
            // the SWA window holds position pos-(n_kv-1-j). Padded slots get a
            // harmless index — their logits are masked to -inf before softmax.
            let rp = |nk: i64, raw: i64| -> Vec<i32> {
                (0..nk).map(|j| if j < raw { (raw - 1 - j).min(self.cfg.rel_extent as i64) as i32 } else { 0 }).collect()
            };
            let rp_f = rp(n_kv_full, n_kv_full_raw);
            let rp_s = rp(n_kv_swa, n_kv_swa_raw);
            t::set(rel_pos_full, rp_f.as_ptr().cast(), 0, rp_f.len() * 4);
            t::set(rel_pos_swa, rp_s.as_ptr().cast(), 0, rp_s.len() * 4);
            // -inf on padded columns (plus a 0 for the appended sink column)
            let mk = |nk: i64, raw: i64| -> Vec<f32> {
                (0..=nk).map(|j| if j < raw || j == nk { 0.0 } else { f32::NEG_INFINITY }).collect()
            };
            let mk_f = mk(n_kv_full, n_kv_full_raw);
            let mk_s = mk(n_kv_swa, n_kv_swa_raw);
            t::set(mask_full, mk_f.as_ptr().cast(), 0, mk_f.len() * 4);
            t::set(mask_swa, mk_s.as_ptr().cast(), 0, mk_s.len() * 4);
            let z = vec![0f32; (hkv * n_rep) as usize];
            t::set(sink_zero, z.as_ptr().cast(), 0, z.len() * 4);
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

        // embed: dequantize the row host-side (embd stays quantized, ~230 MiB not 1.85 GiB)
        {
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
        let prof = std::env::var_os("INKWELL_PROF").is_some();
        let t0 = std::time::Instant::now();
        unsafe {
            assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, g.0), 0, "sched compute");
            self.dump_all();
        }
        if prof { eprintln!("PROF compute={:.1}ms", t0.elapsed().as_secs_f64() * 1e3); }
        let mut logits = vec![0f32; self.cfg.vocab_unpadded as usize];
        if let Some((w, x_dev, y_dev)) = self.out_head.as_mut() {
            let mut normed = vec![0f32; self.cfg.n_embd as usize];
            unsafe { t::get(cur, normed.as_mut_ptr().cast(), 0, normed.len() * 4) };
            x_dev.write(&normed);
            unsafe {
                inkwell_cuda::mmvq_iq4_xs_moe(
                    w.dev_ptr(), x_dev.dev_ptr(), y_dev, 201024, 2304, 2304, 1);
            }
            inkwell_cuda::softcap(y_dev, self.cfg.softcap);
            let all = y_dev.read();
            logits.copy_from_slice(&all[..self.cfg.vocab_unpadded as usize]);
        } else {
            unsafe { t::get(cur, logits.as_mut_ptr().cast(), 0, logits.len() * 4) };
        }
        // (unpadded vocab is a contiguous prefix slice of the padded logits)
        // release the cached split plan so the next step's graph gets re-allocated
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
