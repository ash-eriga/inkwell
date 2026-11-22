//! Decode-time (n_tokens = 1, single sequence) forward runner.
//!
//! Weights: dense tensors on the CUDA buffer; `*_exps` expert tensors live in
//! pinned+mapped host memory and stream over PCIe into own GEMV kernels via a
//! `GGML_OP_CUSTOM` per MoE visit (Strata-style zero-copy offload).
//! KV + conv state live in per-visit device arenas.
use crate::ops::{t, Buf, Ptr, Rt};
use inkwell_gguf::GgufFile;
use inkwell_core::InklingConfig;
use std::collections::HashMap;

const F32: i32 = 0;
const I32: i32 = 26;

/// per-layer weight handles resolved by name
#[derive(Debug, Default)]
struct L {
    attn_norm: Option<Ptr>, attn_q_inp_norm: Option<Ptr>, attn_k_inp_norm: Option<Ptr>, attn_v_inp_norm: Option<Ptr>,
    attn_res_query: Option<Ptr>, ffn_res_query: Option<Ptr>,
    wq: Option<Ptr>, wk: Option<Ptr>, wv: Option<Ptr>, wo: Option<Ptr>,
    attn_rel: Option<Ptr>, attn_rel_bias: Option<Ptr>, attn_rel_bias_pad: Option<Ptr>, attn_rel_clip: Option<Ptr>, attn_q_clip: Option<Ptr>,
    attn_sink_proj: Option<Ptr>, wqkv_gate: Option<Ptr>,
    attn_k_conv: Option<Ptr>, attn_v_conv: Option<Ptr>, attn_post_conv: Option<Ptr>, ffn_post_conv: Option<Ptr>,
    attn_k_conv_proj: Option<Ptr>, attn_v_conv_proj: Option<Ptr>,
    ffn_norm: Option<Ptr>, ffn_gate_inp: Option<Ptr>, ffn_exp_probs_b: Option<Ptr>,
    gate_up_scale: Option<Ptr>, down_scale: Option<Ptr>,
    moe: Option<usize>,
    msg_query: Option<Ptr>, ctrl_down: [Option<Ptr>; 4], ctrl_gain: [Option<Ptr>; 4], ctrl_forget: [Option<Ptr>; 4],
}

/// Everything the decode step needs.
#[derive(Debug)]
pub struct Runner {
    rt: Rt,
    sched: Ptr,
    /// (sched, cpu backend) for pinning expert-chain ops to CPU
    sched_cpu: Option<(Ptr, Ptr)>,
    /// per-step throwaway ggml context (graph nodes), freed after each step
    cur_ctx: Ptr,
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
    /// per-layer MoE weight bindings consumed by the custom op
    moe_args: Vec<Box<crate::moe::MoeArgs>>,
    cfg: InklingConfig,
    layers: Vec<L>,
    // globals
    tok_embd: Ptr, embd_src: (*const u8, i32, usize), embd_row: Ptr, tok_norm: Ptr, output: Ptr, output_norm: Ptr, output_res_query: Ptr,
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
        dev_bytes += 8 << 20; // headroom for materialized views (padded rel_bias)
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
                if pinned.is_empty() { let (f0,_) = inkwell_cuda::mem_info(); eprintln!("MEM before pinned: {f0}"); }
                // copy into cudaHostAlloc'd pinned+mapped memory (file-backed
                // mmap cannot be registered — verified)
                let pb = inkwell_cuda::host_alloc_mapped(ti.nbytes as usize);
                let src = unsafe { gg.mmap.as_ptr().add(gg.data_start as usize + ti.offset as usize) };
                unsafe { std::ptr::copy_nonoverlapping(src, pb.host, ti.nbytes as usize) };
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
        // small CPU tensor carrying the dequantized embedding row each step
        let embd_row = t::new(rt.ctx(), F32, &[2304, 1, 1, 1]);
        { let o = (off_cpu + 127) & !127; cpu_buf.assign(embd_row, o); }
        unsafe { inkwell_ggml::ink_sched_set_tensor_backend(sched.0, embd_row.0, cpu.0) };

        let get = |n: &str| -> Ptr {
            *w.get(n).unwrap_or_else(|| panic!("missing {n}"))
        };
        let opt = |n: &str| -> Option<Ptr> { w.get(n).copied() };

        let mut moe_args: Vec<Box<crate::moe::MoeArgs>> = Vec::new();
        let mut moe_pos: HashMap<usize, usize> = HashMap::new();
        // VRAM LRU expert cache: size from free VRAM after dense weights.
        let (vfree, _vtotal) = inkwell_cuda::mem_info();
        let budget = vfree.saturating_sub(1 << 30).min(4 << 30); // ≤3 GiB, keep 1 GiB headroom
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
            layers.push(l);
        }

        let tok_embd = Ptr(std::ptr::null_mut()); // embd stays host-side
        let tok_norm = opt("token_embd_norm.weight").unwrap_or_else(|| opt("output_norm.weight").expect("tok_norm"));
        let output = get("output.weight");
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
            cur_ctx: Ptr(std::ptr::null_mut()),
            dump_list: std::cell::UnsafeCell::new(Vec::new()),
            in_buf: None,
            rt, sched, dev_buf, cpu_buf, w, pinned, moe_args, cfg: cfg.clone(), layers,
            tok_embd, embd_src, embd_row, tok_norm, output, output_norm, output_res_query,
            k_arena, v_arena, conv_state, max_ctx, swa, visits,
            n_full,
        }
    }
}

// ============ forward ============

impl Runner {
    /// layer helpers
    fn is_swa_layer(&self, il: usize) -> bool { il % 6 != 5 }

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
            eprintln!("DBG {name} ne={n} sum={sum:.6} first3={:?}", &v[..3]);
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
        let mut states = t::reshape3(c, cur, n_embd, 1, 1);
        // upstream prepends each block in vector order -> newest first, cur last
        for b in blocks.iter() {
            states = t::concat(c, t::reshape3(c, *b, n_embd, 1, 1), states, 1);
        }
        states = t::rms_norm(c, states, 1e-6);
        let n_rows = t::ne(query, 1);
        let queries = t::repeat(c, query, t::new(c, F32, &[n_embd, n_rows, 1]));
        let scores = t::soft_max(c, t::mul_mat(c, states, queries), Ptr(std::ptr::null_mut()), 1.0 / (n_embd as f32).sqrt(), 0.0);
        t::mul_mat(c, t::cont(c, t::transpose(c, states)), scores)
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
    fn convolve(&self, x_in: Ptr, w: Ptr, dynamic: Option<Ptr>, slot: usize, region_off: i64, stores: &mut Vec<Ptr>) -> Ptr {
        let c = self.cur_ctx;
        let n_conv = 4i64;
        let state = self.conv_state[slot];
        let ch = t::ne(x_in, 0); // x_in: [ch, 1]
        // past view: [3, ch] at channel offset
        let past = t::view3(c, state, 3, ch, 1, t::nb(state, 1), t::nb(state, 2), region_off * t::nb(state, 1));
        let cur_t = t::cont(c, t::permute(c, x_in, 1, 0, 2, 3)); // [1, ch]
        let x = t::cont(c, t::concat(c, past, cur_t, 0)); // [4, ch]
        // store last 3 rows of x into state region
        let tail = t::view3(c, x, 3, ch, 1, t::nb(x, 1), t::nb(x, 2), 1 * t::nb(x, 0));
        let dst = t::view3(c, state, 3, ch, 1, t::nb(state, 1), t::nb(state, 2), region_off * t::nb(state, 1));
        stores.push(t::cpy(c, tail, dst));
        // conv
        let mut out = t::reshape2(c, t::ssm_conv(c, t::reshape3(c, x, n_conv, ch, 1), w), ch, 1);
        if let Some(dyn_w) = dynamic {
            let n_heads = t::ne(dyn_w, 0) / n_conv;
            let taps = t::reshape3(c, dyn_w, n_conv, n_heads, 1);
            for lag in 0..n_conv {
                let tap = t::view3(c, taps, 1, n_heads, 1, t::nb(taps, 1), t::nb(taps, 2), lag * t::nb(taps, 0));
                let win = t::view3(c, x, 1, ch, 1, t::nb(x, 1), t::nb(x, 2), (n_conv - 1 - lag) * t::nb(x, 0));
                let win = t::reshape3(c, t::cont(c, t::permute(c, win, 1, 0, 2, 3)), ch / n_heads, n_heads, 1);
                let term = t::reshape2(c, t::mul(c, win, tap), ch, 1);
                out = t::add(c, out, term);
            }
        }
        let s = 1.0 / ((if dynamic.is_some() { 2.0 } else { 1.0 }) * n_conv as f32 + 1.0).sqrt();
        t::scale(c, t::add(c, x_in, out), s)
    }

    /// loop control gate: (1-forget)*cur + gain*msg
    fn control(&self, cur: Ptr, msg: Ptr, l: &L, ctrl: usize) -> Ptr {
        let c = self.cur_ctx;
        let gate = t::concat(c, t::rms_norm(c, cur, 1e-6), t::rms_norm(c, msg, 1e-6), 0);
        let gate = t::mul_mat(c, l.ctrl_down[ctrl].expect("ctrl_down"), gate);
        let gain = t::tanh(c, t::mul_mat(c, l.ctrl_gain[ctrl].expect("ctrl_gain"), gate));
        let forget = t::tanh(c, t::mul_mat(c, l.ctrl_forget[ctrl].expect("ctrl_forget"), gate));
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

        // projections
        let q = t::mul_mat(c, l.wq.expect("wq"), inputs[0]);
        let k_dyn = if is_swa { None } else { Some(t::mul_mat(c, l.attn_k_conv_proj.expect("kcp"), inputs[1])) };
        let v_dyn = if is_swa { None } else { Some(t::mul_mat(c, l.attn_v_conv_proj.expect("vcp"), inputs[2])) };
        let k = self.convolve(t::mul_mat(c, l.wk.expect("wk"), inputs[1]), l.attn_k_conv.expect("kconv"), k_dyn, slot, 0, stores);
        let gqa = (hd * hkv) as i64;
        let v = self.convolve(t::mul_mat(c, l.wv.expect("wv"), inputs[2]), l.attn_v_conv.expect("vconv"), v_dyn, slot, gqa, stores);
        let r = t::mul_mat(c, l.attn_rel.expect("rel"), inputs[0]);
        if il == 0 {
            self.dump("rel_raw", r); self.dump("inputs0", inputs[0]);
            // weight raw bytes check
            let wp = l.attn_rel.unwrap();
            let nbytes = t::nbytes(wp);
            let mut vb = vec![0u8; nbytes.min(64) as usize];
            unsafe { t::get(wp, vb.as_mut_ptr().cast(), 0, vb.len()) };
            eprintln!("DBG attn_rel nbytes={nbytes} head={:?}", &vb[..16]);
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
        let n_kv = if is_swa { (pos_k + 1).min(self.swa) } else { pos_k + 1 };
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
        let sinks = t::mul_mat(c, l.attn_sink_proj.expect("sink"), inputs[0]); // [nh, 1]
        let table = t::mul_mat(c, l.attn_rel_bias_pad.or(l.attn_rel_bias).expect("relbias"), r); // [ext+1, nh, 1]
        let table = t::sub(c, table, t::reshape3(c, sinks, 1, nh, 1));
        let table = t::cont(c, t::permute(c, table, 1, 0, 2, 3)); // [nh, ext+1, 1]
        let rel_pos = if is_swa { step.rel_pos_swa } else { step.rel_pos_full };
        let kq_b = t::get_rows(c, table, rel_pos); // [nh, n_kv, 1]
        // permute(2,0,1,3): src dim i goes to dst dim arg_i -> [n_kv, 1, nh, 1]
        let kq_b = t::cont(c, t::permute(c, t::reshape4(c, kq_b, nh, n_kv, 1, 1), 2, 0, 1, 3));
        if il == 0 { self.dump("table-0", table); self.dump("kqb-0", kq_b); }

        // MHA (upstream non-FA path): kq = K^T Q -> [n_kv, 1, nh]
        let k_view = t::view3(c, k_upd, hd, n_kv, hkv, t::nb(k_upd, 1), t::nb(k_upd, 2), 0);
        let q3 = t::permute(c, q, 0, 2, 1, 3); // [hd, 1, nh]
        let kq = t::scale(c, t::mul_mat(c, k_view, q3), 1.0 / (hd as f32).sqrt()); // [n_kv, 1, nh]
        let kq = t::add(c, kq, kq_b);
        // zero-sink trick: append a zero logit column then slice it off post-softmax
        let sink0 = t::reshape4(c, step.sink_zero, 1, 1, nh, 1);
        let kq_ext = t::concat(c, kq, sink0, 0); // [n_kv+1, 1, nh]
        let attn = t::soft_max(c, kq_ext, Ptr(std::ptr::null_mut()), 1.0, 0.0);
        let attn = t::cont(c, t::view4(c, attn, n_kv, 1, nh, 1, t::nb(attn, 1), t::nb(attn, 2), t::nb(attn, 3), 0));
        if il == 0 { self.dump("kq_ext-0", kq_ext); self.dump("attn-0", attn); }

        // weighted V: kqv = V^T @ attn -> [hd, 1, nh]
        let v_view = t::view3(c, v_upd, hd, n_kv, hkv, t::nb(v_upd, 1), t::nb(v_upd, 2), 0);
        let v_t = t::cont(c, t::transpose(c, v_view)); // [n_kv, hd, hkv]
        let kqv = t::mul_mat(c, v_t, attn); // [hd, 1, nh]
        let out = t::reshape2(c, t::cont(c, kqv), hd * nh, 1);

        let mut cur = out;
        cur = t::mul(c, cur, t::sigmoid(c, t::mul_mat(c, l.wqkv_gate.expect("gate"), inputs[0])));
        if il == 0 { self.dump("kqv_out-0", out); }
        cur = self.convolve(t::mul_mat(c, l.wo.expect("wo"), cur), l.attn_post_conv.expect("postconv"), None, slot, 2 * gqa, stores);

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
        let logits = t::mul_mat(c, l.ffn_gate_inp.expect("g"), fcur); // [128, 1]
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
}

impl Runner {
    /// Decode one step: `id` at absolute `pos`. Returns logits [vocab_unpadded].
    ///
    /// # Panics
    /// On graph/alloc/compute failure.
    pub fn step(&mut self, id: u32, pos: i64) -> Vec<f32> {
        self.cur_ctx = Ptr(unsafe { inkwell_ggml::ink_ctx_new() });
        let c = self.cur_ctx;
        let n_embd = self.cfg.n_embd as i64;
        let (hd, nh, hkv) = (128i64, 24i64, 8i64);
        let n_rep = nh / hkv;
        let extent = (self.cfg.rel_extent + 1) as i64;

        // per-visit max n_kv this step (uniform: pos+1 clamped)
        let n_kv_full = pos + 1;
        let n_kv_swa = (pos + 1).min(self.swa);
        let n_kv = n_kv_full; // use largest for the onehot width; views cut per layer
        let _ = n_kv_swa;

        // --- inputs ---
        let ids = t::new(c, I32, &[1]);
        let pos_t = t::new(c, I32, &[1]);
        let kv_idx_full = t::new(c, I32, &[1]);
        let kv_idx_swa = t::new(c, I32, &[1]);
        let rel_pos_full = t::new(c, I32, &[n_kv_full, 1]);
        let rel_pos_swa = t::new(c, I32, &[n_kv_swa, 1]);
        let sink_zero = t::new(c, F32, &[1, 1, hkv, n_rep]);
        // upload input data into cpu-side buffers? These tensors are graph
        // inputs — allocate them in the device buffer and set.
        // (they're tiny; alloc in dev buf of the step)
        if self.in_buf.is_none() {
            self.in_buf = Some(self.rt.buf(self.rt.device_buft(), 16 << 20));
        }
        let buf = self.in_buf.as_ref().unwrap().clone();
        let mut o = 0usize;
        for x in [ids, pos_t, kv_idx_full, kv_idx_swa, rel_pos_full, rel_pos_swa, sink_zero] {
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
            // the SWA window holds position pos-(n_kv-1-j)
            let rp = |nk: i64| -> Vec<i32> {
                (0..nk).map(|j| (nk - 1 - j).min(self.cfg.rel_extent as i64) as i32).collect()
            };
            let rp_f = rp(n_kv_full);
            let rp_s = rp(n_kv_swa);
            t::set(rel_pos_full, rp_f.as_ptr().cast(), 0, rp_f.len() * 4);
            t::set(rel_pos_swa, rp_s.as_ptr().cast(), 0, rp_s.len() * 4);
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

        let step = Step { ids, kv_idx_full, kv_idx_swa, pos: pos_t, rel_pos_full, rel_pos_swa, sink_zero, tau, pos_val: pos };

        // embed: dequantize the row host-side (embd stays quantized, ~230 MiB not 1.85 GiB)
        {
            let (src, ty, stride) = self.embd_src;
            let mut row = vec![0f32; 2304];
            let rc = unsafe { inkwell_ggml::ink_dequant(ty, src.add(id as usize * stride).cast(), row.as_mut_ptr(), 2304) };
            assert_eq!(rc, 0, "dequant embd row");
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
        cur = t::mul_mat(c, self.output, cur);
        if self.cfg.softcap != 0.0 {
            let cap = self.cfg.softcap;
            cur = t::scale(c, t::tanh(c, t::scale(c, cur, 1.0 / cap)), cap);
        }
        t::output(cur);

        // graph
        let tb = std::time::Instant::now();
        let g = Ptr(unsafe { inkwell_ggml::ink_graph_new(c.0, 32768) });
        t::build_forward(g, cur);
        for s in &stores {
            t::build_forward(g, *s);
        }
        for (_, ptr) in unsafe { &*self.dump_list.get() } {
            t::build_forward(g, *ptr);
        }
        let t_build = tb.elapsed();
        let tc = std::time::Instant::now();
        unsafe {
            assert_eq!(inkwell_ggml::ink_sched_compute(self.sched.0, g.0), 0, "sched compute");
            self.dump_all();
        }
        eprintln!("DBG time build={:?} compute={:?}", t_build, tc.elapsed());
        unsafe { inkwell_ggml::ink_tensor_debug(cur.0) };
        let mut logits = vec![0f32; self.cfg.vocab_unpadded as usize];
        unsafe { t::get(cur, logits.as_mut_ptr().cast(), 0, logits.len() * 4) };
        // (unpadded vocab is a contiguous prefix slice of the padded logits)
        // release the cached split plan so the next step's graph gets re-allocated
        unsafe { inkwell_ggml::ink_sched_reset(self.sched.0) };
        unsafe { inkwell_ggml::ink_ctx_free(self.cur_ctx.0) };
        logits
    }
}
