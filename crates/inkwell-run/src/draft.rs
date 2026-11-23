//! DSpark MTP drafter — the checkpoint's own draft head (`mtp.safetensors`).
//!
//! Mirrors `draft.py` (TorchDraft): per verify cycle,
//!   ctx_append: feats[8][4][2304] -> per-layer mix(alpha[l]) -> dflare_norm
//!               -> k/v_proj_ctx -> k_norm_ctx -> rope -> kc/vc[pos]
//!   propose:    x = [embd(anchor)*embed_norm, mask_embed x7] through 3 layers
//!               (attn over ctx cache + the block's own 8 rows), then the
//!               Markov recurrence for 7 steps:
//!               raw = rnn_proj([state|w1[prev]|hid_i|f]) + bias; GRU update;
//!               z_i = unembed(hid_i + ho_b + hidden_out(latent_i));
//!               tok_i = argmax(cap*tanh(z/cap))
//! All weights are EXL3 (same as the target) except bf16 norm vectors and the
//! bf16 `markov_w1` table. No ggml graph — direct kernel launches on stream 0.
//!
//! `feats` layout: `[4 taps][NT rows][2304]` f32 device (runner `block_feats`).

use std::ffi::c_void;

const NB: usize = 8;
const H: usize = 2304;
const NH: usize = 24;
const NKV: usize = 8;
const HD: usize = 128;
const FF: usize = 9216;
const RANK: usize = 256;
const EPS: f32 = 1e-6;
const THETA: f32 = 1e6;
const WINDOW: i64 = 1024;
const VOCAB: usize = 200_058;
const CAP: f32 = 30.0;

/// One EXL3 weight, device-resident (trellis + suh + svh + scratch).
struct Exl {
    trel: inkwell_cuda::DevBuf,
    suh: inkwell_cuda::DevBuf,
    svh: inkwell_cuda::DevBuf,
    k: usize,
    n: usize,
    ka: usize,
    halfk: bool,
    scratch: inkwell_cuda::DevBuf,
}

impl Exl {
    fn load(repo: &inkwell_st::StRepo, base: &str) -> Self {
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
        // scratch must hold 8*k + 64*n f32 for the ntok path (and k + 8n for gemv)
        Self { trel, suh, svh, k, n, ka, halfk, scratch: inkwell_cuda::DevBuf::new(8 * k + 64 * n + 64) }
    }
    /// y [n] = self @ x [k]  (single token)
    unsafe fn gemv(&self, x: *const f32, y: *mut f32) {
        unsafe {
            inkwell_cuda::exl3_gemv_async(self.trel.dev_ptr(), self.suh.dev_ptr(), self.svh.dev_ptr(),
                x, y, self.k, self.n, self.ka, self.halfk, self.scratch.dev_ptr().cast());
        }
    }
    /// y [n, 8] = self @ x [k, 8]  (block of 8)
    unsafe fn gemv8(&self, x: *const f32, y: *mut f32) {
        unsafe {
            inkwell_cuda::exl3_gemv_ntok_async(self.trel.dev_ptr(), self.suh.dev_ptr(), self.svh.dev_ptr(),
                x, y, self.k, self.n, self.ka, self.halfk, self.scratch.dev_ptr().cast());
        }
    }
}

/// Oracle: dequantize an Exl matrix by feeding unit vectors through gemv8.
/// Writes column-major [k][n] f32: W[:,i] at byte offset i*n*4.
#[cfg(feature = "nope")]
fn _unused() {}
unsafe fn dequant_dump(e: &Exl, path: &str) {
    let mut xh = inkwell_cuda::DevBuf::new(NB * e.k);
    let mut y = inkwell_cuda::DevBuf::new(NB * e.n);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut buf = vec![0f32; NB * e.n];
    let mut c0 = 0usize;
    while c0 < e.k {
        let rows = (e.k - c0).min(NB);
        let mut x = vec![0f32; NB * e.k];
        for t in 0..rows { x[t * e.k + c0 + t] = 1.0; }
        xh.write(&x);
        e.gemv8(xh.dev_ptr().cast_const().cast::<f32>(), y.dev_ptr().cast::<f32>());
        inkwell_cuda::DevBuf::sync();
        y.read_bytes(&mut vec_to_bytes_mut(&mut buf));
        for t in 0..rows {
            std::io::Write::write_all(&mut f, f32_as_bytes(&buf[t * e.n..(t + 1) * e.n])).unwrap();
        }
        c0 += rows;
    }
}

/// dump a DevBuf<f32> to a file (n floats)
unsafe fn dev_dump(b: &inkwell_cuda::DevBuf, n: usize, path: &str) {
    let mut v = vec![0f32; n];
    inkwell_cuda::DevBuf::sync();
    b.read_bytes(vec_to_bytes_mut(&mut v));
    std::fs::write(path, f32_as_bytes(&v)).unwrap();
}

/// dump raw bytes from a device ptr
unsafe fn dev_dump_ptr(p: *const f32, n: usize, path: &str) {
    let mut v = vec![0f32; n];
    inkwell_cuda::memcpy_d2h_sync(v.as_mut_ptr().cast(), p.cast(), n * 4);
    std::fs::write(path, f32_as_bytes(&v)).unwrap();
}

fn bf16_to_f32(src: &[u8], dst: &mut [f32]) {
    for (i, d) in dst.iter_mut().enumerate() {
        *d = f32::from_bits((u16::from_le_bytes([src[2 * i], src[2 * i + 1]]) as u32) << 16);
    }
}

/// load a bf16 weight vector, upconvert to f32, upload.
fn norm_w(repo: &inkwell_st::StRepo, name: &str, n: usize) -> inkwell_cuda::DevBuf {
    let data = repo.data(name).unwrap_or_else(|| panic!("{name}"));
    let mut v = vec![0f32; n];
    bf16_to_f32(data, &mut v);
    let mut b = inkwell_cuda::DevBuf::new(n);
    b.write(&v);
    b
}

struct Lw {
    q: Exl, k: Exl, v: Exl, o: Exl,
    kctx: Exl, vctx: Exl,
    gate: Exl, up: Exl, down: Exl,
    in_ln: inkwell_cuda::DevBuf, post_ln: inkwell_cuda::DevBuf, dflare: inkwell_cuda::DevBuf,
    qn: inkwell_cuda::DevBuf, kn: inkwell_cuda::DevBuf, knctx: inkwell_cuda::DevBuf,
}

/// DSpark draft head. One `cycle()` per verify pass:
/// `accept` (ctx_append + bookkeeping) then `propose` -> next `[anchor, 7 drafts]`.
pub struct Mtp {
    w: Vec<Lw>,
    rnn: Exl, hout: Exl,
    /// unembed EXL3 desc (shared with the target runner — not owned)
    un: (*const c_void, *const c_void, *const c_void, usize, usize, usize, bool, *mut f32),
    rnn_b: inkwell_cuda::DevBuf,
    ho_b: inkwell_cuda::DevBuf,
    alpha: inkwell_cuda::DevBuf,          // [4][4] f32 = softmax(fuse_weight)
    m_norm: inkwell_cuda::DevBuf,
    norm: inkwell_cuda::DevBuf,
    mask_embed: inkwell_cuda::DevBuf,
    w1: inkwell_cuda::DevBuf,             // bf16 [V_pad, RANK]
    kc: [inkwell_cuda::DevBuf; 3],
    vc: [inkwell_cuda::DevBuf; 3],
    // block work buffers
    x: inkwell_cuda::DevBuf, h: inkwell_cuda::DevBuf, res: inkwell_cuda::DevBuf,
    qb: inkwell_cuda::DevBuf, kb: inkwell_cuda::DevBuf, vb: inkwell_cuda::DevBuf,
    att: inkwell_cuda::DevBuf, ob: inkwell_cuda::DevBuf,
    gb: inkwell_cuda::DevBuf, ub: inkwell_cuda::DevBuf, act: inkwell_cuda::DevBuf,
    um: inkwell_cuda::DevBuf, kcb: inkwell_cuda::DevBuf, vcb: inkwell_cuda::DevBuf,
    // markov work buffers
    f: inkwell_cuda::DevBuf, state: inkwell_cuda::DevBuf, w1row: inkwell_cuda::DevBuf,
    rnn_in: inkwell_cuda::DevBuf, raw: inkwell_cuda::DevBuf, latent: inkwell_cuda::DevBuf,
    ho: inkwell_cuda::DevBuf, u2: inkwell_cuda::DevBuf, z: inkwell_cuda::DevBuf,
    amax: inkwell_cuda::DevBuf,
    /// host copy of draft logits per block row (probabilistic acceptance)
    pub dz: Vec<Vec<f32>>,
    posd: inkwell_cuda::DevBuf,
    x0: inkwell_cuda::DevBuf,
    // host-side target embedding table + norm (mmap'd repo, 'static)
    embd: *const u8,
    embd_stride: usize,
    embed_norm: Vec<f32>,
    _repo: &'static inkwell_st::StRepo,
    feats: *const f32,
    /// first ctx position ever written (rows below are zeros — mask them out).
    lo: i64,
}

pub fn dev_sync() { unsafe { inkwell_cuda::DevBuf::sync() } }

impl Mtp {
    /// `un_desc`: `(trel, suh, svh, k, n, ka, halfk, scratch)` of the target's
    /// `model.llm.unembed` (device ptrs, borrowed — runner must outlive Mtp).
    /// `feats`: device ptr to the runner's `[4][NT][2304]` tap slab.
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    pub fn new(model_dir: &str, un_desc: (*const c_void, *const c_void, *const c_void, usize, usize, usize, bool, *mut f32),
               feats: *const f32, maxctx: usize) -> Self {
        let repo: &'static inkwell_st::StRepo = Box::leak(Box::new(
            inkwell_st::StRepo::open(std::path::Path::new(model_dir).join("mtp.safetensors"))
                .expect("mtp.safetensors")));
        // target model repo (leaked): embed table + embed_norm for the anchor row
        let mrepo: &'static inkwell_st::StRepo = Box::leak(Box::new(
            inkwell_st::StRepo::open(model_dir).expect("model repo")));
        let et = mrepo.tensor("model.llm.embed.weight").expect("embed.weight");
        let embd = mrepo.data("model.llm.embed.weight").unwrap().as_ptr();
        let embd_row_bytes = et.nbytes() as usize / et.shape[0] as usize;
        let en = mrepo.data("model.llm.embed_norm.weight").unwrap();
        let mut embed_norm = vec![0f32; 2304];
        bf16_to_f32(en, &mut embed_norm);
        let mut w = Vec::new();
        for l in 0..3 {
            let p = format!("layers.{l}");
            w.push(Lw {
                q: Exl::load(repo, &format!("{p}.self_attn.q_proj")),
                k: Exl::load(repo, &format!("{p}.self_attn.k_proj")),
                v: Exl::load(repo, &format!("{p}.self_attn.v_proj")),
                o: Exl::load(repo, &format!("{p}.self_attn.o_proj")),
                kctx: Exl::load(repo, &format!("{p}.self_attn.k_proj_ctx")),
                vctx: Exl::load(repo, &format!("{p}.self_attn.v_proj_ctx")),
                gate: Exl::load(repo, &format!("{p}.mlp.gate_proj")),
                up: Exl::load(repo, &format!("{p}.mlp.up_proj")),
                down: Exl::load(repo, &format!("{p}.mlp.down_proj")),
                in_ln: norm_w(repo, &format!("{p}.input_layernorm.weight"), H),
                post_ln: norm_w(repo, &format!("{p}.post_attention_layernorm.weight"), H),
                dflare: norm_w(repo, &format!("{p}.dflare_norm.weight"), H),
                qn: norm_w(repo, &format!("{p}.self_attn.q_norm.weight"), HD),
                kn: norm_w(repo, &format!("{p}.self_attn.k_norm.weight"), HD),
                knctx: norm_w(repo, &format!("{p}.self_attn.k_norm_ctx.weight"), HD),
            });
        }
        // alpha = softmax(fuse_weight, dim=-1) [4][4] on host
        let fw = repo.data("fuse_weight").unwrap();
        let mut fwh = [0f32; 16];
        bf16_to_f32(fw, &mut fwh);
        for r in 0..4 {
            let mx = (0..4).map(|j| fwh[r * 4 + j]).fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = (0..4).map(|j| (fwh[r * 4 + j] - mx).exp()).collect();
            let s: f32 = ex.iter().sum();
            for j in 0..4 { fwh[r * 4 + j] = ex[j] / s; }
        }
        let mut alpha = inkwell_cuda::DevBuf::new(16);
        alpha.write(&fwh);
        let w1t = repo.tensor("markov_head.markov_w1.weight").unwrap();
        let mut w1 = inkwell_cuda::DevBuf::new_bytes(w1t.nbytes() as usize);
        w1.write_bytes(repo.data("markov_head.markov_w1.weight").unwrap());
        let t = (maxctx + NB + 8) * NKV * HD;
        let slf = Self {
            w,
            rnn: Exl::load(repo, "markov_head.rnn_proj"),
            hout: Exl::load(repo, "markov_head.hidden_out"),
            un: un_desc,
            rnn_b: norm_w(repo, "markov_head.rnn_proj.bias", 3 * RANK),
            ho_b: norm_w(repo, "markov_head.hidden_out.bias", H),
            alpha,
            m_norm: norm_w(repo, "markov_head.dflare_norm.weight", H),
            norm: norm_w(repo, "norm.weight", H),
            mask_embed: norm_w(repo, "mask_embed", H),
            w1,
            kc: [inkwell_cuda::DevBuf::new(t), inkwell_cuda::DevBuf::new(t), inkwell_cuda::DevBuf::new(t)],
            vc: [inkwell_cuda::DevBuf::new(t), inkwell_cuda::DevBuf::new(t), inkwell_cuda::DevBuf::new(t)],
            x: inkwell_cuda::DevBuf::new(NB * H),
            h: inkwell_cuda::DevBuf::new(NB * H),
            res: inkwell_cuda::DevBuf::new(NB * H),
            qb: inkwell_cuda::DevBuf::new(NB * NH * HD),
            kb: inkwell_cuda::DevBuf::new(NB * NKV * HD),
            vb: inkwell_cuda::DevBuf::new(NB * NKV * HD),
            att: inkwell_cuda::DevBuf::new(NB * NH * HD),
            ob: inkwell_cuda::DevBuf::new(NB * H),
            gb: inkwell_cuda::DevBuf::new(NB * FF),
            ub: inkwell_cuda::DevBuf::new(NB * FF),
            act: inkwell_cuda::DevBuf::new(NB * FF),
            um: inkwell_cuda::DevBuf::new(NB * H),
            kcb: inkwell_cuda::DevBuf::new(NB * NKV * HD),
            vcb: inkwell_cuda::DevBuf::new(NB * NKV * HD),
            f: inkwell_cuda::DevBuf::new(H),
            state: inkwell_cuda::DevBuf::new(RANK),
            w1row: inkwell_cuda::DevBuf::new(RANK),
            rnn_in: inkwell_cuda::DevBuf::new(5120),
            raw: inkwell_cuda::DevBuf::new(3 * RANK),
            latent: inkwell_cuda::DevBuf::new(RANK),
            ho: inkwell_cuda::DevBuf::new(H),
            u2: inkwell_cuda::DevBuf::new(H),
            z: inkwell_cuda::DevBuf::new(VOCAB),
            amax: inkwell_cuda::DevBuf::new(1),
            dz: Vec::new(),
            posd: inkwell_cuda::DevBuf::new(NB),
            x0: inkwell_cuda::DevBuf::new(H),
            embd,
            embd_stride: embd_row_bytes,
            embed_norm,
            _repo: repo,
            feats,
            lo: -1,
        };
        if std::env::var_os("INKWELL_MTP_ORACLE").is_some() {
            unsafe {
                std::fs::create_dir_all("/tmp/mtp_dense").unwrap();
                for (l, w) in slf.w.iter().enumerate() {
                    for (nm, e) in [("q", &w.q), ("k", &w.k), ("v", &w.v), ("o", &w.o),
                                    ("kctx", &w.kctx), ("vctx", &w.vctx),
                                    ("gate", &w.gate), ("up", &w.up), ("down", &w.down)] {
                        dequant_dump(e, &format!("/tmp/mtp_dense/l{l}_{nm}.bin"));
                        eprintln!("deq l{l}_{nm} [{}x{}]", e.k, e.n);
                    }
                }
                dequant_dump(&slf.rnn, "/tmp/mtp_dense/rnn.bin");
                dequant_dump(&slf.hout, "/tmp/mtp_dense/hout.bin");
            }
        }
        slf
    }

    fn d(&self, b: &inkwell_cuda::DevBuf) -> *mut f32 { b.dev_ptr().cast() }

    /// ctx_append: write draft-ctx K/V of `feats` rows at `pos[c..c+7]` for all layers.
    /// `feats` = the runner's tap slab from the block at base position `c`.
    pub fn ctx_append(&mut self, c: i64) {
        if std::env::var_os("INKWELL_MTP_NOCTX").is_some() { if self.lo < 0 { self.lo = c; } return; }
        if self.lo < 0 { self.lo = c; }
        if std::env::var_os("INKWELL_MTP_DUMP").is_some() {
            unsafe {
                inkwell_cuda::DevBuf::sync();
                let mut h = vec![0f32; 8];
                let mut b = vec![0u8; 4 * NB * H * 4];
                inkwell_cuda::memcpy_d2h_sync(b.as_mut_ptr().cast(), self.feats.cast(), b.len());
                for tap in 0..4 {
                    let base = tap * NB * H * 4;
                    for (i, v) in h.iter_mut().enumerate() {
                        *v = f32::from_le_bytes(b[base + i * 4..base + i * 4 + 4].try_into().unwrap());
                    }
                    eprintln!("feats tap{tap} row0[0..4]={:?} row7[0..4]={:?}", &h[..4],
                        (0..4).map(|i| f32::from_le_bytes(b[base + 7 * H * 4 + i * 4..base + 7 * H * 4 + i * 4 + 4].try_into().unwrap())).collect::<Vec<_>>());
                }
            }
        }
        unsafe {
            let posh: Vec<i32> = (0..NB).map(|i| (c + i as i64) as i32).collect();
            self.posd.write_bytes(bytemuck_cast_i32(&posh));
            for (l, w) in self.w.iter().enumerate() {
                // u = mix(feats, alpha[l]) -> dflare_norm ; [8][2304]
                inkwell_cuda::draft_mix_async(self.feats, self.d(&self.alpha).add(l * 4),
                    self.d(&w.dflare).cast_const(), self.d(&self.um), NB, NB, H, EPS);
                w.kctx.gemv8(self.d(&self.um), self.d(&self.kcb));
                w.vctx.gemv8(self.d(&self.um), self.d(&self.vcb));
                // k_norm_ctx per head then rope at absolute positions
                inkwell_cuda::draft_rms_async(self.d(&self.kcb), std::ptr::null(),
                    self.d(&w.knctx).cast_const(), self.d(&self.kcb), std::ptr::null_mut(),
                    NB * NKV, HD, EPS);
                inkwell_cuda::draft_rope_async(self.d(&self.kcb), self.d(&self.posd).cast_const().cast::<i32>(),
                    NB, NKV, HD, THETA);
                inkwell_cuda::draft_kvwrite_async(self.d(&self.kc[l]), self.d(&self.vc[l]),
                    self.d(&self.kcb), self.d(&self.vcb),
                    self.d(&self.posd).cast_const().cast::<i32>(), NB, NKV * HD);
            }
        }
    }

    /// propose: returns `[anchor, d1..d7]` at base position `c` (c_new).
    /// `frow` = feats row of the anchor (its index in the last verify block).
    pub fn propose(&mut self, c: i64, anchor: u32, frow: usize) -> [u32; NB] {
        unsafe {
            // x[0] = rms(embd[anchor]) * embed_norm (host) ; rows 1..7 = mask_embed
            let mut e = vec![0f32; H];
            let row = unsafe { std::slice::from_raw_parts(self.embd.add(anchor as usize * self.embd_stride), H * 2) };
            bf16_to_f32(row, &mut e);
            let ss: f32 = e.iter().map(|v| v * v).sum::<f32>() / H as f32;
            let inv = (ss + EPS).sqrt().recip();
            for i in 0..H { e[i] = e[i] * inv * self.embed_norm[i]; }
            self.x0.write(&e);
            inkwell_cuda::memcpy_d2d_async(self.d(&self.x).cast(), self.d(&self.x0).cast(), H * 4);
            // rows 1..7: broadcast mask_embed
            for r in 1..NB {
                inkwell_cuda::memcpy_d2d_async(self.d(&self.x).add(r * H).cast(),
                    self.d(&self.mask_embed).cast(), H * 4);
            }
            let posh: Vec<i32> = (0..NB).map(|i| (c + i as i64) as i32).collect();
            self.posd.write_bytes(bytemuck_cast_i32(&posh));
            if std::env::var_os("INKWELL_MTP_ORACLE").is_some()
                && !std::path::Path::new("/tmp/mtp_in/feats.bin").exists() {
                std::fs::create_dir_all("/tmp/mtp_in").unwrap();
                dev_dump_ptr(self.feats, 4 * NB * H, "/tmp/mtp_in/feats.bin");
                let t = (1024 + NB + 8) * NKV * HD;
                for l in 0..3 {
                    dev_dump(&self.kc[l], t, &format!("/tmp/mtp_in/kc{l}.bin"));
                    dev_dump(&self.vc[l], t, &format!("/tmp/mtp_in/vc{l}.bin"));
                }
                dev_dump(&self.x, NB * H, "/tmp/mtp_in/x.bin");
                std::fs::write("/tmp/mtp_in/meta.json",
                    format!("{{\"c\":{c},\"lo\":{},\"anchor\":{anchor},\"frow\":{frow}}}", self.lo)).unwrap();
                std::fs::write("/tmp/mtp_in/x0.bin", f32_as_bytes(&e)).unwrap();
                dev_dump(&self.f, H, "/tmp/mtp_in/f.bin");
            }
            for (l, w) in self.w.iter().enumerate() {
                if l == 0 {
                    inkwell_cuda::memcpy_d2d_async(self.d(&self.res).cast(), self.d(&self.x).cast(), NB * H * 4);
                    inkwell_cuda::draft_rms_async(self.d(&self.x), std::ptr::null(),
                        self.d(&w.in_ln).cast_const(), self.d(&self.h), std::ptr::null_mut(), NB, H, EPS);
                } else {
                    inkwell_cuda::draft_rms_async(self.d(&self.x), self.d(&self.res),
                        self.d(&w.in_ln).cast_const(), self.d(&self.h), self.d(&self.res), NB, H, EPS);
                }
                if std::env::var_os("INKWELL_MTP_ORACLE").is_some()
                    && !std::path::Path::new(&format!("/tmp/mtp_in/lhin{l}.bin")).exists() {
                    dev_dump(&self.h, NB * H, &format!("/tmp/mtp_in/lhin{l}.bin"));
                }
                w.q.gemv8(self.d(&self.h), self.d(&self.qb));
                w.k.gemv8(self.d(&self.h), self.d(&self.kb));
                w.v.gemv8(self.d(&self.h), self.d(&self.vb));
                if std::env::var_os("INKWELL_MTP_ORACLE").is_some()
                    && !std::path::Path::new(&format!("/tmp/mtp_in/lqraw{l}.bin")).exists() {
                    dev_dump(&self.qb, NB * NH * HD, &format!("/tmp/mtp_in/lqraw{l}.bin"));
                    dev_dump(&self.kb, NB * NKV * HD, &format!("/tmp/mtp_in/lkraw{l}.bin"));
                    dev_dump(&self.vb, NB * NKV * HD, &format!("/tmp/mtp_in/lv{l}.bin"));
                }
                inkwell_cuda::draft_rms_async(self.d(&self.qb), std::ptr::null(),
                    self.d(&w.qn).cast_const(), self.d(&self.qb), std::ptr::null_mut(), NB * NH, HD, EPS);
                inkwell_cuda::draft_rms_async(self.d(&self.kb), std::ptr::null(),
                    self.d(&w.kn).cast_const(), self.d(&self.kb), std::ptr::null_mut(), NB * NKV, HD, EPS);
                inkwell_cuda::draft_rope_async(self.d(&self.qb), self.d(&self.posd).cast_const().cast::<i32>(),
                    NB, NH, HD, THETA);
                inkwell_cuda::draft_rope_async(self.d(&self.kb), self.d(&self.posd).cast_const().cast::<i32>(),
                    NB, NKV, HD, THETA);
                let win_lo = if l == 1 { self.lo } else { (c - WINDOW).max(self.lo) };
                inkwell_cuda::draft_attn_async(self.d(&self.qb), self.d(&self.kc[l]), self.d(&self.vc[l]),
                    self.d(&self.kb), self.d(&self.vb), self.d(&self.att),
                    c as usize, win_lo as usize, NB, NKV, NH / NKV, HD);
                w.o.gemv8(self.d(&self.att), self.d(&self.ob));
                inkwell_cuda::draft_rms_async(self.d(&self.ob), self.d(&self.res),
                    self.d(&w.post_ln).cast_const(), self.d(&self.h), self.d(&self.res), NB, H, EPS);
                w.gate.gemv8(self.d(&self.h), self.d(&self.gb));
                w.up.gemv8(self.d(&self.h), self.d(&self.ub));
                inkwell_cuda::draft_silu_async(self.d(&self.gb), self.d(&self.ub), self.d(&self.act), NB, FF);
                w.down.gemv8(self.d(&self.act), self.d(&self.x));
                if std::env::var_os("INKWELL_MTP_ORACLE").is_some()
                    && std::path::Path::new("/tmp/mtp_in/meta.json").exists()
                    && !std::path::Path::new(&format!("/tmp/mtp_in/lx{l}.bin")).exists() {
                    dev_dump(&self.h, NB * H, &format!("/tmp/mtp_in/lh{l}.bin"));
                    dev_dump(&self.qb, NB * NH * HD, &format!("/tmp/mtp_in/lq{l}.bin"));
                    dev_dump(&self.kb, NB * NKV * HD, &format!("/tmp/mtp_in/lk{l}.bin"));
                    dev_dump(&self.att, NB * NH * HD, &format!("/tmp/mtp_in/latt{l}.bin"));
                    dev_dump(&self.act, NB * FF, &format!("/tmp/mtp_in/lact{l}.bin"));
                    dev_dump(&self.gb, NB * FF, &format!("/tmp/mtp_in/lgb{l}.bin"));
                    dev_dump(&self.ub, NB * FF, &format!("/tmp/mtp_in/lub{l}.bin"));
                    dev_dump(&self.res, NB * H, &format!("/tmp/mtp_in/lres{l}.bin"));
                    dev_dump(&self.x, NB * H, &format!("/tmp/mtp_in/lx{l}.bin"));
                }
            }
            // hid = addrms(x, res, norm); markov uses rows 1..7
            inkwell_cuda::draft_rms_async(self.d(&self.x), self.d(&self.res),
                self.d(&self.norm).cast_const(), self.d(&self.h), std::ptr::null_mut(), NB, H, EPS);
            if std::env::var_os("INKWELL_MTP_ORACLE").is_some()
                && !std::path::Path::new("/tmp/mtp_in/hid.bin").exists() {
                dev_dump(&self.h, NB * H, "/tmp/mtp_in/hid.bin");
            }
            // f = mix(feats[frow], alpha[3], m_norm) -> [2304]
            inkwell_cuda::draft_mix_async(self.feats.add(frow * H), self.d(&self.alpha).add(3 * 4),
                self.d(&self.m_norm).cast_const(), self.d(&self.f), 1, NB, H, EPS);
            if std::env::var_os("INKWELL_MTP_ORACLE").is_some()
                && !std::path::Path::new("/tmp/mtp_in/f2.bin").exists() {
                dev_dump(&self.f, H, "/tmp/mtp_in/f2.bin");
            }
            self.state.zero();
            let mut prev = anchor;
            let mut out = [0u32; NB];
            out[0] = anchor;
            for i in 1..NB {
                // rnn_in = [state | w1[prev] | hid_i | f]
                inkwell_cuda::draft_bf16row_async(self.w1.dev_ptr(), prev as i32, RANK, self.d(&self.w1row));
                let r0 = self.d(&self.rnn_in);
                inkwell_cuda::memcpy_d2d_async(r0.cast(), self.d(&self.state).cast(), RANK * 4);
                inkwell_cuda::memcpy_d2d_async(r0.add(RANK).cast(), self.d(&self.w1row).cast(), RANK * 4);
                inkwell_cuda::memcpy_d2d_async(r0.add(2 * RANK).cast(), self.d(&self.h).add(i * H).cast(), H * 4);
                inkwell_cuda::memcpy_d2d_async(r0.add(2 * RANK + H).cast(), self.d(&self.f).cast(), H * 4);
                self.rnn.gemv(r0.cast_const(), self.d(&self.raw));
                inkwell_cuda::draft_gru_async(self.d(&self.raw), self.d(&self.rnn_b),
                    self.d(&self.state), self.d(&self.latent), RANK);
                self.hout.gemv(self.d(&self.latent), self.d(&self.ho));
                // u = hid_i + ho_b + ho(latent)
                inkwell_cuda::draft_add3_async(self.d(&self.ho), self.d(&self.ho_b),
                    self.d(&self.h).add(i * H), self.d(&self.u2), H);
                let (t_, s_, v_, k_, n_, ka_, hk_, scr_) = self.un;
                inkwell_cuda::exl3_gemv_async(t_, s_, v_, self.d(&self.u2), self.d(&self.z),
                    k_, n_, ka_, hk_, scr_);
                inkwell_cuda::draft_argmax_async(self.d(&self.z), VOCAB, CAP, self.d(&self.amax).cast::<i32>());
                if std::env::var_os("INKWELL_MTP_ORACLE").is_some() {
                    dev_dump(&self.u2, H, &format!("/tmp/mtp_in/u2_{i}.bin"));
                    dev_dump(&self.raw, 3 * RANK, &format!("/tmp/mtp_in/raw_{i}.bin"));
                    dev_dump(&self.state, RANK, &format!("/tmp/mtp_in/state_{i}.bin"));
                }
                inkwell_cuda::DevBuf::sync();
                let mut tok = [0u8; 4];
                self.amax.read_bytes(&mut tok);
                prev = u32::from_ne_bytes(tok);
                out[i] = prev;
                if self.dz.len() < NB { self.dz.resize(NB, Vec::new()); }
                self.dz[i].resize(VOCAB, 0.0);
                inkwell_cuda::memcpy_d2h_sync(self.dz[i].as_mut_ptr().cast(),
                    self.d(&self.z).cast(), VOCAB * 4);
                if i == 1 && std::env::var_os("INKWELL_MTP_BASE").is_some() {
                    // diagnostic: argmax of unembed(hid_1) alone (no markov/ho)
                    let (t_, s_, v_, k_, n_, ka_, hk_, scr_) = self.un;
                    inkwell_cuda::memcpy_d2d_async(self.d(&self.u2).cast(), self.d(&self.h).add(i * H).cast(), H * 4);
                    inkwell_cuda::exl3_gemv_async(t_, s_, v_, self.d(&self.u2), self.d(&self.z), k_, n_, ka_, hk_, scr_);
                    inkwell_cuda::draft_argmax_async(self.d(&self.z), VOCAB, CAP, self.d(&self.amax).cast::<i32>());
                    inkwell_cuda::DevBuf::sync();
                    let mut t2 = [0u8; 4];
                    self.amax.read_bytes(&mut t2);
                    eprintln!("dbg i=1: markov={} base={}", prev, u32::from_ne_bytes(t2));
                }
            }
            out
        }
    }

    /// One draft-and-verify cycle: append ctx rows for the verify block at
    /// `c_old`, then propose. `tgt[j]` = target argmax of verify row j;
    /// `a` = accepted prefix length (drafts matched rows 0..a-1); emits
    /// `a + 1` tokens ending with `tgt[a]`. Returns `[anchor, 7 drafts]` for
    /// the next verify pass at position `c_old + a + 1`.
    #[allow(dead_code)]
    pub fn cycle(&mut self, tgt: &[u32; NB], a: usize, c_old: i64) -> ([u32; NB], i64) {
        self.ctx_append(c_old);
        let c_new = c_old + a as i64 + 1;
        let anchor = tgt[a];
        let out = self.propose(c_new, anchor, a);
        (out, c_new)
    }
}

fn f32_as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
}
fn vec_to_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), v.len() * 4) }
}
fn bytemuck_cast_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) }
}
