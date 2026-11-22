//! Golden test: full MoE chain (gu GEMV -> SITU -> down GEMV -> weighted sum) vs CPU.
use inkwell_ggml::ink_dequant;

fn deq(ty: u32, raw: &[u8], n: usize) -> Vec<f32> {
    let mut w = vec![0f32; n];
    unsafe { ink_dequant(ty as i32, raw.as_ptr().cast(), w.as_mut_ptr(), n as i64) };
    w
}

fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let g = inkwell_gguf::GgufFile::open(&path).expect("gguf");
    const N_E: usize = 2304;
    const N_FF: usize = 1152;
    let ids = [0i32, 3, 5, 7, 11, 13, 64, 127];

    let gu_t = g.tensors.get("blk.0.ffn_gate_up_exps.weight").unwrap();
    let dn_t = g.tensors.get("blk.0.ffn_down_exps.weight").unwrap();
    let gu_ty = gu_t.ggml_type as u32;
    let dn_ty = dn_t.ggml_type as u32;
    println!("gu type={gu_ty} dn type={dn_ty}");
    let gu_rb = (gu_t.nbytes / 128) as usize;
    let dn_rb = (dn_t.nbytes / 128) as usize;
    let gu_base = (g.data_start + gu_t.offset) as usize;
    let dn_base = (g.data_start + dn_t.offset) as usize;

    let sgu_t = g.tensors.get("blk.0.ffn_gate_up_exps.scale");
    let sdn_t = g.tensors.get("blk.0.ffn_down_exps.scale");
    let sgu: Vec<f32> = sgu_t.map(|t| deq(t.ggml_type as u32, &g.mmap[(g.data_start + t.offset) as usize..][..t.nbytes as usize], 128)).unwrap_or_default();
    let sdn: Vec<f32> = sdn_t.map(|t| deq(t.ggml_type as u32, &g.mmap[(g.data_start + t.offset) as usize..][..t.nbytes as usize], 128)).unwrap_or_default();
    println!("sgu.len={} sdn.len={}", sgu.len(), sdn.len());

    let mut s = 0x9e3779b9u64;
    let mut rng = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 33) & 0xffff) as f32 / 65536.0 - 0.5 };
    let x: Vec<f32> = (0..N_E).map(|_| rng()).collect();
    let w: Vec<f32> = (0..8).map(|_| rng().abs() + 0.1).collect();

    // ---- CPU reference ----
    let mut ref_out = vec![0f32; N_E];
    for (si, &e) in ids.iter().enumerate() {
        let gu_raw = &g.mmap[gu_base + e as usize * gu_rb..gu_base + (e as usize + 1) * gu_rb];
        let guf = deq(gu_ty, gu_raw, N_E * 2 * N_FF);
        if si == 0 { let sm: f32 = guf.iter().map(|v| v.abs()).sum(); eprintln!("guf abs sum={sm}"); }
        let sc = sgu.get(e as usize).copied().unwrap_or(1.0);
        let mut hid = vec![0f32; N_FF];
        for c in 0..N_FF {
            let gg = guf[c * N_E..(c + 1) * N_E].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() * sc;
            let uu = guf[(N_FF + c) * N_E..(N_FF + c + 1) * N_E].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() * sc;
            let a = 4.0f32 * (gg / 4.0).tanh() / (1.0 + (-gg).exp());
            hid[c] = a * 25.0 * (uu / 25.0).tanh();
        }
        let dn_raw = &g.mmap[dn_base + e as usize * dn_rb..dn_base + (e as usize + 1) * dn_rb];
        let dnf = deq(dn_ty, dn_raw, N_E * N_FF);
        let sd = sdn.get(e as usize).copied().unwrap_or(1.0);
        for r in 0..N_E {
            let v: f32 = dnf[r * N_FF..(r + 1) * N_FF].iter().zip(&hid).map(|(a, b)| a * b).sum();
            ref_out[r] += w[si] * sd * v;
        }
    }

    // ---- GPU ----
    let mut wbuf = inkwell_cuda::DevBuf::new_bytes(gu_t.nbytes as usize);
    wbuf.write_bytes(&g.mmap[gu_base..gu_base + gu_t.nbytes as usize]);
    let mut dbuf = inkwell_cuda::DevBuf::new_bytes(dn_t.nbytes as usize);
    dbuf.write_bytes(&g.mmap[dn_base..dn_base + dn_t.nbytes as usize]);
    let mut ids_b = inkwell_cuda::DevBuf::new_bytes(32);
    ids_b.write_bytes(unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), 32) });
    let mut xb = inkwell_cuda::DevBuf::new(N_E);
    xb.write(&x);
    let mut wb = inkwell_cuda::DevBuf::new(8);
    wb.write(&w);
    let mut sgu_b = inkwell_cuda::DevBuf::new(8);
    let sguv: Vec<f32> = ids.iter().map(|&e| sgu.get(e as usize).copied().unwrap_or(1.0)).collect();
    sgu_b.write(&sguv);
    let mut sdn_b = inkwell_cuda::DevBuf::new(8);
    let sdnv: Vec<f32> = ids.iter().map(|&e| sdn.get(e as usize).copied().unwrap_or(1.0)).collect();
    sdn_b.write(&sdnv);

    let mut gu = inkwell_cuda::DevBuf::new(8 * 2 * N_FF);
    let mut hid = inkwell_cuda::DevBuf::new(8 * N_FF);
    let mut dn = inkwell_cuda::DevBuf::new(8 * N_E);
    let mut out = inkwell_cuda::DevBuf::new(N_E);
    unsafe {
        inkwell_cuda::mmvq_iq1_s_moe(wbuf.dev_ptr(), &ids_b, gu_rb, &xb, &mut gu, 2 * N_FF, N_E, 8);
        inkwell_cuda::situ_pair(&mut hid, &gu, Some(&sgu_b), 4.0, 25.0, N_FF, 8);
        inkwell_cuda::mmvq_q2_0_moe(dbuf.dev_ptr(), &ids_b, dn_rb, &hid, &mut dn, N_E, N_FF, 8);
        inkwell_cuda::expert_wsum(&mut out, &dn, &wb, Some(&sdn_b), N_E, 8);
    }
    let y = out.read();
    let gud = gu.read();
    let hidd = hid.read();
    eprintln!("gu_dev sum={} hid_dev sum={} out0={}", gud.iter().map(|v| v.abs()).sum::<f32>(), hidd.iter().map(|v| v.abs()).sum::<f32>(), y[0]);
    let mut maxd = 0f32;
    let mut imax = 0;
    for (i, (a, b)) in ref_out.iter().zip(y.iter()).enumerate() {
        let d = (a - b).abs() / a.abs().max(1.0);
        if d > maxd { maxd = d; imax = i; }
    }
    println!("chain max_rel_diff={maxd:.3e} at {imax} (ref={} got={})", ref_out[imax], y[imax]);
}
