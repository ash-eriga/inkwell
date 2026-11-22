//! Stage-by-stage diff: real dumped MoE inputs vs CPU reference.
use inkwell_ggml::ink_dequant;

fn deq(ty: u32, raw: &[u8], n: usize) -> Vec<f32> {
    let mut w = vec![0f32; n];
    unsafe { ink_dequant(ty as i32, raw.as_ptr().cast(), w.as_mut_ptr(), n as i64) };
    w
}

fn main() {
    let d = std::fs::read("/tmp/moe_dump.bin").expect("dump");
    let x: Vec<f32> = d[..9216].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let ids: Vec<i32> = d[9216..9248].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    let w: Vec<f32> = d[9248..9280].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let sgu: Vec<f32> = d[9280..9312].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let sdn: Vec<f32> = d[9312..9344].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    println!("ids={ids:?}\nw={w:?}\nsgu={sgu:?}\nsdn={sdn:?}");

    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    const N_E: usize = 2304; const N_FF: usize = 1152;
    let gu_t = g.tensors.get("blk.0.ffn_gate_up_exps.weight").unwrap();
    let dn_t = g.tensors.get("blk.0.ffn_down_exps.weight").unwrap();
    let gu_rb = (gu_t.nbytes / 128) as usize; let dn_rb = (dn_t.nbytes / 128) as usize;
    let gu_b = (g.data_start + gu_t.offset) as usize;
    let dn_b = (g.data_start + dn_t.offset) as usize;

    // CPU ref per stage
    let mut ref_gu = vec![0f32; 8 * 2 * N_FF];
    let mut ref_hid = vec![0f32; 8 * N_FF];
    let mut ref_dn = vec![0f32; 8 * N_E];
    let mut ref_out = vec![0f32; N_E];
    for (si, &e) in ids.iter().enumerate() {
        let raw = &g.mmap[gu_b + e as usize * gu_rb..gu_b + (e as usize + 1) * gu_rb];
        let m = deq(19, raw, N_E * 2 * N_FF);
        for r in 0..2 * N_FF {
            ref_gu[si * 2 * N_FF + r] = m[r * N_E..(r + 1) * N_E].iter().zip(&x).map(|(a, b)| a * b).sum::<f32>() * sgu[si];
        }
        for c in 0..N_FF {
            let gg = ref_gu[si * 2 * N_FF + c];
            let uu = ref_gu[si * 2 * N_FF + N_FF + c];
            let a = 4.0f32 * (gg / 4.0).tanh() / (1.0 + (-gg).exp());
            ref_hid[si * N_FF + c] = a * 25.0 * (uu / 25.0).tanh();
        }
        let raw = &g.mmap[dn_b + e as usize * dn_rb..dn_b + (e as usize + 1) * dn_rb];
        let m = deq(42, raw, N_E * N_FF);
        for r in 0..N_E {
            ref_dn[si * N_E + r] = m[r * N_FF..(r + 1) * N_FF].iter().zip(&ref_hid[si * N_FF..(si + 1) * N_FF]).map(|(a, b)| a * b).sum::<f32>();
        }
        for r in 0..N_E { ref_out[r] += w[si] * sdn[si] * ref_dn[si * N_E + r]; }
    }

    // GPU stages
    let mut wbuf = inkwell_cuda::DevBuf::new_bytes(gu_t.nbytes as usize);
    wbuf.write_bytes(&g.mmap[gu_b..gu_b + gu_t.nbytes as usize]);
    let mut dbuf = inkwell_cuda::DevBuf::new_bytes(dn_t.nbytes as usize);
    dbuf.write_bytes(&g.mmap[dn_b..dn_b + dn_t.nbytes as usize]);
    let mut ids_b = inkwell_cuda::DevBuf::new_bytes(32);
    ids_b.write_bytes(&d[9216..9248]);
    let mut xb = inkwell_cuda::DevBuf::new(N_E); xb.write(&x);
    let mut wb = inkwell_cuda::DevBuf::new(8); wb.write(&w);
    let mut sgu_b = inkwell_cuda::DevBuf::new(8); sgu_b.write(&sgu);
    let mut sdn_b = inkwell_cuda::DevBuf::new(8); sdn_b.write(&sdn);
    let mut gu = inkwell_cuda::DevBuf::new(8 * 2 * N_FF);
    let mut hid = inkwell_cuda::DevBuf::new(8 * N_FF);
    let mut dn = inkwell_cuda::DevBuf::new(8 * N_E);
    let mut out = inkwell_cuda::DevBuf::new(N_E);
    unsafe {
        inkwell_cuda::mmvq_iq1_s_moe(wbuf.dev_ptr(), &ids_b, gu_rb, &xb, &mut gu, 2 * N_FF, N_E, 0, 8);
    }
    let gud = gu.read();
    inkwell_cuda::situ_pair(&mut hid, &gu, Some(&sgu_b), 4.0, 25.0, N_FF, 8);
    let hidd = hid.read();
    unsafe {
        inkwell_cuda::mmvq_q2_0_moe(dbuf.dev_ptr(), &ids_b, dn_rb, &hid, &mut dn, N_E, N_FF, N_FF, 8);
        inkwell_cuda::expert_wsum(&mut out, &dn, &wb, Some(&sdn_b), N_E, 8);
    }
    let dnd = dn.read();
    let outd = out.read();

    for (name, a, b) in [("gu", &ref_gu, &gud), ("hid", &ref_hid, &hidd), ("dn", &ref_dn, &dnd), ("out", &ref_out, &outd)] {
        let mut maxd = 0f32; let mut imax = 0;
        for (i, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            let d = (p - q).abs() / p.abs().max(1.0);
            if d > maxd { maxd = d; imax = i; }
        }
        println!("{name}: max_rel_diff={maxd:.4e} at {imax} ref={} got={} | ref_abs_sum={}", a[imax], b[imax], a.iter().map(|v| v.abs()).sum::<f32>());
    }
}
