//! Golden test: batched _moe expert GEMV vs CPU dequant+dot (ggml reference).
use inkwell_ggml::ink_dequant;

fn dot_ref(ty: u32, raw: &[u8], x: &[f32], nrows: usize, ncols: usize) -> Vec<f32> {
    let mut w = vec![0f32; nrows * ncols];
    unsafe { ink_dequant(ty as i32, raw.as_ptr().cast(), w.as_mut_ptr(), (nrows * ncols) as i64) };
    (0..nrows).map(|r| w[r * ncols..(r + 1) * ncols].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let g = inkwell_gguf::GgufFile::open(&path).expect("gguf");
    let mut s = 0x9e3779b9u64;
    let mut xs = |n: usize| -> Vec<f32> {
        (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 33) & 0xffff) as f32 / 65536.0 - 0.5 }).collect()
    };

    // (tensor, ggml_type, vals/block, bytes/block)
    for (name, ty, vpb, bpb) in [
        ("blk.5.ffn_gate_up_exps.weight", 16u32, 256usize, 66usize),
        ("blk.9.ffn_gate_up_exps.weight", 17, 256, 74),
        ("blk.0.ffn_gate_up_exps.weight", 19, 256, 50),
        ("blk.16.ffn_gate_up_exps.weight", 22, 256, 82),
        ("blk.5.ffn_down_exps.weight", 40, 64, 36),
        ("blk.0.ffn_down_exps.weight", 42, 64, 18),
    ] {
        let t = g.tensors.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t.ggml_type, ty, "{name} type");
        let (nrows, ncols, nexp) = (t.dims[1] as usize, t.dims[0] as usize, t.dims[2] as usize);
        let row_bytes = (t.nbytes / nexp as u64) as usize;
        let exp = nexp / 2; // test a mid expert
        let base = (g.data_start + t.offset) as usize;
        // copy whole expert table? too big — build a 1-entry table with exp's bytes
        let eraw = &g.mmap[base + exp * row_bytes..base + (exp + 1) * row_bytes];
        let n = 32usize.min(nrows);
        let x = xs(ncols);
        let ref_y = dot_ref(ty, &eraw[..n * (ncols / vpb) * bpb], &x, n, ncols);

        // device copy of just that expert (as a 1-expert table)
        let mut wsrc = inkwell_cuda::DevBuf::new_bytes(row_bytes);
        wsrc.write_bytes(eraw);
        let mut ids = inkwell_cuda::DevBuf::new_bytes(4);
        ids.write_bytes(&[0u8; 4]);
        let mut xb = inkwell_cuda::DevBuf::new(ncols);
        xb.write(&x);
        let mut yb = inkwell_cuda::DevBuf::new(nrows);
        unsafe {
            match ty {
                16 => inkwell_cuda::mmvq_iq2_xxs_moe(wsrc.dev_ptr(), &ids, row_bytes, &xb, &mut yb, nrows, ncols, 0, 1),
                17 => inkwell_cuda::mmvq_iq2_xs_moe(wsrc.dev_ptr(), &ids, row_bytes, &xb, &mut yb, nrows, ncols, 0, 1),
                19 => inkwell_cuda::mmvq_iq1_s_moe(wsrc.dev_ptr(), &ids, row_bytes, &xb, &mut yb, nrows, ncols, 0, 1),
                22 => inkwell_cuda::mmvq_iq2_s_moe(wsrc.dev_ptr(), &ids, row_bytes, &xb, &mut yb, nrows, ncols, 0, 1),
                40 => inkwell_cuda::mmvq_nvfp4_moe(wsrc.dev_ptr(), &ids, row_bytes, &xb, &mut yb, nrows, ncols, 0, 1),
                _ => inkwell_cuda::mmvq_q2_0_moe(wsrc.dev_ptr(), &ids, row_bytes, &xb, &mut yb, nrows, ncols, 0, 1),
            }
        }
        let y = yb.read();
        let mut maxd = 0f32;
        for (i, (a, b)) in ref_y.iter().zip(y.iter()).enumerate() {
            let d = (a - b).abs() / a.abs().max(1.0);
            if d > maxd { maxd = d; }
            if d > 1e-3 { eprintln!("{name} row {i}: ref={a} got={b}"); }
        }
        println!("{name} type={ty} {n}x{ncols} exp={exp} max_rel_diff={maxd:.3e}");
    }
    println!("done");
}

// full-chain check appended via --chain flag path is same file; quick inline test below
