//! Golden test: own GEMV kernels vs CPU dequant+dot reference.
use inkwell_ggml::ink_dequant;

fn dot_ref(ty: u32, raw: &[u8], x: &[f32], nrows: usize, ncols: usize, bpr: usize) -> Vec<f32> {
    let mut w = vec![0f32; nrows * ncols];
    unsafe { ink_dequant(ty as i32, raw.as_ptr().cast(), w.as_mut_ptr(), (nrows * ncols) as i64) };
    let _ = bpr;
    (0..nrows).map(|r| w[r * ncols..(r + 1) * ncols].iter().zip(x).map(|(a, b)| a * b).sum()).collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let g = inkwell_gguf::GgufFile::open(&path).expect("gguf");

    // deterministic pseudo-random x
    let mut xs = |n: usize| -> Vec<f32> {
        let mut s = 0x9e3779b9u64;
        (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 33) & 0xffff) as f32 / 65536.0 - 0.5 }).collect()
    };

    for (name, ty, vals_per_block, _bpb) in [
        ("blk.5.ffn_gate_up_exps.weight", 16, 256usize, 66usize),
        ("blk.0.ffn_gate_up_exps.weight", 19, 256usize, 50usize),
        ("blk.0.ffn_down_exps.weight", 42, 64usize, 18usize),
        ("blk.5.ffn_down_exps.weight", 40, 64usize, 36usize),
    ] {
        let t = g.tensors.get(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(t.ggml_type, ty, "{name} type");
        let (nrows, ncols) = (t.dims[1] as usize, t.dims[0] as usize);
        let bpr = ncols / vals_per_block; // blocks per row
        let bpb = match ty { 16 => 66, 19 => 50, 40 => 36, _ => 18 };
        let n = 64usize.min(nrows);       // test 64 rows of expert 0
        let raw = &g.mmap[(g.data_start + t.offset) as usize..];
        let raw = &raw[..n * bpr * bpb];
        let x = xs(ncols);
        let ref_y = dot_ref(ty, raw, &x, n, ncols, bpr);

        let mut wsrc = inkwell_cuda::DevBuf::new_bytes(raw.len());
        wsrc.write_bytes(raw);
        let mut xb = inkwell_cuda::DevBuf::new(ncols);
        xb.write(&x);
        let mut yb = inkwell_cuda::DevBuf::new(n);
        match ty {
            16 => inkwell_cuda::mmvq_iq2_xxs(&wsrc, &xb, &mut yb, n, ncols),
            19 => inkwell_cuda::mmvq_iq1_s(&wsrc, &xb, &mut yb, n, ncols),
            40 => inkwell_cuda::mmvq_nvfp4(&wsrc, &xb, &mut yb, n, ncols),
            _ => inkwell_cuda::mmvq_q2_0(&wsrc, &xb, &mut yb, n, ncols),
        }
        let y = yb.read();
        let mut maxd = 0f32;
        for (i, (a, b)) in ref_y.iter().zip(y.iter()).enumerate() {
            let d = (a - b).abs() / a.abs().max(1.0);
            if d > maxd { maxd = d; }
            if d > 1e-4 { eprintln!("row {i}: ref={a} got={b}"); }
        }
        println!("{name}: {n}x{ncols} max_rel_diff={maxd:.3e}");
        assert!(maxd < 1e-3);
    }
    // bench: expert GEMV from device-resident weights
    let t = g.tensors.get("blk.5.ffn_gate_up_exps.weight").expect("t");
    let (nrows, ncols) = (t.dims[1] as usize, t.dims[0] as usize);
    let bpr = ncols / 256;
    let raw = &g.mmap[(g.data_start + t.offset) as usize..(g.data_start + t.offset + (nrows * bpr * 66) as u64) as usize];
    let mut wsrc = inkwell_cuda::DevBuf::new_bytes(raw.len());
    wsrc.write_bytes(raw);
    let mut xb = inkwell_cuda::DevBuf::new(ncols);
    xb.write(&xs(ncols));
    let mut yb = inkwell_cuda::DevBuf::new(nrows);
    for _ in 0..3 { inkwell_cuda::mmvq_iq2_xxs(&wsrc, &xb, &mut yb, nrows, ncols) }
    let t0 = std::time::Instant::now();
    let iters = 50;
    for _ in 0..iters { inkwell_cuda::mmvq_iq2_xxs(&wsrc, &xb, &mut yb, nrows, ncols) }
    let dt = t0.elapsed() / iters;
    println!("expert GEMV 2304x2304 IQ2_XXS: {:?} ({:.1} GB/s weight traffic)", dt, raw.len() as f64 / dt.as_secs_f64() / 1e9);
    println!("GEMV kernels: PASS");
}
