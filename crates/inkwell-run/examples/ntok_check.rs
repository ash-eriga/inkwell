// ntok gemv vs per-token gemv on a real REXL3 weight.
fn main() {
    let repo = inkwell_st::StRepo::open("/home/ash/models/inkling").unwrap();
    let base = std::env::var("W").unwrap_or("model.llm.layers.0.attn.wo".into());
    let tr = repo.tensor(&format!("{base}.trellis")).expect("trellis");
    let (k, n) = ((tr.shape[0] * 16) as usize, (tr.shape[1] * 16) as usize);
    let k16 = tr.shape[2] as usize;
    let (ka, halfk) = (k16 / 16, k16 % 16 != 0);
    eprintln!("{base}: k={k} n={n} ka={ka} halfk={halfk}");
    let mut trel = inkwell_cuda::DevBuf::new_bytes(tr.nbytes() as usize);
    trel.write_bytes(repo.data(&format!("{base}.trellis")).unwrap());
    let su = repo.tensor(&format!("{base}.suh")).unwrap();
    let mut suh = inkwell_cuda::DevBuf::new_bytes(su.nbytes() as usize);
    suh.write_bytes(repo.data(&format!("{base}.suh")).unwrap());
    let sv = repo.tensor(&format!("{base}.svh")).unwrap();
    let mut svh = inkwell_cuda::DevBuf::new_bytes(sv.nbytes() as usize);
    svh.write_bytes(repo.data(&format!("{base}.svh")).unwrap());

    let mut x = inkwell_cuda::DevBuf::new(8 * k);
    let xv: Vec<f32> = (0..8 * k).map(|i| ((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0).collect();
    x.write(&xv);
    let mut y_ref = inkwell_cuda::DevBuf::new(8 * n);
    let mut y_nt = inkwell_cuda::DevBuf::new(8 * n);
    let mut scratch = inkwell_cuda::DevBuf::new(8 * k + 64 * n);
    let mut scr1 = inkwell_cuda::DevBuf::new(k + 8 * n);
    unsafe {
        for tt in 0..8 {
            inkwell_cuda::exl3_gemv_async(
                trel.dev_ptr(), suh.dev_ptr(), svh.dev_ptr(),
                x.dev_ptr().cast::<f32>().add(tt * k),
                y_ref.dev_ptr().cast::<f32>().add(tt * n),
                k, n, ka, halfk, scr1.dev_ptr().cast());
        }
        inkwell_cuda::exl3_gemv_ntok_async(
            trel.dev_ptr(), suh.dev_ptr(), svh.dev_ptr(),
            x.dev_ptr().cast(), y_nt.dev_ptr().cast(),
            k, n, ka, halfk, scratch.dev_ptr().cast());
        inkwell_cuda::DevBuf::sync();
    }
    let a = y_ref.read();
    let b = y_nt.read();
    let mut worst = 0f32;
    let mut wi = 0;
    for i in 0..8 * n {
        let d = (a[i] - b[i]).abs();
        if d > worst { worst = d; wi = i; }
    }
    println!("worst|ref-ntok|={worst} at {wi} (ref={} ntok={})", a[wi], b[wi]);
    // timing: 8x single vs 1x ntok
    let t0 = std::time::Instant::now();
    unsafe {
        for _ in 0..20 {
            for tt in 0..8 {
                inkwell_cuda::exl3_gemv_async(
                    trel.dev_ptr(), suh.dev_ptr(), svh.dev_ptr(),
                    x.dev_ptr().cast::<f32>().add(tt * k),
                    y_ref.dev_ptr().cast::<f32>().add(tt * n),
                    k, n, ka, halfk, scr1.dev_ptr().cast());
            }
        }
        inkwell_cuda::DevBuf::sync();
    }
    let t_seq = t0.elapsed() / 20;
    let t0 = std::time::Instant::now();
    unsafe {
        for _ in 0..20 {
            inkwell_cuda::exl3_gemv_ntok_async(
                trel.dev_ptr(), suh.dev_ptr(), svh.dev_ptr(),
                x.dev_ptr().cast(), y_nt.dev_ptr().cast(),
                k, n, ka, halfk, scratch.dev_ptr().cast());
        }
        inkwell_cuda::DevBuf::sync();
    }
    let t_nt = t0.elapsed() / 20;
    println!("8x single: {:?} | 1x ntok: {:?} | {:.2}x", t_seq, t_nt, t_seq.as_secs_f64() / t_nt.as_secs_f64());
}
