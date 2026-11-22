//! GPU smoke test: rmsnorm + clip on the GTX 1060.
fn main() {
    let n = 2304;
    println!("devices: {}", inkwell_cuda::device_count());
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37) - 400.0).collect();
    let w: Vec<f32> = vec![1.5; n];
    let mut dx = inkwell_cuda::DevBuf::new(n);
    let dw = inkwell_cuda::DevBuf::new(n);
    let mut dy = inkwell_cuda::DevBuf::new(n);
    dx.write(&x);
    let mut dw2 = dw;
    dw2.write(&w);
    inkwell_cuda::rmsnorm(&dx, Some(&dw2), &mut dy, n, 1e-6);
    let y = dy.read();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-6).sqrt();
    let exp = x[7] / rms * 1.5;
    println!("y[7]={} expected={} diff={}", y[7], exp, (y[7] - exp).abs());

    // clip
    let mut dc = inkwell_cuda::DevBuf::new(256);
    dc.write(&vec![10.0f32; 256]);
    inkwell_cuda::clip(&mut dc, None, 128, 2, 22.6);
    let c = dc.read();
    let norm = c[..128].iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("clip row0 norm={norm} (<=22.6)");
}
