// minimal situ check
fn main() {
    let n = 16usize;
    let mut gu = inkwell_cuda::DevBuf::new(n * 2);
    let v: Vec<f32> = (0..n * 2).map(|i| if i < n { 1.0 } else { 0.5 }).collect();
    gu.write(&v);
    let mut hid = inkwell_cuda::DevBuf::new(n);
    inkwell_cuda::situ_pair(&mut hid, &gu, None, 4.0, 25.0, n, 1);
    let h = hid.read();
    println!("hid={:?}", &h[..8]);
    let a = 4.0f32 * (1.0f32 / 4.0).tanh() / (1.0 + (-1.0f32).exp());
    let b = 25.0f32 * (0.5f32 / 25.0).tanh();
    println!("expect={}", a * b);
}
