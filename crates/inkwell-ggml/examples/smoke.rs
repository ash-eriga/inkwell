//! Run the ggml-cuda smoke graph.
fn main() {
    let v = inkwell_ggml::smoke();
    #[expect(clippy::cast_precision_loss)]
    let rms = ((0..4096).map(|i| { let x = i as f32 * 0.5 - 1000.0; x * x }).sum::<f32>() / 4096.0 + 1e-6).sqrt();
    let exp = (7.0 * 0.5 - 1000.0) / rms * 2.0;
    println!("y[7]={} expected={exp} diff={}", v[7], (v[7] - exp).abs());
}
