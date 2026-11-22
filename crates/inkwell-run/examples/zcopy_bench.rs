//! Zero-copy expert GEMV: weights read from GGUF mmap over PCIe.
fn main() {
    let path = std::env::args().nth(1).expect("gguf");
    let g = inkwell_gguf::GgufFile::open(&path).expect("gguf");
    let t = g.tensors.get("blk.5.ffn_gate_up_exps.weight").expect("t");
    let (nrows, ncols) = (t.dims[1] as usize, t.dims[0] as usize);
    let nbytes = (nrows * ncols / 256 * 66) as usize;
    let off = (g.data_start + t.offset) as usize;
    // page-align the region start for cudaHostRegister
    let mut host = vec![0u8; nbytes];
    host.copy_from_slice(&g.mmap[off..off + nbytes]);
    let dptr = inkwell_cuda::host_register_mapped(&mut host);
    let w = dptr;
    let mut xb = inkwell_cuda::DevBuf::new(ncols);
    xb.write(&vec![0.01f32; ncols]);
    let mut yb = inkwell_cuda::DevBuf::new(nrows);
    for _ in 0..2 { unsafe { inkwell_cuda::mmvq_iq2_xxs_host(w.cast(), &xb, &mut yb, nrows, ncols) } }
    let t0 = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters { unsafe { inkwell_cuda::mmvq_iq2_xxs_host(w.cast(), &xb, &mut yb, nrows, ncols) } }
    let dt = t0.elapsed() / iters;
    println!("zero-copy expert GEMV: {:?} ({:.1} GB/s over PCIe)", dt, nbytes as f64 / dt.as_secs_f64() / 1e9);
}
