use inkwell_cuda as _;
fn main() {
    let mut w = inkwell_cuda::DevBuf::new_bytes(16 * 9 * 136); // 16 rows x 9 blocks x 136B
    let mut x = inkwell_cuda::DevBuf::new(2304);
    let mut y = inkwell_cuda::DevBuf::new(16);
    w.write_bytes(&vec![0u8; 16 * 9 * 136]);
    x.write(&vec![0.0f32; 2304]);
    let wp: *const std::ffi::c_void = w.dev_ptr();
    let arr = [wp];
    unsafe {
        inkwell_cuda::mmvq_iq4_xs_moe(arr.as_ptr().cast(), x.dev_ptr(), &mut y, 16, 2304, 2304, 1);
    }
    println!("y[0]={}", y.read()[0]);
    // big nrows like lm_head
    let mut w2 = inkwell_cuda::DevBuf::new_bytes(2304 * 9 * 136);
    let mut y2 = inkwell_cuda::DevBuf::new(2304);
    w2.write_bytes(&vec![0u8; 2304 * 9 * 136]);
    let wp2: *const std::ffi::c_void = w2.dev_ptr();
    let arr2 = [wp2];
    unsafe {
        inkwell_cuda::mmvq_iq4_xs_moe(arr2.as_ptr().cast(), x.dev_ptr(), &mut y2, 2304, 2304, 2304, 1);
    }
    println!("ok big y[0]={}", y2.read()[0]);
}
