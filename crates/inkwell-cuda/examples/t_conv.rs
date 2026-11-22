use inkwell_cuda as cu;
fn main() {
    let ch = 2304usize;
    let mut x = cu::DevBuf::new(ch);
    let mut state = cu::DevBuf::new(3 * ch);
    let mut w = cu::DevBuf::new(4 * ch);
    let out = cu::DevBuf::new(ch);
    x.write(&vec![0.5f32; ch]);
    state.write(&vec![1.0f32; 3 * ch]);
    w.write(&vec![0.1f32; 4 * ch]);
    unsafe {
        cu::conv_step_async(x.dev_ptr().cast_const().cast(), state.dev_ptr().cast(), 0,
            w.dev_ptr().cast_const().cast(), std::ptr::null(), ch as i32, 0, 0.4f32,
            out.dev_ptr().cast());
        cu::DevBuf::sync();
    }
    let r = out.read();
    eprintln!("out[0..3]={:?} expect {}", &r[..3], (0.5f32 + 0.4 * 3.5) * 0.4);
    let st = state.read();
    eprintln!("state[0..3]={:?} expect [1,1,0.5]", &st[..3]);
}
