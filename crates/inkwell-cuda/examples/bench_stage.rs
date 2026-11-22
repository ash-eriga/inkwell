use inkwell_cuda as cu;
use std::ffi::c_void;
extern "C" {
    fn cudaEventCreate(e: *mut *mut c_void) -> i32;
    fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
    fn cudaEventSynchronize(e: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
}
fn main() {
    let total = 245 * 1024 * 1024usize;
    let pinned = cu::host_alloc_mapped(total);
    unsafe { std::ptr::write_bytes(pinned.host, 0xAB, total) };
    // fake descs: 128 experts, trellis ptrs into pinned blob at 672KB stride
    let mut descs = vec![0u64; 128 * 4];
    for e in 0..128 {
        let b = pinned.dev as u64 + (e as u64) * 672768 * 3;
        descs[e * 4] = b; descs[e * 4 + 1] = b + 663552; descs[e * 4 + 2] = b + 663552 + 4608;
        descs[e * 4 + 3] = 2;
    }
    let desc_pb = cu::host_alloc_mapped(128 * 32);
    unsafe { std::ptr::copy_nonoverlapping(descs.as_ptr().cast::<u8>(), desc_pb.host, 128 * 32) };
    let ids = [0i32, 3, 7, 11, 22, 44, 66, 99];
    let mut ids_dev = cu::DevBuf::new_bytes(64); ids_dev.write_bytes(unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), 32) });
    let stage = cu::DevBuf::new_bytes(8 * 3 * 672768);
    let sd = cu::DevBuf::new_bytes(128 * 32);
    unsafe {
        let (mut e0, mut e1) = (std::ptr::null_mut(), std::ptr::null_mut());
        cudaEventCreate(&raw mut e0); cudaEventCreate(&raw mut e1);
        cudaEventRecord(e0, std::ptr::null_mut());
        for _ in 0..10 {
            inkwell_cuda::launch_stage_only(desc_pb.dev, ids_dev.dev_ptr() as *const i32, stage.dev_ptr(), sd.dev_ptr());
        }
        cudaEventRecord(e1, std::ptr::null_mut());
        cudaEventSynchronize(e1);
        let mut ms = 0f32; cudaEventElapsedTime(&raw mut ms, e0, e1);
        eprintln!("stage kernel: {:.3} ms/call (16MB pinned->dev)", ms / 10.0);

        // full moe call: x[8*2304] xh gu hid dn w out
        let mut x = cu::DevBuf::new_bytes(8 * 2304 * 4); x.write_bytes(unsafe{std::slice::from_raw_parts(vec![0.01f32;8*2304].as_ptr().cast::<u8>(), 8*2304*4)});
        let mut xh = cu::DevBuf::new_bytes(8 * 2304 * 4);
        let mut gu = cu::DevBuf::new_bytes(8 * 2304 * 4);
        let mut hid = cu::DevBuf::new_bytes(8 * 2304 * 4);
        let mut dn = cu::DevBuf::new_bytes(8 * 2304 * 4);
        let mut wv = cu::DevBuf::new_bytes(64); wv.write_bytes(unsafe{std::slice::from_raw_parts(vec![0.125f32;8].as_ptr().cast::<u8>(), 32)});
        let mut out = cu::DevBuf::new_bytes(2304 * 4);
        cudaEventRecord(e0, std::ptr::null_mut());
        for _ in 0..10 {
            cu::exl3_moe_async(desc_pb.dev, desc_pb.dev, desc_pb.dev,
                ids_dev.dev_ptr() as *const i32, x.dev_ptr().cast::<f32>(),
                xh.dev_ptr().cast::<f32>(), gu.dev_ptr().cast::<f32>(), hid.dev_ptr().cast::<f32>(), dn.dev_ptr().cast::<f32>(),
                wv.dev_ptr().cast::<f32>(), out.dev_ptr().cast::<f32>(), 8,
                stage.dev_ptr(), sd.dev_ptr(), sd.dev_ptr(), sd.dev_ptr());
        }
        cudaEventRecord(e1, std::ptr::null_mut());
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&raw mut ms, e0, e1);
        eprintln!("moe call: {:.3} ms (pinned descs)", ms / 10.0);
    }
}
