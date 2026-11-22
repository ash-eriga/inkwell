use inkwell_ggml as g;
use std::ffi::c_void;
fn main() {
    unsafe {
        let c = g::ink_ctx_new();
        let backend = g::ink_backend_cuda(0);
        let cpu = g::ink_backend_cpu(4);
        let buft = g::ink_cuda_buft(0);
        let cpu_buft = g::ink_cpu_buft();
        let mut bks = [backend, cpu];
        let mut bts = [buft, cpu_buft];
        let sched = g::ink_sched_new(bks.as_mut_ptr(), bts.as_mut_ptr(), 2, 4096);
        // weights: two F32 mats [64,8]
        let buf = g::ink_buf_alloc(buft, 1 << 26);
        let mk = |v: f32, off: usize| {
            let t = g::ink_tensor(c, 0 /*F32*/, 2, [64, 8, 1, 1].as_ptr());
            g::ink_tensor_assign(t, buf, off);
            let d: Vec<f32> = (0..512).map(|i| v + i as f32 * 0.01).collect();
            g::ink_tensor_set(t, d.as_ptr().cast(), 0, 2048);
            t
        };
        let a = mk(1.0, 0);
        let b = mk(100.0, 4096);
        // cat [64,16] at 8192
        let cat = g::ink_tensor(c, 0, 2, [64, 16, 1, 1].as_ptr());
        g::ink_tensor_assign(cat, buf, 8192);
        eprintln!("ptrs: buf={:?} a={:?} b={:?} cat={:?}",
            g::ink_buf_base(buf), g::ink_tensor_data(a), g::ink_tensor_data(b),
            g::ink_tensor_data(cat));
        let mut h = vec![0u8; 2048];
        g::ink_tensor_get(a, h.as_mut_ptr().cast(), 0, 2048);
        g::ink_tensor_set(cat, h.as_ptr().cast(), 0, 2048);
        g::ink_tensor_get(b, h.as_mut_ptr().cast(), 0, 2048);
        g::ink_tensor_set(cat, h.as_ptr().cast(), 2048, 2048);
        // x = ones [64]
        let x = g::ink_tensor(c, 0, 2, [64, 1, 1, 1].as_ptr());
        g::ink_tensor_assign(x, buf, 16384);
        g::ink_tensor_set(x, vec![1f32; 64].as_ptr().cast(), 0, 256);
        let gr = g::ink_graph_new(c, 4096);
        let out = g::ink_mul_mat(c, cat, x);
        g::ink_sched_set_tensor_backend(sched, out, backend);
        let v0 = g::ink_view_1d(c, out, 8, 0);
        let v1 = g::ink_view_1d(c, out, 8, 8);
        g::ink_set_output(out); g::ink_set_output(v0); g::ink_set_output(v1);
        g::ink_build_forward(gr, v0);
        g::ink_build_forward(gr, v1);
        let galloc = g::ink_gallocr(buft);
        assert!(g::ink_alloc_graph(galloc, gr));
        assert_eq!(g::ink_sched_compute(sched, gr), 0);
        eprintln!("post-compute: out={:?} v0={:?} v1={:?}",
            g::ink_tensor_data(out), g::ink_tensor_data(v0), g::ink_tensor_data(v1));
        let mut y = vec![0f32; 16];
        g::ink_tensor_get(out, y.as_mut_ptr().cast(), 0, 64);
        let mut y0 = vec![0f32; 8]; let mut y1 = vec![0f32; 8];
        g::ink_tensor_get(v0, y0.as_mut_ptr().cast(), 0, 32);
        g::ink_tensor_get(v1, y1.as_mut_ptr().cast(), 0, 32);
        let ea: f32 = (0..64).map(|i| 1.0 + i as f32 * 0.01).sum();
        let eb: f32 = (0..64).map(|i| 100.0 + i as f32 * 0.01).sum();
        println!("out[0]={} exp_a={} | out[8]={} exp_b={}", y[0], ea, y[8], eb);
        println!("v0[0]={} v1[0]={}", y0[0], y1[0]);
    }
}
