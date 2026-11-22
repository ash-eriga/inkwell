//! minimal get_rows repro: [4,5] F32 table, pick row 2
use inkwell_run::ops::t;
use inkwell_ggml::Ptr;

fn main() {
    let rt = inkwell_run::Rt::new(4);
    let c = Ptr(unsafe { inkwell_ggml::ink_ctx_new() });
    let table = t::new(c, 0, &[4, 5]);
    let ids = t::new(c, 1, &[1]);
    let out = t::get_rows(c, table, ids);
    let dev = rt.buf(rt.device_buft(), 4096);
    dev.assign(table, 0);
    dev.assign(ids, 512);
    let data: [f32; 20] = std::array::from_fn(|i| i as f32);
    unsafe { t::set(table, data.as_ptr().cast(), 0, 80); t::set(ids, [2i32].as_ptr().cast(), 0, 4) };
    let g = Ptr(unsafe { inkwell_ggml::ink_graph_new(c.0, 64) });
    t::build_forward(g, out);
    unsafe { assert_eq!(inkwell_ggml::ink_sched_compute(rt.sched.0, g.0), 0) };
    let mut v = [0f32; 4];
    unsafe { t::get(out, v.as_mut_ptr().cast(), 0, 16) };
    println!("get_rows(2) = {v:?} (expect [8,9,10,11])");
}
