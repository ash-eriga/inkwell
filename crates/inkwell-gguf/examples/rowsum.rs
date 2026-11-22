use inkwell_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    for n in ["blk.6.loop_msg_query.weight", "blk.7.loop_msg_query.weight", "blk.5.attn_res_query.weight", "blk.6.attn_res_query.weight", "blk.0.ffn_res_query.weight"] {
        if let Some(t) = g.tensors.get(n) {
            let lo=(g.data_start+t.offset) as usize; let d=&g.mmap[lo..lo+t.nbytes as usize];
            let f: &[f32] = unsafe { std::slice::from_raw_parts(d.as_ptr() as *const f32, d.len() / 4) };
            let mut rows = Vec::new();
            // ggml dims [2304, R]: row-major file = column-major ggml; "row" r = col r
            for r in 0..(f.len() / 2304) {
                let s: f64 = f[r * 2304..(r + 1) * 2304].iter().map(|&x| x as f64).sum();
                rows.push(s);
            }
            println!("{n}: {rows:?}");
        } else { println!("{n}: MISSING"); }
    }
}
