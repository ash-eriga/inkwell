use inkwell_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    for n in ["blk.0.attn_norm.weight","token_embd_norm.weight","blk.0.ffn_gate_inp.weight","blk.0.attn_sink_proj.weight","blk.0.attn_post_conv.weight","blk.0.attn_k_conv.weight"] {
        let t = g.tensors.get(n).unwrap();
        let lo=(g.data_start+t.offset) as usize;
        let d=&g.mmap[lo..lo+t.nbytes as usize];
        let f:&[f32]=unsafe{std::slice::from_raw_parts(d.as_ptr() as *const f32, d.len()/4)};
        println!("{n}: n={} sum={:.4} first6={:?}", f.len(), f.iter().map(|&x|x as f64).sum::<f64>(), &f[..6]);
    }
}
