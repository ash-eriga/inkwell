fn main() {
    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    let ti = &g.tensors["token_embd.weight"];
    // row 371: block-aware offset — use dequant whole row
    let n_embd = 2304u64;
    let n_col = ti.dims[1];
    let bs: u64 = 32; // IQ4_XS block? use type_block
    let (per_blk, bytes_blk) = inkwell_gguf::type_block(ti.ggml_type).unwrap();
    let row_bytes = n_embd / per_blk * bytes_blk;
    let src = unsafe { g.mmap.as_ptr().add(g.data_start as usize + ti.offset as usize + (371 * row_bytes) as usize) };
    let mut v = vec![0f32; n_embd as usize];
    let rc = unsafe { inkwell_ggml::ink_dequant(ti.ggml_type as i32, src.cast(), v.as_mut_ptr(), n_embd as i64) };
    let sum: f32 = v.iter().sum();
    println!("ty={} row371 sum={sum} first8={:?} rc={rc}", ti.ggml_type, &v[..8]);
    let _ = n_col;
}
