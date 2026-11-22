fn main() {
    let g = inkwell_gguf::GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    let ti = &g.tensors["token_embd.weight"];
    let n_embd = 2304u64;
    let (per_blk, bytes_blk) = inkwell_gguf::type_block(ti.ggml_type).unwrap();
    let row_bytes = n_embd / per_blk * bytes_blk;
    for row in [0u64, 1, 371, 2, 3] {
        let src = unsafe { g.mmap.as_ptr().add(g.data_start as usize + ti.offset as usize + (row * row_bytes) as usize) };
        let mut v = vec![0f32; n_embd as usize];
        unsafe { inkwell_ggml::ink_dequant(ti.ggml_type as i32, src.cast(), v.as_mut_ptr(), n_embd as i64) };
        println!("row{row} sum={} first3={:?}", v.iter().sum::<f32>(), &v[..3]);
    }
}
