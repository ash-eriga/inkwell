use inkwell_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    let t = g.tensors.get("blk.0.ffn_gate_inp.weight").unwrap();
    let lo=(g.data_start+t.offset) as usize;
    let d=&g.mmap[lo..lo+t.nbytes as usize];
    let f:&[f32]=unsafe{std::slice::from_raw_parts(d.as_ptr() as *const f32, d.len()/4)};
    println!("gate row0 {:?} row2304.. {:?}", &f[..6], &f[2304..2310]);
}
