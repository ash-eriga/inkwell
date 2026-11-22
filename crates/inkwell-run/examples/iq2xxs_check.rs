//! Golden test: inkwell-cuda IQ2_XXS dequant vs ggml's to_float (CPU).
use inkwell_ggml::ink_dequant;

fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let name = std::env::args().nth(2).unwrap_or_else(|| "blk.5.ffn_gate_up_exps.weight".into());
    let g = inkwell_gguf::GgufFile::open(&path).expect("gguf");
    let t = g.tensors.get(&name).expect("tensor");
    assert_eq!(t.ggml_type, 16, "expected IQ2_XXS");
    let expert: u64 = std::env::args().nth(3).map_or(0, |s| s.parse().unwrap());
    // dims [ne0, ne1, n_expert]: expert slice = ne0*ne1 values = ne0*ne1/256 blocks
    let nel = t.dims[0] * t.dims[1];
    let nblocks = nel / 256;
    let bytes = (nblocks * 66) as usize;
    let off = (g.data_start + t.offset + expert * nblocks * 66) as usize;
    let raw = &g.mmap[off..off + bytes];

    // CPU reference via ggml
    let mut ref_y = vec![0f32; nel as usize];
    unsafe { ink_dequant(16, raw.as_ptr().cast(), ref_y.as_mut_ptr(), nel as i64) };

    // GPU kernel
    let mut src = inkwell_cuda::DevBuf::new_bytes(bytes);
    src.write_bytes(raw);
    let mut dst = inkwell_cuda::DevBuf::new(nel as usize);
    inkwell_cuda::dequant_iq2_xxs(&src, &mut dst);
    let y = dst.read();

    let mut max_diff = 0f32;
    let mut nbad = 0usize;
    for (i, (a, b)) in ref_y.iter().zip(y.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff { max_diff = d; }
        if a.to_bits() != b.to_bits() { nbad += 1; if nbad < 5 { eprintln!("diff at {i}: ref={a} got={b}"); } }
    }
    println!("expert {expert}: {nel} vals, max_diff={max_diff:.3e}, mismatched={nbad}");
    assert!(nbad == 0 || max_diff < 1e-6, "kernel mismatch");
    println!("IQ2_XXS dequant: PASS");
}
