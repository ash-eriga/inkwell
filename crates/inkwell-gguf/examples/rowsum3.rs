use inkwell_gguf::GgufFile;
fn main() {
    let g = GgufFile::open(std::env::args().nth(1).unwrap()).unwrap();
    for k in ["blk.0.ffn_gate_up_exps.weight","blk.0.ffn_gate_up_exps.scale","blk.0.ffn_down_exps.weight","blk.0.ffn_down_exps.scale","blk.0.attn_q.weight","blk.6.loop_q_ctrl_down.weight","blk.6.loop_msg_query.weight"] {
        let t = g.tensors.get(k).unwrap();
        println!("{k}: dims={:?} ty={} nbytes={}", t.dims, t.ggml_type, t.nbytes);
    }
}
