use inkwell_gguf::GgufFile;
use inkwell_tokenizer::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: tokids <model-dir> <text> <tokens> <out.ids>");
    let text = args.next().expect("text");
    let n: usize = args.next().expect("tokens").parse().expect("token count");
    let out = args.next().expect("output");
    let gg = if std::path::Path::new(&model).is_dir() {
        format!("{model}/Inkling-2-Mini-UD-IQ2_XXS.gguf")
    } else { model };
    let g = GgufFile::open(&gg).expect("open gguf");
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
    let ids = tok.encode(&std::fs::read_to_string(text).expect("read text"));
    assert!(ids.len() >= n, "need {n} tokens, got {}", ids.len());
    let mut bytes = Vec::with_capacity(n * 4);
    for id in &ids[..n] { bytes.extend_from_slice(&id.to_le_bytes()); }
    std::fs::write(out, bytes).expect("write ids");
}
