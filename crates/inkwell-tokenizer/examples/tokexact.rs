//! Exact-length prompt construction and continuation-id extraction for ACTQ measurements.

use inkwell_gguf::GgufFile;
use inkwell_tokenizer::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: tokexact <model-dir> <input.txt> <tokens> <out.txt>");
    let input = args.next().expect("input");
    let n: usize = args.next().expect("tokens").parse().expect("token count");
    let out = args.next().expect("output");
    let gg = if std::path::Path::new(&model).is_dir() {
        format!("{model}/Inkling-2-Mini-UD-IQ2_XXS.gguf")
    } else { model };
    let g = GgufFile::open(&gg).expect("open gguf");
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");
    let text = std::fs::read_to_string(input).expect("read input");
    let ids = tok.encode(&text);
    assert!(ids.len() >= n, "input has {} tokens, need {n}", ids.len());
    let cut = tok.decode(&ids[..n]).expect("decode prefix");
    let check_ids = tok.encode(&cut);
    assert_eq!(check_ids.len(), n, "roundtrip prefix token count");
    assert_eq!(check_ids, ids[..n], "roundtrip prefix token ids");
    std::fs::write(&out, &cut).expect("write output");
    let mut promptids = Vec::new();
    for &id in &ids[..n] { promptids.extend_from_slice(&id.to_le_bytes()); }
    std::fs::write(format!("{out}.ids"), promptids).expect("write prompt ids");
    if let Some(idsout) = args.next() {
        assert!(ids.len() >= n + 256, "input has {} tokens, need {}", ids.len(), n + 256);
        let mut bytes = Vec::new();
        for &id in ids.iter().skip(n).take(256) { bytes.extend_from_slice(&id.to_le_bytes()); }
        std::fs::write(idsout, bytes).expect("write continuation ids");
        if let Some(targetout) = args.next() {
            let mut target = Vec::new();
            target.extend_from_slice(&ids[n - 1].to_le_bytes());
            for &id in ids.iter().skip(n).take(256) { target.extend_from_slice(&id.to_le_bytes()); }
            std::fs::write(targetout, target).expect("write target ids");
        }
    }
    println!("{out}: {n} tokens, {} bytes", cut.len());
}
