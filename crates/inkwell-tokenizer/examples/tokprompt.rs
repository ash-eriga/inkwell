use inkwell_gguf::GgufFile;
use inkwell_tokenizer::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: tokprompt <model-dir> <tokens> <out.txt>");
    let n: usize = args.next().expect("tokens").parse().expect("token count");
    let out = args.next().expect("output path");
    let gg = if std::path::Path::new(&model).is_dir() {
        format!("{model}/Inkling-2-Mini-UD-IQ2_XXS.gguf")
    } else { model };
    let g = GgufFile::open(&gg).expect("open gguf");
    let tok = Tokenizer::from_gguf(&g).expect("tokenizer");

    let subjects = [
        "The engineer", "A cartographer", "The archivist", "A musician", "The gardener",
        "An electrician", "The teacher", "A surgeon", "The programmer", "A journalist",
    ];
    let verbs = [
        "measured", "repaired", "described", "questioned", "carried", "assembled",
        "photographed", "translated", "remembered", "tested",
    ];
    let objects = [
        "the brass instrument", "a narrow staircase", "the satellite diagram", "an old notebook",
        "the ceramic valve", "a box of letters", "the northern window", "an unreliable engine",
        "the hidden archive", "a cluster of wires",
    ];
    let reasons = [
        "the morning train was late", "nobody had documented the procedure",
        "the weather changed without warning", "the original plan was incomplete",
        "a small detail contradicted the report", "the client asked for evidence",
        "the machine behaved differently under load", "the map omitted a crossing",
    ];
    let pads = [
        "time", "person", "year", "way", "day", "thing", "man", "world",
        "life", "hand", "part", "child", "eye", "woman", "place", "work",
        "week", "case", "point", "company", "number", "group", "problem", "fact",
        "water", "room", "mother", "area", "money", "story", "month", "night",
    ];
    let pads: Vec<&str> = pads.into_iter()
        .filter(|w| tok.encode(&format!(" {w}")).len() == 1)
        .collect();

    let mut text = String::new();
    let mut seed = 0x243f6a8885a308d3u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for sentence_idx in 0.. {
        let s = format!(
            "{} {} {} because {}. ",
            subjects[(rng() as usize) % subjects.len()],
            verbs[(rng() as usize) % verbs.len()],
            objects[(rng() as usize) % objects.len()],
            reasons[(rng() as usize) % reasons.len()],
        );
        let trial = format!("{text}{s}");
        if tok.encode(&trial).len() > n {
            break;
        }
        text = trial;
        if sentence_idx > n {
            panic!("sentence loop did not terminate");
        }
    }
    let mut pi = 0usize;
    loop {
        let got = tok.encode(&text).len();
        if got == n { break; }
        assert!(got < n, "prompt construction overshot: {got} > {n}");
        text.push(' ');
        text.push_str(pads[pi % pads.len()]);
        pi += 1;
    }
    assert_eq!(tok.encode(&text).len(), n, "final prompt token count");
    std::fs::write(&out, &text).expect("write prompt");
    println!("{out}: {n} tokens, {} bytes", text.len());
}
