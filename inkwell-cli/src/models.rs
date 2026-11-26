//! Local EXL3 model discovery and curated Hugging Face EXL3 preset downloader.

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";

/// A curated Hugging Face EXL3 model preset.
#[derive(Debug, Clone, Copy)]
pub struct HfPreset {
    /// Short local directory slug under `~/models/`.
    pub slug: &'static str,
    /// Hugging Face repository ID (`owner/repo`).
    pub repo_id: &'static str,
    /// Human-readable quantization / footprint note.
    pub detail: &'static str,
    /// Approximate download size.
    pub size_label: &'static str,
    /// SHA-256 of `model.safetensors.index.json` (used for fast local deduplication).
    pub index_sha256: &'static str,
}

/// Curated EXL3-only presets for v0.1.
pub const EXL3_PRESETS: &[HfPreset] = &[
    HfPreset {
        slug: "inkling-2-mini-exl3-2.26bpw",
        repo_id: "vcruz305/Inkling-2-Mini-SAGE-REXL3-2.26bpw",
        detail: "EXL3 2.26 bpw · recommended for 6 GB VRAM",
        size_label: "~8.3 GB",
        index_sha256: "2d6289f043f8ba8e99cfcffad03ce3c866f812a2a683ba6cf13e7f31cf5984cc",
    },
    HfPreset {
        slug: "inkling-2-mini-exl3-2.92bpw",
        repo_id: "vcruz305/Inkling-2-Mini-SAGE-REXL3-2.92bpw",
        detail: "EXL3 2.92 bpw · 8–12 GB VRAM",
        size_label: "~10.5 GB",
        index_sha256: "e08f8a1bd22ec73339f434fcf863f730f1345c50da14542ba1de5b9f5f4334a6",
    },
    HfPreset {
        slug: "inkling-2-mini-exl3-4.23bpw",
        repo_id: "vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw",
        detail: "EXL3 4.23 bpw · 16+ GB VRAM",
        size_label: "~14.5 GB",
        index_sha256: "ff7459c15c3bcfc80fc72a0f167693532cf7ba3e1084f5e9320d61e763806567",
    },
];

/// A discovered local EXL3 model directory.
#[derive(Debug, Clone)]
pub struct LocalModel {
    /// Directory name.
    pub name: String,
    /// Absolute path.
    pub path: PathBuf,
    /// Total bytes of non-symlink `.safetensors` shards in this directory.
    pub size_bytes: u64,
    /// SHA-256 of `model.safetensors.index.json`.
    pub index_sha256: String,
    /// Matched curated preset (when SHA-256 matches).
    pub matched_preset: Option<&'static HfPreset>,
}

/// Root directory where local models live (`$INKWELL_MODELS_DIR` or `~/models`).
#[must_use]
pub fn models_dir() -> PathBuf {
    if let Ok(d) = std::env::var("INKWELL_MODELS_DIR") {
        if !d.trim().is_empty() {
            return PathBuf::from(d);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_owned());
    PathBuf::from(home).join("models")
}

fn is_exl3_dir(dir: &Path) -> bool {
    dir.is_dir() && dir.join("model.safetensors.index.json").is_file()
}

fn safetensors_size(dir: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(dir) else { return 0 };
    let mut total = 0u64;
    for entry in rd.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            if let Ok(meta) = fs::symlink_metadata(&p) {
                if !meta.file_type().is_symlink() {
                    total += meta.len();
                }
            }
        }
    }
    total
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
        0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
        0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
        0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
        0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
        0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
        0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
        0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
        0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let rem = data.len() % 64;
    let pad_len = if rem < 56 { 56 - rem } else { 120 - rem };
    let mut buf = Vec::with_capacity(data.len() + pad_len + 8);
    buf.extend_from_slice(data);
    buf.push(0x80);
    buf.resize(data.len() + pad_len, 0);
    buf.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in buf.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[4 * i],
                chunk[4 * i + 1],
                chunk[4 * i + 2],
                chunk[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, delta) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(delta);
        }
    }
    let mut out = String::with_capacity(64);
    for word in h {
        use std::fmt::Write as _;
        let _ = write!(out, "{word:08x}");
    }
    out
}

fn index_sha256(dir: &Path) -> String {
    fs::read(dir.join("model.safetensors.index.json"))
        .map(|b| sha256_hex(&b))
        .unwrap_or_default()
}

/// Scan `models_dir()` for downloaded EXL3 model directories.
#[must_use]
pub fn discover_local_exl3() -> Vec<LocalModel> {
    let root = models_dir();
    let Ok(rd) = fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for entry in rd.flatten() {
        let p = entry.path();
        if is_exl3_dir(&p) {
            let name = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_owned();
            let sha = index_sha256(&p);
            let matched_preset = EXL3_PRESETS.iter().find(|pr| pr.index_sha256 == sha);
            found.push(LocalModel {
                name,
                size_bytes: safetensors_size(&p),
                path: p,
                index_sha256: sha,
                matched_preset,
            });
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

fn missing_presets(local: &[LocalModel]) -> Vec<&'static HfPreset> {
    EXL3_PRESETS
        .iter()
        .filter(|p| !local.iter().any(|m| m.index_sha256 == p.index_sha256))
        .collect()
}

fn format_gib(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
}

fn color(tty: bool, code: &'static str) -> &'static str {
    if tty { code } else { "" }
}

fn render_catalog(local: &[LocalModel], available: &[&HfPreset], vision_enabled: bool) {
    let tty = io::stdout().is_terminal();
    let root = models_dir();
    let v_tag = if vision_enabled {
        format!("{}[vision: ON]{}", color(tty, GREEN), color(tty, RESET))
    } else {
        format!("{}[vision: OFF]{}", color(tty, DIM), color(tty, RESET))
    };

    println!(
        "\n{}{}inkwell models{}  {}({}){}",
        color(tty, BOLD),
        color(tty, CYAN),
        color(tty, RESET),
        color(tty, DIM),
        root.display(),
        color(tty, RESET)
    );

    println!("\n{}Downloaded (EXL3):{}", color(tty, BOLD), color(tty, RESET));
    if local.is_empty() {
        println!(
            "  {}(none found in {}){}",
            color(tty, DIM),
            root.display(),
            color(tty, RESET)
        );
    } else {
        for (i, m) in local.iter().enumerate() {
            let tag = m
                .matched_preset
                .map_or("EXL3", |p| p.detail.split(" · ").next().unwrap_or("EXL3"));
            println!(
                "  {}[{}]{} {:<30} {}{} · {} · {}{}  {v_tag}",
                color(tty, GREEN),
                i + 1,
                color(tty, RESET),
                m.name,
                color(tty, DIM),
                tag,
                format_gib(m.size_bytes),
                m.path.display(),
                color(tty, RESET)
            );
        }
    }

    if !available.is_empty() {
        println!(
            "\n{}Available to download (Hugging Face · EXL3):{}",
            color(tty, BOLD),
            color(tty, RESET)
        );
        for (j, p) in available.iter().enumerate() {
            println!(
                "  {}[{}]{} {:<44} {}{} · {}{}  {v_tag}",
                color(tty, CYAN),
                local.len() + j + 1,
                color(tty, RESET),
                p.repo_id,
                color(tty, DIM),
                p.detail,
                p.size_label,
                color(tty, RESET)
            );
        }
    }
    println!();
}

/// Print all local EXL3 models and not-yet-downloaded Hugging Face EXL3 presets.
pub fn print_model_catalog() {
    let local = discover_local_exl3();
    let available = missing_presets(&local);
    let vision_enabled = std::env::var("INKWELL_VISION").as_deref() == Ok("1");
    render_catalog(&local, &available, vision_enabled);
}

/// Download a curated Hugging Face EXL3 preset into `~/models/<slug>`.
pub fn download_preset(preset: &HfPreset) -> Result<PathBuf, String> {
    let dest = models_dir().join(preset.slug);
    fs::create_dir_all(&dest)
        .map_err(|e| format!("failed to create {}: {e}", dest.display()))?;

    eprintln!(
        "inkwell: downloading {} ({}) -> {} ...",
        preset.repo_id,
        preset.size_label,
        dest.display()
    );

    let hf_ok = Command::new("hf")
        .args([
            "download",
            preset.repo_id,
            "--local-dir",
            &dest.to_string_lossy(),
        ])
        .status();

    match hf_ok {
        Ok(st) if st.success() => return Ok(dest),
        Ok(_) => {
            return Err(format!(
                "hf download failed for {} (check network or `hf auth login`)",
                preset.repo_id
            ));
        }
        Err(_) => {}
    }

    let cli_ok = Command::new("huggingface-cli")
        .args([
            "download",
            preset.repo_id,
            "--local-dir",
            &dest.to_string_lossy(),
        ])
        .status();

    match cli_ok {
        Ok(st) if st.success() => Ok(dest),
        Ok(_) => Err(format!("huggingface-cli download failed for {}", preset.repo_id)),
        Err(_) => Err(format!(
            "neither `hf` nor `huggingface-cli` was found in PATH.\nInstall with: pip install -U huggingface_hub\nThen run: hf download {} --local-dir {}",
            preset.repo_id,
            dest.display()
        )),
    }
}

/// Selection returned by the interactive model picker.
#[derive(Debug, Clone)]
pub struct ModelChoice {
    /// Local path to the selected (or freshly downloaded) EXL3 model directory.
    pub path: String,
    /// Soft vision-tower toggle state (`false` by default in v0.1).
    pub vision_enabled: bool,
}

/// Interactive selector used when `inkwell serve` or `inkwell chat` is invoked without `<model>`.
pub fn select_or_download_model() -> Result<ModelChoice, String> {
    let tty = io::stdout().is_terminal();
    let local = discover_local_exl3();
    let available = missing_presets(&local);
    let mut vision_enabled = std::env::var("INKWELL_VISION").as_deref() == Ok("1");

    let total = local.len() + available.len();
    if !io::stdin().is_terminal() {
        render_catalog(&local, &available, vision_enabled);
        if let Some(first) = local.first() {
            return Ok(ModelChoice {
                path: first.path.to_string_lossy().into_owned(),
                vision_enabled,
            });
        }
        return Err("no local EXL3 models found and stdin is not interactive; pass <model> or run in a terminal".to_owned());
    }

    let default_idx = 1usize;
    loop {
        render_catalog(&local, &available, vision_enabled);
        print!(
            "{}Select model [1-{total}] (default {default_idx}, v=toggle vision, q=quit):{} ",
            color(tty, BOLD),
            color(tty, RESET)
        );
        let _ = io::stdout().flush();
        let mut line = String::new();
        let n = io::stdin()
            .read_line(&mut line)
            .map_err(|e| format!("failed to read selection: {e}"))?;
        if n == 0 {
            return Err("cancelled".to_owned());
        }
        let s = line.trim();
        if s.eq_ignore_ascii_case("q") || s.eq_ignore_ascii_case("quit") {
            return Err("cancelled".to_owned());
        }
        if s.eq_ignore_ascii_case("v") || s.eq_ignore_ascii_case("vision") {
            vision_enabled = !vision_enabled;
            continue;
        }
        let choice: usize = if s.is_empty() {
            default_idx
        } else {
            match s.parse() {
                Ok(v) if (1..=total).contains(&v) => v,
                _ => {
                    println!(
                        "{}Please enter 1-{total}, 'v' to toggle vision tower, or 'q' to quit.{}",
                        color(tty, YELLOW),
                        color(tty, RESET)
                    );
                    continue;
                }
            }
        };

        let path = if choice <= local.len() {
            local[choice - 1].path.to_string_lossy().into_owned()
        } else {
            let preset = available[choice - local.len() - 1];
            download_preset(preset)?.to_string_lossy().into_owned()
        };

        return Ok(ModelChoice {
            path,
            vision_enabled,
        });
    }
}
