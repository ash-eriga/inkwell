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
}

/// Curated EXL3-only presets for v0.1.
pub const EXL3_PRESETS: &[HfPreset] = &[
    HfPreset {
        slug: "inkling-2-mini-exl3-2.26bpw",
        repo_id: "vcruz305/Inkling-2-Mini-SAGE-REXL3-2.26bpw",
        detail: "EXL3 2.26 bpw · recommended for 6 GB VRAM",
        size_label: "~8.3 GB",
    },
    HfPreset {
        slug: "inkling-2-mini-exl3-2.92bpw",
        repo_id: "vcruz305/Inkling-2-Mini-SAGE-REXL3-2.92bpw",
        detail: "EXL3 2.92 bpw · 8–12 GB VRAM",
        size_label: "~10.5 GB",
    },
    HfPreset {
        slug: "inkling-2-mini-exl3-4.23bpw",
        repo_id: "vcruz305/Inkling-2-Mini-SAGE-REXL3-4.23bpw",
        detail: "EXL3 4.23 bpw · 16+ GB VRAM",
        size_label: "~14.5 GB",
    },
];

/// A discovered local EXL3 model directory.
#[derive(Debug, Clone)]
pub struct LocalModel {
    /// Directory name.
    pub name: String,
    /// Absolute path.
    pub path: PathBuf,
    /// Total bytes of `.safetensors` shards in this directory.
    pub size_bytes: u64,
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
            if let Ok(meta) = fs::metadata(&p) {
                total += meta.len();
            }
        }
    }
    total
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
            found.push(LocalModel {
                name,
                size_bytes: safetensors_size(&p),
                path: p,
            });
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

fn format_gib(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
}

fn color(tty: bool, code: &'static str) -> &'static str {
    if tty { code } else { "" }
}

/// Print all local EXL3 models and curated Hugging Face EXL3 presets.
pub fn print_model_catalog() {
    let tty = io::stdout().is_terminal();
    let local = discover_local_exl3();
    let root = models_dir();

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
        println!("  {}(none found in {}){}", color(tty, DIM), root.display(), color(tty, RESET));
    } else {
        for (i, m) in local.iter().enumerate() {
            println!(
                "  {}[{}]{} {:<30} {}EXL3 · {} · {}{}",
                color(tty, GREEN),
                i + 1,
                color(tty, RESET),
                m.name,
                color(tty, DIM),
                format_gib(m.size_bytes),
                m.path.display(),
                color(tty, RESET)
            );
        }
    }

    println!("\n{}Hugging Face presets (EXL3):{}", color(tty, BOLD), color(tty, RESET));
    for (j, p) in EXL3_PRESETS.iter().enumerate() {
        let installed = local.iter().any(|m| m.name == p.slug);
        let badge = if installed { "installed" } else { "download" };
        println!(
            "  {}[{}]{} {:<44} {}{} · {} [{}]{}",
            color(tty, CYAN),
            local.len() + j + 1,
            color(tty, RESET),
            p.repo_id,
            color(tty, DIM),
            p.size_label,
            p.detail,
            badge,
            color(tty, RESET)
        );
    }
    println!();
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

/// Interactive selector used when `inkwell serve` or `inkwell chat` is invoked without `<model>`.
pub fn select_or_download_model() -> Result<String, String> {
    let tty = io::stdout().is_terminal();
    let local = discover_local_exl3();

    print_model_catalog();

    let total = local.len() + EXL3_PRESETS.len();
    if !io::stdin().is_terminal() {
        if let Some(first) = local.first() {
            return Ok(first.path.to_string_lossy().into_owned());
        }
        return Err("no local EXL3 models found and stdin is not interactive; pass <model> or run in a terminal".to_owned());
    }

    let default_idx = 1usize;
    loop {
        print!(
            "{}Select model [1-{total}] (default {default_idx}, q to quit):{} ",
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
        let choice: usize = if s.is_empty() {
            default_idx
        } else {
            match s.parse() {
                Ok(v) if (1..=total).contains(&v) => v,
                _ => {
                    println!(
                        "{}Please enter a number between 1 and {total} (or q).{}",
                        color(tty, YELLOW),
                        color(tty, RESET)
                    );
                    continue;
                }
            }
        };

        if choice <= local.len() {
            let chosen = &local[choice - 1];
            return Ok(chosen.path.to_string_lossy().into_owned());
        }

        let preset = &EXL3_PRESETS[choice - local.len() - 1];
        let dest = models_dir().join(preset.slug);
        if is_exl3_dir(&dest) {
            return Ok(dest.to_string_lossy().into_owned());
        }
        let downloaded = download_preset(preset)?;
        return Ok(downloaded.to_string_lossy().into_owned());
    }
}
