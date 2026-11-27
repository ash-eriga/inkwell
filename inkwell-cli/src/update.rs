//! Self-updater for `inkwell` (`inkwell update [--check | --rollback | --version <tag>]`).
//!
//! Downloads prebuilt release bundles from GitHub Releases, verifies SHA-256 against
//! `SHA256SUMS`, extracts into `~/.local/share/inkwell/versions/<tag>/`, and atomically
//! swaps `~/.local/share/inkwell/current`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const REPO: &str = "ash-eriga/inkwell";
const CURRENT_VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));
const ASSET_SUFFIX: &str = "linux-x86_64-cuda12.tar.gz";

fn home_dir() -> Result<PathBuf, String> {
    std::env::var("HOME")
        .map(PathBuf::from)
        .map_err(|_| "HOME environment variable is not set".to_owned())
}

fn data_root() -> Result<PathBuf, String> {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        if !xdg.trim().is_empty() {
            return Ok(PathBuf::from(xdg).join("inkwell"));
        }
    }
    Ok(home_dir()?.join(".local/share/inkwell"))
}

fn bin_wrapper_path() -> Result<PathBuf, String> {
    Ok(home_dir()?.join(".local/bin/inkwell"))
}

fn fetch_latest_tag() -> Result<String, String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let out = Command::new("curl")
        .args([
            "-fsSL",
            "--connect-timeout",
            "10",
            "--max-time",
            "30",
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            "User-Agent: inkwell-updater",
            &url,
        ])
        .output()
        .map_err(|e| format!("failed to run curl: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "failed to query latest release: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let val: serde_json::Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("invalid GitHub API response: {e}"))?;
    val.get("tag_name")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| "missing tag_name in GitHub release response".to_owned())
}

fn download_file(url: &str, dst: &Path, show_progress: bool) -> Result<(), String> {
    let mut cmd = Command::new("curl");
    if show_progress {
        cmd.arg("-fL#");
    } else {
        cmd.arg("-fsSL");
    }
    let status = cmd
        .args(["--connect-timeout", "15", "-o"])
        .arg(dst)
        .arg(url)
        .status()
        .map_err(|e| format!("failed to run curl: {e}"))?;
    if !status.success() {
        return Err(format!("download failed: {url}"));
    }
    Ok(())
}

fn verify_sha256(tarball: &Path, sums_path: &Path, asset_name: &str) -> Result<(), String> {
    let sums = fs::read_to_string(sums_path)
        .map_err(|e| format!("failed to read SHA256SUMS: {e}"))?;
    let expected = sums
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            let hash = parts.next()?;
            let name = parts.next()?.trim_start_matches('*');
            (name == asset_name).then(|| hash.to_ascii_lowercase())
        })
        .ok_or_else(|| format!("entry for {asset_name} not found in SHA256SUMS"))?;

    let out = Command::new("sha256sum")
        .arg(tarball)
        .output()
        .map_err(|e| format!("failed to run sha256sum: {e}"))?;
    if !out.status.success() {
        return Err("sha256sum failed".to_owned());
    }
    let actual = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if actual != expected {
        return Err(format!(
            "SHA-256 mismatch for {asset_name}: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn atomic_swap_current(root: &Path, target_rel: &Path) -> Result<(), String> {
    let tmp_link = root.join(".current.tmp");
    let current = root.join("current");
    let _ = fs::remove_file(&tmp_link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(target_rel, &tmp_link)
        .map_err(|e| format!("failed to create symlink: {e}"))?;
    fs::rename(&tmp_link, &current)
        .map_err(|e| format!("failed to swap current symlink: {e}"))?;
    Ok(())
}

fn write_wrapper(root: &Path) -> Result<(), String> {
    let wrapper = bin_wrapper_path()?;
    if let Some(parent) = wrapper.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let root_s = root.display();
    let script = format!(
        "#!/usr/bin/env sh\n\
         set -eu\n\
         ROOT=\"{root_s}/current\"\n\
         CUDA_LIB=\"\"\n\
         for d in \"${{INKWELL_CUDA_HOME:-}}/lib64\" \"${{CUDA_HOME:-}}/lib64\" \"$HOME/lfm-gtx1060-bench/cuda-12.6/lib64\" /usr/local/cuda/lib64 /opt/cuda/lib64; do\n\
           if [ -f \"$d/libcudart.so.12\" ]; then CUDA_LIB=\"$d\"; break; fi\n\
         done\n\
         export LD_LIBRARY_PATH=\"$ROOT/lib${{CUDA_LIB:+:$CUDA_LIB}}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\"\n\
         export GGML_PASCAL_GRAPHS=\"${{GGML_PASCAL_GRAPHS:-1}}\"\n\
         export INKWELL_PREFILL_CHUNK=\"${{INKWELL_PREFILL_CHUNK:-512}}\"\n\
         exec \"$ROOT/inkwell\" \"$@\"\n"
    );
    let tmp = wrapper.with_extension("tmp");
    fs::write(&tmp, script).map_err(|e| format!("failed to write wrapper: {e}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("failed to chmod wrapper: {e}"))?;
    }
    fs::rename(&tmp, &wrapper).map_err(|e| format!("failed to install wrapper: {e}"))?;
    Ok(())
}

fn rollback(root: &Path) -> Result<(), String> {
    let versions_dir = root.join("versions");
    let current = root.join("current");
    let cur_name = fs::read_link(&current)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));

    let mut installed: Vec<(std::time::SystemTime, String)> = Vec::new();
    let entries = fs::read_dir(&versions_dir)
        .map_err(|_| "no installed versions found to roll back to".to_owned())?;
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() || !p.join("inkwell").is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if cur_name.as_deref() == Some(name.as_str()) {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        installed.push((mtime, name));
    }
    installed.sort_by(|a, b| b.0.cmp(&a.0));
    let Some((_, prev)) = installed.first() else {
        return Err("no previous version installed to roll back to".to_owned());
    };
    atomic_swap_current(root, &PathBuf::from("versions").join(prev))?;
    println!("Rolled back inkwell to {prev}.");
    Ok(())
}

fn prune_old_versions(root: &Path, keep_tag: &str) {
    let versions_dir = root.join("versions");
    let Ok(entries) = fs::read_dir(&versions_dir) else { return };
    let mut list: Vec<(std::time::SystemTime, PathBuf, String)> = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        list.push((mtime, p, name));
    }
    list.sort_by(|a, b| b.0.cmp(&a.0));
    // Keep current + 1 previous version for instant rollback.
    let mut kept = 0usize;
    for (_, path, name) in list {
        if name == keep_tag || kept < 2 {
            kept += 1;
            continue;
        }
        let _ = fs::remove_dir_all(path);
    }
}

/// Entry point for `inkwell update [--check | --rollback | --version <tag>]`.
pub fn run_update(mut args: std::vec::IntoIter<String>) -> Result<(), String> {
    let mut check_only = false;
    let mut do_rollback = false;
    let mut target_tag: Option<String> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check_only = true,
            "--rollback" => do_rollback = true,
            "--version" => {
                let v = args
                    .next()
                    .ok_or_else(|| "missing value for --version".to_owned())?;
                target_tag = Some(if v.starts_with('v') { v } else { format!("v{v}") });
            }
            "-h" | "--help" => {
                println!(
                    "Usage: inkwell update [--check | --rollback | --version <tag>]\n\
                     \n\
                     Options:\n\
                       --check            Check for a newer release without installing\n\
                       --rollback         Switch back to the previously installed version\n\
                       --version <tag>    Install a specific release tag (e.g. v0.1.0)"
                );
                return Ok(());
            }
            other => return Err(format!("unknown option for update: {other}")),
        }
    }

    let root = data_root()?;
    if do_rollback {
        return rollback(&root);
    }

    let tag = match target_tag {
        Some(t) => t,
        None => fetch_latest_tag()?,
    };

    if check_only {
        if tag == CURRENT_VERSION {
            println!("inkwell {CURRENT_VERSION} is up to date.");
        } else {
            println!("Update available: {CURRENT_VERSION} -> {tag} (run 'inkwell update')");
        }
        return Ok(());
    }

    if tag == CURRENT_VERSION && root.join("versions").join(&tag).join("inkwell").is_file() {
        println!("inkwell {CURRENT_VERSION} is already up to date.");
        return Ok(());
    }

    let asset = format!("inkwell-{tag}-{ASSET_SUFFIX}");
    let base_url = format!("https://github.com/{REPO}/releases/download/{tag}");
    let staging = root.join("staging").join(&tag);
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)
        .map_err(|e| format!("failed to create staging dir: {e}"))?;

    let tarball = staging.join(&asset);
    let sums = staging.join("SHA256SUMS");

    println!("Downloading {asset}...");
    download_file(&format!("{base_url}/{asset}"), &tarball, true)?;
    download_file(&format!("{base_url}/SHA256SUMS"), &sums, false)?;

    println!("Verifying SHA-256...");
    verify_sha256(&tarball, &sums, &asset)?;

    let dest = root.join("versions").join(&tag);
    let _ = fs::remove_dir_all(&dest);
    fs::create_dir_all(&dest).map_err(|e| format!("failed to create {}: {e}", dest.display()))?;

    let status = Command::new("tar")
        .args(["-xzf"])
        .arg(&tarball)
        .args(["-C"])
        .arg(&dest)
        .status()
        .map_err(|e| format!("failed to run tar: {e}"))?;
    if !status.success() {
        let _ = fs::remove_dir_all(&dest);
        return Err("failed to extract release archive".to_owned());
    }
    if !dest.join("inkwell").is_file() {
        let _ = fs::remove_dir_all(&dest);
        return Err("invalid release archive: missing inkwell binary".to_owned());
    }

    atomic_swap_current(&root, &PathBuf::from("versions").join(&tag))?;
    write_wrapper(&root)?;
    let _ = fs::remove_dir_all(&staging);
    prune_old_versions(&root, &tag);

    println!("Updated inkwell to {tag} ({}).", dest.display());
    Ok(())
}
