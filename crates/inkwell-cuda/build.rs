//! Compile the CUDA kernels for `sm_61` with the same toolchain recipe
//! as the llama.cpp baseline (local cuda-12.6 toolkit).
use std::path::{Path, PathBuf};
use std::process::Command;

fn exists_or_env(envs: &[&str], default: &Path) -> PathBuf {
    for e in envs {
        if let Ok(v) = std::env::var(e) {
            if !v.is_empty() {
                return PathBuf::from(v);
            }
        }
    }
    default.to_path_buf()
}

fn main() {
    // Overridable for other build hosts (see third_party/ggml/README.md).
    let env = |k: &str, d: &str| {
        println!("cargo:rerun-if-env-changed={k}");
        std::env::var(k).unwrap_or_else(|_| d.to_string())
    };
    let cuda_s = env("INKWELL_CUDA_HOME", "/home/ash/lfm-gtx1060-bench/cuda-12.6");
    let gxx_s = env("INKWELL_CXX", "/home/ash/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++");
    let cuda = Path::new(&cuda_s);
    let nvcc = cuda.join("bin/nvcc");
    let gxx = Path::new(&gxx_s);
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let outdir = Path::new(&out);
    let src = Path::new("src/kernels.cu");
    let prefill_src = Path::new("src/prefill.cu");
    let ch2dir = Path::new("../../opt/kbench/ch2");

    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", prefill_src.display());
    println!("cargo:rerun-if-changed={}", ch2dir.join("build_nopless.py").display());
    println!("cargo:rerun-if-changed={}", ch2dir.join("patch_sass.py").display());
    println!("cargo:rerun-if-changed={}", ch2dir.join("reasm_remove_loop_nops.py").display());
    println!("cargo:rerun-if-changed={}", ch2dir.join("reasm_augment_repos.py").display());
    println!("cargo:rerun-if-changed={}", ch2dir.join("reasm_assemble_with_repos.py").display());
    println!("cargo:rerun-if-env-changed=INKWELL_NVCC_DEFS");
    println!("cargo:rerun-if-env-changed=INKWELL_CUASM_ROOT");
    println!("cargo:rerun-if-env-changed=CUASMROOT");
    println!("cargo:rerun-if-env-changed=INKWELL_CUASM_PYTHON");
    println!("cargo:rerun-if-env-changed=VENVPY");
    let obj = format!("{out}/kernels.o");
    let status = Command::new(&nvcc)
        .args([
            "-O3", "-std=c++17", "--use_fast_math",
            "-ccbin", gxx.to_str().expect("gxx path"),
            "-gencode", "arch=compute_61,code=sm_61",
            "-c", &src.to_string_lossy(), "-o", &obj,
        ])
        .args(std::env::var("INKWELL_NVCC_DEFS").map(|d| d.split(',').map(|x| format!("-D{}", x)).collect::<Vec<_>>()).unwrap_or_default())
        .status()
        .expect("nvcc");
    assert!(status.success(), "nvcc failed");

    let prefill_obj = format!("{out}/prefill.o");
    let status = Command::new(&nvcc)
        .args([
            "-O3", "-std=c++17", "--use_fast_math",
            "-ccbin", gxx.to_str().expect("gxx path"),
            "-gencode", "arch=compute_61,code=sm_61",
            "-c", &prefill_src.to_string_lossy(), "-o", &prefill_obj,
        ])
        .status()
        .expect("nvcc prefill");
    assert!(status.success(), "nvcc prefill failed");

    // Optional fail-closed CH2 artifact: transform a copy of the cubin embedded
    // in this exact object, never the runtime-linked object itself.  Any missing
    // tool or failed structural check leaves no real cubin and the empty embedded
    // payload below disables the runtime path.
    let real_cubin = outdir.join("ch2nopless.real.cubin");
    let _ = std::fs::remove_file(&real_cubin);
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/ash".into());
    let cuasm_root = exists_or_env(&["INKWELL_CUASM_ROOT", "CUASMROOT"], &Path::new(&home).join(".cache/inkwell-sass-reasm/CuAssembler"));
    let cuasm_py = exists_or_env(&["INKWELL_CUASM_PYTHON", "VENVPY"], &Path::new(&home).join(".cache/inkwell-sass-reasm/cuasm-venv/bin/python"));
    if !cuasm_root.join("bin/cuasm.py").is_file() || !cuasm_py.is_file() {
        println!("cargo:warning=ch2nopless: CuAssembler unavailable; using the normal ptxas kernel (set INKWELL_CUASM_ROOT/INKWELL_CUASM_PYTHON or populate ~/.cache/inkwell-sass-reasm)");
    } else {
        let gen = ch2dir.join("build_nopless.py");
        let work = outdir.join("ch2nopless-work");
        let r = Command::new(&cuasm_py)
            .args([gen.to_str().expect("generator path"), &obj])
            .arg(&real_cubin)
            .arg(&work)
            .arg(cuda.to_str().expect("cuda path"))
            .output()
            .expect("ch2nopless generator");
        if !r.status.success() || !real_cubin.is_file() {
            let _ = std::fs::remove_file(&real_cubin);
            let mut note = String::from_utf8_lossy(&r.stdout).lines().next().unwrap_or("ch2nopless: disabled: generator failed").to_owned();
            if let Ok(f) = std::fs::read_to_string(work.join("failure.txt")) {
                if let Some(line) = f.lines().next() {
                    if !line.is_empty() && !note.contains(line) { note = format!("ch2nopless: disabled: {line}"); }
                }
            }
            println!("cargo:warning={note}");
        }
    }

    // Embed the optional cubin as a byte range in the static kernel library.  An
    // empty range is emitted when generation failed, so no driver module load is
    // attempted and normal runtime compilation remains the only kernel.
    let payload = outdir.join("ch2nopless.payload");
    if real_cubin.is_file() {
        std::fs::copy(&real_cubin, &payload).expect("copy ch2nopless payload");
    } else {
        std::fs::write(&payload, []).expect("write empty ch2nopless payload");
    }
    let asm = outdir.join("ch2nopless_embed.s");
    let payload_escaped = payload.display().to_string().replace('\\', "\\\\").replace('"', "\\\"");
    std::fs::write(&asm, format!(
        ".section .rodata\n.global ink_ch2nopless_cubin_start\nink_ch2nopless_cubin_start:\n.incbin \"{payload_escaped}\"\n.global ink_ch2nopless_cubin_end\nink_ch2nopless_cubin_end:\n"
    )).expect("write ch2nopless assembly");
    let embed_obj = format!("{out}/ch2nopless_embed.o");
    let status = Command::new(gxx)
        .args(["-x", "assembler", "-c"])
        .arg(&asm)
        .args(["-o", &embed_obj])
        .status()
        .expect("assemble ch2nopless embed");
    assert!(status.success(), "ch2nopless embed assembly failed");

    let lib = format!("{out}/libinkwell_kernels.a");
    let status = Command::new("ar")
        .args(["rcs", &lib, &obj, &prefill_obj, &embed_obj])
        .status()
        .expect("ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search={out}");
    println!("cargo:rustc-link-lib=static=inkwell_kernels");
    println!("cargo:rustc-link-search={}/lib64", cuda.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
