//! Compile the CUDA kernels for `sm_61` with the same toolchain recipe
//! as the llama.cpp baseline (local cuda-12.6 toolkit).
use std::path::Path;
use std::process::Command;

fn main() {
    let cuda = Path::new("/home/ash/lfm-gtx1060-bench/cuda-12.6");
    let nvcc = cuda.join("bin/nvcc");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = Path::new("src/kernels.cu");

    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-env-changed=INKWELL_NVCC_DEFS");
    let obj = format!("{out}/kernels.o");
    let status = Command::new(&nvcc)
        .args([
            "-O3", "-std=c++17", "--use_fast_math",
            "-ccbin", "/home/ash/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++",
            "-gencode", "arch=compute_61,code=sm_61",
            "-c", &src.to_string_lossy(), "-o", &obj,
        ])
        .args(std::env::var("INKWELL_NVCC_DEFS").map(|d| d.split(',').map(|x| format!("-D{}", x)).collect::<Vec<_>>()).unwrap_or_default())
        .status()
        .expect("nvcc");
    assert!(status.success(), "nvcc failed");
    let lib = format!("{out}/libinkwell_kernels.a");
    let status = Command::new("ar")
        .args(["rcs", &lib, &obj])
        .status()
        .expect("ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search={out}");
    println!("cargo:rustc-link-lib=static=inkwell_kernels");
    println!("cargo:rustc-link-search={}/lib64", cuda.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
}
