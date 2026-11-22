//! Build the C++ shim against the dp4a ggml build.
use std::process::Command;

fn main() {
    let ggml_inc = "/home/ash/lfm-gtx1060-bench/src-v2/ggml/include";
    let ggml_src = "/home/ash/lfm-gtx1060-bench/src-v2/ggml/src";
    let libdir = "/home/ash/lfm-gtx1060-bench/build-inkwell-dp4a/bin";
    let cxx = "/home/ash/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++";
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");

    println!("cargo:rerun-if-changed=shim.cpp");
    let status = Command::new(cxx)
        .args([
            "-O2", "-std=c++17", "-fPIC",
            &format!("-I{ggml_inc}"), &format!("-I{ggml_src}"),
            "-c", "shim.cpp", "-o", &format!("{out}/shim.o"),
        ])
        .status()
        .expect("g++");
    assert!(status.success(), "shim compile failed");
    let status = Command::new("ar")
        .args(["rcs", &format!("{out}/libinkwell_shim.a"), &format!("{out}/shim.o")])
        .status()
        .expect("ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search={out}");
    println!("cargo:rustc-link-lib=static=inkwell_shim");
    println!("cargo:rustc-link-search={libdir}");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cuda");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
}
