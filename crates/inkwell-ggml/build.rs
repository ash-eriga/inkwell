//! Build the C++ shim against the patched ggml build.
use std::path::PathBuf;
use std::process::Command;

fn first_existing(envs: &[&str], candidates: &[PathBuf], fallback: Option<&str>) -> PathBuf {
    for e in envs {
        println!("cargo:rerun-if-env-changed={e}");
        if let Ok(v) = std::env::var(e) {
            if !v.is_empty() {
                return PathBuf::from(v);
            }
        }
    }
    for c in candidates {
        if c.exists() {
            return c.clone();
        }
    }
    PathBuf::from(fallback.unwrap_or_default())
}

fn main() {
    // Overridable via env vars (see third_party/ggml/README.md).
    let home = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()));
    let repo_ggml = manifest_dir.join("../../third_party/ggml/ggml-src");

    let root = first_existing(
        &["INKWELL_GGML_SRC"],
        &[repo_ggml.clone(), home.join("lfm-gtx1060-bench/src-v2")],
        None,
    );
    let ggml_inc = root.join("ggml/include");
    let ggml_src = root.join("ggml/src");
    assert!(
        ggml_inc.is_dir(),
        "ggml headers not found at {} — build patched ggml per third_party/ggml/README.md and set INKWELL_GGML_SRC",
        ggml_inc.display()
    );

    let libdir = first_existing(
        &["INKWELL_GGML_LIB"],
        &[
            root.join("build-inkwell-dp4a/bin"),
            repo_ggml.join("build-inkwell-dp4a/bin"),
            home.join("lfm-gtx1060-bench/build-inkwell-dp4a/bin"),
        ],
        None,
    );
    assert!(
        libdir.is_dir(),
        "ggml libraries not found at {} — build patched ggml per third_party/ggml/README.md and set INKWELL_GGML_LIB",
        libdir.display()
    );

    let cxx = first_existing(
        &["INKWELL_CXX", "CXX"],
        &[home.join(".local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++")],
        Some("g++"),
    );
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");

    println!("cargo:rerun-if-changed=shim.cpp");
    let status = Command::new(&cxx)
        .args([
            "-O2", "-std=c++17", "-fPIC",
            &format!("-I{}", ggml_inc.display()),
            &format!("-I{}", ggml_src.display()),
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
    println!("cargo:rustc-link-search={}", libdir.display());
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cuda");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
}
