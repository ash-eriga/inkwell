//! Decode-time forward of `inkling2_loop`, orchestrated in Rust with
//! ggml-cuda kernels under it. See `PORTING.md` for the recipe.
#![expect(missing_docs, reason = "scaffold in progress")]

pub mod head_dump;
pub mod moe;
pub mod ops;
pub mod prof;
pub mod draft;
pub mod runner;
