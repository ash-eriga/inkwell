# PLAN — hill-climb Inkwell (restart 2026-11-22)

## Alvos e restrições

- Métrica: decode c1 tok/s, greedy 64 tok, desktop apps ligados (baseline aceito).
- Meta: **15 t/s** (66.7 ms/tok). Baseline: ~8.7 t/s (115 ms/tok).
- Hardware: GTX 1060 6GB (sm_61), 16 GB RAM, driver 580.196.07, CUDA 12.6 local.
- Modelo: Inkling-2-Mini UD-IQ2_XXS GGUF; única troca permitida → `vcruz305/Inkling-2-Mini-SAGE-REXL3-2.26bpw`.
- Qualidade: golden text chinês `"正确答案：A"` deve se manter; reorders fp exigem gold test dedicado.
- SEM megakernels. Rust host-path; kernels CUDA via FFI.
- Ledger narrativo continua em ../E2E_DEBUG.md; este dir = experimentos da skill.

## Inventário

- SM clock 1506 MHz, mem 4252 MHz, P0, 120W. VRAM 6144 MiB (899 usado idle).
- Run: `cargo build -p inkwell-cli --release && LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin GGML_PASCAL_GRAPHS=1 ./target/release/inkwell ~/models/inkling/Inkling-2-Mini-UD-IQ2_XXS.gguf --run`
- nvcc precisa `-ccbin ~/.local/share/mise/installs/conda-gxx-linux-64/13.4.0/bin/x86_64-conda-linux-gnu-g++` (gcc-16 do sistema é incompatível).
- bench: `opt/bench.sh <config> <N>` → ledger.jsonl.

## Log de experimentos

### 2026-11-22 — baseline-iq2xxs
- 6 amostras (bench.sh): 96.6/102.2/111.0/116.5/101.4/104.0 ms/tok → **mediana 103.1 ms/tok = 9.7 t/s**, golden ✓ em 6/6.
- +2 manuais: 115.3, 112.4 ms/tok, golden ✓.
- Bandwidth medida: D2D 163 GB/s best; kernel grande 217 GB/s; PCIe zc ~8.6 GB/s.
- Bytes/token (THEORY.md): GGUF 2686 MiB; REXL3 ~1830 MiB (−32%; attn dense Q8_0→trellis, ctrl F32→BF16, experts −5%).
- REXL3 traz `mtp.safetensors` (drafter MTP: markov_rank 256, block_size 8, target layers 5/23/41/48) → spec decode nativo.
- Ceiling: GGUF ~22 t/s @60GB/s efetivo; REXL3 ~33 t/s. Meta 15 t/s cabe nos dois; REXL3 domina em bytes × EAR (94.66% top-1 vs 90.6%).
- Custo do porte: loader safetensors + kernels EXL3 trellis sm_61 (exllamav3 não suporta Pascal).
