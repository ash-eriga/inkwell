#!/usr/bin/env bash
# Collect real serial greedy tokens (5 varied prompts x 128 tokens = 640 tokens)
# and real DSpark MTP feature taps (12 prompts x 7 drafts = 84 real draft steps)
# under a single flock /tmp/inkwell-gpu.lock.
set -euo pipefail

export LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin
export GGML_PASCAL_GRAPHS=1
export INKWELL_MAX_CTX=512
export INKWELL_CACHE_MARGIN_MB=10
export INKWELL_EXL3_CACHE_K=20
export INKWELL_MOE_V2=1
export INKWELL_MOE_NKY=2

BIN=./target/release/inkwell
MODEL=~/models/inkling
OUTDIR=/tmp/drafter_data
mkdir -p "$OUTDIR/serial" "$OUTDIR/mtp_samples"

PROMPTS=(
  "prose|Once upon a time in a quiet coastal town, an old watchmaker discovered"
  "python|def quicksort(arr):\n    \"\"\"Sort a list in-place using Hoare partition.\"\"\"\n"
  "rust|/// Lock-free ring buffer for single-producer single-consumer queues.\npub struct RingBuf<T> {\n"
  "tech|In modern GPU architectures, memory bandwidth and instruction issue throughput determine kernel performance. Specifically,"
  "factual|The capital of France is"
)

echo "=== PHASE 1: Serial greedy token collection (5 prompts x 128 tok = 640 tok) ==="
for entry in "${PROMPTS[@]}"; do
  name="${entry%%|*}"
  prompt=$(printf '%b' "${entry#*|}")
  echo "[serial] prompt=$name"
  INKWELL_PROMPT="$prompt" INKWELL_GENS=128 \
    "$BIN" "$MODEL" --run < /dev/null > "$OUTDIR/serial/${name}.out" 2> "$OUTDIR/serial/${name}.err"
  grep -E '^(decode |generated text:)' "$OUTDIR/serial/${name}.err" || true
done

echo "=== PHASE 2: Real DSpark MTP feature taps (12 prompts x 7 drafts = 84 draft steps) ==="
# Note: draft.rs INKWELL_MTP_ORACLE dumps (1024+16) rows of kc/vc, so MAX_CTX=1024 is required.
# step_block (nt=8) requires INKWELL_NOCTRLCAT=1 (avoids 1D view on 8-row ctrl_gf_cat).
MTP_PROMPTS=(
  "p00_factual|The capital of France is"
  "p01_prose|Once upon a time in a quiet coastal town, an old watchmaker discovered"
  "p02_python|def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n"
  "p03_rust|fn merge_sort(slice: &mut [i32]) {\n    let n = slice.len();\n"
  "p04_tech|In a mixture-of-experts transformer, each token routes to the top-k experts per layer, which means"
  "p05_math|To prove that the square root of 2 is irrational, suppose for contradiction that"
  "p06_dialogue|\"Are you sure this bridge will hold?\" asked Maya, peering into the fog below. \"Because"
  "p07_json|{\n  \"service\": \"inference-engine\",\n  \"hardware\": {\"gpu\": \"GTX 1060\", \"vram_gb\": 6},\n  \"config\": {"
  "p08_sql|SELECT u.user_id, u.email, COUNT(o.order_id) AS total_orders\nFROM users u\nLEFT JOIN"
  "p09_cuda|__global__ void matrix_transpose(const float* __restrict__ in, float* __restrict__ out, int N) {\n"
  "p10_history|During the Industrial Revolution in eighteenth-century Britain, steam power transformed"
  "p11_science|Photosynthesis converts light energy into chemical energy inside chloroplasts through two main stages:"
)

for entry in "${MTP_PROMPTS[@]}"; do
  name="${entry%%|*}"
  prompt=$(printf '%b' "${entry#*|}")
  rm -rf /tmp/mtp_in
  echo "[mtp] $name"
  INKWELL_MAX_CTX=1024 INKWELL_EXL3_CACHE_K=4 INKWELL_CACHE_MARGIN_MB=40 \
    INKWELL_NOCTRLCAT=1 INKWELL_NO_CTRLFUSE=1 INKWELL_NO_RELGATHER=1 INKWELL_NO_KSLOT=1 INKWELL_NO_GCACHE=1 \
    INKWELL_PROMPT="$prompt" INKWELL_GENS=4 INKWELL_SPEC=mtp INKWELL_SPEC_GENS=1 \
    INKWELL_MTP_ORACLE=1 \
    "$BIN" "$MODEL" --run < /dev/null > "$OUTDIR/mtp_samples/${name}.out" 2> "$OUTDIR/mtp_samples/${name}.err"
  mkdir -p "$OUTDIR/mtp_samples/${name}"
  cp -a /tmp/mtp_in/* "$OUTDIR/mtp_samples/${name}/"
  grep -E '^(spec r1:|   vtok=|   veri=)' "$OUTDIR/mtp_samples/${name}.err" || true
done
echo "=== DONE: collected serial + MTP oracle data ==="
