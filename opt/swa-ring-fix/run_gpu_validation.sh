#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
out=opt/swa-ring-fix/artifacts/gpu
mkdir -p "$out"
export GGML_PASCAL_GRAPHS=1 INKWELL_CACHE_MARGIN_MB=10 INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
cachel=(INKWELL_CACHE_MB_L="0:0,1:8,2:0,3:8,4:8,5:12,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:72,17:88,18:24,19:24,20:28,21:40,22:32,23:40")
lock=/tmp/inkwell-gpu.lock
bin=./target/release/inkwell
model=$HOME/models/inkling

run() {
  name=$1; shift
  echo "START $name $(date -Is)" | tee -a "$out/schedule.log"
  nice -n 19 flock "$lock" env "$@" > "$out/$name.log" 2>&1
  rc=$?
  echo "$rc" > "$out/$name.status"
  echo "END $name rc=$rc $(date -Is)" | tee -a "$out/schedule.log"
}

# ctx512 regression: 3 normal runs, md5/ms/peak summary.
echo "START ctx512 $(date -Is)" | tee -a "$out/schedule.log"
D="$out/ctx512" nice -n 19 flock "$lock" bash ~/.cache/inkwell-ab/tools/validate.sh "$bin" > "$out/ctx512_validate.log" 2>&1
echo $? > "$out/ctx512_validate.status"
echo "END ctx512 rc=$(cat "$out/ctx512_validate.status") $(date -Is)" | tee -a "$out/schedule.log"

# Teacher-forced natural-text oracle comparisons. INKWELL_FORCE_IDS rows are the
# next 128 tokens of the same natural text after the prompt prefix.
for n in 2048 4096; do
  ctx=$((n + 256))
  common=(INKWELL_MAX_CTX=$ctx INKWELL_GENS=128 INKWELL_PROMPT_IDS=opt/swa-ring-fix/artifacts/prompt$n.ids INKWELL_FORCE_IDS=opt/swa-ring-fix/artifacts/force$n.ids)
  if [ "$n" = 4096 ]; then
    rm -f "$out/fixed4k.inkst"
    run tf4k_ring "${cachel[@]}" "${common[@]}" INKWELL_LOGIT_DUMP="$out/tf4k_ring.bin" INKWELL_STATE_SAVE="$out/fixed4k.inkst" "$bin" "$model" --run
  else
    run tf2k_ring "${cachel[@]}" "${common[@]}" INKWELL_LOGIT_DUMP="$out/tf2k_ring.bin" "$bin" "$model" --run
  fi
  run "tf$((n/1024))k_oracle" "${cachel[@]}" "${common[@]}" INKWELL_SWA_ORACLE=1 INKWELL_LOGIT_DUMP="$out/tf$((n/1024))k_oracle.bin" "$bin" "$model" --run
done

# Before/after 4K free-run decode timing. Master does not support
# INKWELL_PROMPT_IDS; command substitution strips the final newline, so this
# baseline is 1 token shorter but still measures 4K-context decode.
master_prompt=$(cat opt/swa-ring-fix/artifacts/prompt4096.txt)
run free4k_master "${cachel[@]}" INKWELL_MAX_CTX=4352 INKWELL_GENS=64 INKWELL_PROMPT="$master_prompt" /home/ash/inkwell/target/release/inkwell "$model" --run
run free4k_fixed "${cachel[@]}" INKWELL_MAX_CTX=4352 INKWELL_GENS=64 INKWELL_PROMPT_IDS=opt/swa-ring-fix/artifacts/prompt4096.ids INKWELL_STATE_LOAD="$out/fixed4k.inkst" "$bin" "$model" --run

# One VTEST8 block starting at pos 4095 crosses the 4096 ring wrap.
run vtest8_wrap_fixed "${cachel[@]}" INKWELL_MAX_CTX=4352 INKWELL_GENS=8 INKWELL_PROMPT_IDS=opt/swa-ring-fix/artifacts/prompt4096.ids INKWELL_STATE_LOAD="$out/fixed4k.inkst" INKWELL_VTEST8=1 "$bin" "$model" --run

echo DONE | tee -a "$out/schedule.log"
