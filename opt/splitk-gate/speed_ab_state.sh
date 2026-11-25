#!/bin/bash
set -eu
if [ "${1:-}" != --inner ]; then
  exec nice -n 19 flock /tmp/inkwell-gpu.lock bash "$0" --inner "$@"
fi
shift
CTX=$1
PAIRS=${2:-4}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ART="$ROOT/opt/splitk-gate/artifacts"
PROMPT=$(cat "$ART/prompts/p$CTX.txt")
if [ "$CTX" = 4096 ]; then MAXCTX=4352; elif [ "$CTX" = 8192 ]; then MAXCTX=8448; else echo "bad ctx" >&2; exit 2; fi
STATE="$ART/state$CTX.bin"
OUT="$ART/results/speed${CTX}-state.out"
: > "$OUT"
run_one() {
  tag=$1; minthr=$2; statemode=$3
  log="$ART/logs/$tag.log"
  extra=()
  if [ "$statemode" = save ]; then extra=(INKWELL_STATE_SAVE="$STATE"); else extra=(INKWELL_STATE_LOAD="$STATE"); fi
  (
    env GGML_PASCAL_GRAPHS=1 INKWELL_MAX_CTX=$MAXCTX INKWELL_CACHE_MARGIN_MB=10 \
      INKWELL_EXL3_CACHE_K=20 INKWELL_MOE_V2=1 INKWELL_MOE_NKY=2 \
      INKWELL_VRAM_CAP_MB=4900 INKWELL_ATTNSPLIT_MIN=$minthr INKWELL_GENS=64 INKWELL_PROMPT="$PROMPT" \
      "${extra[@]}" \
      LD_LIBRARY_PATH="$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin" \
      "$ROOT/target/release/inkwell" "$HOME/models/inkling" --run
  ) >"$log" 2>&1 &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{if($2>m)m=$2} END{print m+0}')
    [ "$m" -gt "$peak" ] && peak=$m
    sleep 0.5
  done
  set +e
  wait "$pid"
  rc=$?
  set -e
  md5=$(grep 'generated text' "$log" | md5sum | cut -c1-8)
  line=$(grep 'decode 8 tok in' "$log" | tail -1 || true)
  ms=$(printf '%s\n' "$line" | python3 -c 'import sys,re; s=sys.stdin.read(); m=re.search(r"in ([0-9.]+)s",s); print(f"{float(m.group(1))*1000/64:.3f}" if m else "NA")')
  printf 'SPEED ctx=%s tag=%s arm=%s thr=%s rc=%s ms_tok=%s md5=%s peak=%s\n' \
    "$CTX" "$tag" "$([ "$minthr" = 320 ] && echo cand || echo ctl)" "$minthr" "$rc" "$ms" "$md5" "$peak" | tee -a "$OUT"
}
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then
    if [ "$i" = 1 ] && [ ! -f "$STATE" ]; then
      run_one "s${CTX}-p${i}-ctl" 12097 save
    else
      run_one "s${CTX}-p${i}-ctl" 12097 load
    fi
    run_one "s${CTX}-p${i}-cand" 320 load
  else
    run_one "s${CTX}-p${i}-cand" 320 load
    run_one "s${CTX}-p${i}-ctl" 12097 load
  fi
done
echo "SPEED_DONE ctx=$CTX" | tee -a "$OUT"
