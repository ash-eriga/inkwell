#!/bin/bash
# Spec-decode / serial A/B harness: N runs → JSONL lines in opt/ledger.jsonl
# usage: spec_bench.sh <config-name> <N> [serial|spec]
#   serial: INKWELL_GENS tokens through step()            → ms_tok
#   spec:   INKWELL_SPEC_GENS tokens through the MTP loop  → tok_s, emit, verify_ms, draft_ms
# Env overrides (INKWELL_*) are passed through, so candidates set their knobs
# in the calling shell. Refuses to run if an inkwell process is already alive.
set -u
CFG=$1; N=${2:-3}; MODE=${3:-spec}
cd ~/inkwell || exit 1
if pgrep -x inkwell >/dev/null; then echo "REFUSE: inkwell process already running" >&2; exit 2; fi
export LD_LIBRARY_PATH=~/lfm-gtx1060-bench/cuda-12.6/lib64:~/lfm-gtx1060-bench/build-inkwell-dp4a/bin
export GGML_PASCAL_GRAPHS=1
mkdir -p /tmp/ikw
GOLD='capital of the United States is Washington'
med() { sort -n | awk '{a[NR]=$1} END{ if (NR==0) {print "nan"} else {printf "%.3f", a[int((NR+1)/2)]} }'; }
for i in $(seq 1 "$N"); do
    t0=$(date +%s.%N)
    if [ "$MODE" = serial ]; then
        out=$(timeout 400 ./target/release/inkwell ~/models/inkling --run 2>&1)
    else
        out=$(INKWELL_SPEC=mtp timeout 400 ./target/release/inkwell ~/models/inkling --run 2>&1)
    fi
    rc=$?
    wall=$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
    echo "$out" > "/tmp/ikw/${CFG}.${i}.log"
    txt=$(echo "$out" | grep -oP '(generated|spec) text: "\K[^"]*' | head -1)
    gold=no; echo "$txt" | grep -q "$GOLD" && gold=yes
    if [ "$MODE" = serial ]; then
        s=$(echo "$out" | grep -oP 'decode 8 tok in \K[0-9.]+s' | head -1 | tr -d 's')
        gens=$(echo "$out" | grep -c '^gen ')
        if [ -n "$s" ] && [ "$rc" = 0 ]; then
            ms=$(awk -v s="$s" -v g="$gens" 'BEGIN{printf "%.2f", s*1000/g}')
            sw=$(echo "$out" | grep -oP 'stepwall=\K[0-9.]+' | med)
            printf '{"ts":"%s","config":"%s","mode":"serial","i":%d,"rc":%d,"wall_s":%s,"decode_s":%s,"gens":%d,"ms_tok":%s,"stepwall_med":%s,"golden":"%s","text":"%s"}\n' \
                "$(date -Is)" "$CFG" "$i" "$rc" "$wall" "$s" "$gens" "$ms" "$sw" "$gold" "$txt" | tee -a opt/ledger.jsonl
        else
            printf '{"ts":"%s","config":"%s","mode":"serial","i":%d,"rc":%d,"error":"no decode line","tail":"%s"}\n' \
                "$(date -Is)" "$CFG" "$i" "$rc" "$(echo "$out" | tail -3 | tr '\n' '|')" | tee -a opt/ledger.jsonl
        fi
    else
        tok=$(echo "$out" | grep -oP '^spec: \K[0-9]+(?= tok in)' | head -1)
        if [ -n "$tok" ] && [ "$rc" = 0 ]; then
            read -r tps rounds acc emit < <(echo "$out" | grep -oP '^spec: [0-9]+ tok in [0-9.]+s = \K[0-9.]+ tok/s \([0-9]+ rounds, accept [0-9]+%, [0-9.]+ tok/round' | awk '{gsub(/[^0-9.]/," ",$0); print $1, $2, $3, $4}')
            tot=$(echo "$out" | grep -oP '^spec: [0-9]+ tok in \K[0-9.]+' | head -1)
            ver=$(echo "$out" | grep -oP '^spec r[0-9]+: verify=\K[0-9.]+' | med)
            drf=$(echo "$out" | grep -oP '^ +draft=\K[0-9.]+' | med)
            ver_n=$(echo "$out" | grep -cP '^spec r[0-9]+: verify=')
            printf '{"ts":"%s","config":"%s","mode":"spec","i":%d,"rc":%d,"wall_s":%s,"tok":%s,"rounds":%s,"tok_s":%s,"accept_pct":%s,"emit":%s,"verify_ms_med":%s,"draft_ms_med":%s,"verify_n":%s,"golden":"%s","text":"%s"}\n' \
                "$(date -Is)" "$CFG" "$i" "$rc" "$wall" "$tok" "${rounds:-0}" "${tps:-nan}" "${acc:-nan}" "${emit:-nan}" "$ver" "$drf" "$ver_n" "$gold" "$txt" | tee -a opt/ledger.jsonl
        else
            printf '{"ts":"%s","config":"%s","mode":"spec","i":%d,"rc":%d,"wall_s":%s,"error":"no spec line","tail":"%s"}\n' \
                "$(date -Is)" "$CFG" "$i" "$rc" "$wall" "$(echo "$out" | tail -4 | tr '\n' '|')" | tee -a opt/ledger.jsonl
        fi
    fi
done
# summary (this config's last N lines)
echo "== summary $CFG ($MODE, last $N) =="
grep -F "\"config\":\"$CFG\"" opt/ledger.jsonl | grep -F "\"mode\":\"$MODE\"" | tail -n "$N" > /tmp/ikw/sum.jsonl
if [ "$MODE" = serial ]; then
    echo -n "ms_tok: "; grep -oP '"ms_tok":\K[0-9.]+' /tmp/ikw/sum.jsonl | tr '\n' ' '; echo
    echo -n "median ms_tok: "; grep -oP '"ms_tok":\K[0-9.]+' /tmp/ikw/sum.jsonl | med; echo
else
    echo -n "tok_s:  "; grep -oP '"tok_s":\K[0-9.]+' /tmp/ikw/sum.jsonl | tr '\n' ' '; echo
    echo -n "emit:   "; grep -oP '"emit":\K[0-9.]+' /tmp/ikw/sum.jsonl | tr '\n' ' '; echo
    echo -n "verify: "; grep -oP '"verify_ms_med":\K[0-9.]+' /tmp/ikw/sum.jsonl | tr '\n' ' '; echo
    echo -n "draft:  "; grep -oP '"draft_ms_med":\K[0-9.]+' /tmp/ikw/sum.jsonl | tr '\n' ' '; echo
    echo -n "golden: "; grep -oP '"golden":\K[0-9a-z]+' /tmp/ikw/sum.jsonl | tr '\n' ' '; echo
    echo -n "median tok_s: "; grep -oP '"tok_s":\K[0-9.]+' /tmp/ikw/sum.jsonl | med; echo
fi
