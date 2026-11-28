#!/usr/bin/env bash
set -u
BIN=${1:-/home/ash/.cache/inkwell-front4e/bin/inkwell-f4e-cand}
/home/ash/inkwell/opt/tools/gpu-run -p normal -t f4e-ctx512-a -- opt/tools/benchq.sh f4e-ctx512-a "$BIN"
/home/ash/inkwell/opt/tools/gpu-run -p normal -t f4e-ctx512-b -- opt/tools/benchq.sh f4e-ctx512-b "$BIN"
