#!/usr/bin/env bash
set -u
BIN=$1
exec /home/ash/inkwell/opt/tools/gpu-run -p normal -t f3d-ctx512 -- opt/tools/benchq.sh f3d-ctx512 "$BIN"
