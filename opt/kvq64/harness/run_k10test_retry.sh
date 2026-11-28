#!/usr/bin/env bash
# K10V6 controlled write-roundtrip + attention-consistency test (retry —
# first attempt missed LD_LIBRARY_PATH for the test binary).
set -u
export LD_LIBRARY_PATH=$HOME/lfm-gtx1060-bench/cuda-12.6/lib64:$HOME/lfm-gtx1060-bench/build-inkwell-dp4a/bin
echo '== K10V6 controlled roundtrip+attention =='
/home/ash/.cache/inkwell-intkv/kvq-attn-test-k10
echo "k10test_rc=$?"
