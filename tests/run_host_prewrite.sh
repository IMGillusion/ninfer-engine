#!/bin/bash
set -euo pipefail
cd /src
INC='-I. -Isrc -Iinclude -Ithird_party -Isrc/targets/qwen3_6/export -Isrc/targets/qwen3_6_27b/export -Isrc/targets/qwen3_6_35b_a3b/export -I/usr/local/cuda/include'
for v in qwen3_6_27b qwen3_6_35b_a3b; do
 g++ -std=c++20 -fsyntax-only $INC src/targets/$v/impl/variant.cpp
 printf 'SYNTAX PASS %s\n' "$v"
done
g++ -std=c++20 -fsyntax-only $INC src/runtime/engine/engine.cpp
printf 'SYNTAX PASS engine.cpp\n'
for test in host_kv_prewrite owner_spill_batch_bridge disk_kv_batch_probe disk_kv_flush_concurrency; do
 g++ -std=c++20 -O2 -Isrc tests/${test}_test.cpp src/core/disk_kv_bridge.cpp src/core/disk_kv_store.cpp -pthread -o /tmp/$test
 /tmp/$test
done
g++ -std=c++20 -O2 $INC tests/owner_spill_plan_test.cpp src/targets/qwen3_6/impl/runtime/prefix_identity.cpp -o /tmp/plan
/tmp/plan
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc tests/host_kv_prewrite_test.cpp src/core/disk_kv_bridge.cpp src/core/disk_kv_store.cpp -pthread -o /tmp/prewrite-asan
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 /tmp/prewrite-asan
printf 'ALL TESTS PASS\n'
