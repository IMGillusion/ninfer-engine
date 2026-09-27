#!/usr/bin/env bash
set -euo pipefail
cd /src
inc=(-I. -Isrc -Iinclude -Ithird_party -Isrc/targets/qwen3_6/export -Isrc/targets/qwen3_6_27b/export -Isrc/targets/qwen3_6_35b_a3b/export -I/usr/local/cuda/include)
for v in qwen3_6_27b qwen3_6_35b_a3b; do
 g++ -std=c++20 -fsyntax-only "${inc[@]}" src/targets/$v/impl/variant.cpp
 echo "$v syntax PASS"
done
for t in seed_lifecycle_test disk_kv_read_ticket_test disk_kv_batch_probe_test disk_kv_async_ticket_test owner_spill_batch_bridge_test; do
 g++ -std=c++20 -O2 -pthread -Isrc tests/$t.cpp src/core/disk_kv_bridge.cpp src/core/disk_kv_store.cpp -o /tmp/$t
 /tmp/$t
done
g++ -std=c++20 -O2 -pthread -Isrc tests/owner_spill_batch_test.cpp -o /tmp/batch
/tmp/batch
g++ -std=c++20 -O2 -pthread "${inc[@]}" tests/owner_spill_plan_test.cpp src/targets/qwen3_6/impl/runtime/prefix_identity.cpp -o /tmp/plan
/tmp/plan
g++ -std=c++20 -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc tests/seed_lifecycle_test.cpp src/core/disk_kv_bridge.cpp src/core/disk_kv_store.cpp -o /tmp/lifecycle-san
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 /tmp/lifecycle-san
if [[ ${SEED_GPU_TEST:-0} == 1 ]]; then
 g++ -std=c++20 -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections "${inc[@]}" tests/state_image_chunk_test.cpp src/targets/qwen3_6/impl/state/state_image.cpp src/core/linear_attention_state.cpp src/core/cyclic_kv_cache.cpp src/core/tensor.cpp src/core/dtype.cpp src/core/layout.cpp -x c++ src/core/arena.cu src/core/device.cu -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -o /tmp/state_chunk
 /tmp/state_chunk
fi
