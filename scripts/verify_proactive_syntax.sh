#!/usr/bin/env bash
set -euo pipefail
cd /src
inc=(-I. -Isrc -Iinclude -Ithird_party -Isrc/targets/qwen3_6/export -Isrc/targets/qwen3_6_27b/export -Isrc/targets/qwen3_6_35b_a3b/export -I/usr/local/cuda/include)
for v in qwen3_6_27b qwen3_6_35b_a3b; do
  g++ -std=c++20 -fsyntax-only "${inc[@]}" src/targets/$v/impl/variant.cpp
  echo "$v syntax PASS"
done
g++ -std=c++20 -fsyntax-only "${inc[@]}" src/runtime/engine/engine.cpp && echo "engine.cpp syntax PASS"
