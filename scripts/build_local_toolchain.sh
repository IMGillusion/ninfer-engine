#!/usr/bin/env bash
# Local build toolchain for this network: the vllm nightly base image is gone
# and registry pulls stall mid-layer, so the toolchain is assembled from apt
# (aliyun debian mirror + NVIDIA's official repo, which IS reachable) inside a
# persistent gcc:13 container. Produces the same CUDA 13.1 / sm_120a binary as
# the old Dockerfile.local flow; Dockerfile.runtime13 packages it.
#
#   docker rm -f nb-build
#   docker run -d --name nb-build -v "$(pwd):/src" -w /src \
#       docker.m.daocloud.io/library/gcc:13 sleep infinity
#   bash scripts/build_local_toolchain.sh      # idempotent
#   docker exec nb-build bash -c "cmake --build /build --parallel \
#       --target ninfer ninfer-serve ninfer-perplexity"
set -euo pipefail
cd /src

grep -qs aliyun /etc/apt/sources.list.d/debian.sources || cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: https://mirrors.aliyun.com/debian
Suites: bookworm bookworm-updates
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    gnupg wget ca-certificates python3 python3-pip ninja-build pkg-config \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev

if [ ! -x /usr/local/cuda-13.1/bin/nvcc ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        cuda-nvcc-13-1 cuda-cudart-dev-13-1
fi
# Vendored NVTX headers, same as Dockerfile.local copies into the toolkit.
cp -r /src/nvtx3 /usr/local/cuda-13.1/include/ 2>/dev/null || true

[ -d /build ] || cmake -S /src -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=OFF \
    -DNINFER_BUILD_BENCHMARKS=OFF \
    -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
echo "toolchain ready: cmake --build /build --parallel --target ninfer ninfer-serve ninfer-perplexity"
