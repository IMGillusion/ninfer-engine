// CPU link injection executes the real StateImage conv/recurrent/hidden/DFlash loop.
#include <ninfer/targets/qwen3_6/state_image.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
static int calls, drains, fail_at;
static cudaError_t submit_error, drain_error;
static bool owner_alive, scratch_alive, disk_written;
static std::vector<std::function<void()>> pending;
extern "C" cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind kind, cudaStream_t) {
    assert(owner_alive && scratch_alive && !disk_written && kind == cudaMemcpyDeviceToHost);
    if (++calls == fail_at) { return submit_error; }
    pending.emplace_back([=] { std::memcpy(d, s, n); });
    return cudaSuccess;
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) {
    assert(owner_alive && scratch_alive && !disk_written);
    ++drains;
    if (drain_error != cudaSuccess) { return drain_error; }
    for (auto& f : pending) { f(); }
    pending.clear();
    return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t) { return "injected-state-error"; }
static void run(bool dflash, int failure, cudaError_t error = cudaErrorInvalidValue,
                cudaError_t sync = cudaSuccess) {
    calls = drains = 0; fail_at = failure; submit_error = error; drain_error = sync;
    pending.clear(); owner_alive = scratch_alive = true; disk_written = false;
    StateImageSpec spec{.linear = {.layers=2, .conv_channels=5, .conv_width=3,
        .value_heads=2, .value_head_dim=4, .key_head_dim=3, .slot_count=3,
        .conv_dtype=DType::BF16}, .hidden=7};
    if (dflash) { spec.dflash_local = DFlashLocalStateSpec{2,17,2,4}; }
    LayoutBuilder builder;
    const auto layout = plan_state_image_device_pool(builder, spec);
    const auto size = builder.finish(256);
    std::unique_ptr<void, decltype(&std::free)> storage(std::aligned_alloc(256, size), &std::free);
    assert(storage);
    auto* raw = static_cast<std::byte*>(storage.get());
    for (size_t i=0; i<size; ++i) { raw[i] = std::byte((i*17+i/256)%251); }
    StateImageDevicePool pool({raw,size}, layout);
    const auto& host = pool.host_layout();
    std::vector<std::byte> out(host.image_bytes, std::byte{0xa5});
    auto expected = out;
    const int slot = 2;
    auto expect = [&](size_t offset, Tensor tensor) {
        std::memcpy(expected.data()+offset, tensor.data, tensor.bytes());
    };
    for (unsigned l=0; l<2; ++l) {
        expect(host.linear_conv.offset+l*host.linear_conv_layer_bytes, pool.linear().conv_slot(l,slot));
        expect(host.linear_recurrent.offset+l*host.linear_recurrent_layer_bytes, pool.linear().recurrent_slot(l,slot));
    }
    expect(host.continuation_hidden.offset, pool.continuation_hidden_slot(slot));
    if (dflash) {
        for (unsigned l=0; l<2; ++l) {
            const auto v = pool.dflash_local()->layer_view(l);
            expect(host.dflash_local_k->offset+l*host.dflash_local_layer_bytes, v.k.slice(3,slot,1));
            expect(host.dflash_local_v->offset+l*host.dflash_local_layer_bytes, v.v.slice(3,slot,1));
        }
    }
    const bool ok = pool.spill_copy_to_host(slot, {out.data(), &host}, nullptr);
    assert(drains==1 && pending.empty());
    if (failure) { assert(!ok && calls==failure && !disk_written); }
    else { assert(ok && out==expected); disk_written=true; }
    owner_alive = scratch_alive = false;
}
int main() {
    for (bool dflash : {false,true}) {
        run(dflash,0);
        const int total=calls;
        assert(total==(dflash ? 9 : 5));
        // Every plane, including both DFlash layers and hidden, not only endpoints.
        for (int f=1; f<=total; ++f) {
            run(dflash,f);
            run(dflash,f,cudaErrorMemoryAllocation);
        }
        for (int mode=0; mode<3; ++mode) {
            const auto pid=fork(); assert(pid>=0);
            if (!pid) {
                if (mode==0) { run(dflash,3,cudaErrorIllegalAddress); }
                if (mode==1) { run(dflash,3,cudaErrorInvalidValue,cudaErrorLaunchFailure); }
                if (mode==2) { run(dflash,0,cudaSuccess,cudaErrorLaunchFailure); }
                _exit(99);
            }
            int status=0; assert(waitpid(pid,&status,0)==pid);
            assert(WIFEXITED(status) && WEXITSTATUS(status)==70);
        }
    }
    puts("spill_copy_state_planes PASS: real common/DFlash loops, every-plane failure, payload/padding, six fatal subprocess exits=70");
}
