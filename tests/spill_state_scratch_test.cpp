#include <bits/stdc++.h>
#define private public
#include "core/disk_kv_bridge.h"
#undef private
#include "targets/qwen3_6/impl/runtime/spill_state_scratch.h"
using namespace ninfer;
using namespace ninfer::targets::qwen3_6;
static unsigned allocations=0, copies=0;
extern "C" cudaError_t cudaMallocHost(void** p, size_t n) { ++allocations; *p=std::malloc(n); return *p?cudaSuccess:cudaErrorMemoryAllocation; }
extern "C" cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
extern "C" cudaError_t cudaMemcpyAsync(void* d,const void* s,size_t n,cudaMemcpyKind,cudaStream_t) { ++copies; std::memcpy(d,s,n); return cudaSuccess; }
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
extern "C" const char* cudaGetErrorString(cudaError_t) { return "scratch-test"; }
extern "C" const char* cudaGetErrorName(cudaError_t) { return "scratch-test"; }
int main() {
    StateImageSpec spec{.linear={.layers=2,.conv_channels=5,.conv_width=3,.value_heads=2,.value_head_dim=4,.key_head_dim=3,.slot_count=3,.conv_dtype=DType::BF16},.hidden=7};
    LayoutBuilder builder; auto layout=plan_state_image_device_pool(builder,spec);
    auto size=builder.finish(256);
    std::unique_ptr<void,decltype(&std::free)> backing(std::aligned_alloc(256,size),&std::free);
    std::memset(backing.get(),0x37,size);
    StateImageDevicePool device({static_cast<std::byte*>(backing.get()),size},layout);
    HostStatePool host(device.host_layout(),16);
    auto scratch=host.allocate(); assert(scratch);
    // Saturate the configured shared pool; spill must never allocate another slot.
    while(host.allocate()) {}
    assert(host.occupied()==16 && allocations==1);
    const auto startup_allocations=allocations;
    StateScratchLease gate;
    auto pressure=gate.try_acquire(); assert(pressure);
    StateScratchLease::Lease legacy;
    auto attempt=[&] { return runtime_support::spill_state_to_scratch(device,2,&host,scratch,gate,legacy,nullptr); };
    assert(!attempt() && copies==0 && allocations==startup_allocations);
    pressure.reset();
    auto view=attempt(); assert(view && copies==5);
    const size_t n=device.host_layout().image_bytes;
    std::vector<std::byte> expected(view->data,view->data+n);
    const std::string root="/tmp/spill-state-scratch-test";
    std::filesystem::remove_all(root);
    {
        DiskKVBridge bridge({.base_path=root,.main_page_stride=64,.backend_page_stride=64,.state_page_stride=n,.capacity_bytes=4*1024*1024});
        // Move ownership into pressure progress, exactly as a retained borrow.
        pressure=std::move(legacy); assert(pressure && !legacy);
        SpillTicket ticket;
        std::unique_lock lock(bridge.families_[2].store->mu_);
        const DiskKVIdentity id{.lo=19,.frontier=64};
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        do {
            ticket=bridge.try_submit(id,DiskKVKind::StateImage,{view->data,n});
            assert(std::chrono::steady_clock::now()<deadline);
            if (!ticket) { std::this_thread::yield(); }
        } while (!ticket); // Admission may transiently contend with the worker queue lock.
        assert(DiskKVBridge::poll(ticket)==SpillStatus::Pending);
        std::memset(backing.get(),0x92,size);
        for(int i=0;i<100;++i) {
            assert(!attempt()); // legacy skips; no self-wait and no overwrite
            assert(!gate.try_acquire()); // seed excluded too
            assert(std::equal(expected.begin(),expected.end(),view->data));
        }
        assert(copies==5 && allocations==startup_allocations && host.occupied()==16);
        lock.unlock();
        bridge.wait_idle(); assert(DiskKVBridge::poll(ticket)==SpillStatus::Written);
        // Even an already-terminal ticket retains protection until cleanup.
        assert(!attempt());
        std::vector<std::byte> read(n);
        assert(bridge.restore_page(id,DiskKVKind::StateImage,read) && read==expected);
        ticket.reset(); pressure.reset();
        auto second=attempt(); assert(second && copies==10);
        assert(!std::equal(expected.begin(),expected.end(),second->data));
        legacy.reset();
        assert(allocations==startup_allocations && host.occupied()==16);
    }
    std::filesystem::remove_all(root);
    std::puts("state scratch PASS: busy skip=100 borrow readback=1 startup mallocHost=1 extra=0 occupied=16");
}
