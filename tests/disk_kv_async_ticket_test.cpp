#include "core/disk_kv_bridge.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
using namespace ninfer;
static SpillTicket submit(DiskKVBridge& b, DiskKVIdentity id, DiskKVKind kind,
                          std::span<const std::byte> bytes, bool probe=false) {
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    SpillTicket t;
    while (!(t=probe?b.try_probe(id,kind):b.try_submit(id,kind,bytes))) {
        assert(std::chrono::steady_clock::now()<end); std::this_thread::yield();
    }
    return t;
}
static SpillStatus poll(SpillTicket t) {
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(20);
    while(DiskKVBridge::poll(t)==SpillStatus::Pending) {
        assert(std::chrono::steady_clock::now()<end); std::this_thread::yield();
    }
    return DiskKVBridge::poll(t);
}
int main() {
    const std::string root="/tmp/ninfer-async-ticket-test";
    std::filesystem::remove_all(root);
    constexpr size_t stride=4096;
    std::vector<std::byte> bytes(stride,std::byte{37}), dst(stride);
    DiskKVIdentity id{.lo=1,.hi=2,.tag=3,.frontier=64};
    {
        DiskKVBridge b({.base_path=root,.main_page_stride=stride,
                       .backend_page_stride=stride,.state_page_stride=stride,
                       .capacity_bytes=32*1024*1024});
        assert(poll(submit(b,id,DiskKVKind::MainKV,{},true))==SpillStatus::Missing);
        for (auto kind:{DiskKVKind::MainKV,DiskKVKind::BackendKV,DiskKVKind::StateImage}) {
            auto t=submit(b,id,kind,bytes);
            assert(poll(t)==SpillStatus::Written);
            assert(b.restore_page(id,kind,dst) && bytes==dst);
            assert(poll(submit(b,id,kind,{},true))==SpillStatus::Present);
        }
        // Invalid write is explicitly Failed, never a queued-success acknowledgement.
        assert(poll(submit(b,id,DiskKVKind::MainKV,std::span(bytes).first(1)))==SpillStatus::Failed);
        std::vector<SpillTicket> tickets;
        for (unsigned i=0;i<2048;++i) {
            auto key=id;key.lo=100+i;
            tickets.push_back(submit(b,key,DiskKVKind::MainKV,bytes));
        }
        for(auto& t:tickets) assert(poll(t)==SpillStatus::Written);
        b.wait_idle();
    }
    {
        DiskKVBridge b({.base_path=root,.main_page_stride=stride,
                       .backend_page_stride=stride,.state_page_stride=stride,
                       .capacity_bytes=32*1024*1024});
        for(unsigned i=0;i<2048;++i) {
            auto key=id;key.lo=100+i;
            assert(b.restore_page(key,DiskKVKind::MainKV,dst) && dst==bytes);
        }
    }
    std::filesystem::remove_all(root);
    std::puts("PASS async probe/write/failure, all 3 families, 2048-ticket drain and reopen");
}
