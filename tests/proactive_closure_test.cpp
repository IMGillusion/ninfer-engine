#include "core/proactive_closure.h"
#include <cassert>
#include <filesystem>
#include <iostream>
using namespace ninfer;
int main() {
    ProactiveWatermarks water;
    std::array<std::pair<std::uint64_t,std::uint64_t>,1> usage{{{89,100}}};
    assert(!water.update(usage)); usage[0].first=90; assert(water.update(usage));
    usage[0].first=80; assert(water.update(usage)); usage[0].first=75; assert(!water.update(usage));
    const auto root=std::filesystem::temp_directory_path()/"proactive-closure-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    DiskKVBridge bridge({.base_path=root.string(),.main_page_stride=64,.backend_page_stride=64,
        .state_page_stride=256,.capacity_bytes=1024*1024});
    std::uint64_t released=0;
    {
        ProactiveClosure closure;
        for (unsigned k=0;k<3;++k) closure.entries.push_back({{k+1,5,7,64},static_cast<DiskKVKind>(k),k==2?256U:64U});
        assert(closure.protect(bridge));
        ProactiveClosure competing; competing.entries=closure.entries; assert(!competing.protect(bridge));
        auto copy=[&](std::size_t i,std::span<std::byte> out) { std::fill(out.begin(),out.end(),std::byte(i+17)); return true; };
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        for (;;) {
            const auto r=closure.tick(bridge,copy);
            assert(r!=ProactiveClosure::Result::Failed);
            if(r==ProactiveClosure::Result::Ready) break;
            assert(released==0); // prewrite never counts as reclaimed
            assert(std::chrono::steady_clock::now()<deadline);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        assert(closure.verified_bytes==384);
        for (std::size_t i=0;i<closure.entries.size();++i) {
            const auto& e=closure.entries[i];
            assert(!bridge.drop_page(e.id,e.kind));
            std::vector<std::byte> payload(e.bytes,std::byte(99));
            // Force the store well past capacity while closure lease is held.
            for(unsigned n=0;n<300;++n) assert(bridge.spill_page_sync({n+1000,0,0,64},e.kind,payload));
            assert(bridge.restore_page(e.id,e.kind,payload));
            for(auto b:payload) assert(b==std::byte(i+17));
        }
        released=closure.verified_bytes; // CPU fixture release, not GPU measurement
    }
    assert(released==384);
    assert(bridge.drop_page({1,5,7,64},DiskKVKind::MainKV));
    {
        ProactiveClosure failed; failed.entries.push_back({{88,0,0,64},DiskKVKind::MainKV,64});
        assert(failed.protect(bridge));
        assert(failed.tick(bridge,[](auto,auto){return false;})==ProactiveClosure::Result::Failed);
        assert(failed.verified_bytes==0);
    }
    {
        ProactiveClosure cancelled; cancelled.entries.push_back({{99,0,0,64},DiskKVKind::MainKV,64});
        assert(cancelled.protect(bridge));
        cancelled.tick(bridge,[](auto,std::span<std::byte> out){std::fill(out.begin(),out.end(),std::byte(44));return true;});
        cancelled.cancelled=true;
        assert(cancelled.tick(bridge,[](auto,auto){assert(false);return false;})==ProactiveClosure::Result::Failed);
    }
    bridge.wait_idle();
    std::cout<<"PASS watermarks high=90 low=75; full-state=256 main=64 backend=64 readback=384; lease LRU+drop protected; failure/cancel retained; prewrite reclaimed=0\n";
}
