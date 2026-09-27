#include "core/disk_kv_bridge.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
using namespace ninfer;
int main() {
    for (bool sleep : {true, false}) {
        const std::string root = sleep ? "/tmp/ninfer-poll-sleep" : "/tmp/ninfer-poll-yield";
        std::filesystem::remove_all(root);
        {
            DiskKVBridge bridge({.base_path=root, .main_page_stride=4096,
                                 .capacity_bytes=16*1024*1024});
            std::vector<std::byte> bytes(4096, std::byte{42});
            const auto start = std::chrono::steady_clock::now();
            auto boundary = [&] {
                if (sleep) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                else std::this_thread::yield();
                assert(std::chrono::steady_clock::now()-start < std::chrono::seconds(30));
            };
            for (unsigned i=0; i<256; ++i) {
                DiskKVIdentity id{.lo=i+1, .hi=7, .tag=1, .frontier=64};
                SpillTicket ticket;
                while (!(ticket=bridge.try_probe(id,DiskKVKind::MainKV))) boundary();
                do { boundary(); } while (DiskKVBridge::poll(ticket)==SpillStatus::Pending);
                assert(DiskKVBridge::poll(ticket)==SpillStatus::Missing);
                while (!(ticket=bridge.try_submit(id,DiskKVKind::MainKV,bytes))) boundary();
                do { boundary(); } while (DiskKVBridge::poll(ticket)==SpillStatus::Pending);
                assert(DiskKVBridge::poll(ticket)==SpillStatus::Written);
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            bridge.wait_idle();
            std::printf("%s: 256 sequential probe+write pairs, 4096 bytes/page, %.3f ms\n",sleep?"1ms boundary":"yield boundary",ms);
        }
        std::filesystem::remove_all(root);
    }
}
