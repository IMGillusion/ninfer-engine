// Read-throughput of the disk KV tier against the store's lock design:
// single reader vs N in flight. The store used to hold its mutex across the
// 1.18MB CRC+memcpy, which serialized every reader at ~1.2 ms/page; this bench
// shows what the same volume does now that the payload work moved outside the lock.
#include "core/disk_kv_bridge.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using Clock = std::chrono::steady_clock;

static void until(auto f, const char* what) {
    const auto end = Clock::now() + std::chrono::seconds(120);
    while (!f()) {
        if (Clock::now() > end) { std::fprintf(stderr, "timeout: %s\n", what); std::exit(2); }
        std::this_thread::yield();
    }
}

int main(int argc, char** argv) {
    const std::size_t pages = argc > 1 ? std::stoul(argv[1]) : 1461;
    const std::size_t stride = 1179648;
    const std::string root = "/bench/read-bench";
    std::filesystem::remove_all(root);
    const std::size_t budget = (pages + 512) * stride * 4;  // 85/15 split across families
    DiskKVBridge bridge({.base_path = root, .main_page_stride = stride,
                         .backend_page_stride = stride, .state_page_stride = stride,
                         .capacity_bytes = budget});
    std::vector<std::byte> src(stride, std::byte{7}), dst(stride);
    std::vector<DiskKVIdentity> ids;
    ids.reserve(pages);
    const auto w0 = Clock::now();
    for (std::size_t i = 0; i < pages; ++i) {
        const DiskKVIdentity id{1, 2, static_cast<std::uint32_t>(i), 64};
        ids.push_back(id);
        src[0] = std::byte{static_cast<unsigned char>(i)};
        until([&] { return bridge.spill_page_sync(id, DiskKVKind::MainKV, src); }, "spill_page_sync");
    }
    bridge.wait_idle();
    const double write_s = std::chrono::duration<double>(Clock::now() - w0).count();
    std::printf("write %zu pages  %.2f ms/page  %.0f MB/s\n", pages,
                write_s * 1000.0 / static_cast<double>(pages),
                static_cast<double>(pages) * stride / 1048576.0 / write_s);

    auto read_all = [&](std::size_t inflight, const char* label) {
        const auto t0 = Clock::now();
        std::size_t done = 0, full = 0;
        while (done < ids.size()) {
            const std::size_t n = std::min(inflight, ids.size() - done);
            std::vector<ReadTicket> tickets;
            tickets.reserve(n);
            while (tickets.size() < n) {
                auto ticket = bridge.try_read(ids[done + tickets.size()], DiskKVKind::MainKV, dst);
                if (!ticket) { ++full; std::this_thread::yield(); continue; }
                tickets.push_back(std::move(ticket));
            }
            for (auto& ticket : tickets) {
                until([&] { return bridge.poll(ticket) != DiskReadResult::Pending; }, "read");
                const auto r = bridge.poll(ticket);
                if (r != DiskReadResult::Success) {
                    std::fprintf(stderr, "%s: read failed page=%zu result=%d\n", label, done, static_cast<int>(r));
                    std::exit(3);
                }
            }
            done += n;
        }
        const double s = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("%-10s in-flight=%-3zu  %.2f ms/page  %.0f MB/s  total=%.2fs  queue_full=%zu\n",
                    label, inflight, s * 1000.0 / static_cast<double>(pages),
                    static_cast<double>(pages) * stride / 1048576.0 / s, s, full);
    };

    read_all(1, "seq");
    read_all(4, "parallel");
    read_all(8, "parallel");
    read_all(16, "parallel");
    read_all(32, "parallel");
    return 0;
}
