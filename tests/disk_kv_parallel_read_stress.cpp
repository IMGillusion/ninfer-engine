// Concurrency safety of the lock-free read payload path: readers must never see a
// slot recycled underneath them (which would surface as a spurious BadCRC/BadIdentity).
// Readers hammer existing pages while a writer keeps inserting new ones to force
// continuous LRU eviction.
#include "core/disk_kv_bridge.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

using namespace ninfer;

int main() {
    const std::size_t stride = 65536;            // small so the store really fills up
    const std::size_t slots = 24;                // capacity in pages
    const std::string root = "/tmp/parallel-read-stress";
    std::filesystem::remove_all(root);
    DiskKVBridge bridge({.base_path = root, .main_page_stride = stride,
                         .capacity_bytes = (slots + 2) * stride * 4});
    std::vector<std::byte> payload(stride, std::byte{0xAB});
    // Seed a few known pages that readers will keep reading.
    std::vector<DiskKVIdentity> stable;
    for (std::uint32_t i = 0; i < 4; ++i) {
        const DiskKVIdentity id{7, 7, i, 64};
        std::fill(payload.begin(), payload.end(), std::byte{static_cast<unsigned char>(i + 1)});
        if (!bridge.spill_page_sync(id, DiskKVKind::MainKV, payload)) { std::fprintf(stderr, "seed spill failed\n"); return 2; }
        stable.push_back(id);
    }
    bridge.wait_idle();
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> reads{0}, mismatches{0}, missing{0}, errors{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 6; ++t) {
        readers.emplace_back([&, t] {
            std::vector<std::byte> dst(stride);
            while (!stop.load(std::memory_order_relaxed)) {
                const auto& id = stable[(t + reads.load()) % stable.size()];
                auto ticket = bridge.try_read(id, DiskKVKind::MainKV, dst);
                if (!ticket) { std::this_thread::yield(); continue; }
                while (bridge.poll(ticket) == DiskReadResult::Pending) { std::this_thread::yield(); }
                const auto r = bridge.poll(ticket);
                if (r == DiskReadResult::Success) {
                    // The bytes must be internally consistent: whole page of one colour.
                    const auto first = dst[0];
                    for (std::size_t i = 0; i < dst.size(); i += 4096) {
                        if (dst[i] != first) { ++mismatches; break; }
                    }
                    reads.fetch_add(1);
                } else if (r == DiskReadResult::Missing) {
                    ++missing;                      // legal: LRU may drop it
                } else {
                    ++errors;                       // BadCRC/BadIdentity = UNSAFE
                }
            }
        });
    }
    // Writer: keep inserting new identities so eviction runs continuously.
    for (std::uint32_t i = 0; i < 4000 && errors.load() == 0; ++i) {
        const DiskKVIdentity id{9, 9, i, 64};
        payload[0] = std::byte{static_cast<unsigned char>(i)};
        bridge.spill_page_sync(id, DiskKVKind::MainKV, payload);
    }
    stop = true;
    for (auto& r : readers) { r.join(); }
    bridge.wait_idle();
    std::printf("reads=%llu missing=%llu mismatches=%llu corrupt_errors=%llu\n",
                (unsigned long long)reads.load(), (unsigned long long)missing.load(),
                (unsigned long long)mismatches.load(), (unsigned long long)errors.load());
    return (mismatches.load() == 0 && errors.load() == 0) ? 0 : 1;
}
