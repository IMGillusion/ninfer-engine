// Bridge round-trip test (per-page content addressing): spill one page, free the
// host copy, restore it back, verify byte-identity, dedupe, LRU eviction, and
// miss fallback.
//
// Exercises DiskKVBridge directly. page_stride uses a realistic main-KV value
// (1.516 MiB) so the byte geometry matches the engine.
// Build: g++ -std=c++20 -O2 -I . tests/disk_kv_bridge_test.cpp
//         src/core/disk_kv_bridge.cpp src/core/disk_kv_store.cpp -o dkb_test
// Run:    ./dkb_test <tmpdir>

#include "core/disk_kv_bridge.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s (line %d)\n", msg, __LINE__); ++failures; } \
    else { std::printf("ok:   %s\n", msg); } \
} while (0)

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/dkb";
    const std::size_t stride = 1589632;  // 1.516 MiB, typical main-KV page

    // ~30 main pages (~46 MB) so LRU triggers after a few fills.
    DiskKVBridge b(DiskKVBridge::Options{
        .base_path            = dir,
        .main_page_stride     = stride,
        .backend_page_stride  = stride,
        .state_page_stride    = 0,            // state family disabled here
        .capacity_bytes       = 30 * stride,  // split 60/30/10 across enabled families
        .verify_crc           = true});
    CHECK(b.enabled(), "bridge enabled");
    CHECK(b.path(DiskKVKind::MainKV).find("main") != std::string::npos, "main family path");
    CHECK(b.slot_count(DiskKVKind::MainKV) >= 12, "main family has slot budget");

    auto make_bytes = [&](std::uint8_t seed) {
        std::vector<std::byte> page(stride);
        for (std::size_t i = 0; i < page.size(); ++i) {
            page[i] = static_cast<std::byte>((i * 131 + seed) & 0xFF);
        }
        return page;
    };

    // ---- 1. spill + restore one page, verify identity ----
    DiskKVIdentity id{.lo = 0xA11CE, .hi = 0xBEEF, .tag = 0, .frontier = 640};
    auto src = make_bytes(17);
    CHECK(b.spill_page(id, DiskKVKind::MainKV, std::span<const std::byte>(src.data(), src.size())),
          "spill one page");
    b.wait_idle();
    CHECK(b.contains(id, DiskKVKind::MainKV), "identity now restorable");
    // One live slot occupies a PITCH (header + page, 4KiB-aligned), not the
    // bare page stride.
    const std::size_t pitch =
        (stride + DiskKVStore::kSlotHeaderSize + 4095) & ~static_cast<std::size_t>(4095);
    CHECK(b.used_bytes(DiskKVKind::MainKV) == pitch, "used bytes == one slot pitch");

    std::vector<std::byte> dst(stride, std::byte{0});
    CHECK(b.restore_page(id, DiskKVKind::MainKV, std::span<std::byte>(dst.data(), dst.size())),
          "restore_page hits");
    CHECK(std::memcmp(src.data(), dst.data(), stride) == 0, "round-trip bytes identical");

    // ---- 2. dedupe: same identity again is a no-op ----
    const DiskKVBridgeStats before = b.stats();
    CHECK(b.spill_page(id, DiskKVKind::MainKV, std::span<const std::byte>(src.data(), src.size())),
          "re-spill same identity");
    const DiskKVBridgeStats after = b.stats();
    CHECK(after.spill_dups == before.spill_dups + 1, "dedupe counted");
    CHECK(after.spills == before.spills, "dedupe did not add a spill");

    // ---- 3. miss fallback: unknown identity must miss ----
    DiskKVIdentity cold{.lo = 999, .hi = 1000, .tag = 0, .frontier = 777};
    CHECK(!b.contains(cold, DiskKVKind::MainKV), "unknown identity absent");
    std::vector<std::byte> missbuf(stride);
    CHECK(!b.restore_page(cold, DiskKVKind::MainKV,
                          std::span<std::byte>(missbuf.data(), missbuf.size())),
          "unknown identity restore misses");
    // Same digest, different tag = different KV -> must also miss.
    DiskKVIdentity retag{.lo = 0xA11CE, .hi = 0xBEEF, .tag = 1, .frontier = 640};
    CHECK(!b.contains(retag, DiskKVKind::MainKV), "same digest, different tag is a distinct key");

    // ---- 4. LRU eviction: fill the family, then one more spill evicts the
    // coldest; a just-touched key must survive.
    const std::uint32_t main_cap = b.slot_count(DiskKVKind::MainKV);
    CHECK(main_cap >= 4, "main capacity large enough for the scenario");
    b.wait_idle();
    for (std::uint32_t i = 0; i + 1 < main_cap; ++i) {
        DiskKVIdentity p{.lo = 1000 + i, .hi = 2000 + i, .tag = 0,
                         .frontier = static_cast<std::uint32_t>(64 * (i + 1))};
        auto page = make_bytes(static_cast<std::uint8_t>(100 + (i & 0xFF)));
        CHECK(b.spill_page(p, DiskKVKind::MainKV,
                           std::span<const std::byte>(page.data(), page.size())),
              "fill page");
    }
    CHECK(b.spill_page(id, DiskKVKind::MainKV,
                       std::span<const std::byte>(src.data(), src.size())), "spill id (warm)");
    b.wait_idle();
    b.touch(id, DiskKVKind::MainKV);
    // Overflow: spill fresh ids until an LRU eviction actually happens (a
    // fresh id may hash to a DEAD slot first — that is a plain insert, not an
    // eviction; with main_cap live slots a few trials guarantee one).
    const DiskKVBridgeStats pre_evict = b.stats();
    std::uint32_t tries = 0;
    for (; tries < main_cap * 4; ++tries) {
        DiskKVIdentity p_next{.lo = 5000 + tries, .hi = 6000 + tries, .tag = 0,
                              .frontier = static_cast<std::uint32_t>(64 * (2000 + tries))};
        auto page = make_bytes(static_cast<std::uint8_t>(150 + (tries & 0xFF)));
        CHECK(b.spill_page(p_next, DiskKVKind::MainKV,
                           std::span<const std::byte>(page.data(), page.size())),
              "overflow spill");
        b.wait_idle();
        if (b.stats().evicted_slots > pre_evict.evicted_slots) { break; }
    }
    CHECK(tries < main_cap * 4, "LRU eviction actually triggered");
    b.wait_idle();
    CHECK(b.contains(id, DiskKVKind::MainKV), "touched key survived eviction");

    // ---- 5. family isolation ----
    DiskKVIdentity bid{.lo = 5, .hi = 5, .tag = 0, .frontier = 64};
    auto bpage = make_bytes(55);
    CHECK(b.spill_page(bid, DiskKVKind::BackendKV,
                       std::span<const std::byte>(bpage.data(), bpage.size())),
          "spill backend page");
    b.wait_idle();
    CHECK(b.used_bytes(DiskKVKind::BackendKV) == pitch, "backend family counts separately");
    CHECK(!b.contains(bid, DiskKVKind::MainKV), "backend key absent from main family");
    std::vector<std::byte> missbuf2(stride);
    CHECK(!b.restore_page(retag, DiskKVKind::MainKV,
                          std::span<std::byte>(missbuf2.data(), missbuf2.size())),
          "distinct-tag page misses restore");

    // ---- 6. stats sanity ----
    const DiskKVBridgeStats s = b.stats();
    CHECK(s.spills > 0 && s.spill_bytes > 0, "spill stats counted");
    CHECK(s.restores >= 1, "restore stats counted");
    CHECK(s.restore_misses >= 2, "miss stats counted");
    CHECK(s.evicted_slots >= 1, "eviction stats counted");

    // ---- 7. probe_prefix: contiguous prefix check ----
    {
        // Build a small chain: pages 0,1,2 all restorable; page 3 not.
        std::vector<DiskKVIdentity> chain;
        for (std::uint32_t i = 0; i < 3; ++i) {
            DiskKVIdentity c{.lo = 5000 + i, .hi = 6000 + i, .tag = 0, .frontier = 64 * (i + 1)};
            auto page = make_bytes(static_cast<std::uint8_t>(300 + i));
            b.spill_page(c, DiskKVKind::MainKV, std::span<const std::byte>(page.data(), page.size()));
            chain.push_back(c);
        }
        b.wait_idle();
        chain.push_back(DiskKVIdentity{.lo = 99999, .hi = 99999, .tag = 0, .frontier = 777});
        CHECK(b.probe_prefix(chain, DiskKVKind::MainKV) == 3, "probe stops at first miss");
    }

    std::printf("\n%s: %d failure(s)\n", failures ? "TEST FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
