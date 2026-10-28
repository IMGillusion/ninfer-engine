#pragma once
// Test fixture: spill one page and return only once it is physically
// readable. The engine's contains() deliberately reports submitted-but-
// in-flight writes as present (the probe/promised contract), so a fixture
// that waits on contains would race the write worker; restore_page reads
// through the store and only succeeds once the bytes actually landed.
#include "core/disk_kv_bridge.h"

#include <chrono>
#include <span>
#include <thread>
#include <vector>

inline bool test_spill_wait(ninfer::DiskKVBridge& bridge, const ninfer::DiskKVIdentity& id,
                            ninfer::DiskKVKind kind, std::span<const std::byte> bytes,
                            std::chrono::milliseconds) {
    if (!bridge.spill_page_wait(id, kind, bytes, std::chrono::milliseconds(30000))) { return false; }
    std::vector<std::byte> landed(bytes.size());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30000);
    while (!bridge.restore_page(id, kind, landed)) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::vector<std::byte>(bytes.begin(), bytes.end()) == landed;
}
