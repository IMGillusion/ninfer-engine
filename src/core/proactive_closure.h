#pragma once
#include <cstdlib>
#include "core/disk_kv_bridge.h"
#include <functional>

namespace ninfer {
// One independently scheduled closure. Only owned pageable bytes cross the
// boundary; no owner claim, host pin or startup scratch lease survives tick().
class ProactiveClosure {
public:
    ~ProactiveClosure() {
        // Teardown only: the accepted CPU read must not outlive its destination.
        while (!drained()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    static constexpr std::size_t max_payload_bytes = 256ULL << 20;
    static constexpr std::size_t max_entries = 32768;
    struct Entry { DiskKVIdentity id; DiskKVKind kind; std::size_t bytes; };
    enum class Result { Pending, Ready, Failed };
    std::vector<Entry> entries;
    std::size_t cursor = 0;
    std::uint64_t verified_bytes = 0;
    bool failed = false;
    bool cancelled = false;

    bool protect(DiskKVBridge& bridge) {
        if (entries.empty() || entries.size() > max_entries) return false;
        std::array<std::vector<DiskKVIdentity>,3> ids;
        for (const auto& e : entries) {
            if (!e.bytes || e.bytes > max_payload_bytes) return false;
            ids[static_cast<unsigned>(e.kind)].push_back(e.id);
        }
        lease = bridge.try_protect_closure(ids);
        return bool(lease);
    }
    template<class Copy> Result tick(DiskKVBridge& bridge, Copy&& copy) {
        if (cancelled || failed) return Result::Failed;
        if (cursor == entries.size()) return Result::Ready;
        const auto& e = entries[cursor];
        if (read) {
            auto r = DiskKVBridge::poll(read);
            if (r == DiskReadResult::Pending) return Result::Pending;
            read.reset();
            if (r != DiskReadResult::Success) { failed = true; return Result::Failed; }
            verified_bytes += e.bytes;
            std::vector<std::byte>().swap(payload);
            ++cursor;
            written = false;
            return cursor == entries.size() ? Result::Ready : Result::Pending;
        }
        if (write) {
            const auto r = DiskKVBridge::poll(write);
            if (r == SpillStatus::Pending) return Result::Pending;
            write.reset();
            if (r != SpillStatus::Written && r != SpillStatus::Present) {
                failed = true; return Result::Failed;
            }
            written = true;
        }
        if (written) {
            if (payload.empty()) payload.resize(e.bytes);
            read = bridge.try_read_priority(e.id, e.kind, payload);
            return Result::Pending;
        }
        if (payload.empty()) {
            payload.resize(e.bytes);
            if (!copy(cursor, std::span<std::byte>(payload))) {
                failed = true; return Result::Failed;
            }
        }
        write = bridge.try_submit_owned_priority(e.id, e.kind, payload);
        return Result::Pending;
    }
    // A read borrows our own buffer, never runtime state. Cancellation gives
    // admission priority immediately; drain only this CPU buffer asynchronously.
    bool drained() const noexcept {
        return (!write || DiskKVBridge::poll(write) != SpillStatus::Pending) &&
               (!read || DiskKVBridge::poll(read) != DiskReadResult::Pending);
    }
private:
    std::unique_ptr<DiskKVBridge::ClosureLease> lease;
    std::vector<std::byte> payload;
    SpillTicket write;
    ReadTicket read;
    bool written = false;
};

struct ProactiveWatermarks {
    bool active = false;
    // The maintenance must pre-spill while the pools still have HEADROOM. At the
    // old 0.90 start the on-demand path had already paid the eviction on the
    // admission path: measured 65.7s of D2H+write for one never-spilled 93K
    // session, surfacing as a 128s request, while the NEXT eviction of the same
    // session cost 5s (96% already on disk). Start while one session's worth of
    // slack remains; stop with two. Overridable for threshold sweeps.
    long double start = .80L, stop = .65L;
    ProactiveWatermarks() {
        if (const char* s = std::getenv("NINFER_PROACTIVE_START")) { const auto v = std::strtold(s, nullptr); if (v > 0 && v <= 1) start = v; }
        if (const char* s = std::getenv("NINFER_PROACTIVE_STOP")) { const auto v = std::strtold(s, nullptr); if (v > 0 && v <= 1) stop = v; }
        if (stop > start) stop = start;
    }
    bool update(std::span<const std::pair<std::uint64_t,std::uint64_t>> usage) {
        bool high = false, low = true;
        for (auto [used, capacity] : usage) if (capacity) {
            const auto ratio = static_cast<long double>(used) / capacity;
            high |= ratio >= start;
            low &= ratio <= stop;
        }
        if (!active && high) active = true;
        else if (active && low) active = false;
        return active;
    }
};
} // namespace ninfer
