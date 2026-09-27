#pragma once
#include "core/disk_kv_bridge.h"
#include <array>
#include <cstdlib>
#include <cstring>

namespace ninfer {
// Engine-thread-only bounded scheduling. Worker receives owned bytes, never an owner.
template<class Candidate, std::size_t Capacity = 4096>
class HostKVPrewrite {
public:
    static constexpr std::size_t byte_limit = 8 * 1024 * 1024;
    static constexpr auto time_limit = std::chrono::microseconds(250);
    struct Stats { std::uint64_t registered=0, full=0, invalid=0, busy=0, written=0, failed=0; } stats;
    static bool configured() noexcept {
        const char* v = std::getenv("NINFER_HOST_KV_PREWRITE");
        return v && std::strcmp(v, "1") == 0;
    }
    bool enabled = configured();
    bool add(const Candidate& c) noexcept {
        if (!enabled) return false;
        if (count_ == Capacity) { ++stats.full; return false; }
        queue_[(head_ + count_) % Capacity] = c; ++count_; ++stats.registered; return true;
    }
    std::size_t pending() const noexcept { return count_; }
    bool has_work() const noexcept { return enabled && (count_ || ticket_); }
    std::size_t buffered() const noexcept { return bytes_.size() + in_flight_bytes_; }
    // validate must check owner generation, row membership, page epoch and digest.
    // copy executes a short pin/memcpy/unpin on this thread; no CUDA or disk I/O.
    template<class Bridge, class Validate, class Copy>
    void tick(Bridge& bridge, Validate validate, Copy copy) noexcept {
        if (!enabled) return;
        const auto start = std::chrono::steady_clock::now();
        try {
            if (ticket_) {
                const auto status = DiskKVBridge::poll(ticket_);
                if (status == SpillStatus::Pending) return;
                if (status == SpillStatus::Written || status == SpillStatus::Present) ++stats.written;
                else ++stats.failed;
                ticket_.reset(); in_flight_bytes_ = 0;
            }
            // One owned page in flight, one copy maximum per tick. Invalid scanning is bounded.
            for (std::size_t inspected=0; count_ && inspected<32; ++inspected) {
                if (std::chrono::steady_clock::now()-start >= time_limit) return;
                auto& c=queue_[head_];
                if (!validate(c)) { ++stats.invalid; pop(); continue; }
                if (bytes_.empty()) {
                    if (!copy(c, id_, kind_, bytes_) || bytes_.empty() || bytes_.size()>byte_limit) {
                        ++stats.invalid; pop(); return;
                    }
                }
                // A memcpy is non-preemptible; if it consumed the quantum, submit
                // next boundary after revalidation rather than extending this tick.
                if (std::chrono::steady_clock::now()-start >= time_limit) return;
                const auto size=bytes_.size();
                ticket_=bridge.try_submit_owned(id_,kind_,bytes_);
                if (!ticket_) { ++stats.busy; return; }
                in_flight_bytes_=size; pop(); return;
            }
        } catch (...) { ++stats.failed; if (count_) pop(); }
    }
private:
    void pop() noexcept { bytes_.clear(); head_=(head_+1)%Capacity; --count_; }
    std::array<Candidate,Capacity> queue_{};
    std::size_t head_=0,count_=0,in_flight_bytes_=0;
    std::vector<std::byte> bytes_;
    DiskKVIdentity id_{};
    DiskKVKind kind_=DiskKVKind::MainKV;
    SpillTicket ticket_;
};
} // namespace ninfer
