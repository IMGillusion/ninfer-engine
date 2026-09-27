#pragma once
#include <atomic>
#include <utility>

namespace ninfer {
// One startup-reserved image shared by legacy spill, pressure spill and seed.
// Acquisition never waits: a scheduler callback may run while pressure owns it.
// The holder must retain its lease until CUDA and every borrowed ticket drain.
class StateScratchLease {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) { reset(); owner_ = std::exchange(other.owner_, nullptr); }
            return *this;
        }
        ~Lease() { reset(); }
        explicit operator bool() const noexcept { return owner_ != nullptr; }
        void reset() noexcept {
            if (auto* owner = std::exchange(owner_, nullptr)) {
                owner->busy_.clear(std::memory_order_release);
            }
        }
    private:
        friend class StateScratchLease;
        explicit Lease(StateScratchLease* owner) : owner_(owner) {}
        StateScratchLease* owner_ = nullptr;
    };
    Lease try_acquire() noexcept {
        if (busy_.test_and_set(std::memory_order_acquire)) { return {}; }
        return Lease(this);
    }
private:
    std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
};
} // namespace ninfer
