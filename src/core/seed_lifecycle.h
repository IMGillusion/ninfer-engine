#pragma once
#include "core/disk_kv_bridge.h"
#include <stdexcept>

namespace ninfer {
// Scheduler-owned publication guard. Reads borrow startup scratch; cancellation
// is sticky and never releases it before the worker's terminal acknowledgement.
class SeedLifecycle {
public:
    enum class Phase { Empty, Preparing, Prepared, Finalized, Aborted };
    void begin_prepare() {
        if (phase_ != Phase::Empty) { throw std::logic_error("sequence prepared twice"); }
        phase_ = Phase::Preparing;
    }
    void prepared() {
        if (phase_ != Phase::Preparing) { throw std::logic_error("invalid seed prepare completion"); }
        phase_ = Phase::Prepared;
    }
    bool is_prepared() const noexcept { return phase_ == Phase::Prepared; }
    bool cancel_and_drained(const ReadTicket& read) noexcept {
        cancelled_ = true;
        return !read || DiskKVBridge::poll(read) != DiskReadResult::Pending;
    }
    void finalize() {
        if (phase_ != Phase::Prepared || cancelled_) {
            throw std::logic_error("invalid seed final publication");
        }
        phase_ = Phase::Finalized;
    }
    void abort(const ReadTicket& read) {
        if (!cancel_and_drained(read)) { throw std::logic_error("seed abort before read drain"); }
        if (phase_ == Phase::Finalized) { throw std::logic_error("seed abort after publication"); }
        phase_ = Phase::Aborted;
    }
    Phase phase() const noexcept { return phase_; }
private:
    Phase phase_ = Phase::Empty;
    bool cancelled_ = false;
};
}
