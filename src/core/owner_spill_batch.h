#pragma once
#include "core/disk_kv_bridge.h"
#include <functional>
#include <stdexcept>

namespace ninfer {
// Scheduler-owned batch. Bytes remain borrowed until every accepted ticket is
// terminal, including failure/cancellation; submission and acknowledgement are
// separate monotonic cursors. No callback touches bytes on a Busy retry.
struct OwnerSpillBatch {
    std::vector<DiskKVIdentity> batch_ids;
    std::vector<SpillTicket> batch_tickets;
    std::vector<std::size_t> missing;
    std::vector<std::span<const std::byte>> payloads;
    ProbeBatchTicket probe;
    std::uint32_t batch_submit = 0;
    std::size_t acknowledged = 0;
    DiskKVKind batch_kind = DiskKVKind::MainKV;
    std::size_t batch_stride = 0;
    bool batch_copy_pending = false; // prepared bytes; not a CUDA operation
    bool classified = false;
    bool failed = false;

    bool pending() const {
        if (probe && !DiskKVBridge::poll(probe)) return true;
        for (const auto& t : batch_tickets)
            if (DiskKVBridge::poll(t) == SpillStatus::Pending) return true;
        return false;
    }
    enum class Result { Pending, Complete, Failed };
    template<class Bridge, class Prepare, class Ack>
    Result progress(Bridge& bridge, Prepare&& prepare, Ack&& ack) {
        if (failed) return Result::Failed;
        if (!probe) {
            probe = bridge.try_probe_batch(batch_ids, batch_kind);
            if (!probe) return Result::Pending;
        }
        if (!DiskKVBridge::poll(probe)) return Result::Pending;
        if (!classified) {
            if (probe->size != batch_ids.size()) { failed = true; return Result::Failed; }
            for (std::size_t i = 0; i < batch_ids.size(); ++i) {
                const auto status = probe->results[i];
                if (status == SpillStatus::Present) ack(status);
                else if (status == SpillStatus::Missing) missing.push_back(i);
                else failed = true;
            }
            classified = true;
            if (failed) return Result::Failed;
        }
        if (!batch_copy_pending) {
            // prepare must be retry-safe until it returns true.
            if (!prepare(missing, payloads)) return Result::Pending;
            if (payloads.size() != missing.size()) throw std::logic_error("spill payload count");
            batch_copy_pending = true;
        }
        while (acknowledged < batch_tickets.size()) {
            const auto status = DiskKVBridge::poll(batch_tickets[acknowledged]);
            if (status == SpillStatus::Pending) break;
            ++acknowledged;
            if (status == SpillStatus::Failed) { failed = true; return Result::Failed; }
            ack(status);
        }
        // Reserve before accepting any borrowed write; push_back must not
        // throw after the bridge has acquired a borrow.
        batch_tickets.reserve(missing.size());
        while (batch_submit < missing.size()) {
            auto t = bridge.try_submit(batch_ids[missing[batch_submit]], batch_kind,
                                       payloads[batch_submit]);
            if (!t) return Result::Pending;
            batch_tickets.push_back(std::move(t));
            ++batch_submit;
        }
        return acknowledged == missing.size() ? Result::Complete : Result::Pending;
    }
    void reset() {
        if (pending()) throw std::logic_error("spill batch reset while borrowed");
        *this = OwnerSpillBatch{};
    }
};
} // namespace ninfer
