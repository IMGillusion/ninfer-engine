#pragma once
#include "core/state_scratch_lease.h"
#include <ninfer/targets/qwen3_6/state_image.h>
#include <optional>

namespace ninfer::targets::qwen3_6::runtime_support {
// Checked D2H into the startup reservation. No pool allocation on this path.
// Caller retains lease through synchronous consumption or asynchronous ticket ack.
inline std::optional<HostStateImageConstView> spill_state_to_scratch(
    StateImageDevicePool& device, std::int32_t source,
    HostStatePool* host, std::optional<HostStateSlotHandle> scratch,
    StateScratchLease& gate, StateScratchLease::Lease& lease, cudaStream_t stream) {
    if (!host || !scratch) { return std::nullopt; }
    if (!lease) {
        lease = gate.try_acquire();
        if (!lease) { return std::nullopt; }
    }
    if (!device.spill_copy_to_host(source, host->writable_view(*scratch), stream)) {
        return std::nullopt;
    }
    return host->view(*scratch);
}
} // namespace ninfer::targets::qwen3_6::runtime_support
