#pragma once

#include <ninfer/targets/qwen3_6/runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::runtime_support {

// Checkpoint requirements, not historical execution progress or available pages,
// define the chain to preserve. Never clamp this to mapped/committed capacity:
// a hole inside this requirement must remain a spill failure.
inline TargetKVRequirement owner_spill_requirement(const ContinuationSummary& summary) {
    TargetKVRequirement result;
    const auto include = [&](const CheckpointSummary& checkpoint) {
        result.main_frontier = std::max(result.main_frontier, checkpoint.required_kv.main_frontier);
        result.backend_frontier = std::max(result.backend_frontier, checkpoint.required_kv.backend_frontier);
        result.main_pages = std::max(result.main_pages, checkpoint.required_kv.main_pages);
        result.backend_pages = std::max(result.backend_pages, checkpoint.required_kv.backend_pages);
    };
    if (summary.endpoint) { include(*summary.endpoint); }
    if (summary.rewrite) { include(*summary.rewrite); }
    for (const auto& anchor : summary.long_anchors) { include(anchor); }
    return result;
}

inline bool owner_spill_page_mapped(std::uint32_t page, std::uint32_t mapped) noexcept {
    return page < mapped;
}

inline std::uint32_t owner_spill_required_columns(std::uint32_t frontier,
                                                  std::uint32_t page,
                                                  std::uint32_t page_size) noexcept {
    return std::min(page_size, frontier - page * page_size);
}

inline bool owner_spill_page_committed(std::uint32_t required,
                                       std::uint32_t committed) noexcept {
    return committed >= required;
}

// PrefixShortlistDigests::size() counts tokens; at(size()) is the final digest.
// Spill stages exclude frontier zero before reaching this check.
inline bool owner_spill_digest_available(std::uint32_t frontier, std::size_t tokens) noexcept {
    return frontier <= tokens;
}

} // namespace ninfer::targets::qwen3_6::runtime_support
