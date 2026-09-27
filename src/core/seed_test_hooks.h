#pragma once
// Compile-time opt-in. No environment variable, allocation or callback in production.
#ifdef NINFER_SEED_TEST_HOOKS
#include <cstddef>
#include <cstdint>
namespace ninfer::seed_test {
inline void (*stage_hook)(const char*) = nullptr;
inline void (*snapshot_hook)(std::uint64_t, std::uint64_t, std::uint32_t,
                             std::uint32_t, const std::byte*, std::size_t) = nullptr;
inline void stage(const char* name) { if (stage_hook) stage_hook(name); }
inline void snapshot(std::uint64_t lo, std::uint64_t hi, std::uint32_t tag,
                     std::uint32_t frontier, const std::byte* data, std::size_t bytes) {
    if (snapshot_hook) snapshot_hook(lo, hi, tag, frontier, data, bytes);
}
}
#define NINFER_SEED_STAGE(name) ::ninfer::seed_test::stage(name)
#define NINFER_SEED_SNAPSHOT(...) ::ninfer::seed_test::snapshot(__VA_ARGS__)
#else
#define NINFER_SEED_STAGE(name) ((void)0)
#define NINFER_SEED_SNAPSHOT(...) ((void)0)
#endif
