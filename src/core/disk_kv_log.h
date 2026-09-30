#pragma once
// Lightweight logging for the disk-KV (L3) subsystem.
//
// src/core must not depend on upper layers, so this header is self-contained
// and exposes a settable sink for the upper stack to wire its own logger in.
// The DEFAULT renderer already matches the engine's visual log format
// (PrettyLogFormatter: "YYYY-MM-DD HH:MM:SS.mmm LEVEL message", local/Beijing
// wall time), so an unwired sink still produces engine-style lines:
//
//   2026-09-27 14:03:22.123 INFO  [l3-spill] owner ok | id=3 | ...
//
// Severity travels as a char: 'I' info, 'W' warn, 'E' error, 'D' trace/debug
// (trace-class records are additionally gated by their NINFER_*_TRACE env
// switches at the call sites and stay silent in production).
//
// Message convention: one line per event, pipe-separated fields
// ("req#N done | field | field" style), no trailing newline, no timestamp in
// the message itself (the renderer owns it).

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <functional>
#include <string>
#include <sys/time.h>

namespace ninfer {

using DiskKvLogSink =
    std::function<void(char severity, const char* component, const std::string& msg)>;

inline DiskKvLogSink& disk_kv_log_sink() {
    static DiskKvLogSink sink;
    return sink;
}

// Upper layers (serve/startup) may install a sink to route records into the
// engine's own logger. Passing an empty function restores the default.
inline void set_disk_kv_log_sink(DiskKvLogSink sink) {
    disk_kv_log_sink() = std::move(sink);
}

inline const char* disk_kv_log_level_name(char severity) noexcept {
    switch (severity) {
    case 'E': return "ERROR";
    case 'W': return "WARN ";
    case 'D': return "DEBUG";
    default:  return "INFO ";
    }
}

// Default renderer: engine-style "timestamp LEVEL [component] msg" on stderr.
// The sink, when installed, fully replaces this rendering.
inline void disk_kv_log(char severity, const char* component, const std::string& msg) {
    if (auto& sink = disk_kv_log_sink(); sink) {
        sink(severity, component, msg);
        return;
    }
    ::timeval tv{};
    ::gettimeofday(&tv, nullptr);
    std::time_t wall = static_cast<std::time_t>(tv.tv_sec);
    std::tm local{};
    ::localtime_r(&wall, &local);
    std::fprintf(stderr, "%04d-%02d-%02d %02d:%02d:%02d.%03d %s [%s] %s\n",
                 local.tm_year + 1900, local.tm_mon + 1, local.tm_mday,
                 local.tm_hour, local.tm_min, local.tm_sec,
                 static_cast<int>(tv.tv_usec / 1000),
                 disk_kv_log_level_name(severity), component, msg.c_str());
    std::fflush(stderr);
}

// printf-style convenience wrapper so call sites keep their fprintf format
// strings verbatim (minus the "[l3-xxx] " prefix and trailing newline).
inline void disk_kv_logf(char severity, const char* component, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    const int n = std::vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    std::string msg(n > 0 ? static_cast<std::size_t>(n) : 0U, '\0');
    if (n > 0) {
        std::vsnprintf(msg.data(), static_cast<std::size_t>(n) + 1U, fmt, copy);
    }
    va_end(copy);
    disk_kv_log(severity, component, msg);
}

} // namespace ninfer
