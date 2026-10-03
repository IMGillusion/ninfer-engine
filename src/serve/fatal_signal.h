#pragma once
// Fatal-signal attribution for the serving process. Production engines die
// of SIGSEGV/SIGBUS with no diagnostic when memory is corrupted (observed:
// a GPF inside libc from a corrupted stack while the docker stdout stream
// was already dead — nothing recorded anywhere, restart policy wiped the
// evidence). This handler writes one attributed line — signal, fault
// address, thread id, best-effort backtrace — to the request-log file
// (O_APPEND write(2): survives a dead stdout pipe) and to stderr, then
// restores the default disposition and re-raises so the container's restart
// policy and exit status keep their original semantics.
//
// Handler constraints: async-signal-safe calls only (write/open/close/
// backtrace_symbols_fd — the glibc fd variant does not allocate); a re-entry
// or nested fault exits immediately instead of recursing.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>

#include <execinfo.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace ninfer {
namespace serve {
namespace fatal {

inline std::atomic<bool> reentered{false};
inline const char* log_path = nullptr;  // set once before any signal can arrive

inline void write_all(int fd, const char* data, std::size_t n) {
    while (n > 0) {
        const ssize_t w = ::write(fd, data, n);
        if (w <= 0) { return; }
        data += w;
        n -= static_cast<std::size_t>(w);
    }
}

inline void handler(int sig, siginfo_t* info, void* /*context*/) {
    if (reentered.exchange(true)) { _exit(70); }
    char buf[192];
    const int n = std::snprintf(
        buf, sizeof buf,
        "{\"event\":\"fatal_signal\",\"signal\":%d,\"signame\":\"%s\","
        "\"addr\":\"%p\",\"tid\":%ld}\n",
        sig,
        sig == SIGSEGV ? "SIGSEGV" : sig == SIGBUS  ? "SIGBUS"
        : sig == SIGABRT ? "SIGABRT"
                         : sig == SIGFPE ? "SIGFPE" : sig == SIGILL ? "SIGILL" : "?",
        info != nullptr ? info->si_addr : nullptr, static_cast<long>(::gettid()));
    if (n > 0) { write_all(STDERR_FILENO, buf, static_cast<std::size_t>(n)); }
    int fd = -1;
    if (log_path != nullptr) {
        fd = ::open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    }
    if (fd >= 0 && n > 0) { write_all(fd, buf, static_cast<std::size_t>(n)); }
    // Best-effort stack: the glibc fd variants do not allocate, so this stays
    // usable on a corrupted heap. A nested fault re-enters this handler once
    // and exits through the guard above.
    void* frames[32];
    const int depth = ::backtrace(frames, 32);
    if (depth > 0) {
        if (fd >= 0) { ::backtrace_symbols_fd(frames, depth, fd); }
        ::backtrace_symbols_fd(frames, depth, STDERR_FILENO);
    }
    if (fd >= 0) { ::close(fd); }
    ::signal(sig, SIG_DFL);
    ::raise(sig);
    _exit(70);  // unreachable when raise() kills the process
}

inline void install(const char* request_log_path) {
    log_path = request_log_path;
    struct sigaction sa {};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    ::sigaction(SIGSEGV, &sa, nullptr);
    ::sigaction(SIGBUS, &sa, nullptr);
    ::sigaction(SIGFPE, &sa, nullptr);
    ::sigaction(SIGILL, &sa, nullptr);
    ::sigaction(SIGABRT, &sa, nullptr);
}

} // namespace fatal
} // namespace serve
} // namespace ninfer
