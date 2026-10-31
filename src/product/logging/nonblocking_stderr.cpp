#include "product/logging/nonblocking_stderr.h"

#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <thread>

namespace ninfer::product {
namespace {

std::atomic<bool> g_installed{false};

void drain_loop(int pipe_fd, int real_fd) {
    std::array<std::byte, 8192> buffer{};
    for (;;) {
        const ssize_t n = ::read(pipe_fd, buffer.data(), buffer.size());
        if (n > 0) {
            // The real stderr may back up when the log consumer dies; the
            // drain thread may then block or the write may fail - either is
            // contained here. Business threads only see the non-blocking
            // pipe side and can never wedge on a log write.
            std::size_t off = 0;
            while (off < static_cast<std::size_t>(n)) {
                const ssize_t w = ::write(real_fd, buffer.data() + off,
                                          static_cast<std::size_t>(n) - off);
                if (w > 0) {
                    off += static_cast<std::size_t>(w);
                    continue;
                }
                if (w < 0 && errno == EINTR) { continue; }
                if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                    ::pollfd p{real_fd, POLLOUT, 0};
                    ::poll(&p, 1, 200);
                    continue;
                }
                break;   // EPIPE / EBADF: the consumer is gone; drop the rest.
            }
            continue;
        }
        if (n == 0) { return; }   // write end closed: done.
        if (errno == EINTR) { continue; }
        if (errno != EAGAIN && errno != EWOULDBLOCK) { return; }
        ::pollfd p{pipe_fd, POLLIN, 0};
        ::poll(&p, 1, 200);
    }
}

} // namespace

bool install_nonblocking_stderr() {
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) { return true; }
    int fds[2] = {-1, -1};
    if (::pipe(fds) != 0) { g_installed.store(false); return false; }
    // Stable copy of the original stderr for the drain thread. Source fd
    // first, then the command, then the floor.
    const int real_fd = ::fcntl(2, F_DUPFD_CLOEXEC, 10);
    if (real_fd < 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        g_installed.store(false);
        return false;
    }
    if (::dup2(fds[1], 2) != 2) {
        ::close(real_fd);
        ::close(fds[0]);
        ::close(fds[1]);
        g_installed.store(false);
        return false;
    }
    ::close(fds[1]);
    // The business-facing write side MUST be non-blocking: a full pipe (log
    // consumer dead) turns write(2) into a permanent block that wedges the
    // caller with whatever lock it holds - the observed 5-hour engine
    // deadlock. With O_NONBLOCK a full pipe just drops the line.
    const int out_flags = ::fcntl(2, F_GETFL);
    if (out_flags >= 0) { (void)::fcntl(2, F_SETFL, out_flags | O_NONBLOCK); }
    // Drain side reads through poll; non-blocking is harmless either way.
    const int in_flags = ::fcntl(fds[0], F_GETFL);
    if (in_flags >= 0) { (void)::fcntl(fds[0], F_SETFL, in_flags | O_NONBLOCK); }
    std::thread(drain_loop, fds[0], real_fd).detach();
    return true;
}

} // namespace ninfer::product
