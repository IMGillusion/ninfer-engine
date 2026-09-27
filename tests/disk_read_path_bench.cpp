// Cold-read path comparison on the PRODUCTION store file: mmap + memcpy (what the
// store does today) versus pread() into a buffer, each at several concurrency
// levels. The store's mmap fault path measured ~420MB/s cold while four direct
// read() streams reached ~883MB/s, so the fix - if any - is which syscall path
// the payload travels.
//
// Usage: path-bench <store-file> <region-MB-per-stream> <threads...>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;

static double run_mmap(const std::string& path, std::size_t bytes_per_thread, int threads) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("open"); std::exit(2); }
    struct stat st{};
    ::fstat(fd, &st);
    std::byte* base = static_cast<std::byte*>(::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
    if (base == MAP_FAILED) { std::perror("mmap"); std::exit(2); }
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        std::vector<std::byte> sink(1 << 20);
        for (;;) {
            const std::size_t id = next.fetch_add(1);
            const std::size_t begin = id * bytes_per_thread;
            if (begin + bytes_per_thread > static_cast<std::size_t>(st.st_size)) { return; }
            for (std::size_t off = 0; off < bytes_per_thread; off += sink.size()) {
                std::memcpy(sink.data(), base + begin + off,
                            std::min(sink.size(), bytes_per_thread - off));
            }
        }
    };
    const auto t0 = Clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) { pool.emplace_back(worker); }
    for (auto& t : pool) { t.join(); }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    ::munmap(base, st.st_size);
    ::close(fd);
    const double mb = static_cast<double>(bytes_per_thread) * threads / 1048576.0;
    return mb / s;
}

static double run_pread(const std::string& path, std::size_t bytes_per_thread, int threads, bool direct) {
    const int fd = ::open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) { std::perror("open"); std::exit(2); }
    struct stat st{};
    ::fstat(fd, &st);
    const std::size_t block = 1 << 20;
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        void* buf = ::aligned_alloc(4096, block);   // O_DIRECT needs aligned I/O
        for (;;) {
            const std::size_t id = next.fetch_add(1);
            const std::size_t begin = id * bytes_per_thread;
            if (begin + bytes_per_thread > static_cast<std::size_t>(st.st_size)) { return; }
            for (std::size_t off = 0; off < bytes_per_thread; off += block) {
                const ssize_t n = ::pread(fd, buf, std::min(block, bytes_per_thread - off),
                                          static_cast<off_t>(begin + off));
                if (n <= 0) { return; }
            }
        }
    };
    const auto t0 = Clock::now();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) { pool.emplace_back(worker); }
    for (auto& t : pool) { t.join(); }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    ::close(fd);
    const double mb = static_cast<double>(bytes_per_thread) * threads / 1048576.0;
    return mb / s;
}

int main(int argc, char** argv) {
    const std::string path = argv[1];
    const std::size_t per = static_cast<std::size_t>(std::stoul(argv[2])) * 1024 * 1024;
    // One mode per invocation so the caller can drop the page cache in between:
    // a combined run warms the region for the later modes and compares nothing.
    const std::string mode = argv[3];
    std::vector<int> threads;
    for (int i = 4; i < argc; ++i) { threads.push_back(std::stoi(argv[i])); }
    for (int t : threads) {
        const double mb_s = mode == "mmap"   ? run_mmap(path, per, t)
                          : mode == "direct" ? run_pread(path, per, t, true)
                                             : run_pread(path, per, t, false);
        std::printf("%-6s threads=%-2d  %7.0f MB/s\n", mode.c_str(), t, mb_s);
        std::fflush(stdout);
    }
    return 0;
}
