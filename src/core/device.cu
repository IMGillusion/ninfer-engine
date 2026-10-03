#include "core/device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    // Report and exit WITHOUT stdio or abort(): both production crashes
    // (SIGSEGV inside libc, GPF at warn+0x88 via the abort@plt call) died
    // inside this error-report path itself — each was the first CUDA error
    // of the process life, and the message was lost to a frozen stdout
    // pipe. Raw write(2) only: the record lands in the request log (path
    // parsed from our own --request-log-jsonl cmdline argument, the same
    // file the fatal-signal handler uses) and on stderr non-blocking (a
    // dead stdout pipe must not wedge the exit), then _Exit(70) — no
    // stream flush, no raise, no error-path machinery — so the restart
    // policy takes over with the evidence saved.
    char message[512];
    const int n = std::snprintf(
        message, sizeof message,
        "{\"event\":\"cuda_error\",\"file\":\"%s\",\"line\":%d,"
        "\"expr\":\"%s\",\"code\":%d,\"name\":\"%s\",\"detail\":\"%s\"}\n",
        file, line, expr, static_cast<int>(err), cudaGetErrorName(err),
        cudaGetErrorString(err));
    char cmdline[4096] = {};
    int log_fd         = -1;
    {
        const int fd = ::open("/proc/self/cmdline", O_RDONLY);
        if (fd >= 0) {
            const ssize_t got = ::read(fd, cmdline, sizeof cmdline - 1);
            ::close(fd);
            for (ssize_t i = 0; i + 1 < got; ++i) {
                if (cmdline[i] == '\0' &&
                    std::strcmp(&cmdline[i + 1], "--request-log-jsonl") == 0) {
                    log_fd = ::open(cmdline + i + 20, O_WRONLY | O_CREAT | O_APPEND, 0644);
                    break;
                }
            }
        }
    }
    if (log_fd >= 0) {
        std::size_t written = 0;
        while (n > 0 && written < static_cast<std::size_t>(n)) {
            const ssize_t w = ::write(log_fd, message + written,
                                      static_cast<std::size_t>(n) - written);
            if (w <= 0) { break; }
            written += static_cast<std::size_t>(w);
        }
        ::close(log_fd);
    }
    // Best-effort stderr, non-blocking so a dead stdout pipe cannot stall the exit.
    const int stderr_flags = ::fcntl(STDERR_FILENO, F_GETFL, 0);
    if (stderr_flags >= 0) { ::fcntl(STDERR_FILENO, F_SETFL, stderr_flags | O_NONBLOCK); }
    if (n > 0) { (void)::write(STDERR_FILENO, message, static_cast<std::size_t>(n)); }
    std::_Exit(70);  // restart required: device state can no longer be trusted
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}

} // namespace ninfer
