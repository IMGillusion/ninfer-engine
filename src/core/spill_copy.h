#pragma once
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace ninfer {
// Shared by real CUDA plane loops and fault-injection tests. Never release
// source/destination ownership until finish() has proved the stream drained.
template<class Backend> class SpillCopy {
    Backend backend_;
    int submit_error_ = 0;
    bool finished_ = false;
public:
    explicit SpillCopy(Backend backend) : backend_(std::move(backend)) {}
    ~SpillCopy() { if (!finished_) (void)finish(); }
    template<class F> void submit(F&& operation) {
        if (!submit_error_) submit_error_ = backend_.submit(std::forward<F>(operation));
    }
    bool finish() {
        const int drain_error = backend_.drain();
        if (drain_error || (submit_error_ && !backend_.recoverable(submit_error_))) {
            // Must not unwind: CUDA may still own pinned memory. Production
            // backend exits without destructors; supervisor must recreate CUDA.
            backend_.fatal(submit_error_, drain_error);
            std::abort(); // backend contract: fatal never returns
        }
        finished_ = true;
        return submit_error_ == 0;
    }
};
}
