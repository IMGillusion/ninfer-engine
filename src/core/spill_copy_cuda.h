#pragma once
#include "core/disk_kv_log.h"
#include "core/spill_copy.h"
#include <cuda_runtime.h>
namespace ninfer {
struct SpillCudaBackend {
    cudaStream_t stream;
    template<class F> int submit(F&& f) const { return static_cast<int>(f()); }
    int drain() const { return static_cast<int>(cudaStreamSynchronize(stream)); }
    bool recoverable(int error) const {
        // Narrow allowlist: unknown/sticky/context errors require process restart.
        return error == cudaErrorInvalidValue || error == cudaErrorMemoryAllocation;
    }
    [[noreturn]] void fatal(int submit_error, int drain_error) const {
        disk_kv_logf('E', "l3-spill-cuda",
                     "fatal CUDA failure | submit=%d (%s) drain=%d (%s) | inference stopped, restart process required",
                     submit_error, cudaGetErrorString(static_cast<cudaError_t>(submit_error)),
                     drain_error, cudaGetErrorString(static_cast<cudaError_t>(drain_error)));
        std::_Exit(70); // no unwinding/freeing memory still potentially owned by CUDA
    }
};
}
