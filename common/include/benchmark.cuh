#pragma once
#include "utils.cuh"

namespace FlashLab::Benchmark {

    template <typename T>
    void benchmark_kernel (
        std::function<T(cudaStream_t)> bound_function, 
        cudaStream_t stream, 
        size_t bytes_moved,
        float cpu_ref_time, 
        size_t num_repeats, 
        size_t num_warmups, 
        bool flush_l2_cache);

}