#pragma once
#include "utils.cuh"
#include <iostream>    // For std::cerr, std::endl
#include <functional>  // For std::function and std::bind

template <typename T>
void benchmark_kernel (std::function<T(cudaStream_t)> bound_function, cudaStream_t stream, size_t num_repeats = 100, size_t num_warmups = 100, float cpu_ref_time, bool flush_l2_cache) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int device = 0;
    int l2_size = 0;
    float *d_F;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
    size_t sizeF = l2_size * 2;

    // ALLOCATE
    CUDA_CHECK(cudaMalloc(&d_F, sizeF));

    // WARMUP
    for (size_t i{0}; i < num_warmups; ++i) {
        bound_function(stream);
    }

    // EXECUTION LOOP
    float time = 0.0f;
    float partial_time = 0.0f;

    for (size_t i = 0; i < num_repeats; ++i) {
        if (flush_l2_cache) {
            CUDA_CHECK(cudaMemsetAsync(d_F, 0, sizeF, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CHECK_LAST_CUDA_ERROR();
        }
        CUDA_CHECK(cudaEventRecord(start, stream));
        bound_function(stream)
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&partial_time, start, stop));
        time += partial_time
    }

    // CALC AVG_TIME
    float avg_ms = time / iterations;

    // BANDWIDTH CALCULATION
    // Formula: Bytes moved = (Read N * d + write N) * 4 Bytes, time = avg_ms
    double gb = (double)(N * d + N) * sizeof(float) / 1e9;
    double bandwidth = gb / (avg_ms / 1000.0);

    // PRINT RESULT
    std::printf("-- Benchmark Result --\n");
    std::printf("Average Time:  %.4f ms\n", avg_ms);
    std::printf("Throughput:    %.2f GB/s\n", bandwidth);
    std::printf("Speedup from CPU:  %.2fx\n", cpu_ref_time / avg_ms);
    std::cout << "Flush: " << std::boolalpha << flush_l2_cache << std::endl;

    // FREE FLUSH MEMORY
    CUDA_CHECK(cudaFree(d_F));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}