# CUDA SOTA IMPLEMENTATION PRACTICE

A "no-tutorial" journey into implementing State-of-the-Art AI kernels in CUDA.

## The Mission

To learn by doing. Starting from basic kernels and slowly ramping it up, towards kernels like FlashAttn. The current target is **FlashAttention**, implemented from scratch.

## Implementation Roadmap

Kernels implemented so far:
- [x] Reduction Algorithm: from PMPP
- [x] Reduction Algorithm: Warp Shuffling (xor and down)
- [x] rowSum using __shfl_xor_sync()
- [x] rowMax using __shfl_xor_sync()
- [ ] Online Softmax

## Rules

- [ ] Full `CUDA_CHECK` error handling on every API call.
- [ ] Proper C++ Namespacing and Header/Source separation.
- [ ] `build.sh` scripting for reproducible experiments.
- [ ] Benchmarking GB/s and % of Peak Bandwidth.

## Benchmarking

I have included a cold and hot start benchmarking function in common/benchmarking.cuh. For a cold start, L2 cache is flushed before every kernel launch and for a hot start, the kernel is launched, without L2 flush, iteratively and averaged over time to yield statistics. I don't think I will but maybe later on I can sample these kernel launches and do statistics on them... That may be for some time later.

## Worklog - Temporary

May 10, 2026 <br>
For now, I worked on organizing my github and getting comfortable with .cu and .cuh. I experimented with launch commands using flags and I think this workflow will keep everything much more clearer. Especially having a template is great since any local changes to the kernel doesn't affect the previous kernels I wrote. With this, I hope that I can create benchmarking functions that can work with multiple kernels :)

May 11, 2026<br>
Implemented shuffle intrinsics for warp-level reduction primitives. Did a small benchmark test. I will write a more generalized/encompassing benchmarking func. that I can just use to compare kernels more quickly in the future. Implemented XOR warp shuffling.

May 12, 2026<br>
Implemented rowSum using XOR warp shuffling that can handle matrices of all sizes. Organized my folder system. I'm still learning how the compiler communicates with .cu and .cuh files. Changed libraries from C to C++ and changed some function calls to start with std::. I'm still trying to figure out why C++ function calls are preffered and when they do not matter. Also experimented with template but I need a more solid example where it really makes a difference.

May 13, 2026<br>
Tried to implement a benchmark function that calculated bandwidth, speedup from cpu and average time by omitting warmup times and averaging over multiple iterations. Turns out, on silicon time is much much more smaller than the first multiple runs. Here is a small result of how matrix size affects my throughput.

To prevent my GPU from using L2 cache and cheat the benchmark, I decided to use matrix sizes bigger than 16k x 1024. (67 MB).

| Size | Time (ms) | Bandwidth (GB/s) | Bandwidth Utilization (%) |
| :--- | :--- | :--- | :--- |
| 16384 x 1024 | 0.2832 | 237.18 | 74.1 |
| 32768 x 1024 | 0.5628 | 238.70 | 75.6 |
| 65536 x 1024 | 1.1193 | 240.06 | 75.0 |

May 14, 2026<br>
Implemented Cache Flushing on benchmarking. Increasing matrix sizes more than what the L2 cache size allows yields a very similar effect as to L2 flushing and generally speaking the 'performance cheating' given by L2 cache is quite irrelevant in real life applications where matrix sizes, for example, exceed 16k x 16k. Thus I find it quite irrelevant. Yet I still need to research proper cuda error and stream handling and learn how to make a more encompassing benchmarking function that can handle all my kernels. I think it has something to do with templates. I also plan to implement more c++ and also experiment with namespaces in my functions.

May 15, 2026<br>
Experimented with std::function and std::bind to create a more Type-Agnostic Benchmarking Wrapper. This will be important later when I have multiple kernels with different arguments and different streams where benchmarking must be injected into a specific kernel and to a specific stream. Furhtermore, a single benchmark function compatible with all kernels is great for rapid prototyping, reducing the extra work to write kernel specific benchmark functions again and again. I still have some changes to make into the benchmark function such as modifying the pointers into void* for it to really be generic. Even though these concept are still a bit vauge to me, I find this practice of writing elegant and generic code to be really helpful in creating a more organized workflow and also paving the way for more kernels to be implemented efficiently. I'm satisfied with today :) 