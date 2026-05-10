# CUDA Experiments
I hope to shape this repo into a collection of projects that helped me learn CUDA.

## CUDA Fundamentals
With CUDA Fundamentals, I hope to explore the fundamentals of CUDA programming. I plan to implement basic kernels and slowly ramp it up and see where it goes.

Kernels implemented so far:
- [x] Reduction Algorithm: from PMPP
- [ ] Reduction Algorithm: Warp Shuffling
- [ ] build.sh file to run launch command (later)

#### RUN COMMANDS
```
nvcc -I../common main.cu reduction.cu ../common/utils.cu -o test_run
```

#### To-Do:

* Implement matrix row reduction
* Explore alternatives for reductions including warp shuffling

### Reduction Algorithm

Reduction algorithm's are used to extract a single value from an array. Generally speaking, with proper index mapping, it can be also applied to matrices and then a rowSum or a rowMax can be achieved. However, the idea after it is to utilize the most threads within a warp to reduce **warp divergence**. That is, a warp (32 threads) following the same control path is much more favorable then some of it executing one part and the rest of it another part (seen most commonly in if-else control statements). Warp divergence can in turn increases the overhead time.

Here is the algorithm with a small size illustration I made:

![image](images/reductionAlgo.png)

I hope that this makes everything obvious and if not:

Stride is a global offset index and the goal is to enable every thread to contribute in the partial sum. Every thread adds two elements and so at every loop the size is halved, thus the necessary amount of thread will be halved too. The number of threads that will work to compute d elements is d/2, which will be equal to stride at every point in the loop. To ensure that, we introduce an if statement (tid < stride) inside the loop and after each write, we do a __syncthreads() call to ensure all threads finish their sum before looking into each others values. Otherwise a race condition occurs. By the way, one thing I missed while making this illustration is that you can actually launch half the array size of threads since in the first loop already half of the threads are not being used. Or even better, you can do the first reduction while loading it from GMEM to SMEM. Say the array size is 64, then while uploading to SMEM, let each thread fetch 2 data with a stride of 32 and load to SMEM by adding those two. Then the SMEM size will be 32, much smaller, and the 32 threads than can finish up the reduction with the above algorithm.
