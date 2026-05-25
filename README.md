# **CUDA SOTA IMPLEMENTATION PRACTICE**

A "no-tutorial" journey into implementing State-of-the-Art AI kernels in CUDA.

## **The Mission**

To learn by doing. Starting from basic kernels and slowly ramping it up, towards SOTA papers. The current target is **FlashAttention**, implemented from scratch.

# 🛡️ **CURRENT GOAL**

**FlashAttn** implementation, starting from softmax. Follow along from ./src/ and ./research/ to see explanations and math behind each of the kernels I write.

## 🚀 **Implementation Roadmap**

### **Prerequisite Kernels**
- [x] Reduction Algorithm: from PMPP
- [x] Reduction Algorithm: Warp Shuffling (xor and down)
- [x] rowSum using __shfl_xor_sync()
- [x] rowMax using __shfl_xor_sync()
- [x] Online Softmax
### **Road to FlashAttn**
- [x] **FlashAttn Part 1**: QK_Matmul
- [ ] **FlashAttn Part 2**: S_Softmax
- [ ] **FlashAttn Part 3**: PV_Matmul
- [ ] **Final Part**: Fusing: Flash Attention!

## **(Repetitive) Launch commands**

**Clone a branch or the main. E.g.:**

```
git clone https://github.com/borymory/cuda-experiments.git
```

```
cd cuda-experiments
```

```
chmod +x scripts/build.sh
```

**Run using build.sh and specify path for the kernel. E.g.:**

```
./scripts/build.sh src/03_flashAttn/flashAttn_fundamentals
```

**Run kernel using**

```
./bin/test_run
```

## **Rules** 

- [x] 🔥 Full `CUDA_CHECK` error handling on every API call.
- [x] 🔥 Proper C++ Namespacing and Header/Source separation.
- [x] 🔥 `build.sh` scripting for reproducible experiments.
- [x] 🔥 Benchmarking GB/s and % of Peak Bandwidth.

## **Benchmarking** 📚

I have included a cold and hot start benchmarking function in common/include/benchmark.cuh. For a cold start, L2 cache is flushed before every kernel launch and for a hot start, the kernel is launched, without L2 flush, iteratively and averaged over time to yield statistics. I don't think I will but maybe later on I can sample these kernel launches and do statistics on them... That may be for some time later. More on this lateron.

## **Directory**

```
cuda-experiments/
├── README.md                # You are HERE
├── images/                  # basic diagrams
│
├── common/                  # Utilities used by all kernels
│   ├── include/
│   │   ├── utils.cuh        # Header for utils.cu
│   │   └── benchmark.cuh    # Templated timing & GB/s
│   └── src/
│       └── utils.cu         # Tools: Init matrix, validate etc...
│
├── research/                # Notes, generally indexology and explanations
│
├── src/                     # Implementation folders
│   ├── 01_reductions/       # Reduction algorithm
│   ├── 02_softmax/          # Milestone 2: Numerical Stability
│   └── 03_flashAttn/        # Milestone 3: The Main Goal
│       └── fundamentals/    # QK_matmul
│           ├── kernel.cu    # QK_matmul kernel + launch wrapper
│           ├── kernel.cuh   # Declarations
│           └── main.cu      # Verification vs CPU reference
│
├── templates/               # The "Blueprint" for starting new kernels (unused)
│
├── scripts/                 # Build automation"
│   └── build.sh
│
└── bin/                     # Compiled binaries
    └── test_run             # The current executable (by build.sh)
```

## **Worklog - Temporary** 🔏

### **LOG 0**

For now, I worked on organizing my github and getting comfortable with .cu and .cuh. I experimented with launch commands using flags and I think this workflow will keep everything much more clearer. Especially having a template is great since any local changes to the kernel doesn't affect the previous kernels I wrote. With this, I hope that I can create benchmarking functions that can work with multiple kernels :)

### **LOG 1**

Implemented shuffle intrinsics for warp-level reduction primitives. Did a small benchmark test. I will write a more generalized/encompassing benchmarking func. that I can just use to compare kernels more quickly in the future. Implemented XOR warp shuffling.

### **LOG 2**

Implemented rowSum using XOR warp shuffling that can handle matrices of all sizes. Organized my folder system. I'm still learning how the compiler communicates with .cu and .cuh files. Changed libraries from C to C++ and changed some function calls to start with std::. I'm still trying to figure out why C++ function calls are preffered and when they do not matter. Also experimented with template but I need a more solid example where it really makes a difference.

### **LOG 3**

Tried to implement a benchmark function that calculated bandwidth, speedup from cpu and average time by omitting warmup times and averaging over multiple iterations. Turns out, on silicon time is much much more smaller than the first multiple runs. Here is a small result of how matrix size affects my throughput.

To prevent my GPU from using L2 cache and cheat the benchmark, I decided to use matrix sizes bigger than 16k x 1024. (67 MB).

| Size | Time (ms) | Bandwidth (GB/s) | Bandwidth Utilization (%) |
| :--- | :--- | :--- | :--- |
| 16384 x 1024 | 0.2832 | 237.18 | 74.1 |
| 32768 x 1024 | 0.5628 | 238.70 | 75.6 |
| 65536 x 1024 | 1.1193 | 240.06 | 75.0 |

### **LOG 4**

Implemented Cache Flushing on benchmarking. Increasing matrix sizes more than what the L2 cache size allows yields a very similar effect as to L2 flushing and generally speaking the 'performance cheating' given by L2 cache is quite irrelevant in real life applications where matrix sizes, for example, exceed 16k x 16k. Thus I find it quite irrelevant. Yet I still need to research proper cuda error and stream handling and learn how to make a more encompassing benchmarking function that can handle all my kernels. I think it has something to do with templates. I also plan to implement more c++ and also experiment with namespaces in my functions.

### **LOG 5**

Experimented with std::function and std::bind to create a more Type-Agnostic Benchmarking Wrapper. This will be important later when I have multiple kernels with different arguments and different streams where benchmarking must be injected into a specific kernel and to a specific stream. Furhtermore, a single benchmark function compatible with all kernels is great for rapid prototyping, reducing the extra work to write kernel specific benchmark functions again and again. I still have some changes to make into the benchmark function such as modifying the pointers into void* for it to really be generic. Even though these concept are still a bit vauge to me, I find this practice of writing elegant and generic code to be really helpful in creating a more organized workflow and also paving the way for more kernels to be implemented efficiently. I'm satisfied with today :)

### **LOG 6**

Started working on Online Softmax and the flashAttn math. The motivation behind FlashAttn becomes much more easier when you try to map Matmul -> Softmax -> Matmul into block-rows of the output matrix O. I noticed that the online softmax trick is used twice, which isn't stated clearly in the paper itself: 1) when calculating P_{ij} 2) when calculating the j-th block of O_i, we do an online-softmax-like update to the old O_i, of block j-1. Anyhow, most efforts right now will be going to keeping softmax folder clean. I actually want to implement a basic flashattn first and then create anoter one that can handle different data types.

### **LOG 7**

Implemented a single row softmax kernel using warp-level primitives. Transforming that logic onto a softmax kernel that works with matrices is only a matter of proper indexology, of which I plan to do tomorrow. Next is possibly flashAttn. I need to work around the places where softmax updates will be made in flashAttn algorithm. Included static_cast<> in main.cu for safe casting.

### **LOG 8**

Implemented matrix online softmax kernel. Experimented with mixed precision and used forceinline for function overloading in softmax kernel. I want to keep the kernels type-agnostic and I want to make them suitable for common use. However that will take time and I am sensitive over its complexity. I'd rather have a mediocre efficiency kernel than a super fast but highly complex code, for now. Maybe in the future I will grab on some hardware and test the other case. But for nowi readable and maintainable code is my highest priority. I will implement flash attention with that goal in mind.

### **LOG 9**

Sketched out flash attention kernel. I worked around the math and I am still stuck between the hardest design choice: registers or shared memory when storing the intermediate chunk S_ij. I initially though registers though I didn't want to directly go through the elements per thread calculation again and again without a naive flashattn implementation so I decided to settle with shared memory and use appropriate chunk sizes Bc and Br. I think the most hard part is not being able to test the intermediate code. Maybe there is a way that I am not aware of yet but it seems and feels like I am writing kernels in the dark for an hour or something before I am able to test it. The worse part is to run the kernel, see it giving inaccurate answers, turning back to code and sadly seeing that you have to proof check the math, not debug it like seperating part of the codes and running them. The data size is big, the assumptions are getting complexer. I think solidifying each part of the basics is a really good way to really minimize simple but hard to notice mistakes ruining the kernel, like uncoalesced mem accesses/out-of-bound reaches/wrong matmul indexology (I think I currently suffer from that while implementing flashAttn rn). Anyhow, we will see how it sticks in time. :)

### **LOG 10**

Been working on flashAttn for some time now. I've decided to take a step back and divide the flash attention to three parts: QK matmul, S softmax, PV matmul. Right now I am working on the QK matmul and hoping that it will run properly on two or more blocks. Right now it works for a single block (meaning matmul works, but indexology for the following blocks are a bit shaky) and I maintain a huge list of assumptions that are made in writing the kernel. (I always wonder how researchers that design these kernels approach this problem.) I've also checked out Karpathy's llm.c. It's pretty cool, if you haven't came across it by now please go check it out! The kernels inside are not magnificent but for a beginner it is incredibly valuable as what you see later are basically built from those 'simple' implementations. I may later on work on some simple but efficient CUDA kernels inside the dev folder of it, if it also happens that I buy a GPU.

I mentioned a concern above. I wonder how other people tackle designing new kernels? Do you guys draw it or approach by index first? I hope you guys don't start by doing the math and index one by one without visualizing... I've never seen that happen though it would've scared me to see someone that coherent and efficient with indexology.

Multiple blocks didn't work because there was an indexology problem (always an indexology problem) at the CPU code. I always assumed that I have written my CPU code correctly which really stung me today. I haven't been logging here for some time now so this can be considered as a mega update here :)

Also added emojis to make this place a bit pretty. Because it deserves to be so.

### **LOG 11**

Kicked things of with finalizing QK_matmul. It passed all the tests I planned on doing for now. I listed them quickly in ./research. Feel free to check it out real quick as the next fundamental kernels for Flash Attention are built on top of it. Currently I am working and, as of writing, finished S_softmax. It is built onto QK_matmul assuming that it works (and it does). I tested my first attempt and was really happy to see that all my tests were a success but then I saw the sad reality that my CPU was calculating -nan. I am surprised how -nan isn't notified by C++ when I try to calculate relative error. 

## **Stretch Want-To-Do's**

* **Start exploring llm.c. Replicate it with 'minimal' cheating, run it.**
* **Try to implement other papers in CUDA**
* **Continue with FlashAttn papers**
* **Adopt an open project**
* **Experiment with tensor cores**