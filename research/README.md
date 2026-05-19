# Research

Includes kernel related explanation, math and notes, all specific. Related kernels can be found in ./src from root.

## Rules

Must-be in every kernel written, applied to every kernel and main to maintain compatibility.

- [x] Full `CUDA_CHECK` error handling on every API call.
- [x] Proper C++ Namespacing and Header/Source separation.
- [x] `build.sh` scripting for reproducible experiments.
- [x] Benchmarking GB/s and % of Peak Bandwidth.

## online_softmax.md

Online softmax, derived from standard softmax we are familiar with. Starting from theory to CPU implementation to CUDA implementation. Next, implemented inside FlashAttn.

## Worklog:

### **LOG 0**


Created the research environment, setted up the main goal and specified rules for consistency. Created online_softmax.md.

### **LOG 1**

Started working on online_softmax.md which will build the basis towards FlashAttn. Why it is necessary will be visible later, when we will need to express the attention mechanism and the importance of calculating the output matrix O without materializing the large intermediate matrix S.