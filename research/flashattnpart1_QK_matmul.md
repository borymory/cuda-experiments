# **QK_matmul**

This is the QK^T matrix multiplication part of FlashAttention written in CUDA.

## **Main idea**

The main idea was to assign a warp into every row of Q. Thus a block occupies a block row of shape (Br, d). The thread block collabratively loads a chunk (row-tile) of K and does matmul and stores it intermediately in S_ij SMEM. To verify the kernel against the CPU, I made it load back from S_ij to the global memory.

## **Assumptions**

Since this and two other parts will fuse and make the first naive kernel, I've decided to keep it simple and use the following assumptions.

* **Br and Bc is a multiple of 32.**
* **Br divides N perfectly. (no half blocks in Q)**
* **Bc divides N perfectly. (no half blocks in K)**
* **Br divides Bc perfectly. (no half blocks working on collabrative loading)**
* **D is a multiple of 32. (no warp divergence)**

## **TEST**

| N | d | Result|
| :--- | :--- | :---|
| 32 | 32 | ✅ |
| 64 | 32 | ✅ |
| 128 | 32 | ✅ |
| 32 | 64 | ✅ |
| 64 | 64 | ✅ |
| 128 | 64 | ✅ |
| 1024 | 32 | ✅ |
| 1024 | 64 | ✅ |

## **Possible Problems**

This code is really inefficient. Bank conflicts are present in the loading of K_j.