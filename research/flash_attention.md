# **Flash Attention**

## **Initial Thoughts and Roadmap**

My first attempt was to just imagine the kernel itself, the thread mapping and write it down as the operation flows. It went horribly wrong. Especially when I had to take a break from calculating a threads matmul indeology to going back to write again another Global memory to Shared memory load loop. This back and forth was really mentally exhausting and when it came to test it againsts a CPU, I couldn't find a way to debug the whole kernel at once. This led me to take another approach.

In my second attempt Flash Attention is broken down to three fused operations: QK_matmul, S_softmax, and PV_matmul. I decided initially to build each part seperately then do a fusion. However after writing QK_matmul, I noticed that I can just write S_softmax on top of it. After implementing PV_matmul I noticed that the resulting kernel is just the naive flash attention implementation without needing additional fusing. This manner of writing kernels proved much easier to be since at each time the number of unknown sources of errors are really concentrated to a small point of the code on which I can work with quite well. Whether the problem is indexology, wrong code, or just wrong math, when you know surely that the error is inside a narrow line of code, solving it becomes really simple.

## **Challenges**

The one thing I found not so helpful while implementing Flash Attention was the pseudocode that was presented in the paper. Algorithms like Flash Attention are  for parallel execution. However the pseudocode provided was written in a linear manner (as for a CPU) instead of in a parallel manner, which reduces its reproducability. Maybe a better way of sharing pseudocodes for parallel operations is giving the math of the fused operations as an overview alongside examples over what each thread/computing unit can do. What I mean is the pseudocode itself can be kept since it indeed gives a good overview of the operation itself but there can also be an operation or sequences of operations defined that provide the execution path for a thread. A good example of this I found in NVIDIA's **[online softmax paper](https://arxiv.org/abs/1805.02867)** (Algorithm 3 and section 3.1), when it defines an operation between threads which can be 1-to-1 implemented in CUDA with ease using warp-level primitives.

Another thing that I got confused by in the pseduocode of Flash Attention was in the linear-fashion for loops over every block. At each iteration, it loads a chunk of Q, K or V from HBM to SRAM which feels as if the linear presudocode is written in a parallel manner even though it is not. I feel like there must be a clear distinction between the linear and parallel fashion code.

## **Drawings and Diagrams for FlashAttention**

Before that I have a quick question. Do people say flashAttn, flash attention, Flash Attention or FlashATTN? Each one of them sounds very cool. Especially the capslock makes it even cooler. FlashAttn. As if it is a medicine. Anyhow, here are the following diagrams for what a thread block in the naive Flash Attention kernel does:

I've found it much more important to set the thread block size dependent on Br than Bc since Br determiens the chunk of the output matrix that the thread block will work on which is of much more higher priority. But then I came across the problem that if we have a thread block mapped in shape of (Br, 32), how can I use the same mapping to load (Bc, d)? I settled for a two for loop where the inner loop uses 32 threads per row to load d many elements with a simple stride. The outer loop then goves Br elements down the row until all Bc rows are covered. To prevent indexology work such as boundary checks and to keep the code simple, I assumed that Bc is divisible by Br. 

## **Current Problems**

I don't have that much practice with benchmarking and reiteratively optimizing. I feel like that is exactly the next goal after the naive implementation. Most and foremost, for it really to be **Flash** Attention, I need to express bottlenecks. One that I can think of right now is the bank conflict presented my SMEM accesses.

## **Iterative Optimziation**

### **Naive FlashAttn Bottleneck Discussion**
