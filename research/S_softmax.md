# **S_softmax**

This is the P = softmax(S) part of FlashAttention written in CUDA. It builds on the previous part: QK_matmul.

## **Main idea**

I've decided to take in QK_matmul results, write it into registers while doing online softmax in thread granularity to prepare for the more efficient warp shuffle online softmax method. This prevents the need of accessing SMEM multiple times. Afterwards the normalized values are sent to GMEM for CPU verification.

## **Assumptions**

The assumptions are the same as in QK_matmul. Nothing new here.

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

Online softmax implementation itself isn't that inefficient, considering we kept the running statistics inside registers the whole process. I'd say the inefficiency right now is mostly due to the SMEM accesses of K_j (bank conflicts) inherited from QK_matmul.