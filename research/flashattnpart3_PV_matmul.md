# **PV_matmul_**

This is the O = PV part of FlashAttention written in CUDA. It builds on the previous parts: QK_matmul and P = softmax(S)

## **Main idea**

The chunk of P is shape (Br, Bc) and chunk of V is shape (Bc, d). This means the resulting output chunk O is shape (Br, d). So for the mapping, I have decided that each thread in a warp is responsible for d/32 many elements in a row of chunk O. We keep each O value inside registers during the execution of the kernel and update it accordingly to the fused attention math and only after the block load loop finishes we write the result back to HBM. This wraps up the naive flash attention kernel and testing it against CPU we get the below results.

## **Assumptions**

The assumptions are the same as in the previous parts. Nothing new here.

## **TEST**



## **Possible Problems**

Haven't checked it yet.