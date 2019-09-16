# GEMM_AVX2

#Introduction

Fast avx2/fma3 dgemm and sgemm subroutines for large matrices (dimension 3000-40000), written in C and assembly, slightly outperform Intel MKL 2019 (update 4) on Core i9 9900K and Ryzen 7 3700X, able to achieve >95% 1-thread theoretical performance and >90% OpenMP multithread theoretical performance. 



#Dynamic libraries

DGEMM.so and SGEMM.so


#Top performances

i9-9900K, avx-offset=6, dual-channel DDR4-2400: 

1-thread dgemm: DGEMM.so 67.5 GFLOPS; OpenBLAS(Haswell,recent update) 65.8 GFLOPS; MKL(2018-libgomp) 66-67 GFLOPS; Theoretical 70.4 GFLOPS.

1-thread sgemm: SGEMM.so 135-136 GFLOPS; MKL(2018-libgomp) 134-135 GFLOPS; OpenBLAS(Haswell) 120 GFLOPS; Theoretical 141 GFLOPS.


r7-3700X, 3.6 GHz, dual-channel DDR4-2133:

1-thread dgemm: DGEMM.so 55.1 GFLOPS; MKL 55.0 GFLOPS; OpenBLAS(Haswell,recent update) 53.0 GFLOPS; Theoretical 57.6 GFLOPS.

1-thread sgemm: SGEMM.so 111 GFLOPS; MKL 110 GFLOPS; OpenBLAS(Haswell) 104 GFLOPS; Theoretical 115 GFLOPS.


#Function interfaces in C:


omp-paralleled: void dgemm_(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in DGEMM.so; void sgemm_(char *transa,char *transb,int *m,int *n,int *k,float *alpha,float *a,int *lda,float *b,int *ldb,float *beta,float *c,int *ldc), in SGEMM.so.


#Function namings in source codes:

load/gemmblk: The role of the function: load = "load elements from main matrix and pack them into a matrix block"; dgemmblk = "do dgemm of matrix blocks".

irreg/reg/tail: The matrix block dealing with: 
         irreg = "block size smaller than defined size, e.g. a block at the edge of the main matrix";
           reg = "block size the same as defined";
          tail = "the block's m dimension smaller than the defined size, but the other dimension identical to defined size".

a/b:Load and pack elements from matrix A or B

c/r:Load from column-major or row-major source matrix

ccc:All matrix blocks are column-major for gemm


#Tuning the performance:

Choosing the best kernel: there are 3 types of kernels, differ in the blocking of C matrix in 12 ymm registers: 512bitx6column, 768bitx4column and 1024bitx3column.  Usually the 768bitx4column kernel is the most efficient one. 

Adjusting parameters for blocking and prefetch: see the README files in kernel directories for details. 


#Test programs:

For gemm benchmarking tools, see my repository GEMM_AVX2_FMA3 for details. 

There's a FMA3 dgemm GFLOPS test tool in the folder MAX_GFLOPS. 

Before testing on AMD Zen processors, please set the environment variable "MKL_DEBUG_CPU_TYPE" to 5, which tells MKL to use the AVX2 code path. 



#Comments:

Any optimizations to the gemm codes are welcomed~

