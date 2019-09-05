# GEMM_AVX2

#Introduction

Fast avx2/fma3 dgemm and sgemm subroutines for large matrices, written in C and assembly, with performances comparable to Intel MKL(2018).



#Dynamic libraries

DGEMM.so and DGEMM_LARGEMEM.so, the latter consumes more memory but runs faster.

SGEMM.so (experimental)


#Top performances

i9-9900K, avx-offset=6, dual-channel DDR4-2400: 

1-thread dgemm: DGEMM.so 65.3-66.2 GFLOPS; DGEMM_LARGEMEM.so 67.5 GFLOPS; OpenBLAS(Haswell,recent update) 65.8 GFLOPS; MKL(2018-libgomp) 66-67 GFLOPS; Theoretical 70.4 GFLOPS.

1-thread sgemm: SGEMM.so 135 GFLOPS; MKL(2018-libgomp) 133 GFLOPS; OpenBLAS(Haswell) 120 GFLOPS; Theoretical 141 GFLOPS.

8-thread dgemm: DGEMM.so 455-465 GFLOPS; DGEMM_LARGEMEM.so 488 GFLOPS; OpenBLAS(Haswell,recent update) 470 GFLOPS; MKL(2018-libgomp) 474 GFLOPS; Theoretical 525 GFLOPS.

8-thread sgemm: SGEMM.so 988 GFLOPS; MKL(2018-libgomp) 965 GFLOPS; Theoretical 1050 GFLOPS.

r7-3700X, 3.6 GHz, dual-channel DDR4-2133:

1-thread dgemm: DGEMM.so 52-54 GFLOPS; DGEMM_LARGEMEM.so 54.8 GFLOPS; MKL 55 GFLOPS; OpenBLAS(Haswell,recent update) 53.0 GFLOPS; Theoretical 57.6 GFLOPS.

1-thread sgemm: SGEMM.so 111 GFLOPS; MKL 110-111 GFLOPS; OpenBLAS(Haswell) 104 GFLOPS; Theoretical 115 GFLOPS.


#Function interfaces in C:

1-thread only: void dgemmserial(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in DGEMM.so

omp-paralleled: void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in DGEMM.so and DGEMM_LARGEMEM.so; void sgemm(char *transa,char *transb,int *m,int *n,int *k,float *alpha,float *a,int *lda,float *b,int *ldb,float *beta,float *c,int *ldc), in SGEMM.so.


#Function namings in source codes:

load/dgemmblk: The role of the function: load = "load elements from main matrix and pack them into a matrix block"; dgemmblk = "do dgemm of matrix blocks".

irreg/reg/tail: The matrix block dealing with: 
         irreg = "block size smaller than defined size, e.g. a block at the edge of the main matrix";
           reg = "block size the same as defined";
          tail = "the block's m dimension smaller than defined size, but the other dimension identical to defined size".

a/b:Load and pack elements from matrix A or B

c/r:Load from column-major or row-major main matrix

ccc:All matrix blocks are column-major for dgemm

_ac/_ar:The matrix A is column-major(transa='N') or row-major(transa='T')



#Attached test programs:

There're 2 dgemm test programs attached (General_Benchmark_*.c) for debugging and benchmarking purposes. Compilation of them requires installation of Intel MKL (version 2018 is ok). Please compile them with gcc and link them to MKL with 32-bit integer interface.

Before testing on zen processors, please set the environment variable "MKL_DEBUG_CPU_TYPE" to 5, which tells MKL to use AVX2 code path. Setting of "GOMP_CPU_AFFINITY" is encouraged to improve reproducibility.



#Comments:

Any optimizations to the gemm codes are welcomed~
Unlike the completely-tested "DGEMM.so", the libraries "DGEMM_LARGEMEM.so" and "SGEMM.so" haven't been thoroughly tested so may become buggy in some rare cases. The author would be grateful if some experts could help him check the codes.
