# DGEMM_AVX2

#Introduction

Fast avx2/fma3 dgemm subroutines for large matrices, written in C and assembly, with efficiencies comparable to Intel MKL(2018).

#Top performance 

i9-9900K, avx-offset=6, dual-channel DDR4-2400: 

1 thread: DGEMM.so 65.3-66.2 GFLOPS; DGEMM_LARGEMEM.so 67.5 GFLOPS; OpenBLAS(Haswell,recent update) 65.3 GFLOPS; MKL(2018-libgomp) 66-67 GFLOPS; Theoretical 70.4 GFLOPS

8 threads: DGEMM.so 455-465 GFLOPS; DGEMM_LARGEMEM.so 488 GFLOPS; OpenBLAS(Haswell,recent update) 460-470 GFLOPS; MKL(2018-libgomp) 474 GFLOPS; Theoretical 524 GFLOPS

r7-3700X, 3.6 GHz, dual-channel DDR4-2133:

1 thread: DGEMM.so 52-54 GFLOPS; DGEMM_LARGEMEM.so 54.8 GFLOPS; OpenBLAS(Haswell,recent update) 52.8 GFLOPS; Theoretical 57.6 GFLOPS


#Dynamic libraries

DGEMM.so and DGEMM_LARGEMEM.so, the latter consumes more memory but runs faster.



#Function interfaces in C:

1-thread: void dgemmserial(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in DGEMM.so

omp-paralleled: void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in both libraries.




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



#Comments:

Any optimizations to the dgemm codes are welcomed~
Unlike the completely-tested "DGEMM.so", the library "DGEMM_LARGEMEM.so" haven't been thoroughly tested so may become buggy in some rare cases. The author would be grateful if some experts could help him check the codes.
