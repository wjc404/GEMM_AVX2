# GEMM_AVX2

# Introduction

Fast avx2/fma3 sgemm and dgemm subroutines for large matrices, written in C and assembly, able to outperform Intel MKL(2019 update 4) after tuning.


# Interface in C

omp-paralleled: void dgemm_(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc); void sgemm_(char *transa,char *transb,int *m,int *n,int *k,float *alpha,float *a,int *lda,float *b,int *ldb,float *beta,float *c,int *ldc).


# How to tune

Please edit "dgemm_tune.h" and "sgemm_tune.h". Benchmarking tools can be downloaded from my repository "GEMM_AVX2_FMA3".

# Comments:

Any optimizations to the gemm codes are welcomed~
