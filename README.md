# DGEMM_AVX2

#Introduction

Fast avx2/fma3 dgemm subroutines for large matrices, written in C and assembly, with efficiencies comparable to Intel MKL(2018).


#Dynamic libraries

DGEMM.so and DGEMM_LARGEMEM.so, the latter consumes more memory but runs faster.



#Function interface in C:

1-thread: void dgemmserial(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in DGEMM.so

omp-paralleled: void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc), in both libraries.




#Function naming in source codes:

load/dgemmblk: The role of the function: load elements from main matrix and pack them into a matrix block, or do dgemm of block matrices

irreg/reg/tail: The matrix block dealing with: 
         irreg: Block size smaller than defined size, e.g. a block at the edge of the main matrix;
           reg: Block size the same as defined;
          tail: The block's m dimension smaller than defined size, but the other dimension identical to defined size.

a/b:Load and pack elements from matrix A or B

c/r:Load from column-major or row-major main matrix

ccc:All matrix blocks are column-major

_ac/_ar:The matrix A is column-major(transa='N') or row-major(transa='T')



#Attached test programs:

2 DGEMM test codes are also attached (General_Benchmark_*.c). Compilation of them requires installation of Intel MKL 2018.



#Comments:

Any optimizations to the dgemm codes are welcomed~
Unlike the completely-tested "DGEMM.so", the library "DGEMM_LARGEMEM.so" haven't been thoroughly tested so may become buggy in some rare cases. The author would be grateful if some experts could help him check the codes.
