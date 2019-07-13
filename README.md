# DGEMM_AVX2
A fast avx2/fma3 dgemm subroutine for large matrices, written in C and assembly, with 97-99% single-thread performance of Intel MKL(2018) (or 101-105% of OpenBLAS(Haswell))


Interface in C:

1-thread: void dgemmserial(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc)

omp-paralleled: void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc)


Function naming in dgemm.c:

load/dgemmblk: The role of the function: load elements from main matrix and pack them into a matrix block, or do dgemm of block matrices

irreg/reg/tail: The matrix block dealing with
         irreg: Block size smaller than defined size, e.g. a block at the edge of the main matrix
           reg: Block size the same as defined
          tail: The block's m dimension smaller than defined size, but the other dimension identical to defined size

a/b/c:Load and pack elements from matrix A or B

c/r:Load from column-major or row-major main matrix

ccc:All matrix blocks are column-major

_ac/_ar:The matrix A is column-major(transa='N') or row-major(transa='T')
