//from trsm_kernel_LN.c of OpenBLAS
/* m*m matrix A (at L side) before packing: 
d y y ... y
0 d y ... y
0 0 d ... y
. . .   . 
. . .   . 
. . .   . 
0 0 0 ... d
*/

#include "common.h"

static inline void solve(BLASLONG m, BLASLONG n, FLOAT *a, FLOAT *b, FLOAT *c, BLASLONG ldc) { //solve m*n elements of b
//a[m][m]: column-major
//b[m][n]: row-major
//c[n][ldc]: column-major
  FLOAT a0, b0;
  int i, j, k;
  for (i = m - 1; i >= 0; i--) {
    a0 = a[i*m+i]; //reciprocal of the original value
    for (j = 0; j < n; j ++) {
      b0 = c[j*ldc+i]*a0;
      c[j*ldc+i] = b[i*n+j] = b0;
      for (k = 0; k < i; k ++) c[j*ldc+k] -= b0*a[i*n+k];
    }
  }
}
//int GEMM_KERNEL(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT *C, BLASLONG ldc);

#define COMPUTE(a_copy,b_copy) {\
  aa = a + m_count * k;\
  cc = c + m_count;\
  if (k - kk > 0)\
    GEMM_KERNEL(a_copy,b_copy,k - kk,-1.0,aa + a_copy * kk,b + b_copy * kk,cc,ldc);\
  solve(a_copy,b_copy,aa + (kk-a_copy) * a_copy,b + (kk-a_copy) * b_copy, cc, ldc);\
  kk -= a_copy;\
}

int CNAME(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT dummy1, FLOAT *a, FLOAT *b, FLOAT *c, BLASLONG ldc, BLASLONG offset){
  BLASLONG i, j;
  FLOAT *aa, *cc;
  BLASLONG kk,m_count;
  j = n / GEMM_UNROLL_N; m_count=m;
  while (j > 0) {
    kk = m + offset; //kk = num_of_unsolved_b_elements_in_the_column
    for (i = 1; i < GEMM_UNROLL_M; i *= 2){
	  if (m & i) {
        m_count -= i;
	    COMPUTE(i,GEMM_UNROLL_N)
	  }
    }
    for(i=m/GEMM_UNROLL_M;i>0;i--) {
      m_count -= GEMM_UNROLL_M;
      COMPUTE(GEMM_UNROLL_M,GEMM_UNROLL_N)
    }
    b += GEMM_UNROLL_N * k;
    c += GEMM_UNROLL_N * ldc;
    j --;
  }
  if (n % GEMM_UNROLL_N > 0) {
    j = (GEMM_UNROLL_N >> 1);
    while (j > 0) {
      if (n & j) {
	    kk = m + offset; m_count = m;
	    for (i = 1; i < GEMM_UNROLL_M; i *= 2){
	      if (m & i) {
            m_count -= i;
	        COMPUTE(i,j)
	      }
	    }
        for(i=m/GEMM_UNROLL_M;i>0;i--){
          m_count -= GEMM_UNROLL_M;
          COMPUTE(GEMM_UNROLL_M,j)
	    }
	    b += j * k;
	    c += j * ldc;
      }
      j = j >> 1;
    }
  }
  return 0;
}
