//from trsm_kernel_LT.c of OpenBLAS
#include "common.h"

#define GEMM_KERNEL GEMM_KERNEL_N

static void solve(BLASLONG m, BLASLONG n, FLOAT *a, FLOAT *b, FLOAT *c, BLASLONG ldc) {
  FLOAT a0, b0;
  int i, j, k;
  for (i = 0; i < m; i++) {
    a0 = a[i*m+i];
    for (j = 0; j < n; j ++) {
      b0 = c[j*ldc+i] * a0;
      b[i*n+j] = c[j*ldc+i] = b0;
      for (k = i + 1; k < m; k ++) c[j*ldc+k] -= b0 * a[i*m+k];
    }
  }
}

#define COMPUTE(a_copy,b_copy) {\
  aa = a + m_count * k;\
  cc = ch + m_count;\
  if (kk > 0)\
    GEMM_KERNEL(a_copy, b_copy, kk, -1.0, aa, bh, cc, ldc);\
  solve(a_copy, b_copy, aa + kk * a_copy, bh + kk * b_copy, cc, ldc);\
  kk += a_copy;\
}

int CNAME(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT dummy1, FLOAT *a, FLOAT *b, FLOAT *c, BLASLONG ldc, BLASLONG offset){
  FLOAT *aa, *cc, *bh, *ch;
  BLASLONG kk, m_count;
  BLASLONG i, j;
  bh = b; ch = c;
  j = n / GEMM_UNROLL_N;
  while (j > 0) {
    kk = offset; m_count = 0;
    aa = a;
    cc = ch;
    i = m / GEMM_UNROLL_M;
    while (i > 0) {
      COMPUTE(GEMM_UNROLL_M,GEMM_UNROLL_N)
      m_count += GEMM_UNROLL_M;
      i --;
    }
    if (m % GEMM_UNROLL_M > 0) {
      i = GEMM_UNROLL_M >> 1;
      while (i > 0) {
        if (m & i) {
          COMPUTE(i,GEMM_UNROLL_N)
          m_count += i;
        }
        i = i >> 1;
      }
    }
    bh += GEMM_UNROLL_N * k;
    ch += GEMM_UNROLL_N * ldc;
    j --;
  }
  if (n % GEMM_UNROLL_N > 0) {
    j = GEMM_UNROLL_N >> 1;
    while (j > 0) {
      if (n & j) {
        kk = offset; m_count = 0;
        i = m / GEMM_UNROLL_M;
        while (i > 0) {
          COMPUTE(GEMM_UNROLL_M,j)
          m_count += GEMM_UNROLL_M;
          i --;
        }
        if (m % GEMM_UNROLL_M > 0) {
          i = GEMM_UNROLL_M >> 1;
          while (i > 0) {
	        if (m & i) {
              COMPUTE(i,j)
              m_count += i;
	        }
            i = i >> 1;
          }
        }
        bh += j * k;
        ch += j * ldc;
      }
      j = j >> 1;
    }
  }
  return 0;
}
