# ifdef DOUBLE
 # define IRREG_SIZE 8
 # define IRREG_VEC_TYPE __m256d
 # define IRREG_VEC_ZERO _mm256_setzero_pd
 # define IRREG_VEC_LOADA _mm256_load_pd
 # define IRREG_VEC_LOADU _mm256_loadu_pd
 # define IRREG_VEC_MASKLOAD _mm256_maskload_pd
 # define IRREG_VEC_STOREU _mm256_storeu_pd
 # define IRREG_VEC_MASKSTORE _mm256_maskstore_pd
 # define IRREG_VEC_BROAD _mm256_broadcast_sd
 # define IRREG_VEC_FMADD _mm256_fmadd_pd
 # define IRREG_VEC_ADD _mm256_add_pd
# else
 # define IRREG_SIZE 4
 # define IRREG_VEC_TYPE __m256
 # define IRREG_VEC_ZERO _mm256_setzero_ps
 # define IRREG_VEC_LOADA _mm256_load_ps
 # define IRREG_VEC_LOADU _mm256_loadu_ps
 # define IRREG_VEC_MASKLOAD _mm256_maskload_ps
 # define IRREG_VEC_STOREU _mm256_storeu_ps
 # define IRREG_VEC_MASKSTORE _mm256_maskstore_ps
 # define IRREG_VEC_BROAD _mm256_broadcast_ss
 # define IRREG_VEC_FMADD _mm256_fmadd_ps
 # define IRREG_VEC_ADD _mm256_add_ps
# endif
# if GEMM_UNROLL_N == 2
 # include "gemm_edge_kernel_unroll2.h"
# endif
# if GEMM_UNROLL_N == 3
 # include "gemm_edge_kernel_unroll3.h"
# endif
# if GEMM_UNROLL_N == 4
 # include "gemm_edge_kernel_unroll4.h"
# endif
# if GEMM_UNROLL_N == 6
 # include "gemm_edge_kernel_unroll6.h"
# endif

static void gemmblkirregkccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int kdim){
  register IRREG_VEC_TYPE t1,t2,t3,t4,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<GEMM_BLOCK_DIM_N;ccol+=GEMM_UNROLL_N){//loop over cblk-columns, calculate GEMM_UNROLL_N columns of cblk in each iteration.
   cpref=ctemp;
   INIT_ncol
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_ncol
  }
}
static void gemmblkirregnccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int ndim){
  register IRREG_VEC_TYPE t1,t2,t3,t4,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<=ndim-GEMM_UNROLL_N;ccol+=GEMM_UNROLL_N){//loop over cblk-columns, calculate GEMM_UNROLL_N columns of cblk in each iteration.
   cpref=ctemp;
   INIT_ncol
   atemp=ablk;
   for(acol=0;acol<GEMM_BLOCK_DIM_K;acol+=8){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    KERNELk2
    KERNELk2
    KERNELk2
    KERNELk2
   }
   STORE_C_ncol
  }
  for(;ccol<ndim;ccol++){
   INIT_1col
   atemp=ablk;
   for(acol=0;acol<GEMM_BLOCK_DIM_K;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_1col
  }
}
static void gemmblkirregccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int mdim,int ndim,int kdim){
  register IRREG_VEC_TYPE t1,t2,t3,t4,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
# ifdef DOUBLE
  __m256i ml1 = _mm256_setr_epi32(0,-(mdim>0),0,-(mdim>1),0,-(mdim>2),0,-(mdim>3));
  __m256i ml2 = _mm256_setr_epi32(0,-(mdim>4),0,-(mdim>5),0,-(mdim>6),0,-(mdim>7));
 # if GEMM_UNROLL_N < 6
  __m256i ml3 = _mm256_setr_epi32(0,-(mdim>8),0,-(mdim>9),0,-(mdim>10),0,-(mdim>11));
 # endif
 # if GEMM_UNROLL_N < 4
  __m256i ml4 = _mm256_setr_epi32(0,-(mdim>12),0,-(mdim>13),0,-(mdim>14),0,-(mdim>15));
 # endif
 # if GEMM_UNROLL_N < 3
  __m256i ml5 = _mm256_setr_epi32(0,-(mdim>16),0,-(mdim>17),0,-(mdim>18),0,-(mdim>19));
  __m256i ml6 = _mm256_setr_epi32(0,-(mdim>20),0,-(mdim>21),0,-(mdim>22),0,-(mdim>23));
 # endif
# else
  __m256i ml1 = _mm256_setr_epi32(-(mdim>0),-(mdim>1),-(mdim>2),-(mdim>3),-(mdim>4),-(mdim>5),-(mdim>6),-(mdim>7));
  __m256i ml2 = _mm256_setr_epi32(-(mdim>8),-(mdim>9),-(mdim>10),-(mdim>11),-(mdim>12),-(mdim>13),-(mdim>14),-(mdim>15));
 # if GEMM_UNROLL_N < 6
  __m256i ml3 = _mm256_setr_epi32(-(mdim>16),-(mdim>17),-(mdim>18),-(mdim>19),-(mdim>20),-(mdim>21),-(mdim>22),-(mdim>23));
 # endif
 # if GEMM_UNROLL_N < 4
  __m256i ml4 = _mm256_setr_epi32(-(mdim>24),-(mdim>25),-(mdim>26),-(mdim>27),-(mdim>28),-(mdim>29),-(mdim>30),-(mdim>31));
 # endif
 # if GEMM_UNROLL_N < 3
  __m256i ml5 = _mm256_setr_epi32(-(mdim>32),-(mdim>33),-(mdim>34),-(mdim>35),-(mdim>36),-(mdim>37),-(mdim>38),-(mdim>39));
  __m256i ml6 = _mm256_setr_epi32(-(mdim>40),-(mdim>41),-(mdim>42),-(mdim>43),-(mdim>44),-(mdim>45),-(mdim>46),-(mdim>47));
 # endif
# endif
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<=ndim-GEMM_UNROLL_N;ccol+=GEMM_UNROLL_N){//loop over cblk-columns, calculate GEMM_UNROLL_N columns of cblk in each iteration.
   cpref=ctemp;
   INIT_ncol
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREEDGEM_C_ncol
  }
  for(;ccol<ndim;ccol++){
   INIT_1col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREEDGEM_C_1col
  }
}
