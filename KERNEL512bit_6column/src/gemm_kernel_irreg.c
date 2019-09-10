#ifdef DOUBLE
 #define IRREG_SIZE 8
 #define IRREG_VEC_TYPE __m256d
 #define IRREG_VEC_ZERO _mm256_setzero_pd
 #define IRREG_VEC_LOADA _mm256_load_pd
 #define IRREG_VEC_LOADU _mm256_loadu_pd
 #define IRREG_VEC_MASKLOAD _mm256_maskload_pd
 #define IRREG_VEC_STOREU _mm256_storeu_pd
 #define IRREG_VEC_MASKSTORE _mm256_maskstore_pd
 #define IRREG_VEC_BROAD _mm256_broadcast_sd
 #define IRREG_VEC_FMADD _mm256_fmadd_pd
 #define IRREG_VEC_ADD _mm256_add_pd
#else
 #define IRREG_SIZE 4
 #define IRREG_VEC_TYPE __m256
 #define IRREG_VEC_ZERO _mm256_setzero_ps
 #define IRREG_VEC_LOADA _mm256_load_ps
 #define IRREG_VEC_LOADU _mm256_loadu_ps
 #define IRREG_VEC_MASKLOAD _mm256_maskload_ps
 #define IRREG_VEC_STOREU _mm256_storeu_ps
 #define IRREG_VEC_MASKSTORE _mm256_maskstore_ps
 #define IRREG_VEC_BROAD _mm256_broadcast_ss
 #define IRREG_VEC_FMADD _mm256_fmadd_ps
 #define IRREG_VEC_ADD _mm256_add_ps
#endif

#define INIT_1col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)ctemp,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+64/IRREG_SIZE-1),_MM_HINT_T0);\
}
#define INIT_6col {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c4=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c5=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c6=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c7=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c8=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c9=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c10=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);cpref+=ldc;\
   c11=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c12=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE-1),_MM_HINT_T0);\
}
#define KERNELkr {\
   a1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   a2=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c1=IRREG_VEC_FMADD(a1,b1,c1);c2=IRREG_VEC_FMADD(a2,b1,c2);\
}
#define KERNELk1 {\
   KERNELkr\
   b2=IRREG_VEC_BROAD(btemp);btemp++;\
   c3=IRREG_VEC_FMADD(a1,b2,c3);c4=IRREG_VEC_FMADD(a2,b2,c4);\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c5=IRREG_VEC_FMADD(a1,b1,c5);c6=IRREG_VEC_FMADD(a2,b1,c6);\
   b2=IRREG_VEC_BROAD(btemp);btemp++;\
   c7=IRREG_VEC_FMADD(a1,b2,c7);c8=IRREG_VEC_FMADD(a2,b2,c8);\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c9=IRREG_VEC_FMADD(a1,b1,c9);c10=IRREG_VEC_FMADD(a2,b1,c10);\
   b2=IRREG_VEC_BROAD(btemp);btemp++;\
   c11=IRREG_VEC_FMADD(a1,b2,c11);c12=IRREG_VEC_FMADD(a2,b2,c12);\
}
#define KERNELk2 {\
    _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
    _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
    KERNELk1\
    _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
    _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
    KERNELk1\
}
#define STORE_C_1col_multa1(c1,c2,a1) {\
   c1=IRREG_VEC_FMADD(a1,IRREG_VEC_LOADU(ctemp),c1);\
   c2=IRREG_VEC_FMADD(a1,IRREG_VEC_LOADU(ctemp+32/IRREG_SIZE),c2);\
   IRREG_VEC_STOREU(ctemp,c1);\
   IRREG_VEC_STOREU(ctemp+32/IRREG_SIZE,c2);\
   ctemp+=ldc;\
}
#define STORE_C_1col_nomult(c1,c2) {\
   c1=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp),c1);\
   c2=IRREG_VEC_ADD(IRREG_VEC_LOADU(ctemp+32/IRREG_SIZE),c2);\
   IRREG_VEC_STOREU(ctemp,c1);\
   IRREG_VEC_STOREU(ctemp+32/IRREG_SIZE,c2);\
   ctemp+=ldc;\
}
#define STOREIRREGM_C_1col(c1,c2,a1) {\
   c1=IRREG_VEC_FMADD(a1,IRREG_VEC_MASKLOAD(ctemp,ml1),c1);\
   c2=IRREG_VEC_FMADD(a1,IRREG_VEC_MASKLOAD(ctemp+32/IRREG_SIZE,ml2),c2);\
   IRREG_VEC_MASKSTORE(ctemp,ml1,c1);\
   IRREG_VEC_MASKSTORE(ctemp+32/IRREG_SIZE,ml2,c2);\
   ctemp+=ldc;\
}
static void gemmblkirregkccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int kdim,FLOAT * __restrict__ beta){
  register IRREG_VEC_TYPE a1,a2,b1,b2,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<BlkDimN;ccol+=6){//loop over cblk-columns, calculate 6 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_6col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   a1=IRREG_VEC_BROAD(beta);
   STORE_C_1col_multa1(c1,c2,a1)
   STORE_C_1col_multa1(c3,c4,a1)
   STORE_C_1col_multa1(c5,c6,a1)
   STORE_C_1col_multa1(c7,c8,a1)
   STORE_C_1col_multa1(c9,c10,a1)
   STORE_C_1col_multa1(c11,c12,a1)
  }
}
static void gemmblkirregnccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int ndim){
  register IRREG_VEC_TYPE a1,a2,b1,b2,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<ndim-5;ccol+=6){//loop over cblk-columns, calculate 6 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_6col
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol+=8){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    KERNELk2
    KERNELk2
    KERNELk2
    KERNELk2
   }
   STORE_C_1col_nomult(c1,c2)
   STORE_C_1col_nomult(c3,c4)
   STORE_C_1col_nomult(c5,c6)
   STORE_C_1col_nomult(c7,c8)
   STORE_C_1col_nomult(c9,c10)
   STORE_C_1col_nomult(c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_1col_nomult(c1,c2)
  }
}
static void gemmblkirregccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int mdim,int ndim,int kdim,FLOAT * __restrict__ beta){
  register IRREG_VEC_TYPE a1,a2,b1,b2,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;__m256i ml1,ml2;
  FLOAT *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
#ifdef DOUBLE
  ml1=_mm256_setr_epi32(0,-(mdim>0),0,-(mdim>1),0,-(mdim>2),0,-(mdim>3));
  ml2=_mm256_setr_epi32(0,-(mdim>4),0,-(mdim>5),0,-(mdim>6),0,-(mdim>7));
#else
  ml1=_mm256_setr_epi32(-(mdim>0),-(mdim>1),-(mdim>2),-(mdim>3),-(mdim>4),-(mdim>5),-(mdim>6),-(mdim>7));
  ml2=_mm256_setr_epi32(-(mdim>8),-(mdim>9),-(mdim>10),-(mdim>11),-(mdim>12),-(mdim>13),-(mdim>14),-(mdim>15));
#endif
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<ndim-5;ccol+=6){//loop over cblk-columns, calculate 6 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_6col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   a1=IRREG_VEC_BROAD(beta);
   STOREIRREGM_C_1col(c1,c2,a1)
   STOREIRREGM_C_1col(c3,c4,a1)
   STOREIRREGM_C_1col(c5,c6,a1)
   STOREIRREGM_C_1col(c7,c8,a1)
   STOREIRREGM_C_1col(c9,c10,a1)
   STOREIRREGM_C_1col(c11,c12,a1)
  }
  c9=IRREG_VEC_BROAD(beta);
  for(;ccol<ndim;ccol++){
   INIT_1col
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREIRREGM_C_1col(c1,c2,c9)
  }
}
