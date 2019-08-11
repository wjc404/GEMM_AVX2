# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h> //AVX2
# include <omp.h>

# define BlkUnitM 6 //fixed!
# define BlkUnitN 64 //fixed!
# define BlkUnitK BlkUnitN
# define BlkDimM (BlkUnitM*4)
# define BlkDimN (BlkUnitN*4)
# define BlkDimK (BlkUnitK*4)
//compilation command: gcc -fopenmp --shared -fPIC -march=haswell -O3 sgemm.c sgemm.S -o SGEMM.so

// below are functions written in assembly
extern void sgemmblkregccc(float *abufferctpos,float *bblk,float *cstartpos,int ldc);//carry >90% sgemm calculations
extern void sgemmblktailccc(float *abufferctpos,float *bblk,float *cstartpos,int ldc,int mdim);
extern void timedelay();//produce nothing besides a delay(~3 us), with no system calls

void load_irreg_a_c(float *astartpos,float *ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;float *aread,*awrite;
  aread=astartpos;awrite=ablk;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      *(awrite+arow)=*(aread+arow);
    }
    for(;arow<BlkDimM;arow++){
      *(awrite+arow)=0.0;
    }
    aread+=lda;awrite+=BlkDimM;
  }
}
void load_irreg_a_r(float *astartpos,float *ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;float *aread,*awrite;
  aread=astartpos;awrite=ablk;
  for(arow=0;arow<mdim;arow++){
    for(acol=0;acol<kdim;acol++){
      *(awrite+acol*BlkDimM)=*(aread+acol);
    }
    aread+=lda;awrite++;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<BlkDimM-mdim;arow++){
      *(awrite+arow)=0.0;
    }
    awrite+=BlkDimM;
  }
}
void load_reg_a_c(float *astartpos,float *ablk,int lda){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,BlkDimK);}
void load_reg_a_r(float *astartpos,float *ablk,int lda){load_irreg_a_r(astartpos,ablk,lda,BlkDimM,BlkDimK);}
void load_tail_a_c(float *astartpos,float *ablk,int lda,int mdim){load_irreg_a_c(astartpos,ablk,lda,mdim,BlkDimK);}
void load_tail_a_r(float *astartpos,float *ablk,int lda,int mdim){load_irreg_a_r(astartpos,ablk,lda,mdim,BlkDimK);}
void load_irregk_a_c(float *astartpos,float *ablk,int lda,int kdim){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,kdim);}
void load_irregk_a_r(float *astartpos,float *ablk,int lda,int kdim){load_irreg_a_r(astartpos,ablk,lda,BlkDimM,kdim);}
void load_reg_b_c(float *bstartpos,float *bblk,int ldb,float *alpha){
 float *inb1,*inb2,*inb3,*inb4,*outb;
 int bcol,brow;
 outb=bblk;
 inb1=bstartpos;
 inb2=inb1+ldb;
 inb3=inb2+ldb;
 inb4=inb3+ldb;
 for(bcol=0;bcol<BlkUnitN;bcol++){
  for(brow=0;brow<BlkUnitK;brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;
  inb4-=(bcol==BlkUnitN-1)*(ldb*BlkDimN);
  for(;brow<2*BlkUnitK;brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;
  inb3-=(bcol==BlkUnitN-1)*(ldb*BlkDimN);
  for(;brow<3*BlkUnitK;brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;
  inb2-=(bcol==BlkUnitN-1)*(ldb*BlkDimN);
  for(;brow<BlkDimK;brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb-BlkDimK;
  inb2+=ldb-BlkDimK;
  inb3+=ldb-BlkDimK;
  inb4+=ldb-BlkDimK;
 }
}
void load_reg_b_r(float *bstartpos,float *bblk,int ldb,float *alpha){
  register __m128 bi1,bi2,bi3,bi4,bt1,bt2,bt3,bt4,bb;
  float *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
  bb=_mm_broadcast_ss(alpha);
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;int bshift=4*ldb-BlkDimN;
  for(brow=0;brow<BlkUnitK;brow+=4){
    bout=bblk+brow*4;
    for(bcol=0;bcol<BlkUnitN;bcol++){
      bi1=_mm_loadu_ps(bin1);bin1+=4;bi2=_mm_loadu_ps(bin2);bin2+=4;
      bi3=_mm_loadu_ps(bin3);bin3+=4;bi4=_mm_loadu_ps(bin4);bin4+=4;
      bt1=_mm_mul_ps(bi1,bb);bt2=_mm_mul_ps(bi2,bb);bt3=_mm_mul_ps(bi3,bb);bt4=_mm_mul_ps(bi4,bb);
      _mm_store_ps(bout,bt1);_mm_store_ps(bout+4,bt2);_mm_store_ps(bout+8,bt3);_mm_store_ps(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*BlkUnitK;brow+=4){
    bout=bblk+brow*4+(BlkUnitN-1)*4*BlkDimK+3;
    *(bout+0)=*bin1*(*alpha);bin1++;
    *(bout+4)=*bin2*(*alpha);bin2++;
    *(bout+8)=*bin3*(*alpha);bin3++;
    *(bout+12)=*bin4*(*alpha);bin4++;
    bout=bblk+brow*4;
    for(bcol=1;bcol<BlkUnitN;bcol++){
      bi1=_mm_loadu_ps(bin1);bin1+=4;bi2=_mm_loadu_ps(bin2);bin2+=4;
      bi3=_mm_loadu_ps(bin3);bin3+=4;bi4=_mm_loadu_ps(bin4);bin4+=4;
      bt1=_mm_mul_ps(bi1,bb);bt2=_mm_mul_ps(bi2,bb);bt3=_mm_mul_ps(bi3,bb);bt4=_mm_mul_ps(bi4,bb);
      _mm_store_ps(bout,bt1);_mm_store_ps(bout+4,bt2);_mm_store_ps(bout+8,bt3);_mm_store_ps(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);*(bout+2)=*(bin1+2)*(*alpha);bin1+=3;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);*(bout+6)=*(bin2+2)*(*alpha);bin2+=3;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);*(bout+10)=*(bin3+2)*(*alpha);bin3+=3;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);*(bout+14)=*(bin4+2)*(*alpha);bin4+=3;
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*BlkUnitK;brow+=4){
    bout=bblk+brow*4+(BlkUnitN-1)*4*BlkDimK+2;
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);bin1+=2;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);bin2+=2;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);bin3+=2;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);bin4+=2;
    bout=bblk+brow*4;
    for(bcol=1;bcol<BlkUnitN;bcol++){
      bi1=_mm_loadu_ps(bin1);bin1+=4;bi2=_mm_loadu_ps(bin2);bin2+=4;
      bi3=_mm_loadu_ps(bin3);bin3+=4;bi4=_mm_loadu_ps(bin4);bin4+=4;
      bt1=_mm_mul_ps(bi1,bb);bt2=_mm_mul_ps(bi2,bb);bt3=_mm_mul_ps(bi3,bb);bt4=_mm_mul_ps(bi4,bb);
      _mm_store_ps(bout,bt1);_mm_store_ps(bout+4,bt2);_mm_store_ps(bout+8,bt3);_mm_store_ps(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);bin1+=2;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);bin2+=2;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);bin3+=2;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);bin4+=2;
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<4*BlkUnitK;brow+=4){
    bout=bblk+brow*4+(BlkUnitN-1)*4*BlkDimK+1;
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);*(bout+2)=*(bin1+2)*(*alpha);bin1+=3;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);*(bout+6)=*(bin2+2)*(*alpha);bin2+=3;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);*(bout+10)=*(bin3+2)*(*alpha);bin3+=3;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);*(bout+14)=*(bin4+2)*(*alpha);bin4+=3;
    bout=bblk+brow*4;
    for(bcol=1;bcol<BlkUnitN;bcol++){
      bi1=_mm_loadu_ps(bin1);bin1+=4;bi2=_mm_loadu_ps(bin2);bin2+=4;
      bi3=_mm_loadu_ps(bin3);bin3+=4;bi4=_mm_loadu_ps(bin4);bin4+=4;
      bt1=_mm_mul_ps(bi1,bb);bt2=_mm_mul_ps(bi2,bb);bt3=_mm_mul_ps(bi3,bb);bt4=_mm_mul_ps(bi4,bb);
      _mm_store_ps(bout,bt1);_mm_store_ps(bout+4,bt2);_mm_store_ps(bout+8,bt3);_mm_store_ps(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    *(bout+0)=*bin1*(*alpha);bin1++;
    *(bout+4)=*bin2*(*alpha);bin2++;
    *(bout+8)=*bin3*(*alpha);bin3++;
    *(bout+12)=*bin4*(*alpha);bin4++;
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
void load_irreg_b_c(float *bstartpos,float *bblk,int ldb,int ndim,int kdim,float *alpha){//dense rearr(old) lazy mode
  float *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;bout=bblk;
  for(bcol=0;bcol<ndim-3;bcol+=4){
    for(brow=0;brow<kdim;brow++){
      *bout=*bin1*(*alpha);bin1++;bout++;
      *bout=*bin2*(*alpha);bin2++;bout++;
      *bout=*bin3*(*alpha);bin3++;bout++;
      *bout=*bin4*(*alpha);bin4++;bout++;
    }
    bin1+=4*ldb-kdim;
    bin2+=4*ldb-kdim;
    bin3+=4*ldb-kdim;
    bin4+=4*ldb-kdim;
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      *bout=*bin1*(*alpha);bin1++;bout++;
    }
    bin1+=ldb-kdim;
  }
}
void load_irreg_b_r(float *bstartpos,float *bblk,int ldb,int ndim,int kdim,float *alpha){//dense rearr(old) lazy mode
  float *bin,*bout;int bcol,brow;register __m128 btmp,bmul;
  bin=bstartpos;bmul=_mm_broadcast_ss(alpha);
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*4;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      btmp=_mm_loadu_ps(bin);
      btmp=_mm_mul_ps(btmp,bmul);
      _mm_storeu_ps(bout,btmp);
      bin+=4;bout+=4*kdim;
    }
    bout-=3*brow;
    for(;bcol<ndim;bcol++){
      *bout=*bin*(*alpha);bin++;bout+=kdim;
    }
    bin+=ldb-ndim;
  }
}
void sgemmblkirregkccc(float *ablk,float *bblk,float *cstartpos,int ldc,int kdim,float *beta){
  register __m256 a1,a2,a3,b1,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;float *atemp,*btemp,*ctemp,*cpref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<BlkDimN;ccol+=4){//loop over cblk-columns, calculate 4 columns of cblk in each iteration.
   cpref=cstartpos+(ccol+4)%BlkDimN*ldc+(ccol+4)/BlkDimN*BlkDimM;
   b1=_mm256_broadcast_ss(beta);
   c1=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c2=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c3=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c4=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c5=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c6=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c7=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c8=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c9=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c10=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c11=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c12=_mm256_mul_ps(b1,_mm256_loadu_ps(ctemp));ctemp-=3*ldc+16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   atemp=ablk;
   for(acol=0;acol<kdim;acol++){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    a1=_mm256_load_ps(atemp);atemp+=8;
    a2=_mm256_load_ps(atemp);atemp+=8;
    a3=_mm256_load_ps(atemp);atemp+=8;
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c1=_mm256_fmadd_ps(a1,b1,c1);c2=_mm256_fmadd_ps(a2,b1,c2);c3=_mm256_fmadd_ps(a3,b1,c3);
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c4=_mm256_fmadd_ps(a1,b1,c4);c5=_mm256_fmadd_ps(a2,b1,c5);c6=_mm256_fmadd_ps(a3,b1,c6);
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c7=_mm256_fmadd_ps(a1,b1,c7);c8=_mm256_fmadd_ps(a2,b1,c8);c9=_mm256_fmadd_ps(a3,b1,c9);
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c10=_mm256_fmadd_ps(a1,b1,c10);c11=_mm256_fmadd_ps(a2,b1,c11);c12=_mm256_fmadd_ps(a3,b1,c12);
   }
   _mm256_storeu_ps(ctemp,c1);ctemp+=8;_mm256_storeu_ps(ctemp,c2);ctemp+=8;_mm256_storeu_ps(ctemp,c3);ctemp+=ldc-16;
   _mm256_storeu_ps(ctemp,c4);ctemp+=8;_mm256_storeu_ps(ctemp,c5);ctemp+=8;_mm256_storeu_ps(ctemp,c6);ctemp+=ldc-16;
   _mm256_storeu_ps(ctemp,c7);ctemp+=8;_mm256_storeu_ps(ctemp,c8);ctemp+=8;_mm256_storeu_ps(ctemp,c9);ctemp+=ldc-16;
   _mm256_storeu_ps(ctemp,c10);ctemp+=8;_mm256_storeu_ps(ctemp,c11);ctemp+=8;_mm256_storeu_ps(ctemp,c12);ctemp+=ldc-16;
  }
}
#define KERNELm24n4k2 {\
    a1=_mm256_load_ps(atemp);atemp+=8;   _mm_prefetch((char *)(atemp+64),_MM_HINT_T0);\
    a2=_mm256_load_ps(atemp);atemp+=8;\
    a3=_mm256_load_ps(atemp);atemp+=8;   _mm_prefetch((char *)(btemp+128),_MM_HINT_T0);\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c1=_mm256_fmadd_ps(a1,b1,c1);c2=_mm256_fmadd_ps(a2,b1,c2);c3=_mm256_fmadd_ps(a3,b1,c3);\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c4=_mm256_fmadd_ps(a1,b1,c4);c5=_mm256_fmadd_ps(a2,b1,c5);c6=_mm256_fmadd_ps(a3,b1,c6);\
    b1=_mm256_broadcast_ss(btemp);btemp++;   _mm_prefetch((char *)(atemp+64),_MM_HINT_T0);\
    c7=_mm256_fmadd_ps(a1,b1,c7);c8=_mm256_fmadd_ps(a2,b1,c8);c9=_mm256_fmadd_ps(a3,b1,c9);\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c10=_mm256_fmadd_ps(a1,b1,c10);c11=_mm256_fmadd_ps(a2,b1,c11);c12=_mm256_fmadd_ps(a3,b1,c12);\
    a1=_mm256_load_ps(atemp);atemp+=8;\
    a2=_mm256_load_ps(atemp);atemp+=8;   _mm_prefetch((char *)(atemp+64),_MM_HINT_T0);\
    a3=_mm256_load_ps(atemp);atemp+=8;\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c1=_mm256_fmadd_ps(a1,b1,c1);c2=_mm256_fmadd_ps(a2,b1,c2);c3=_mm256_fmadd_ps(a3,b1,c3);\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c4=_mm256_fmadd_ps(a1,b1,c4);c5=_mm256_fmadd_ps(a2,b1,c5);c6=_mm256_fmadd_ps(a3,b1,c6);\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c7=_mm256_fmadd_ps(a1,b1,c7);c8=_mm256_fmadd_ps(a2,b1,c8);c9=_mm256_fmadd_ps(a3,b1,c9);\
    b1=_mm256_broadcast_ss(btemp);btemp++;\
    c10=_mm256_fmadd_ps(a1,b1,c10);c11=_mm256_fmadd_ps(a2,b1,c11);c12=_mm256_fmadd_ps(a3,b1,c12);\
}
void sgemmblkirregnccc(float *ablk,float *bblk,float *cstartpos,int ldc,int ndim){
  register __m256 a1,a2,a3,b1,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  float *atemp,*btemp,*ctemp,*cpref,*apref;int ccol,acol;
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<ndim-3;ccol+=4){//loop over cblk-columns, calculate 5 columns of cblk in each iteration.
   cpref=cstartpos+(ccol+4)%ndim*ldc+(ccol+4)/ndim*BlkDimM;
   c1=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c2=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c3=_mm256_loadu_ps(ctemp);ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c4=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c5=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c6=_mm256_loadu_ps(ctemp);ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c7=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c8=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c9=_mm256_loadu_ps(ctemp);ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c10=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c11=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c12=_mm256_loadu_ps(ctemp);ctemp-=3*ldc+16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol+=8){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    KERNELm24n4k2
    KERNELm24n4k2
    KERNELm24n4k2
    KERNELm24n4k2
   }
   _mm256_storeu_ps(ctemp,c1);ctemp+=8;_mm256_storeu_ps(ctemp,c2);ctemp+=8;_mm256_storeu_ps(ctemp,c3);ctemp+=ldc-16;
   _mm256_storeu_ps(ctemp,c4);ctemp+=8;_mm256_storeu_ps(ctemp,c5);ctemp+=8;_mm256_storeu_ps(ctemp,c6);ctemp+=ldc-16;
   _mm256_storeu_ps(ctemp,c7);ctemp+=8;_mm256_storeu_ps(ctemp,c8);ctemp+=8;_mm256_storeu_ps(ctemp,c9);ctemp+=ldc-16;
   _mm256_storeu_ps(ctemp,c10);ctemp+=8;_mm256_storeu_ps(ctemp,c11);ctemp+=8;_mm256_storeu_ps(ctemp,c12);ctemp+=ldc-16;
  }
  cpref=cstartpos+BlkDimM;
  for(;ccol<ndim;ccol++){
   c1=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c2=_mm256_loadu_ps(ctemp);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c3=_mm256_loadu_ps(ctemp);ctemp-=16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol++){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    a1=_mm256_load_ps(atemp);atemp+=8;
    a2=_mm256_load_ps(atemp);atemp+=8;
    a3=_mm256_load_ps(atemp);atemp+=8;
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c1=_mm256_fmadd_ps(a1,b1,c1);c2=_mm256_fmadd_ps(a2,b1,c2);c3=_mm256_fmadd_ps(a3,b1,c3);
   }
   _mm256_storeu_ps(ctemp,c1);ctemp+=8;_mm256_storeu_ps(ctemp,c2);ctemp+=8;_mm256_storeu_ps(ctemp,c3);ctemp+=ldc-16;
  }
}
void sgemmblkirregccc(float *ablk,float *bblk,float *cstartpos,int ldc,int mdim,int ndim,int kdim,float *beta){
  register __m256 a1,a2,a3,b1,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;__m256i ml1,ml2,ml3;
  float *atemp,*btemp,*ctemp,*cpref,*apref;int ccol,acol;
  ml1=_mm256_setr_epi32(-(mdim>0),-(mdim>1),-(mdim>2),-(mdim>3),-(mdim>4),-(mdim>5),-(mdim>6),-(mdim>7));
  ml2=_mm256_setr_epi32(-(mdim>8),-(mdim>9),-(mdim>10),-(mdim>11),-(mdim>12),-(mdim>13),-(mdim>14),-(mdim>15));
  ml3=_mm256_setr_epi32(-(mdim>16),-(mdim>17),-(mdim>18),-(mdim>19),-(mdim>20),-(mdim>21),-(mdim>22),-(mdim>23));
  ctemp=cstartpos;btemp=bblk;
  for(ccol=0;ccol<ndim-3;ccol+=4){//loop over cblk-columns, calculate 5 columns of cblk in each iteration.
   cpref=cstartpos+(ccol+4)%ndim*ldc+(ccol+4)/ndim*BlkDimM;
   b1=_mm256_broadcast_ss(beta);
   c1=_mm256_maskload_ps(ctemp,ml1);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c2=_mm256_maskload_ps(ctemp,ml2);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c3=_mm256_maskload_ps(ctemp,ml3);ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c1=_mm256_mul_ps(c1,b1);c2=_mm256_mul_ps(c2,b1);c3=_mm256_mul_ps(c3,b1);
   c4=_mm256_maskload_ps(ctemp,ml1);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c5=_mm256_maskload_ps(ctemp,ml2);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c6=_mm256_maskload_ps(ctemp,ml3);ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c4=_mm256_mul_ps(c4,b1);c5=_mm256_mul_ps(c5,b1);c6=_mm256_mul_ps(c6,b1);
   c7=_mm256_maskload_ps(ctemp,ml1);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c8=_mm256_maskload_ps(ctemp,ml2);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c9=_mm256_maskload_ps(ctemp,ml3);ctemp+=ldc-16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c7=_mm256_mul_ps(c7,b1);c8=_mm256_mul_ps(c8,b1);c9=_mm256_mul_ps(c9,b1);
   c10=_mm256_maskload_ps(ctemp,ml1);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c11=_mm256_maskload_ps(ctemp,ml2);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c12=_mm256_maskload_ps(ctemp,ml3);ctemp-=3*ldc+16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c10=_mm256_mul_ps(c10,b1);c11=_mm256_mul_ps(c11,b1);c12=_mm256_mul_ps(c12,b1);
   atemp=ablk;//bpref=bblk+BlkDimK*((i+5)%BlkDimN);
   for(acol=0;acol<kdim;acol++){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    a1=_mm256_load_ps(atemp);atemp+=8;
    a2=_mm256_load_ps(atemp);atemp+=8;   _mm_prefetch((char *)(btemp+128),_MM_HINT_T0);
    a3=_mm256_load_ps(atemp);atemp+=8;
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c1=_mm256_fmadd_ps(a1,b1,c1);c2=_mm256_fmadd_ps(a2,b1,c2);c3=_mm256_fmadd_ps(a3,b1,c3);
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c4=_mm256_fmadd_ps(a1,b1,c4);c5=_mm256_fmadd_ps(a2,b1,c5);c6=_mm256_fmadd_ps(a3,b1,c6);
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c7=_mm256_fmadd_ps(a1,b1,c7);c8=_mm256_fmadd_ps(a2,b1,c8);c9=_mm256_fmadd_ps(a3,b1,c9);
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c10=_mm256_fmadd_ps(a1,b1,c10);c11=_mm256_fmadd_ps(a2,b1,c11);c12=_mm256_fmadd_ps(a3,b1,c12);
   }
   _mm256_maskstore_ps(ctemp,ml1,c1);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml2,c2);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml3,c3);ctemp+=ldc-16;
   _mm256_maskstore_ps(ctemp,ml1,c4);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml2,c5);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml3,c6);ctemp+=ldc-16;
   _mm256_maskstore_ps(ctemp,ml1,c7);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml2,c8);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml3,c9);ctemp+=ldc-16;
   _mm256_maskstore_ps(ctemp,ml1,c10);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml2,c11);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml3,c12);ctemp+=ldc-16;
  }
  cpref=cstartpos+BlkDimM;
  c9=_mm256_broadcast_ss(beta);
  for(;ccol<ndim;ccol++){
   c1=_mm256_maskload_ps(ctemp,ml1);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=16;
   c2=_mm256_maskload_ps(ctemp,ml2);ctemp+=8;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=7;
   c3=_mm256_maskload_ps(ctemp,ml3);ctemp-=16;_mm_prefetch((char *)cpref,_MM_HINT_T0);cpref+=ldc-23;
   c1=_mm256_mul_ps(c1,c9);c2=_mm256_mul_ps(c2,c9);c3=_mm256_mul_ps(c3,c9);
   atemp=ablk;
   for(acol=0;acol<kdim;acol++){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    a1=_mm256_load_ps(atemp);atemp+=8;
    a2=_mm256_load_ps(atemp);atemp+=8;
    a3=_mm256_load_ps(atemp);atemp+=8;
    b1=_mm256_broadcast_ss(btemp);btemp++;
    c1=_mm256_fmadd_ps(a1,b1,c1);c2=_mm256_fmadd_ps(a2,b1,c2);c3=_mm256_fmadd_ps(a3,b1,c3);
   }
   _mm256_maskstore_ps(ctemp,ml1,c1);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml2,c2);ctemp+=8;
   _mm256_maskstore_ps(ctemp,ml3,c3);ctemp+=ldc-16;
  }
}
void synproc(int tid,int threads,int *workprogress){//workprogress[] must be shared among all threads
  int waitothers,ctid,temp;
  workprogress[16*tid]++;
  temp=workprogress[16*tid];
  for(waitothers=1;waitothers;timedelay()){
    waitothers=0;
    for(ctid=0;ctid<threads;ctid++){
      if(workprogress[16*ctid]<temp) waitothers = 1;
    }
  }
}//this function is for synchronization of threads before/after load_abuffer
void load_abuffer_ac(float *aheadpos,float *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*BlkDimK,LDA);
  load_tail_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*BlkDimK,LDA,EdgeM);
}
void load_abuffer_ar(float *aheadpos,float *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*BlkDimK,LDA);
  load_tail_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*BlkDimK,LDA,EdgeM);
}
void load_abuffer_irregk_ac(float *aheadpos,float *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*kdim,LDA,kdim);
  load_irreg_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*kdim,LDA,EdgeM,kdim);
}
void load_abuffer_irregk_ar(float *aheadpos,float *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*kdim,LDA,kdim);
  load_irreg_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*kdim,LDA,EdgeM,kdim);
}
void sgemmcolumn(float *abuffer,float *bblk,float *cheadpos,int BlksM,int EdgeM,int LDC){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    sgemmblkregccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC);
    MCT+=BlkDimM;
  }
  sgemmblktailccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,EdgeM);
}
void sgemmcolumnirregn(float *abuffer,float *bblk,float *cheadpos,int BlksM,int EdgeM,int LDC,int ndim){
  int MCT=0;int BlkCtM;float beta=1.0;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    sgemmblkirregnccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,ndim);
    MCT+=BlkDimM;
  }
  sgemmblkirregccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,EdgeM,ndim,BlkDimK,&beta);
}
void sgemmcolumnirregk(float *abuffer,float *bblk,float *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,float *beta){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    sgemmblkirregkccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,kdim,beta);
    MCT+=BlkDimM;
  }
  sgemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,EdgeM,BlkDimN,kdim,beta);
}
void sgemmcolumnirreg(float *abuffer,float *bblk,float *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,int ndim,float *beta){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    sgemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,BlkDimM,ndim,kdim,beta);
    MCT+=BlkDimM;
  }
  sgemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,EdgeM,ndim,kdim,beta);
}
void cmultbeta(float *c,int ldc,int m,int n,float beta){
  int i,j;float *C0,*C;
  C0=c;
  for(i=0;i<n;i++){
    C=C0;
    for(j=0;j<m;j++){
      *C*=beta;C++;
    }
    C0+=ldc;
  }
}
void sgemm(char *transa,char *transb,int *m,int *n,int *k,float *alpha,float *a,int *lda,float *bstart,int *ldb,float *beta,float *cstart,int *ldc){
//assume column-major storage with arguments passed by addresses (FORTRAN style)
//a:matrix with m rows and k columns if transa=N
//b:matrix with k rows and n columns if transb=N
//c:product matrix with m rows and n columns
 const int M = *m;/* const int N = *n; */const int K = *k;
 float BETA = 1.0;
 const int LDA = *lda;const int LDB = *ldb;const int LDC=*ldc;
 const char TRANSA = *transa;const char TRANSB = *transb;
 const int BlksM = (M-1)/BlkDimM+1;const int EdgeM = M-(BlksM-1)*BlkDimM;//the m-dim of edges
 const int BlksK = (K-1)/BlkDimK+1;const int EdgeK = K-(BlksK-1)*BlkDimK;//the k-dim of edges
 int *workprogress, *cchunks;const int numthreads=omp_get_max_threads();int i; //for parallel execution
 //cchunk[] for dividing tasks, workprogress[] for recording the progresses of all threads and synchronization.
 //synchronization is necessary here since abuffer[] is shared between threads.
 //if abuffer[] is thread-private, the bandwidth of memory will limit the performance.
 //synchronization by openmp functions can be expensive, so handcoded funcion (synproc) is used instead.
 float *abuffer; //abuffer[]: store 256 columns of matrix a
 if((*alpha) == 0.0 && (*beta) != 1.0) cmultbeta(cstart,LDC,M,(*n),(*beta));//limited by memory bendwidth so no need for parallel execution
 if((*alpha) != 0.0){//then do C=alpha*AB+beta*C
  abuffer = (float *)aligned_alloc(4096,(BlkDimM*BlkDimK*BlksM)*sizeof(float));
  workprogress = (int *)calloc(20*numthreads,sizeof(int));
  cchunks = (int *)malloc((numthreads+1)*sizeof(int));
  for(i=0;i<=numthreads;i++) cchunks[i]=(*n)*i/numthreads;
#pragma omp parallel
 {
  int tid = omp_get_thread_num();
  float *c = cstart + LDC * cchunks[tid];
  float *b;
  if(TRANSB=='N' || TRANSB=='n') b = bstart + LDB * cchunks[tid];
  else b = bstart + cchunks[tid];
  const int N = cchunks[tid+1]-cchunks[tid];
  const int BlksN = (N-1)/BlkDimN+1; const int EdgeN = N-(BlksN-1)*BlkDimN;//the n-dim of edges
  int BlkCtM,BlkCtN,BlkCtK,MCT,NCT,KCT;//loop counters over blocks
  //MCT,NCT and KCT are used to locate the current position of matrix blocks
  float *bblk = (float *)aligned_alloc(4096,(BlkDimN*BlkDimK)*sizeof(float)); //thread-private bblk[]
  if(TRANSA=='N' || TRANSA=='n'){
   if(TRANSB=='N' || TRANSB=='n'){//CASE NN
    if(tid==0) load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK); //only the master thread can write abuffer
    synproc(tid,numthreads,workprogress); //before the calculations, child threads need to wait here until the master finish writing abuffer
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_c(b+NCT*LDB,bblk,LDB,BlkDimN,EdgeK,alpha);
     sgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    sgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);//before updating abuffer, the master thread need to wait here until all child threads finish calculation with current abuffer
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0) load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM); //only the master thread can write abuffer
     synproc(tid,numthreads,workprogress);//before the calculations, child threads need to wait here until the master finish writing abuffer
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      sgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     sgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     synproc(tid,numthreads,workprogress);//before updating abuffer, the master thread need to wait here until all child threads finish calculation with current abuffer
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
   else{//CASE NY
    if(tid==0) load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    synproc(tid,numthreads,workprogress);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_r(b+NCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     sgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    sgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0) load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      sgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     sgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     synproc(tid,numthreads,workprogress);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
  }
  else{
   if(TRANSB=='N' || TRANSB=='n'){//case YN
    if(tid==0) load_abuffer_irregk_ar(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    synproc(tid,numthreads,workprogress);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_c(b+NCT*LDB,bblk,LDB,BlkDimN,EdgeK,alpha);
     sgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    sgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     if(tid==0) load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      sgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     sgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     synproc(tid,numthreads,workprogress);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
   else{//case YY
    if(tid==0) load_abuffer_irregk_ar(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    synproc(tid,numthreads,workprogress);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_r(b+NCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     sgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    sgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     if(tid==0) load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      sgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     sgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     synproc(tid,numthreads,workprogress);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
  }
  free(bblk);bblk=NULL;
 }//out of openmp region
  free(cchunks);cchunks=NULL;
  free(workprogress);workprogress=NULL;
  free(abuffer);abuffer=NULL;
 }
}
