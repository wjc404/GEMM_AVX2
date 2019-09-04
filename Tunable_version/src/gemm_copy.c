static void load_irreg_a_c(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
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
static void load_irreg_a_r(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
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
static void load_reg_a_c(FLOAT *astartpos,FLOAT *ablk,int lda){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,BlkDimK);}
static void load_reg_a_r(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow;FLOAT *ar1,*ar2,*ar3,*ar4,*awrite;
  for(arow=0;arow<BlkDimM;arow+=4){
    ar1=astartpos+arow*lda;
    ar2=ar1+lda;ar3=ar2+lda;ar4=ar3+lda;
    awrite=ablk+arow;
    for(acol=0;acol<BlkDimK;acol++){
      *(awrite+0)=*(ar1+acol);
      *(awrite+1)=*(ar2+acol);
      *(awrite+2)=*(ar3+acol);
      *(awrite+3)=*(ar4+acol);
      awrite+=BlkDimM;
    }
  }
}
static void load_tail_a_c(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_c(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_tail_a_r(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_r(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_irregk_a_c(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,kdim);}
static void load_irregk_a_r(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_r(astartpos,ablk,lda,BlkDimM,kdim);}
static void load_reg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
 FLOAT *inb1,*inb2,*inb3,*inb4,*outb;
 int bcol,brow;
 outb=bblk;
 inb1=bstartpos;
 inb2=inb1+ldb;
 inb3=inb2+ldb;
 inb4=inb3+ldb;
 for(bcol=0;bcol<(BlkDimN/4);bcol++){
  for(brow=0;brow<(BlkDimK/4);brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;
  inb4-=(bcol==(BlkDimN/4)-1)*(ldb*BlkDimN);
  for(;brow<2*(BlkDimK/4);brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;
  inb3-=(bcol==(BlkDimN/4)-1)*(ldb*BlkDimN);
  for(;brow<3*(BlkDimK/4);brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   *(outb+3)=(*inb4)*(*alpha);inb4++;
   outb+=4;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;
  inb2-=(bcol==(BlkDimN/4)-1)*(ldb*BlkDimN);
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
#ifdef DOUBLE
 #define FLOATVEC __m256d
 #define LOADVEC _mm256_loadu_pd
 #define MULVEC _mm256_mul_pd
 #define STOREVEC _mm256_storeu_pd
 #define BROADVEC _mm256_broadcast_sd
#else
 #define FLOATVEC __m128
 #define LOADVEC _mm_loadu_ps
 #define MULVEC _mm_mul_ps
 #define STOREVEC _mm_storeu_ps
 #define BROADVEC _mm_broadcast_ss
#endif
static void load_reg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  register FLOATVEC bi1,bi2,bi3,bi4,bt1,bt2,bt3,bt4,bb;
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
  bb=BROADVEC(alpha);
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;int bshift=4*ldb-BlkDimN;
  for(brow=0;brow<(BlkDimK/4);brow+=4){
    bout=bblk+brow*4;
    for(bcol=0;bcol<(BlkDimN/4);bcol++){
      bi1=LOADVEC(bin1);bin1+=4;bi2=LOADVEC(bin2);bin2+=4;
      bi3=LOADVEC(bin3);bin3+=4;bi4=LOADVEC(bin4);bin4+=4;
      bt1=MULVEC(bi1,bb);bt2=MULVEC(bi2,bb);bt3=MULVEC(bi3,bb);bt4=MULVEC(bi4,bb);
      STOREVEC(bout,bt1);STOREVEC(bout+4,bt2);STOREVEC(bout+8,bt3);STOREVEC(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*(BlkDimK/4);brow+=4){
    bout=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+3;
    *(bout+0)=*bin1*(*alpha);bin1++;
    *(bout+4)=*bin2*(*alpha);bin2++;
    *(bout+8)=*bin3*(*alpha);bin3++;
    *(bout+12)=*bin4*(*alpha);bin4++;
    bout=bblk+brow*4;
    for(bcol=1;bcol<(BlkDimN/4);bcol++){
      bi1=LOADVEC(bin1);bin1+=4;bi2=LOADVEC(bin2);bin2+=4;
      bi3=LOADVEC(bin3);bin3+=4;bi4=LOADVEC(bin4);bin4+=4;
      bt1=MULVEC(bi1,bb);bt2=MULVEC(bi2,bb);bt3=MULVEC(bi3,bb);bt4=MULVEC(bi4,bb);
      STOREVEC(bout,bt1);STOREVEC(bout+4,bt2);STOREVEC(bout+8,bt3);STOREVEC(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);*(bout+2)=*(bin1+2)*(*alpha);bin1+=3;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);*(bout+6)=*(bin2+2)*(*alpha);bin2+=3;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);*(bout+10)=*(bin3+2)*(*alpha);bin3+=3;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);*(bout+14)=*(bin4+2)*(*alpha);bin4+=3;
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*(BlkDimK/4);brow+=4){
    bout=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+2;
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);bin1+=2;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);bin2+=2;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);bin3+=2;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);bin4+=2;
    bout=bblk+brow*4;
    for(bcol=1;bcol<(BlkDimN/4);bcol++){
      bi1=LOADVEC(bin1);bin1+=4;bi2=LOADVEC(bin2);bin2+=4;
      bi3=LOADVEC(bin3);bin3+=4;bi4=LOADVEC(bin4);bin4+=4;
      bt1=MULVEC(bi1,bb);bt2=MULVEC(bi2,bb);bt3=MULVEC(bi3,bb);bt4=MULVEC(bi4,bb);
      STOREVEC(bout,bt1);STOREVEC(bout+4,bt2);STOREVEC(bout+8,bt3);STOREVEC(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);bin1+=2;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);bin2+=2;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);bin3+=2;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);bin4+=2;
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<4*(BlkDimK/4);brow+=4){
    bout=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+1;
    *(bout+0)=*bin1*(*alpha);*(bout+1)=*(bin1+1)*(*alpha);*(bout+2)=*(bin1+2)*(*alpha);bin1+=3;
    *(bout+4)=*bin2*(*alpha);*(bout+5)=*(bin2+1)*(*alpha);*(bout+6)=*(bin2+2)*(*alpha);bin2+=3;
    *(bout+8)=*bin3*(*alpha);*(bout+9)=*(bin3+1)*(*alpha);*(bout+10)=*(bin3+2)*(*alpha);bin3+=3;
    *(bout+12)=*bin4*(*alpha);*(bout+13)=*(bin4+1)*(*alpha);*(bout+14)=*(bin4+2)*(*alpha);bin4+=3;
    bout=bblk+brow*4;
    for(bcol=1;bcol<(BlkDimN/4);bcol++){
      bi1=LOADVEC(bin1);bin1+=4;bi2=LOADVEC(bin2);bin2+=4;
      bi3=LOADVEC(bin3);bin3+=4;bi4=LOADVEC(bin4);bin4+=4;
      bt1=MULVEC(bi1,bb);bt2=MULVEC(bi2,bb);bt3=MULVEC(bi3,bb);bt4=MULVEC(bi4,bb);
      STOREVEC(bout,bt1);STOREVEC(bout+4,bt2);STOREVEC(bout+8,bt3);STOREVEC(bout+12,bt4);
      bout+=4*BlkDimK;
    }
    *(bout+0)=*bin1*(*alpha);bin1++;
    *(bout+4)=*bin2*(*alpha);bin2++;
    *(bout+8)=*bin3*(*alpha);bin3++;
    *(bout+12)=*bin4*(*alpha);bin4++;
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_irreg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
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
static void load_irreg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;register FLOATVEC btmp,bmul;
  bin=bstartpos;bmul=BROADVEC(alpha);
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*4;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      btmp=LOADVEC(bin);
      btmp=MULVEC(btmp,bmul);
      STOREVEC(bout,btmp);
      bin+=4;bout+=4*kdim;
    }
    bout-=3*brow;
    for(;bcol<ndim;bcol++){
      *bout=*bin*(*alpha);bin++;bout+=kdim;
    }
    bin+=ldb-ndim;
  }
}
