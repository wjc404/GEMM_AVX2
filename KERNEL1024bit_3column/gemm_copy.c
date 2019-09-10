# define ZERO_VALUE 1.0
static void load_irreg_a_c(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
  aread=astartpos;awrite=ablk;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      *(awrite+arow)=*(aread+arow);
    }
    for(;arow<BlkDimM;arow++){
      *(awrite+arow)=ZERO_VALUE;
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
      *(awrite+arow)=ZERO_VALUE;
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
 FLOAT *inb1,*inb2,*inb3,*outb;
 int bcol,brow;
 outb=bblk;
 inb1=bstartpos;
 inb2=inb1+ldb;
 inb3=inb2+ldb;
 for(bcol=0;bcol<(BlkDimN/3);bcol++){
  for(brow=0;brow<(BlkDimK/3);brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   outb+=3;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;
  inb3-=(bcol==(BlkDimN/3)-1)*(ldb*BlkDimN);
  for(;brow<2*(BlkDimK/3);brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   outb+=3;
  }
  inb1+=ldb;inb2+=ldb;inb3+=ldb;
  inb2-=(bcol==(BlkDimN/3)-1)*(ldb*BlkDimN);
  for(;brow<BlkDimK;brow++){
   *(outb+0)=(*inb1)*(*alpha);inb1++;
   *(outb+1)=(*inb2)*(*alpha);inb2++;
   *(outb+2)=(*inb3)*(*alpha);inb3++;
   outb+=3;
  }
  inb1+=ldb-BlkDimK;
  inb2+=ldb-BlkDimK;
  inb3+=ldb-BlkDimK;
 }
}
#define bcopy_3col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bout[2]=bin1[2]*ALPHA;bin1+=3;\
  bout[3]=bin2[0]*ALPHA;bout[4]=bin2[1]*ALPHA;bout[5]=bin2[2]*ALPHA;bin2+=3;\
  bout[6]=bin3[0]*ALPHA;bout[7]=bin3[1]*ALPHA;bout[8]=bin3[2]*ALPHA;bin3+=3;\
  bout[9]=bin4[0]*ALPHA;bout[10]=bin4[1]*ALPHA;bout[11]=bin4[2]*ALPHA;bin4+=3;\
  bout+=3*BlkDimK;\
}
#define bcopy_2col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bin1+=2;\
  bout[3]=bin2[0]*ALPHA;bout[4]=bin2[1]*ALPHA;bin2+=2;\
  bout[6]=bin3[0]*ALPHA;bout[7]=bin3[1]*ALPHA;bin3+=2;\
  bout[9]=bin4[0]*ALPHA;bout[10]=bin4[1]*ALPHA;bin4+=2;\
}
#define bcopy_1col {\
  bout[0]=bin1[0]*ALPHA;bin1++;\
  bout[3]=bin2[0]*ALPHA;bin2++;\
  bout[6]=bin3[0]*ALPHA;bin3++;\
  bout[9]=bin4[0]*ALPHA;bin4++;\
}
static void load_reg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;FLOAT ALPHA=*alpha;
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;int bshift=4*ldb-BlkDimN;
  for(brow=0;brow<(BlkDimK/3);brow+=4){
    bout=bblk+brow*3;
    for(bcol=0;bcol<(BlkDimN/3);bcol++) bcopy_3col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*(BlkDimK/3);brow+=4){
    bout=bblk+brow*3+((BlkDimN/3)-1)*3*BlkDimK+2;
    bcopy_1col
    bout=bblk+brow*3;
    for(bcol=1;bcol<(BlkDimN/3);bcol++) bcopy_3col
    bcopy_2col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<BlkDimK;brow+=4){
    bout=bblk+brow*3+((BlkDimN/3)-1)*3*BlkDimK+1;
    bcopy_2col
    bout=bblk+brow*3;
    for(bcol=1;bcol<(BlkDimN/3);bcol++) bcopy_3col
    bcopy_1col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_irreg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin1,*bin2,*bin3,*bout;int bcol,brow;
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bout=bblk;
  for(bcol=0;bcol<ndim-2;bcol+=3){
    for(brow=0;brow<kdim;brow++){
      *bout=*bin1*(*alpha);bin1++;bout++;
      *bout=*bin2*(*alpha);bin2++;bout++;
      *bout=*bin3*(*alpha);bin3++;bout++;
    }
    bin1+=3*ldb-kdim;
    bin2+=3*ldb-kdim;
    bin3+=3*ldb-kdim;
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      *bout=*bin1*(*alpha);bin1++;bout++;
    }
    bin1+=ldb-kdim;
  }
}
static void load_irreg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;FLOAT ALPHA=*alpha;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*3;
    for(bcol=0;bcol<ndim-2;bcol+=3){
      bout[0]=bin[0]*ALPHA;
      bout[1]=bin[1]*ALPHA;
      bout[2]=bin[2]*ALPHA;
      bin+=3;bout+=3*kdim;
    }
    bout-=2*brow;
    for(;bcol<ndim;bcol++){
      *bout=*bin*ALPHA;bin++;bout+=kdim;
    }
    bin+=ldb-ndim;
  }
}
