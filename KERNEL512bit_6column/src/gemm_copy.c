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
#define bc_copy_1row {\
   *(outb+0)=(*inb1)*ALPHA;inb1++;\
   *(outb+1)=(*inb2)*ALPHA;inb2++;\
   *(outb+2)=(*inb3)*ALPHA;inb3++;\
   *(outb+3)=(*inb4)*ALPHA;inb4++;\
   *(outb+4)=(*inb5)*ALPHA;inb5++;\
   *(outb+5)=(*inb6)*ALPHA;inb6++;\
   outb+=6;\
}
static void load_reg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
 FLOAT *inb1,*inb2,*inb3,*inb4,*inb5,*inb6,*outb;FLOAT ALPHA=*alpha;
 int bcol,brow;
 outb=bblk;
 inb1=bstartpos;
 inb2=inb1+ldb;
 inb3=inb2+ldb;
 inb4=inb3+ldb;
 inb5=inb4+ldb;
 inb6=inb5+ldb;
 for(bcol=0;bcol<(BlkDimN/6);bcol++){
  for(brow=0;brow<(BlkDimK/6);brow++) bc_copy_1row
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;inb5+=ldb;inb6+=ldb;
  inb6-=(bcol==(BlkDimN/6)-1)*(ldb*BlkDimN);
  for(;brow<2*(BlkDimK/6);brow++) bc_copy_1row
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;inb5+=ldb;inb6+=ldb;
  inb5-=(bcol==(BlkDimN/6)-1)*(ldb*BlkDimN);
  for(;brow<3*(BlkDimK/6);brow++) bc_copy_1row
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;inb5+=ldb;inb6+=ldb;
  inb4-=(bcol==(BlkDimN/6)-1)*(ldb*BlkDimN);
  for(;brow<4*(BlkDimK/6);brow++) bc_copy_1row
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;inb5+=ldb;inb6+=ldb;
  inb3-=(bcol==(BlkDimN/6)-1)*(ldb*BlkDimN);
  for(;brow<5*(BlkDimK/6);brow++) bc_copy_1row
  inb1+=ldb;inb2+=ldb;inb3+=ldb;inb4+=ldb;inb5+=ldb;inb6+=ldb;
  inb2-=(bcol==(BlkDimN/6)-1)*(ldb*BlkDimN);
  for(;brow<BlkDimK;brow++) bc_copy_1row
  inb1+=ldb-BlkDimK;
  inb2+=ldb-BlkDimK;
  inb3+=ldb-BlkDimK;
  inb4+=ldb-BlkDimK;
  inb5+=ldb-BlkDimK;
  inb6+=ldb-BlkDimK;
 }
}
#define bcopy_6col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bout[2]=bin1[2]*ALPHA;bout[3]=bin1[3]*ALPHA;bout[4]=bin1[4]*ALPHA;bout[5]=bin1[5]*ALPHA;bin1+=6;\
  bout[6]=bin2[0]*ALPHA;bout[7]=bin2[1]*ALPHA;bout[8]=bin2[2]*ALPHA;bout[9]=bin2[3]*ALPHA;bout[10]=bin2[4]*ALPHA;bout[11]=bin2[5]*ALPHA;bin2+=6;\
  bout[12]=bin3[0]*ALPHA;bout[13]=bin3[1]*ALPHA;bout[14]=bin3[2]*ALPHA;bout[15]=bin3[3]*ALPHA;bout[16]=bin3[4]*ALPHA;bout[17]=bin3[5]*ALPHA;bin3+=6;\
  bout[18]=bin4[0]*ALPHA;bout[19]=bin4[1]*ALPHA;bout[20]=bin4[2]*ALPHA;bout[21]=bin4[3]*ALPHA;bout[22]=bin4[4]*ALPHA;bout[23]=bin4[5]*ALPHA;bin4+=6;\
  bout+=6*BlkDimK;\
}
#define bcopy_5col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bout[2]=bin1[2]*ALPHA;bout[3]=bin1[3]*ALPHA;bout[4]=bin1[4]*ALPHA;bin1+=5;\
  bout[6]=bin2[0]*ALPHA;bout[7]=bin2[1]*ALPHA;bout[8]=bin2[2]*ALPHA;bout[9]=bin2[3]*ALPHA;bout[10]=bin2[4]*ALPHA;bin2+=5;\
  bout[12]=bin3[0]*ALPHA;bout[13]=bin3[1]*ALPHA;bout[14]=bin3[2]*ALPHA;bout[15]=bin3[3]*ALPHA;bout[16]=bin3[4]*ALPHA;bin3+=5;\
  bout[18]=bin4[0]*ALPHA;bout[19]=bin4[1]*ALPHA;bout[20]=bin4[2]*ALPHA;bout[21]=bin4[3]*ALPHA;bout[22]=bin4[4]*ALPHA;bin4+=5;\
}
#define bcopy_4col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bout[2]=bin1[2]*ALPHA;bout[3]=bin1[3]*ALPHA;bin1+=4;\
  bout[6]=bin2[0]*ALPHA;bout[7]=bin2[1]*ALPHA;bout[8]=bin2[2]*ALPHA;bout[9]=bin2[3]*ALPHA;bin2+=4;\
  bout[12]=bin3[0]*ALPHA;bout[13]=bin3[1]*ALPHA;bout[14]=bin3[2]*ALPHA;bout[15]=bin3[3]*ALPHA;bin3+=4;\
  bout[18]=bin4[0]*ALPHA;bout[19]=bin4[1]*ALPHA;bout[20]=bin4[2]*ALPHA;bout[21]=bin4[3]*ALPHA;bin4+=4;\
}
#define bcopy_3col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bout[2]=bin1[2]*ALPHA;bin1+=3;\
  bout[6]=bin2[0]*ALPHA;bout[7]=bin2[1]*ALPHA;bout[8]=bin2[2]*ALPHA;bin2+=3;\
  bout[12]=bin3[0]*ALPHA;bout[13]=bin3[1]*ALPHA;bout[14]=bin3[2]*ALPHA;bin3+=3;\
  bout[18]=bin4[0]*ALPHA;bout[19]=bin4[1]*ALPHA;bout[20]=bin4[2]*ALPHA;bin4+=3;\
}
#define bcopy_2col {\
  bout[0]=bin1[0]*ALPHA;bout[1]=bin1[1]*ALPHA;bin1+=2;\
  bout[6]=bin2[0]*ALPHA;bout[7]=bin2[1]*ALPHA;bin2+=2;\
  bout[12]=bin3[0]*ALPHA;bout[13]=bin3[1]*ALPHA;bin3+=2;\
  bout[18]=bin4[0]*ALPHA;bout[19]=bin4[1]*ALPHA;bin4+=2;\
}
#define bcopy_1col {\
  bout[0]=bin1[0]*ALPHA;bin1++;\
  bout[6]=bin2[0]*ALPHA;bin2++;\
  bout[12]=bin3[0]*ALPHA;bin3++;\
  bout[18]=bin4[0]*ALPHA;bin4++;\
}
static void load_reg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;FLOAT ALPHA=*alpha;
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;int bshift=4*ldb-BlkDimN;
  for(brow=0;brow<(BlkDimK/6);brow+=4){
    bout=bblk+brow*6;
    for(bcol=0;bcol<(BlkDimN/6);bcol++) bcopy_6col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*(BlkDimK/6);brow+=4){
    bout=bblk+brow*6+((BlkDimN/6)-1)*6*BlkDimK+5;
    bcopy_1col
    bout=bblk+brow*6;
    for(bcol=1;bcol<(BlkDimN/6);bcol++) bcopy_6col
    bcopy_5col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*(BlkDimK/6);brow+=4){
    bout=bblk+brow*6+((BlkDimN/6)-1)*6*BlkDimK+4;
    bcopy_2col
    bout=bblk+brow*6;
    for(bcol=1;bcol<(BlkDimN/6);bcol++) bcopy_6col
    bcopy_4col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<4*(BlkDimK/6);brow+=4){
    bout=bblk+brow*6+((BlkDimN/6)-1)*6*BlkDimK+3;
    bcopy_3col
    bout=bblk+brow*6;
    for(bcol=1;bcol<(BlkDimN/6);bcol++) bcopy_6col
    bcopy_3col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<5*(BlkDimK/6);brow+=4){
    bout=bblk+brow*6+((BlkDimN/6)-1)*6*BlkDimK+2;
    bcopy_4col
    bout=bblk+brow*6;
    for(bcol=1;bcol<(BlkDimN/6);bcol++) bcopy_6col
    bcopy_2col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<BlkDimK;brow+=4){
    bout=bblk+brow*6+((BlkDimN/6)-1)*6*BlkDimK+1;
    bcopy_5col
    bout=bblk+brow*6;
    for(bcol=1;bcol<(BlkDimN/6);bcol++) bcopy_6col
    bcopy_1col
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_irreg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin1,*bin2,*bin3,*bin4,*bin5,*bin6,*bout;int bcol,brow;FLOAT ALPHA=*alpha;
  bin1=bstartpos;bin2=bin1+ldb;bin3=bin2+ldb;bin4=bin3+ldb;bin5=bin4+ldb;bin6=bin5+ldb;bout=bblk;
  for(bcol=0;bcol<ndim-5;bcol+=6){
    for(brow=0;brow<kdim;brow++){
      *bout=(*bin1)*ALPHA;bin1++;bout++;
      *bout=(*bin2)*ALPHA;bin2++;bout++;
      *bout=(*bin3)*ALPHA;bin3++;bout++;
      *bout=(*bin4)*ALPHA;bin4++;bout++;
      *bout=(*bin5)*ALPHA;bin5++;bout++;
      *bout=(*bin6)*ALPHA;bin6++;bout++;
    }
    bin1+=6*ldb-kdim;
    bin2+=6*ldb-kdim;
    bin3+=6*ldb-kdim;
    bin4+=6*ldb-kdim;
    bin5+=6*ldb-kdim;
    bin6+=6*ldb-kdim;
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      *bout=(*bin1)*ALPHA;bin1++;bout++;
    }
    bin1+=ldb-kdim;
  }
}
static void load_irreg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout;int bcol,brow;FLOAT ALPHA=*alpha;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*6;
    for(bcol=0;bcol<ndim-5;bcol+=6){
      bout[0]=bin[0]*ALPHA;
      bout[1]=bin[1]*ALPHA;
      bout[2]=bin[2]*ALPHA;
      bout[3]=bin[3]*ALPHA;
      bout[4]=bin[4]*ALPHA;
      bout[5]=bin[5]*ALPHA;
      bin+=6;bout+=6*kdim;
    }
    bout-=5*brow;
    for(;bcol<ndim;bcol++){
      *bout=(*bin)*ALPHA;bin++;bout+=kdim;
    }
    bin+=ldb-ndim;
  }
}
