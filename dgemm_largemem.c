# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h> //AVX2

# define BlkUnitM 3 //fixed!
# define BlkUnitN 64 //fixed!
# define BlkUnitK BlkUnitN
# define BlkDimM (BlkUnitM*4)
# define BlkDimN (BlkUnitN*4)
# define BlkDimK (BlkUnitK*4)
//compilation command: gcc -fopenmp --shared -fPIC -march=haswell -O3 dgemm_largemem.c dgemm.S -o DGEMM_LARGEMEM.so
//DGEMM_LARGEMEM.so requires more memory space than DGEMM.so, ran faster (~2%) than the latter.
//currently no parallelization

void load_irreg_b_c(double *bstartpos,double *bblk,int ldb,int ndim,int kdim,double *alpha){//dense rearr(old) lazy mode
  double *bin1,*bin2,*bin3,*bin4,*bout;int bcol,brow;
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
void load_irreg_b_r(double *bstartpos,double *bblk,int ldb,int ndim,int kdim,double *alpha){//dense rearr(old) lazy mode
  double *bin,*bout;int bcol,brow;register __m256d btmp,bmul;
  bin=bstartpos;bmul=_mm256_broadcast_sd(alpha);
  for(brow=0;brow<kdim;brow++){
    bout=bblk+brow*4;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      btmp=_mm256_loadu_pd(bin);
      btmp=_mm256_mul_pd(btmp,bmul);
      _mm256_storeu_pd(bout,btmp);
      bin+=4;bout+=4*kdim;
    }
    bout-=3*brow;
    for(;bcol<ndim;bcol++){
      *bout=*bin*(*alpha);bin++;bout+=kdim;
    }
    bin+=ldb-ndim;
  }
}
// below are functions written in assembly
extern void load_reg_a_c(double *astartpos,double *ablk,int lda);
extern void load_reg_a_r(double *astartpos,double *ablk,int lda);
extern void load_tail_a_c(double *astartpos,double *ablk,int lda,int mdim);
extern void load_tail_a_r(double *astartpos,double *ablk,int lda,int mdim);
extern void load_irregk_a_c(double *astartpos,double *ablk,int lda,int kdim);//lazy mode
extern void load_irregk_a_r(double *astartpos,double *ablk,int lda,int kdim);//lazy mode
extern void load_irreg_a_c(double *astartpos,double *ablk,int lda,int mdim,int kdim);//sparse lazy mode
extern void load_irreg_a_r(double *astartpos,double *ablk,int lda,int mdim,int kdim);//sparse lazy mode
extern void load_reg_b_c(double *bstartpos,double *bblk,int ldb,double *alpha);
extern void load_reg_b_r(double *bstartpos,double *bblk,int ldb,double *alpha);
extern void dgemmblkregccc(double *abufferctpos,double *bblk,double *cstartpos,int ldc);
extern void dgemmblktailccc(double *abufferctpos,double *bblk,double *cstartpos,int ldc,int mdim);
extern void dgemmblkirregkccc(double *ablk,double *bblk,double *cstartpos,int ldc,int kdim,double *beta);
extern void dgemmblkirregnccc(double *ablk,double *bblk,double *cstartpos,int ldc,int ndim);
extern void dgemmblkirregccc(double *ablk,double *bblk,double *cstartpos,int ldc,int mdim,int ndim,int kdim,double *beta);
void load_abuffer_ac(double *aheadpos,double *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*BlkDimK,LDA);
  load_tail_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*BlkDimK,LDA,EdgeM);
}
void load_abuffer_ar(double *aheadpos,double *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*BlkDimK,LDA);
  load_tail_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*BlkDimK,LDA,EdgeM);
}
void load_abuffer_irregk_ac(double *aheadpos,double *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*kdim,LDA,kdim);
  load_irreg_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*kdim,LDA,EdgeM,kdim);
}
void load_abuffer_irregk_ar(double *aheadpos,double *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*kdim,LDA,kdim);
  load_irreg_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*kdim,LDA,EdgeM,kdim);
}
void dgemmcolumn(double *abuffer,double *bblk,double *cheadpos,int BlksM,int EdgeM,int LDC){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    dgemmblkregccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC);
    MCT+=BlkDimM;
  }
  dgemmblktailccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,EdgeM);
}
void dgemmcolumnirregn(double *abuffer,double *bblk,double *cheadpos,int BlksM,int EdgeM,int LDC,int ndim){
  int MCT=0;int BlkCtM;double beta=1.0;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    dgemmblkirregnccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,ndim);
    MCT+=BlkDimM;
  }
  dgemmblkirregccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,EdgeM,ndim,BlkDimK,&beta);
}
void dgemmcolumnirregk(double *abuffer,double *bblk,double *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,double *beta){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    dgemmblkirregkccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,kdim,beta);
    MCT+=BlkDimM;
  }
  dgemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,EdgeM,BlkDimN,kdim,beta);
}
void dgemmcolumnirreg(double *abuffer,double *bblk,double *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,int ndim,double *beta){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    dgemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,BlkDimM,ndim,kdim,beta);
    MCT+=BlkDimM;
  }
  dgemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,EdgeM,ndim,kdim,beta);
}
void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc){//dgemm function with 1 thread
//assume column-major storage with arguments passed by addresses (FORTRAN style)
//a:matrix with m rows and k columns if transa=N
//b:matrix with k rows and n columns if transb=N
//c:product matrix with m rows and n columns
 const int M = *m;const int N = *n;const int K = *k;
 double BETA = 1.0;
 const int LDA = *lda;const int LDB = *ldb;const int LDC=*ldc;
 const char TRANSA = *transa;const char TRANSB = *transb;
 const int BlksM = (M-1)/BlkDimM+1;const int EdgeM = M-(BlksM-1)*BlkDimM;//the m-dim of edges
 const int BlksN = (N-1)/BlkDimN+1;const int EdgeN = N-(BlksN-1)*BlkDimN;//the n-dim of edges
 const int BlksK = (K-1)/BlkDimK+1;const int EdgeK = K-(BlksK-1)*BlkDimK;//the k-dim of edges
 int BlkCtM,BlkCtN,BlkCtK,MCT,NCT,KCT;//loop counters over blocks(tiles)
 //MCT,NCT and KCT are used to locate the current position of matrix blocks
 int mdim,ndim,kdim;
 double *ablk,*bblk,*abuffer; //abuffer[]: store 256 columns of matrix a
 if((*alpha) != (double)0.0){//then do C+=alpha*AB
  ablk=(double *)aligned_alloc(4096,(BlkDimM*BlkDimK)*sizeof(double));
  bblk=(double *)aligned_alloc(64,(BlkDimN*BlkDimK)*sizeof(double));
  abuffer = (double *)aligned_alloc(4096,(BlkDimM*BlkDimK*BlksM)*sizeof(double));
  if(TRANSA=='N' || TRANSA=='n'){
   if(TRANSB=='N' || TRANSB=='n'){//CASE NN
    load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_c(b+NCT*LDB,bblk,LDB,BlkDimN,EdgeK,alpha);
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
   else{//CASE NY
    load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_r(b+NCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
  }
  else{
   if(TRANSB=='N' || TRANSB=='n'){//case YN
    load_abuffer_irregk_ar(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_c(b+NCT*LDB,bblk,LDB,BlkDimN,EdgeK,alpha);
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
   else{//case YY
    load_abuffer_irregk_ar(a,abuffer,LDA,BlksM,EdgeM,EdgeK);
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_r(b+NCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
  }
  free(abuffer);abuffer=NULL;
  free(bblk);free(ablk);
  bblk=NULL;ablk=NULL;
 }
}
