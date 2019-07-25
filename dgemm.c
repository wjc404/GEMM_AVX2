# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h> //AVX2
# include <omp.h>

# define BlkUnitM 3 //fixed!
# define BlkUnitN 64 //fixed!
# define BlkUnitK BlkUnitN
# define BlkDimM (BlkUnitM*4)
# define BlkDimN (BlkUnitN*4)
# define BlkDimK (BlkUnitK*4)
//compilation command: gcc dgemm.c dgemm.S -fopenmp --shared -fPIC -march=haswell -O3 -o DGEMM.so

//major part: reg N, reg K, reg or irreg M

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
extern void dgemmblkregccc_ac(double *ablk,double *bblk,double *cstartpos,int ldc,double *aprefpos,int lda,double *nextablk);
extern void dgemmblkregccc_ar(double *ablk,double *bblk,double *cstartpos,int ldc,double *aprefpos,int lda);
extern void dgemmblktailccc(double *ablk,double *bblk,double *cstartpos,int ldc,int mdim);
extern void dgemmblkirregkccc(double *ablk,double *bblk,double *cstartpos,int ldc,int kdim,double *beta);
extern void dgemmblkirregnccc(double *ablk,double *bblk,double *cstartpos,int ldc,int ndim);
extern void dgemmblkirregccc(double *ablk,double *bblk,double *cstartpos,int ldc,int mdim,int ndim,int kdim,double *beta);

void dgemmcolumn_ac(double *aheadpos,double *ablk,double *ablk2,double *bblk,double *cheadpos,int BlksM,int EdgeM,int LDA,int LDC){
  load_reg_a_c(aheadpos,ablk,LDA);
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-2;BlkCtM+=2){
    dgemmblkregccc_ac(ablk,bblk,cheadpos+MCT,LDC,aheadpos+MCT+BlkDimM,LDA,ablk2);
    dgemmblkregccc_ac(ablk2,bblk,cheadpos+MCT+BlkDimM,LDC,aheadpos+MCT+BlkDimM*2,LDA,ablk);
    MCT+=BlkDimM*2;
  }
  if(BlkCtM==BlksM-2) {
    dgemmblkregccc_ac(ablk,bblk,cheadpos+MCT,LDC,aheadpos+MCT+BlkDimM,LDA,ablk2);
    MCT+=BlkDimM;
    dgemmblktailccc(ablk2,bblk,cheadpos+MCT,LDC,EdgeM);
  }
  else dgemmblktailccc(ablk,bblk,cheadpos+MCT,LDC,EdgeM);
}
void dgemmcolumn_ar(double *aheadpos,double *ablk,double *ablk2,double *bblk,double *cheadpos,int BlksM,int EdgeM,int LDA,int LDC){
  int MCT,BlkCtM;MCT=0;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    load_reg_a_r(aheadpos+MCT*LDA,ablk,LDA);//aheadpos=a+KCT
    dgemmblkregccc_ar(ablk,bblk,cheadpos+MCT,LDC,aheadpos+(MCT+BlkDimM)*LDA,LDA);//cheadpos=c+NCT*LDC
    MCT+=BlkDimM;
  }
  load_tail_a_r(aheadpos+MCT*LDA,ablk,LDA,EdgeM);
  dgemmblktailccc(ablk,bblk,cheadpos+MCT,LDC,EdgeM);
}
void dgemmserial(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc){//dgemm function with 1 thread
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
 double *ablk,*bblk,*ablk2;
 ablk=(double *)aligned_alloc(4096,(BlkDimM*BlkDimK)*sizeof(double));
 ablk2=(double *)aligned_alloc(4096,(BlkDimM*BlkDimK)*sizeof(double));
 bblk=(double *)aligned_alloc(64,(BlkDimN*BlkDimK)*sizeof(double));
 if((*alpha) != (double)0.0 || (*beta) != (double)1.0){//then do C=alpha*AB+beta*C
  if(TRANSA=='N' || TRANSA=='n'){
   if(TRANSB=='N' || TRANSB=='n'){//CASE NN
    KCT=0;
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     MCT=0;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      load_irregk_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeK);
      dgemmblkirregkccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeK,beta);
      MCT+=BlkDimM;
     }
     load_irreg_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeM,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,BlkDimN,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,EdgeK,alpha);
    for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
     MCT=BlkCtM*BlkDimM;
     load_irregk_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,BlkDimM,EdgeN,EdgeK,beta);
    }
    MCT=BlkDimM*(BlksM-1);
    load_irreg_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeM,EdgeK);
    dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,EdgeK,beta);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      dgemmcolumn_ac(a+KCT*LDA,ablk,ablk2,bblk,c+NCT*LDC,BlksM,EdgeM,LDA,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      MCT=BlkCtM*BlkDimM;
      load_reg_a_c(a+KCT*LDA+MCT,ablk,LDA);
      dgemmblkirregnccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeN);
     }
     MCT=(BlksM-1)*BlkDimM;
     load_tail_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeM);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,BlkDimK,&BETA);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
   else{//CASE NY
    KCT=0;
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     MCT=0;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      load_irregk_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeK);
      dgemmblkirregkccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeK,beta);
      MCT+=BlkDimM;
     }
     load_irreg_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeM,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,BlkDimN,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
     MCT=BlkCtM*BlkDimM;
     load_irregk_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,BlkDimM,EdgeN,EdgeK,beta);
    }
    MCT=BlkDimM*(BlksM-1);
    load_irreg_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeM,EdgeK);
    dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,EdgeK,beta);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      dgemmcolumn_ac(a+KCT*LDA,ablk,ablk2,bblk,c+NCT*LDC,BlksM,EdgeM,LDA,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      MCT=BlkCtM*BlkDimM;
      load_reg_a_c(a+KCT*LDA+MCT,ablk,LDA);
      dgemmblkirregnccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeN);
     }
     MCT=(BlksM-1)*BlkDimM;
     load_tail_a_c(a+KCT*LDA+MCT,ablk,LDA,EdgeM);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,BlkDimK,&BETA);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
  }
  else{
   if(TRANSB=='N' || TRANSB=='n'){//case YN
    KCT=0;
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     MCT=0;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      load_irregk_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeK);
      dgemmblkirregkccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeK,beta);
      MCT+=BlkDimM;
     }
     load_irreg_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeM,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,BlkDimN,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,EdgeK,alpha);
    for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
     MCT=BlkCtM*BlkDimM;
     load_irregk_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,BlkDimM,EdgeN,EdgeK,beta);
    }
    MCT=BlkDimM*(BlksM-1);
    load_irreg_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeM,EdgeK);
    dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,EdgeK,beta);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      dgemmcolumn_ar(a+KCT,ablk,ablk2,bblk,c+NCT*LDC,BlksM,EdgeM,LDA,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      MCT=BlkCtM*BlkDimM;
      load_reg_a_r(a+MCT*LDA+KCT,ablk,LDA);
      dgemmblkirregnccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeN);
     }
     MCT=(BlksM-1)*BlkDimM;
     load_tail_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeM);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,BlkDimK,&BETA);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
   else{//case YY
    KCT=0;
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     MCT=0;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,BlkDimN,EdgeK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      load_irregk_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeK);
      dgemmblkirregkccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeK,beta);
      MCT+=BlkDimM;
     }
     load_irreg_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeM,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,BlkDimN,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
     MCT=BlkCtM*BlkDimM;
     load_irregk_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeK);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,BlkDimM,EdgeN,EdgeK,beta);
    }
    MCT=BlkDimM*(BlksM-1);
    load_irreg_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeM,EdgeK);
    dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,EdgeK,beta);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      dgemmcolumn_ar(a+KCT,ablk,ablk2,bblk,c+NCT*LDC,BlksM,EdgeM,LDA,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
      MCT=BlkCtM*BlkDimM;
      load_reg_a_r(a+MCT*LDA+KCT,ablk,LDA);
      dgemmblkirregnccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeN);
     }
     MCT=(BlksM-1)*BlkDimM;
     load_tail_a_r(a+MCT*LDA+KCT,ablk,LDA,EdgeM);
     dgemmblkirregccc(ablk,bblk,c+NCT*LDC+MCT,LDC,EdgeM,EdgeN,BlkDimK,&BETA);
     KCT+=BlkDimK;
    }//loop BlkCtK++
   }
  }
 }
 free(ablk);free(bblk);free(ablk2);
 ablk=NULL;bblk=NULL;ablk2=NULL;
}
void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *beta,double *c,int *ldc){//dgemm wrapper caller, paralleled via GNU openmp
  int *nchunk,*mchunk;double *apos;
  int numthreads=omp_get_max_threads();
  int i,nummchunk,mth;
  nchunk=(int *)malloc(sizeof(int)*(numthreads+1));
  for(i=0;i<numthreads;i++) nchunk[i]=(long)(*n)*i/numthreads; nchunk[numthreads]=*n;
  nummchunk=(*m)/6000+1;
  mchunk=(int *)malloc(sizeof(int)*(nummchunk+1));
  for(i=0;i<nummchunk;i++) mchunk[i]=(long)(*m)*i/nummchunk; mchunk[nummchunk]=*m;
  for(i=0;i<nummchunk;i++){
   mth=mchunk[i+1]-mchunk[i];
   if((*transa)=='T') apos=a+mchunk[i]*(*lda); else apos=a+mchunk[i];
   omp_set_num_threads(numthreads);
   #pragma omp parallel
   {
    int thr=omp_get_thread_num();
    int nth=nchunk[thr+1]-nchunk[thr];
    double *cpos=c+nchunk[thr]*(*ldc)+mchunk[i];double *bpos;
    if((*transb)=='T') bpos=b+nchunk[thr]; else bpos=b+nchunk[thr]*(*ldb);
    dgemmserial(transa,transb,&mth,&nth,k,alpha,apos,lda,bpos,ldb,beta,cpos,ldc);
   }
  }
  free(mchunk);mchunk=NULL;
  free(nchunk);nchunk=NULL;
}
