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
//compilation command: gcc -fopenmp --shared -fPIC -march=haswell -O3 dgemm_largemem.c dgemm.S -o DGEMM_LARGEMEM.so
//DGEMM_LARGEMEM.so requires more memory space(m*2kB) than DGEMM.so, ran faster (~2-5%) than the latter.

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
extern void timedelay();//produce nothing besides a delay(~3 us), with no system calls 
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
void dgemm(char *transa,char *transb,int *m,int *n,int *k,double *alpha,double *a,int *lda,double *bstart,int *ldb,double *beta,double *cstart,int *ldc){//dgemm function paralleled via gnu-openmp. top performance: 486GFLOPS(93% theoretical) for 8 threads on i9-9900K at 4.1 GHz (while Intel MKL(2018) gave 474 GFLOPS)
//assume column-major storage with arguments passed by addresses (FORTRAN style)
//a:matrix with m rows and k columns if transa=N
//b:matrix with k rows and n columns if transb=N
//c:product matrix with m rows and n columns
 const int M = *m;/* const int N = *n; */const int K = *k;
 double BETA = 1.0;
 const int LDA = *lda;const int LDB = *ldb;const int LDC=*ldc;
 const char TRANSA = *transa;const char TRANSB = *transb;
 const int BlksM = (M-1)/BlkDimM+1;const int EdgeM = M-(BlksM-1)*BlkDimM;//the m-dim of edges
 const int BlksK = (K-1)/BlkDimK+1;const int EdgeK = K-(BlksK-1)*BlkDimK;//the k-dim of edges
 int *workprogress, *cchunks;const int numthreads=omp_get_max_threads();int i; //for parallel execution
 //cchunk[] for dividing tasks, workprogress[] for recording the progresses of all threads and synchronization.
 //unlike the implementation in DGEMM.so, synchronization is necessary here since abuffer[] is shared between threads.
 //if abuffer[] is thread-private, the bandwidth of memory will limit the performance.
 //synchronization by openmp functions can be expensive, so handcoded funcion (synproc) is used instead.
 double *abuffer; //abuffer[]: store 256 columns of matrix a
 if((*alpha) != (double)0.0 || (*beta) != (double)1.0){//then do C=alpha*AB+beta*C
  abuffer = (double *)aligned_alloc(4096,(BlkDimM*BlkDimK*BlksM)*sizeof(double));
  workprogress = (int *)calloc(20*numthreads,sizeof(int));
  cchunks = (int *)malloc((numthreads+1)*sizeof(int));
  for(i=0;i<=numthreads;i++) cchunks[i]=(*n)*i/numthreads;
#pragma omp parallel
 {
  int tid = omp_get_thread_num();
  double *c = cstart + LDC * cchunks[tid];
  double *b;
  if(TRANSB=='N' || TRANSB=='n') b = bstart + LDB * cchunks[tid];
  else b = bstart + cchunks[tid];
  const int N = cchunks[tid+1]-cchunks[tid];
  const int BlksN = (N-1)/BlkDimN+1; const int EdgeN = N-(BlksN-1)*BlkDimN;//the n-dim of edges
  int BlkCtM,BlkCtN,BlkCtK,MCT,NCT,KCT;//loop counters over blocks
  //MCT,NCT and KCT are used to locate the current position of matrix blocks
  double *bblk = (double *)aligned_alloc(4096,(BlkDimN*BlkDimK)*sizeof(double)); //thread-private bblk[]
  if(TRANSA=='N' || TRANSA=='n'){
   if(TRANSB=='N' || TRANSB=='n'){//CASE NN
    if(tid==0) load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK); //only the master thread can write abuffer
    synproc(tid,numthreads,workprogress); //before the calculations, child threads need to wait here until the master finish writing abuffer
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_c(b+NCT*LDB,bblk,LDB,BlkDimN,EdgeK,alpha);
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);//before updating abuffer, the master thread need to wait here until all child threads finish calculation with current abuffer
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0) load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM); //only the master thread can write abuffer
     synproc(tid,numthreads,workprogress);//before the calculations, child threads need to wait here until the master finish writing abuffer
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0) load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     if(tid==0) load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
     dgemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    dgemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     if(tid==0) load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      dgemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     dgemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
