# include <stdio.h>
# include <stdlib.h>
# include <immintrin.h> //AVX2
# include <omp.h>
//compilation command: gcc -DDOUBLE -fopenmp --shared -fPIC -march=haswell -O3 gemm_kernel.S gemm_driver.c -o DGEMM.so
//compilation command: gcc -fopenmp --shared -fPIC -march=haswell -O3 gemm_kernel.S gemm_driver.c -o SGEMM.so
#ifdef DOUBLE
 #define FLOAT double
 #define CNAME dgemm_
 #define BlkDimM 12
#else
 #define FLOAT float
 #define CNAME sgemm_
 #define BlkDimM 24
#endif
# include "gemm_copy.c"
# include "gemm_kernel_irreg.c"
extern void gemmblkregccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc);//carry >90% gemm calculations
extern void gemmblktailccc(FLOAT *abufferctpos,FLOAT *bblk,FLOAT *cstartpos,int ldc,int mdim);
extern void timedelay();//produce nothing besides a delay(~3 us), with no system calls
static void synproc(int tid,int threads,int *workprogress){//workprogress[] must be shared among all threads
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
static void load_abuffer_ac(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*BlkDimK,LDA);
  load_tail_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*BlkDimK,LDA,EdgeM);
}
static void load_abuffer_ar(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM){
  int i;
  for(i=0;i<BlksM-1;i++) load_reg_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*BlkDimK,LDA);
  load_tail_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*BlkDimK,LDA,EdgeM);
}
static void load_abuffer_irregk_ac(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*kdim,LDA,kdim);
  load_irreg_a_c(aheadpos+i*BlkDimM,abuffer+i*BlkDimM*kdim,LDA,EdgeM,kdim);
}
static void load_abuffer_irregk_ar(FLOAT *aheadpos,FLOAT *abuffer,int LDA,int BlksM,int EdgeM,int kdim){
  int i;
  for(i=0;i<BlksM-1;i++) load_irregk_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*kdim,LDA,kdim);
  load_irreg_a_r(aheadpos+i*BlkDimM*LDA,abuffer+i*BlkDimM*kdim,LDA,EdgeM,kdim);
}
static void gemmcolumn(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkregccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC);
    MCT+=BlkDimM;
  }
  gemmblktailccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,EdgeM);
}
static void gemmcolumnirregn(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC,int ndim){
  int MCT=0;int BlkCtM;FLOAT beta=1.0;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkirregnccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,ndim);
    MCT+=BlkDimM;
  }
  gemmblkirregccc(abuffer+MCT*BlkDimK,bblk,cheadpos+MCT,LDC,EdgeM,ndim,BlkDimK,&beta);
}
static void gemmcolumnirregk(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,FLOAT *beta){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkirregkccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,kdim,beta);
    MCT+=BlkDimM;
  }
  gemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,EdgeM,BlkDimN,kdim,beta);
}
static void gemmcolumnirreg(FLOAT *abuffer,FLOAT *bblk,FLOAT *cheadpos,int BlksM,int EdgeM,int LDC,int kdim,int ndim,FLOAT *beta){
  int MCT=0;int BlkCtM;
  for(BlkCtM=0;BlkCtM<BlksM-1;BlkCtM++){
    gemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,BlkDimM,ndim,kdim,beta);
    MCT+=BlkDimM;
  }
  gemmblkirregccc(abuffer+MCT*kdim,bblk,cheadpos+MCT,LDC,EdgeM,ndim,kdim,beta);
}
static void cmultbeta(FLOAT *c,int ldc,int m,int n,FLOAT beta){
  int i,j;FLOAT *C0,*C;
  C0=c;
  for(i=0;i<n;i++){
    C=C0;
    for(j=0;j<m;j++){
      *C*=beta;C++;
    }
    C0+=ldc;
  }
}
void CNAME(char *transa,char *transb,int *m,int *n,int *k,FLOAT *alpha,FLOAT *a,int *lda,FLOAT *bstart,int *ldb,FLOAT *beta,FLOAT *cstart,int *ldc){
//assume column-major storage with arguments passed by addresses (FORTRAN style)
//a:matrix with m rows and k columns if transa=N
//b:matrix with k rows and n columns if transb=N
//c:product matrix with m rows and n columns
 const int M = *m;/* const int N = *n; */const int K = *k;
 FLOAT BETA = 1.0;
 const int LDA = *lda;const int LDB = *ldb;const int LDC=*ldc;
 const char TRANSA = *transa;const char TRANSB = *transb;
 const int BlksM = (M-1)/BlkDimM+1;const int EdgeM = M-(BlksM-1)*BlkDimM;//the m-dim of edges
 const int BlksK = (K-1)/BlkDimK+1;const int EdgeK = K-(BlksK-1)*BlkDimK;//the k-dim of edges
 int *workprogress, *cchunks;const int numthreads=omp_get_max_threads();int i; //for parallel execution
 //cchunk[] for dividing tasks, workprogress[] for recording the progresses of all threads and synchronization.
 //synchronization is necessary here since abuffer[] is shared between threads.
 //if abuffer[] is thread-private, the bandwidth of memory will limit the performance.
 //synchronization by openmp functions can be expensive, so handcoded funcion (synproc) is used instead.
 FLOAT *abuffer; //abuffer[]: store 256 columns of matrix a
 if((*alpha) == 0.0 && (*beta) != 1.0) cmultbeta(cstart,LDC,M,(*n),(*beta));//limited by memory bendwidth so no need for parallel execution
 if((*alpha) != 0.0){//then do C=alpha*AB+beta*C
  abuffer = (FLOAT *)aligned_alloc(4096,(BlkDimM*BlkDimK*BlksM)*sizeof(FLOAT));
  workprogress = (int *)calloc(20*numthreads,sizeof(int));
  cchunks = (int *)malloc((numthreads+1)*sizeof(int));
  for(i=0;i<=numthreads;i++) cchunks[i]=(*n)*i/numthreads;
#pragma omp parallel
 {
  int tid = omp_get_thread_num();
  FLOAT *c = cstart + LDC * cchunks[tid];
  FLOAT *b;
  if(TRANSB=='N' || TRANSB=='n') b = bstart + LDB * cchunks[tid];
  else b = bstart + cchunks[tid];
  const int N = cchunks[tid+1]-cchunks[tid];
  const int BlksN = (N-1)/BlkDimN+1; const int EdgeN = N-(BlksN-1)*BlkDimN;//the n-dim of edges
  int BlkCtM,BlkCtN,BlkCtK,MCT,NCT,KCT;//loop counters over blocks
  //MCT,NCT and KCT are used to locate the current position of matrix blocks
  FLOAT *bblk = (FLOAT *)aligned_alloc(4096,(BlkDimN*BlkDimK)*sizeof(FLOAT)); //thread-private bblk[]
  if(TRANSA=='N' || TRANSA=='n'){
   if(TRANSB=='N' || TRANSB=='n'){//CASE NN
    if(tid==0) load_abuffer_irregk_ac(a,abuffer,LDA,BlksM,EdgeM,EdgeK); //only the master thread can write abuffer
    synproc(tid,numthreads,workprogress); //before the calculations, child threads need to wait here until the master finish writing abuffer
    for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
     NCT=BlkDimN*BlkCtN;
     load_irreg_b_c(b+NCT*LDB,bblk,LDB,BlkDimN,EdgeK,alpha);
     gemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    gemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);//before updating abuffer, the master thread need to wait here until all child threads finish calculation with current abuffer
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0) load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM); //only the master thread can write abuffer
     synproc(tid,numthreads,workprogress);//before the calculations, child threads need to wait here until the master finish writing abuffer
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      gemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     gemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
     gemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    gemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=1;BlkCtK<BlksK;BlkCtK++){
     if(tid==0) load_abuffer_ac(a+KCT*LDA,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      gemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     gemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
     gemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_c(b+NCT*LDB,bblk,LDB,EdgeN,EdgeK,alpha);
    gemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     if(tid==0) load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_c(b+NCT*LDB+KCT,bblk,LDB,alpha);
      gemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_c(b+NCT*LDB+KCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     gemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
     gemmcolumnirregk(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,beta);
    }
    NCT=BlkDimN*(BlksN-1);
    load_irreg_b_r(b+NCT,bblk,LDB,EdgeN,EdgeK,alpha);
    gemmcolumnirreg(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeK,EdgeN,beta);
    synproc(tid,numthreads,workprogress);
    KCT=EdgeK;
    for(BlkCtK=0;BlkCtK<BlksK-1;BlkCtK++){
     if(tid==0) load_abuffer_ar(a+KCT,abuffer,LDA,BlksM,EdgeM);
     synproc(tid,numthreads,workprogress);
     for(BlkCtN=0;BlkCtN<BlksN-1;BlkCtN++){
      NCT=BlkCtN*BlkDimN;
      load_reg_b_r(b+KCT*LDB+NCT,bblk,LDB,alpha);
      gemmcolumn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC);
     }//loop BlkCtN++
     NCT=(BlksN-1)*BlkDimN;
     load_irreg_b_r(b+KCT*LDB+NCT,bblk,LDB,EdgeN,BlkDimK,alpha);
     gemmcolumnirregn(abuffer,bblk,c+NCT*LDC,BlksM,EdgeM,LDC,EdgeN);
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
