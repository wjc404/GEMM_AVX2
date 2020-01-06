# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <dlfcn.h>
# include <string.h>
# define MAX_DIFF 1.0e-2
# define STATIC_STRMM_NAME strmm
//gcc -fopenmp strmm_test.c -Wl,--start-group /home/wang/intel/mkl/lib/intel64/libmkl_intel_lp64.a /home/wang/intel/mkl/lib/intel64/libmkl_gnu_thread.a /home/wang/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -o strmm_test

void STATIC_STRMM_NAME(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *a, int *lda, float *b, int *ldb);//statically linked with MKL's STRMM, interface = lp64.
void find_maximum_diff(float *array1, float *array2, int dimension,int *max_position){
    int count=0,maxno=-1;float diff=MAX_DIFF,temp=0.0;
    for(count=0;count<dimension;count++){
      temp = (float)fabs((double)array1[count]-(double)array2[count]);
      if(temp > diff){
        maxno = count + 1;
        diff = temp;
      }
    }
    if(maxno > 0) printf("The 2 fp32 elements at position %d are inconsistent, difference: %e \n",maxno,diff);
    *max_position = maxno;
}
int main(int argc, char* argv[]){ //command-line usage: strmm_test [m] [n] [side] [uplo] [transa] [diag]
/* first process input parameters */
    int m,n,lda,ldb;char side,uplo,transa,diag;
    if (argc >= 2) m = atoi(argv[1]); //1st input parameter
    else m = 10;
    if (m < 1) m = 1;
    if (argc >= 3) n = atoi(argv[2]); //2nd input parameter
    else n = 10;
    if (n < 1) n = 1;
    ldb = m;
    if (argc >= 4) side = *argv[3]; //3rd input parameter
    else side = 'L';
    if (side == 'L' || side == 'l') {side = 'L'; lda = m;}
    else {side = 'R'; lda = n;}
    if (argc >= 5) uplo = *argv[4]; //4th input parameter
    else uplo = 'U';
    if (uplo == 'U' || uplo == 'u') uplo = 'U';
    else uplo = 'L';
    if (argc >= 6) transa = *argv[5]; //5th input parameter
    else transa = 'N';
    if (transa == 'N' || transa == 'n') transa = 'N';
    else transa = 'T';
    if (argc >= 7) diag = *argv[6]; //6th input parameter
    else diag = 'N';
    if (diag == 'U' || diag == 'u') diag = 'U';
    else diag = 'N';
/* then perform dynamic linking to the subroutine being tested */
    void (*strmmroutine)(char *side, char *uplo, char *transa, char *diag, int *m, int *n, float *alpha, float *a, int *lda, float *b, int *ldb);
    char strmmpath[200],strmmname[18];void *handle;char *DLERR;
    dlerror();
    printf("Enter your strmm library path, including the file name:");
    scanf("%s",strmmpath);
    handle = dlopen(strmmpath,RTLD_LAZY);
    DLERR = dlerror();
    if (DLERR) {  
      printf ("Error locating your strmm library: %s\n",DLERR);  
      exit(1);  
    }
    printf("Enter the function name of strmm in your library:");
    scanf("%s",strmmname);
    strmmroutine = dlsym(handle,strmmname);
    DLERR = dlerror();
    if (DLERR) {
      printf ("Error locating strmm function in your library: %s\n",DLERR);
      strmmroutine=NULL;dlclose(handle);handle=NULL;
      exit(1);
    }
/* now start comparing test */
    float *A = (float*) malloc(sizeof(float)*lda*lda);
    float *B0 = (float*) malloc(sizeof(float)*m*n);
    float *B1 = (float*) malloc(sizeof(float)*m*n);
    float *B2 = (float*) malloc(sizeof(float)*m*n);
    int elem_count;srand((unsigned)time(NULL));
    for(elem_count=0;elem_count<lda*lda;elem_count++) A[elem_count] = (float)rand()/RAND_MAX*(float)rand()/RAND_MAX;
    for(elem_count=0;elem_count<m*n;elem_count++) B0[elem_count] = (float)rand()/RAND_MAX*(float)rand()/RAND_MAX;
    int m_subdim,n_subdim;
    int max_no = -1;float max_diff = 0.0;float alpha = 2.0;
    for(m_subdim = 1;m_subdim <= m;m_subdim ++){
      printf("testing m = %d\n",m_subdim);
      for(n_subdim = 1;n_subdim <= n;n_subdim++){
        memcpy(B1,B0,sizeof(float)*m*n);
        STATIC_STRMM_NAME(&side,&uplo,&transa,&diag,&m_subdim,&n_subdim,&alpha,A,&lda,B1,&ldb);
        memcpy(B2,B0,sizeof(float)*m*n);
        (*strmmroutine)(&side,&uplo,&transa,&diag,&m_subdim,&n_subdim,&alpha,A,&lda,B2,&ldb);
        find_maximum_diff(B1,B2,m*n,&max_no);
        if(max_no > 0){
          printf("  Test failed at n = %d.\n", n_subdim);
          m_subdim=m+1;n_subdim=n+1;
        }
      }
    }
    if(max_no == -1)  printf("All tests passed with m <= %d, n <= %d, side = %c, uplo = %c, transa = %c and diag = %c.\n", m, n, side, uplo, transa, diag);
/* clean up and leave */
    free(A);A=NULL;free(B0);B0=NULL;free(B1);B1=NULL;free(B2);B2=NULL;
    strmmroutine=NULL;
    dlclose(handle);
    DLERR = dlerror();
    handle=NULL;
    if (DLERR) {
      printf ("Error in closing your strmm library:%s\n",DLERR);
      exit(1);
    }
    return 0;
}
