#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include "/home/wang/intel/mkl/include/mkl_vsl.h"
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <string.h>
//this program is for tests (about performances and outputs) of user-provided dtrmm library against Intel MKL
/*an example of compilation command
gcc -fopenmp dtrmmtest.c -Wl,--start-group /home/wang/intel/mkl/lib/intel64/libmkl_intel_lp64.a /home/wang/intel/mkl/lib/intel64/libmkl_gnu_thread.a /home/wang/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -o dtrmmtest
*/

int main(int argc, char* argv[]) // command line: ./dtrmmtest [niter] [m] [n] [upperbound] [lowerbound] [side] [uplo] [transa] [diag]
{
// variables blow for dynamic linking
        void (*dtrmmroutine1)(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb);
        void dtrmm(char *side, char *uplo, char *transa, char *diag, int *m, int *n, double *alpha, double *a, int *lda, double *b, int *ldb);
        void *handle1;
        char dtrmmpath1[200],dtrmmname1[18]; // get linking info from stdin
        char *DLERR;
// variables blow for main dtrmm test
	double *A,*B1,*B2;
        double alpha = 2.0;
	char side,uplo,transa,diag;

        long ops,availmem,occupmem,maxelem;
	int i,j,m,n,niters,seed,error,lda,ldb;
	double walltime1,walltime2;
        double maxdif,tempdif,upperbound,lowerbound;
        VSLStreamStatePtr stream; // pointer to MKL_stream_state structure
        struct sysinfo s_info; // for getting memory info
        struct timeval starttime,endtime; // for linux timing
        double *tscs; //for storing time

// get dimension(default=2000) of matrices and number(default=4) of iterations
	if (argc >= 2) niters = atoi(argv[1]);
	else niters = 4;
        if (niters < 1 || niters > 300) niters = 1;
	if (argc >= 3) m = atoi(argv[2]);
	else m = 2000;
        if (m < 3) m = 3; ldb = m;
	if (argc >= 4) n = atoi(argv[3]);
	else n = 2000;
        if (n < 3) n = 3;
// get range of random numbers filled in A and B, default [0,10)
	if (argc >= 5) upperbound = atof(argv[4]);
	else upperbound = 10.0;
	if (argc >= 6) lowerbound = atof(argv[5]);
	else lowerbound = 0.0;
        if (upperbound <= lowerbound) { // check the order
                upperbound = upperbound + lowerbound;
                lowerbound = upperbound - lowerbound;
                upperbound = upperbound - lowerbound;
        }
	if (argc >= 7) side = *argv[6];
	else side = 'L';
        if(side=='L' || side=='l') {side = 'L';lda = m;}
        else {side = 'R';lda = n;}
	if (argc >= 8) uplo = *argv[7];
	else uplo = 'U';
        if(uplo=='U' || uplo=='u') {uplo = 'U';}
        else {uplo = 'L';}
	if (argc >= 9) transa = *argv[8];
	else transa = 'N';
        if(transa=='N' || transa=='n') {transa = 'N';}
        else if(transa=='C' || transa=='c') {transa = 'C';}
        else {transa = 'T';}
	if (argc >= 10) diag = *argv[9];
	else diag = 'N';
        if(diag=='U' || diag=='u') {diag = 'U';}
        else {diag = 'N';}

//print test informations
        printf("A SIMPLE DTRMM TESTING PROGRAM\n");
        printf("Matrix elements will be stored in column-major order.\n");
        printf("Operation on A: %c\n",transa);
        printf("Type of triangular A: unit?%c; uplo?%c; side?%c\n",diag,uplo,side);
        printf("DTRMM will be called in this way:\n");
        printf("    <dtrmmname>(&side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb)\n");
        printf("    A: triangular matrix with dim = m or n; B: normal m*n matrix.\n");
        printf("    DTRMM performs B = alpha op(A) B  or  B = alpha B op(A).\n");
        printf("Please make sure that the dtrmm library you provided can respond normally to the call above\n\n");

// load the tested cgemm routine
        dlerror();
        printf("Enter your dtrmm library path(path to the *.so file):");
        scanf("%s",dtrmmpath1);
        handle1 = dlopen(dtrmmpath1,RTLD_LAZY);
        DLERR = dlerror();
        if (DLERR) {printf ("Error locating the dtrmm library: %s\n",DLERR); exit(1);}
        printf("Enter the function name of dtrmm in your library(e.g., dtrmm_):");
        scanf("%s",dtrmmname1);
        dtrmmroutine1 = dlsym(handle1,dtrmmname1);
        DLERR = dlerror();
        if (DLERR) {
            printf ("Error locating dtrmm function in the library: %s\n",DLERR);
            dtrmmroutine1=NULL;dlclose(handle1);handle1=NULL;exit(1);
        }
        sleep(5); // pause for viewing linking results

// check memory availability
        error = sysinfo(&s_info);
        if (error != 0){
            printf ("Cannot get memory info, now exit\n");
            dtrmmroutine1=NULL;dlclose(handle1);handle1=NULL;exit(1);
        }
        availmem = (long)s_info.freeram*(long)s_info.mem_unit/(long)sizeof(double); // available memory in QWORDs
        occupmem = (long)m*(long)n*2+(long)m*(long)m+100000000; //spare 0.8GB for scratch use
        printf("Available memory in qwords: %ld\nMemory required in qwords: %ld\n",availmem,occupmem);
        if (occupmem > availmem){
            printf("Memory not enough, test aborted. Please adjust input parameters m or n.\n");exit(0);
        }

// allocate space for matrices and counter vector
	A = (double*) malloc(sizeof(double)*lda*lda);
	B1 = (double*) malloc(sizeof(double)*m*n);
	B2 = (double*) malloc(sizeof(double)*m*n);
	tscs = (double*) malloc(sizeof(double)*(2*niters));	// Counter vector
        if(A==NULL || B1==NULL || B2==NULL || tscs==NULL){
            printf("Memory allocation for arrays failed, now exit\n");
            if(A!=NULL) free(A);
            if(B1!=NULL) free(B1);
            if(B2!=NULL) free(B2);
            if(tscs!=NULL) free(tscs);
            tscs=NULL;A=B1=B2=NULL;
            dtrmmroutine1=NULL;dlclose(handle1);handle1=NULL;exit(1);
        }

// print test information
        printf("Matrix dimensions lower than 3 will be reset to 3\n");
	printf("Dimensions of matrix A: %d * %d\n",lda,lda);
	printf("Dimensions of matrix B: %d * %d\n",m,n);
	printf("Number of iterations %d \n",niters);
        printf("Elements of A and B will be generated randomly at the start of every iteration in the range [ %e , %e )\n",lowerbound,upperbound);
        srand((unsigned)time(NULL));
        for (i=0; i<2*niters; ++i) tscs[i] = 0.0;
        printf("Now start DTRMM iterations \n\n");

// **start DTRMM-compare iterations here**
    for (i=0; i<niters; ++i) {
        if(niters < 100) printf("Iteration %d:\n",i+1);
// initialization of matrices with random numbers
        for (j=0; j<lda; j++){
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,lda,&A[j*lda],lowerbound,upperbound);
            vslDeleteStream(&stream);
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,(m*n/lda),&B1[j*(m*n/lda)],lowerbound,upperbound);
            vslDeleteStream(&stream);
            memcpy(&B2[j*(m*n/lda)],&B1[j*(m*n/lda)],(m*n/lda)*sizeof(double));
        }

        gettimeofday(&starttime,0);
        dtrmm(&side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B1, &ldb);
        gettimeofday(&endtime,0);
        tscs[2*i] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100) printf("First 5 DP elements of product matrix B1: %e, %e, %e, %e, %e\n",B1[0],B1[1],B1[2],B1[3],B1[4]);

        gettimeofday(&starttime,0);
        (*dtrmmroutine1)(&side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B2, &ldb);
        gettimeofday(&endtime,0);
        tscs[2*i+1] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100) printf("First 5 DP elements of product matrix B2: %e, %e, %e, %e, %e\n",B2[0],B2[1],B2[2],B2[3],B2[4]);

// compare matrices B1 and B2
        maxdif = 0.0;maxelem = 0;
        for (j=0; j<m*n ; j++){
            tempdif = fabs(B1[j]-B2[j]);
	    if(tempdif > maxdif) {maxdif = tempdif;maxelem = j+1;}
        }
        printf("The max diff of paired DP elements in the 2 B matrices in iteration %d : element no. %ld: %e\n\n",i+1,maxelem,maxdif);
    }

// **end of iterations, print calculated FLOPS of routine 1 and 2 in each iteration**
        dtrmmroutine1=NULL;dlclose(handle1);DLERR = dlerror();handle1=NULL;
        if (DLERR) {
            printf ("Error in closing libraries:%s\n",DLERR);
            free(A);free(B1);free(B2);free(tscs);tscs=NULL;A=B1=B2=NULL;
            exit(1);
        }
	if(diag=='U') ops = (long)m*(long)n*(long)(lda);
	else          ops = (long)m*(long)n*(long)(lda+1);
        printf("SUMMARY of the test:\n");
	printf("FP operation count per dtrmm call:%ld\n",ops);
        printf("tested library: %s.\n",dtrmmpath1);
        printf("MKL library: 2019, linked with libgomp, 32-bit integer interface.\n");
	printf("Iter\tSeconds-MKL\tGFLOPS-MKL\tSeconds-test\tGFLOPS-test\n");
        for (i=0; i<niters; ++i) {
	    walltime1 = tscs[2*i]/1000000; //wall time in sec (routine 1)
	    walltime2 = tscs[2*i+1]/1000000; //wall time in sec (routine 2)
	    printf(" %d\t  %f\t %f\t    %f\t %f\n",i+1,walltime1,(double)ops/walltime1/1.0e9,walltime2,(double)ops/walltime2/1.0e9);
	}
        free(A);free(B1);free(B2);free(tscs);tscs=NULL;A=B1=B2=NULL;
        return 0;
}
