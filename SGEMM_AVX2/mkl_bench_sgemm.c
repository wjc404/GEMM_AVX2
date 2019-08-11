#include <stdio.h> // for printf(),...
#include <stdlib.h> // for rand(), malloc(), exit(), atoi(),...
#include <time.h> // for time() as seed
#include <math.h> // for fabs()
#include <dlfcn.h> // for dynamic linking of sgemm function to specified CPU/GPU libraries
#include "/opt/intel/mkl/include/mkl_vsl.h" // for MKL-based random number generator
#include <sys/time.h> // for timing of sgemm
#include <sys/sysinfo.h> // for determining available memory
#include <unistd.h> // for sleep()
//this program is for tests (about performances and outputs) of user-provided sgemm library against Intel MKL
/*an example of compilation command
gcc -fopenmp mkl_bench_sgemm.c -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -o mkl_bench_sgemm
*/

int main(int argc, char* argv[]) // command line: ./general_benchmark_dev [niter] [m] [n] [k] [upperbound] [lowerbound] [transa] [transb]
{
// variables blow for dynamic linking
        void (*sgemmroutine1)(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
        void sgemm(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);
        void *handle1;
        char sgemmpath1[200],sgemmname1[18]; // get linking info from stdin
        char *DLERR;
// variables blow for main sgemm test
	float *A,*B,*C;
        float alpha=1.0;
        float beta=1.0;
	char trsa,trsb;

        long ops,availmem,occupmem,maxelem;
	int i,j,k,m,n,niters,seed,error,lda,ldb,ldc;
	float walltime1,walltime2,maxdif,tempdif,upperbound,lowerbound;
        VSLStreamStatePtr stream; // pointer to MKL_stream_state structure
        struct sysinfo s_info; // for getting memory info
        struct timeval starttime,endtime; // for linux timing
        float *tscs; //for storing time

// variables blow for testing of dynamic linking
        float a[8]={1.1,1.2,1.3,1.4,-1.7,-2.1,-4.1,0.9};
        float b[12]={-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2};
        float c[6]={1.78,-0.13,3.14,-0.69,4.5,-1.25}; //reference product
        float d[6]={0.0,0.0,0.0,0.0,0.0,0.0};
        int tempm=2;
        int tempn=3;
        int tempk=4;
        int templda,templdb,templdc;

// get dimension(default=2000) of matrices and number(default=4) of iterations
	if (argc >= 2) {
		niters = atoi(argv[1]);
	} else {
		niters = 4;
	}
        if (niters < 1 || niters > 300) niters = 1;
	if (argc >= 3) {
		m = atoi(argv[2]);
	} else {
		m = 2000;
	}
        if (m < 10) m = 10;
	if (argc >= 4) {
		n = atoi(argv[3]);
	} else {
		n = 2000;
	}
        if (n < 10) n = 10;
	if (argc >= 5) {
		k = atoi(argv[4]);
	} else {
		k = 2000;
	}
        if (k < 10) k = 10;
        templdc = tempm; ldc = m;
// get range of random numbers filled in A and B, default [0,10)
	if (argc >= 6) {
		upperbound = atof(argv[5]);
	} else {
		upperbound = (float) 10;
	}
	if (argc >= 7) {
		lowerbound = atof(argv[6]);
	} else {
		lowerbound = (float) 0;
	}
        if (upperbound <= lowerbound) { // check the order
                upperbound = upperbound + lowerbound;
                lowerbound = upperbound - lowerbound;
                upperbound = upperbound - lowerbound;
        }
	if (argc >= 8) {
		trsa = *argv[7];
	} else {
		trsa = 'N';
	}
        if(trsa=='N' || trsa=='n'){
                trsa = 'N'; lda = m; templda = tempm;
        }else{
                trsa = 'T'; lda = k; templda = tempk;
                a[0]=1.1;a[1]=1.3;a[2]=-1.7;a[3]=-4.1;a[4]=1.2;a[5]=1.4;a[6]=-2.1;a[7]=0.9;
        }
	if (argc >= 9) {
		trsb = *argv[8];
	} else {
		trsb = 'N';
	}
        if(trsb=='N' || trsb=='n'){
                trsb = 'N'; ldb = k; templdb = tempk;
        }else{
                trsb = 'T'; ldb = n; templdb = tempn;
                b[0]=-0.1;b[1]=-0.5;b[2]=-0.9;b[3]=-0.2;b[4]=-0.6;b[5]=-1.0;b[6]=-0.3;b[7]=-0.7;b[8]=-1.1;b[9]=-0.4;b[10]=-0.8;b[11]=-1.2;
        }
        printf("A SIMPLE SGEMM TESTING PROGRAM\n");
        printf("Matrix elements will be stored in column-major order.\n");
        printf("Transposition: matrix A: %c; matrix B: %c.\n",trsa,trsb);
        printf("SGEMM will be called in this way:\n");
        printf("    <sgemmname>(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)\n");
        printf("Please make sure that the sgemm library you provided can respond normally to the call above\n\n");
// load the tested sgemm routine
        dlerror();
        printf("Enter your sgemm library path(path to the *.so file):");
        scanf("%s",sgemmpath1);
        handle1 = dlopen(sgemmpath1,RTLD_LAZY);
        DLERR = dlerror();
        if (DLERR) {  
            printf ("Error locating the library: %s\n",DLERR);  
            exit(1);  
        }
        printf("Enter the function name of sgemm in your library(e.g., sgemm):");
        scanf("%s",sgemmname1);
        sgemmroutine1 = dlsym(handle1,sgemmname1);
        DLERR = dlerror();
        if (DLERR) {
            printf ("Error locating sgemm function in your library: %s\n",DLERR);
            sgemmroutine1=NULL;dlclose(handle1);handle1=NULL;
            exit(1);
        }
        printf("Now test sgemm with small matrices, identical results indicate successful linking\n");
        (*sgemmroutine1)(&trsa, &trsb, &tempm, &tempn, &tempk, &alpha, a, &templda, b, &templdb, &beta, d, &templdc);
        printf("Reference product matrix:         %e  %e  %e  %e  %e  %e\n",c[0],c[1],c[2],c[3],c[4],c[5]);
        printf("Product matrix from your library: %e  %e  %e  %e  %e  %e\n\n",d[0],d[1],d[2],d[3],d[4],d[5]);

        sleep(5); // pause for viewing linking results
// check memory availability
        error = sysinfo(&s_info);
        if (error != 0){
            printf ("Cannot get memory info, now exit\n");
            sgemmroutine1=NULL;
            dlclose(handle1);handle1=NULL;
            exit(1);
        }
        availmem = (long)s_info.freeram*(long)s_info.mem_unit/(long)sizeof(float); // available memory for floats in DWORDs
        occupmem = (long)m*(long)n+(long)m*(long)k+(long)n*(long)k+100000000;
        printf("Available memory in dwords: %ld\nMemory required in dwords: %ld\n",availmem,occupmem);//debug
        if (occupmem > availmem){
            lda = ldb = ldc = k = n = m = (int)sqrt(availmem/3-100000000);
            printf("Matrix dimensions reset to %d-%d-%d due to memory limitations\n",m,n,k);
        }
	beta = 1.0;
// allocate space for matrices and counter vector
	A = (float*) malloc(sizeof(float)*m*k);
	B = (float*) malloc(sizeof(float)*k*n);
	C = (float*) malloc(sizeof(float)*m*n);
	tscs = (float*) malloc(sizeof(float)*(2*niters));	// Counter vector
        if(A==NULL || B==NULL || C==NULL || tscs==NULL){
            printf("Memory allocation for arrays failed, now exit\n");
            if(A!=NULL) free(A);
            if(B!=NULL) free(B);
            if(C!=NULL) free(C);
            if(tscs!=NULL) free(tscs);
            tscs=NULL;A=B=C=NULL;
            sgemmroutine1=NULL;
            dlclose(handle1);handle1=NULL;
            exit(1);
        }
// print test information
        printf("Matrix dimensions lower than 10 will be reset to 10\n");
	printf("Dimensions of matrix A: %d * %d\n",m,k);
	printf("Dimensions of matrix B: %d * %d\n",k,n);
	printf("Number of iterations %d \n",niters);
        printf("Elements of A and B will be generated randomly at the start of every iteration in the range [ %e , %e )\n",lowerbound,upperbound);
        printf("Expected value of every element in the product matrix: %e\n",2*(lowerbound+upperbound)*(lowerbound+upperbound)/4*k);
        srand((unsigned)time(NULL));
        for (i=0; i<2*niters; ++i) tscs[i] = 0.0;
        printf("Now start SGEMM iterations \n\n");
// **start SGEMM-compare iterations here**
    for (i=0; i<niters; ++i) {
        if(niters < 100) printf("Iteration %d:\n",i+1);
// initialization of matrices
#pragma omp parallel for
	for (j=0; j<m*n ; j++) C[j] = 0.0;
//#pragma omp parallel for private(seed) private(stream)
        for (j=0; j<k; j++){  // Randomly generate double-precision numbers to fill the matrices(A and B); Use efficient MKL implementation (far more efficient than rand())
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,m,&A[j*m],lowerbound,upperbound);
            vslDeleteStream(&stream);
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n,&B[j*n],lowerbound,upperbound);
            vslDeleteStream(&stream);
        }
        if(niters < 100){
            printf("First 5 elements of matrix A: %e, %e, %e, %e, %e\n",A[0],A[1],A[2],A[3],A[4]);//DEBUG
            printf("First 5 elements of matrix B: %e, %e, %e, %e, %e\n",B[0],B[1],B[2],B[3],B[4]);//DEBUG
        }
// start C = 2AB
	alpha = 2.0;beta = -1.0;
        gettimeofday(&starttime,0);
        sgemm(&trsa, &trsb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        gettimeofday(&endtime,0);
        tscs[2*i] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100) printf("First 5 elements of product matrix (AB)test: %e, %e, %e, %e, %e\n",C[0],C[1],C[2],C[3],C[4]);//DEBUG
// start C = 2C - 4AB
	alpha = -4.0;beta = 2.0;
        gettimeofday(&starttime,0);
        (*sgemmroutine1)(&trsa, &trsb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        gettimeofday(&endtime,0);
        tscs[2*i+1] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100) printf("First 5 elements of matrix (AB)test-(AB)mkl: %e, %e, %e, %e, %e\n",C[0],C[1],C[2],C[3],C[4]);//DEBUG
// compare matrices C and [0]
        maxdif = 0.00;maxelem = 0;
        for (j=0; j<m*n ; j++){
            tempdif = fabs(C[j]);
	    if(tempdif > maxdif) {maxdif = tempdif;maxelem = j;}
        }
        printf("Max abs of elements in '(AB)test-(AB)mkl' in iteration %d : element no. %ld: %e\n\n",i+1,maxelem,maxdif);
    }
// **end of iterations, print calculated FLOPS of routine 1 and 2 in each iteration**
        sgemmroutine1=NULL;
        dlclose(handle1);
        DLERR = dlerror();
        handle1=NULL;
        if (DLERR) {
            printf ("Error in closing libraries:%s\n",DLERR);
            free(A);free(B);free(C);free(tscs);tscs=NULL;A=B=C=NULL;
            exit(1);
        }
	ops = (long)2*(long)m*(long)n*(long)k; //FP operations in one sgemm run
        printf("SUMMARY of the test:\n");
	printf("FP operation count per sgemm call:%ld\n",ops);
        printf("lib1: %s.\n",sgemmpath1);
        printf("MKL library: 2018, linked with libgomp, 32-bit integer interface.\n");
	printf("Iter\tSeconds-MKL2018\tGFLOPS-MKL2018\tSeconds-lib1\tGFLOPS-lib1\n");
        for (i=0; i<niters; ++i) {
	    walltime1 = tscs[2*i]/1000000; //wall time in sec (routine 1)
	    walltime2 = tscs[2*i+1]/1000000; //wall time in sec (routine 2)
	    printf(" %d\t  %f\t %f\t    %f\t %f\n",i+1,walltime1,(float)ops/walltime1/1.0e9,walltime2,(float)ops/walltime2/1.0e9);
	}
        free(A);free(B);free(C);free(tscs);tscs=NULL;A=B=C=NULL;
        return 0;
}
