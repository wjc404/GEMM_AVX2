#include <stdio.h> // for printf(),...
#include <stdlib.h> // for rand(), malloc(), exit(), atoi(),...
#include <time.h> // for time() as seed
#include <math.h> // for fabs()
#include <dlfcn.h> // for dynamic linking of dgemm function to specified CPU/GPU libraries
#include "/opt/intel/mkl/include/mkl_vsl.h" // for MKL-based random number generator
#include <sys/time.h> // for timing of dgemm
#include <sys/sysinfo.h> // for determining available memory
#include <unistd.h> // for sleep()
//this program is for comparative test (about performances and outputs) of 2 dgemm libraries provided by user.
/* an example of compilation command
gcc -fopenmp General_Benchmark_DEV.c -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_sequential.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -o general_benchmark_dev
*/
// command line: ./general_benchmark_dev [niter] [m] [n] [k] [upperbound] [lowerbound] [transa] [transb]
// please set OMP_NUM_THREADS before calling the program
int main(int argc, char* argv[]) 
{
// variables blow for dynamic linking
        void (*dgemmroutine1)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
        void (*dgemmroutine2)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
        void *handle1, *handle2;
        char dgemmpath1[200],dgemmpath2[200],dgemmname1[18],dgemmname2[18]; // get linking info from stdin
        char *DLERR;
// variables blow for main dgemm test
	double *A,*B,*C;
        double alpha=1.0;
        double beta=1.0;
	char trsa,trsb;

        long ops,availmem,occupmem,maxelem;
	int i,j,k,m,n,niters,seed,error,lda,ldb,ldc;
	double walltime1,walltime2,maxdif,tempdif,upperbound,lowerbound;
        VSLStreamStatePtr stream; // pointer to MKL_stream_state structure
        struct sysinfo s_info; // for getting memory info
        struct timeval starttime,endtime; // for linux timing
        double *tscs; //for storing time

// variables blow for testing of dynamic linking
        double a[8]={1.1,1.2,1.3,1.4,-1.7,-2.1,-4.1,0.9};
        double b[12]={-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-1.1,-1.2};
        double c[6]={1.78,-0.13,3.14,-0.69,4.5,-1.25}; //reference product
        double d[6]={0.0,0.0,0.0,0.0,0.0,0.0};
        double e[6]={0.0,0.0,0.0,0.0,0.0,0.0};
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
        if (m < 1) m = 1;
	if (argc >= 4) {
		n = atoi(argv[3]);
	} else {
		n = 2000;
	}
        if (n < 1) n = 1;
	if (argc >= 5) {
		k = atoi(argv[4]);
	} else {
		k = 2000;
	}
        if (k < 1) k = 1;
        templdc = tempm; ldc = m;
// get range of random numbers filled in A and B, default [0,10)
	if (argc >= 6) {
		upperbound = atof(argv[5]);
	} else {
		upperbound = (double) 10;
	}
	if (argc >= 7) {
		lowerbound = atof(argv[6]);
	} else {
		lowerbound = (double) 0;
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
        printf("A SIMPLE DGEMM TESTING PROGRAM\n");
        printf("Matrix elements will be stored in column-major order.\n");
        printf("Transposition: matrix A: %c; matrix B: %c.\n",trsa,trsb);
        printf("DGEMM will be called in this way:\n");
        printf("    dgemm(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)\n");
        printf("Please make sure that the dgemm routines you want to test can respond normally to the call above\n");
        printf("Please note that dgemm functions in different BLAS libraries can have different names\n");
        printf("    e.g. Intel MKL and NVIDIA nvblas use the name 'dgemm' but OpenBLAS uses 'dgemm_'\n\n");
// load the first dgemm routine
        dlerror();
        printf("Enter the library path of dgemm routine 1 (path to the *.so file):");
        scanf("%s",dgemmpath1);
        handle1 = dlopen(dgemmpath1,RTLD_LAZY);
        DLERR = dlerror();
        if (DLERR) {  
            printf ("Error locating the first library: %s\n",DLERR);  
            exit(1);  
        }
        printf("Enter the function name of dgemm in routine 1 (e.g., dgemm):");
        scanf("%s",dgemmname1);
        dgemmroutine1 = dlsym(handle1,dgemmname1);
        DLERR = dlerror();
        if (DLERR) {
            printf ("Error locating dgemm in the first library: %s\n",DLERR);
            dgemmroutine1=NULL;dlclose(handle1);handle1=NULL;
            exit(1);
        }
        printf("Now test dgemm routine 1 with small matrices, identical results indicate successful linking\n");
        (*dgemmroutine1)(&trsa, &trsb, &tempm, &tempn, &tempk, &alpha, a, &templda, b, &templdb, &beta, d, &templdc); //test of routine 1
        printf("Elements of reference product matrix: %e  %e  %e  %e  %e  %e\n",c[0],c[1],c[2],c[3],c[4],c[5]);
        printf("Elements of routine-1 product matrix: %e  %e  %e  %e  %e  %e\n\n",d[0],d[1],d[2],d[3],d[4],d[5]);
// load the second dgemm routine
        printf("Enter the library path of dgemm routine 2 (path to the *.so file):");
        scanf("%s",dgemmpath2); //adjustment from icc-version
        handle2 = dlopen(dgemmpath2,RTLD_LAZY);
        DLERR = dlerror();
        if (DLERR) {  
            printf ("Error locating the second library: %s\n",DLERR);
            dgemmroutine1=NULL;dlclose(handle1);handle1=NULL;
            exit(1);  
        }
        printf("Enter the function name of dgemm in routine 2 (e.g., dgemm):");
        scanf("%s",dgemmname2); //adjustment from icc-version
        dgemmroutine2 = dlsym(handle2,dgemmname2);
        DLERR = dlerror();
        if (DLERR) {
            printf ("Error locating dgemm in the second library: %s\n",DLERR);
            dgemmroutine2=NULL;dlclose(handle2);handle2=NULL;dgemmroutine1=NULL;dlclose(handle1);handle1=NULL;
            exit(1);
        }
        printf("Now test dgemm routine 2 with small matrices, identical results indicate successful linking\n");
        (*dgemmroutine2)(&trsa, &trsb, &tempm, &tempn, &tempk, &alpha, a, &templda, b, &templdb, &beta, e, &templdc); //test of routine 2
        printf("Elements of reference product matrix: %e  %e  %e  %e  %e  %e\n",c[0],c[1],c[2],c[3],c[4],c[5]);
        printf("Elements of routine-2 product matrix: %e  %e  %e  %e  %e  %e\n\n",e[0],e[1],e[2],e[3],e[4],e[5]);
        sleep(5); // pause for viewing linking results
// check memory availability
        error = sysinfo(&s_info);
        if (error != 0){
            printf ("Cannot get memory info, now exit\n");
            dgemmroutine2=dgemmroutine1=NULL;
            dlclose(handle1);dlclose(handle2);handle1=handle2=NULL;
            exit(1);
        }
        availmem = (long)s_info.freeram*(long)s_info.mem_unit/(long)sizeof(double); // available memory for doubles
        occupmem = (long)m*(long)n+(long)m*(long)k+(long)n*(long)k+100000000; //0.8 GB preserved for DGEMM libraries
        printf("Available memory in QWords: %ld\nMemory required in QWords: %ld\n",availmem,occupmem);//debug
        if (occupmem > availmem){ // if the matrices will occupy more than half of the free memory
            lda = ldb = ldc = k = n = m = (int)sqrt(availmem/3-100000000);
            printf("Matrix dimensions reset to %d-%d-%d due to memory limitations\n",m,n,k);
        }
// allocate space for matrices and counter vector
	A = (double*) malloc(sizeof(double)*m*k);
	B = (double*) malloc(sizeof(double)*k*n);
	C = (double*) malloc(sizeof(double)*m*n);
	tscs = (double*) malloc(sizeof(double)*(2*niters));	// Counter vector
        if(A==NULL || B==NULL || C==NULL || tscs==NULL){
            printf("Memory allocation for arrays failed, now exit\n");
            if(A!=NULL) free(A);
            if(B!=NULL) free(B);
            if(C!=NULL) free(C);
            if(tscs!=NULL) free(tscs);
            tscs=NULL;A=B=C=NULL;
            dgemmroutine2=dgemmroutine1=NULL;
            dlclose(handle1);dlclose(handle2);handle1=handle2=NULL;
            exit(1);
        }
// print test information
	printf("Dimensions of matrix A: %d * %d\n",m,k);
	printf("Dimensions of matrix B: %d * %d\n",k,n);
	printf("Number of iterations %d \n",niters);
        printf("Elements of A and B will be generated randomly at the start of every iteration in the range [ %e , %e )\n",lowerbound,upperbound);
        printf("Expected value of every element in the product matrix: %e\n",2*(lowerbound+upperbound)*(lowerbound+upperbound)/4*k);
        srand((unsigned)time(NULL));
        for (i=0; i<2*niters; ++i) tscs[i] = 0.0;
        printf("Now start DGEMM iterations \n\n");
// **start DGEMM-compare iterations here**
    for (i=0; i<niters; ++i) {
        if(niters < 100) printf("Iteration %d:\n",i+1);
// initialization of matrices
#pragma omp parallel for
	for (j=0; j<m*n ; j++) C[j] = 0.0;
#pragma omp parallel for private(seed) private(stream)
        for (j=0; j<k; j++){  // Randomly generate double-precision numbers to fill the matrices(A and B); Use efficient MKL implementation (far more efficient than rand())
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,m,&A[j*m],lowerbound,upperbound);
            vslDeleteStream(&stream);
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,n,&B[j*n],lowerbound,upperbound);
            vslDeleteStream(&stream);
        }
        if(niters < 100 && m*n >= 5){
            printf("First 5 elements of matrix A: %e, %e, %e, %e, %e\n",A[0],A[1],A[2],A[3],A[4]);//DEBUG
            printf("First 5 elements of matrix B: %e, %e, %e, %e, %e\n",B[0],B[1],B[2],B[3],B[4]);//DEBUG
        }
// start C = 2 * (AB)1
	beta = 1.0;
	alpha = 2.0;
        gettimeofday(&starttime,0);
        (*dgemmroutine1)(&trsa, &trsb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        gettimeofday(&endtime,0);
        tscs[2*i] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100 && m*n >= 5) printf("First 5 elements of product matrix (AB)1: %e, %e, %e, %e, %e\n",C[0],C[1],C[2],C[3],C[4]);//DEBUG
// start C = 0.5 * C - (AB)2
        beta = 0.5;
	alpha = -1.0;
        gettimeofday(&starttime,0);
        (*dgemmroutine2)(&trsa, &trsb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        gettimeofday(&endtime,0);
        tscs[2*i+1] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100 && m*n >= 5) printf("First 5 elements of matrix (AB)1-(AB)2: %e, %e, %e, %e, %e\n",C[0],C[1],C[2],C[3],C[4]);//DEBUG
// compare matrices C and [0]
        maxdif = 0.00;maxelem = 0;
        for (j=0; j<m*n ; j++){
            tempdif = fabs(C[j]);
	    if(tempdif > maxdif) {maxdif = tempdif;maxelem = j;}
        }
        printf("Max abs of elements in '(AB)1-(AB)2' in iteration %d : element no. %ld: %e\n\n",i+1,maxelem,maxdif);
    }
// **end of iterations, print calculated FLOPS of routine 1 and 2 in each iteration**
        dgemmroutine2=dgemmroutine1=NULL;
        dlclose(handle1);
        dlclose(handle2);
        DLERR = dlerror();
        handle1=handle2=NULL;
        if (DLERR) {
            printf ("Error in closing libraries:%s\n",DLERR);
            free(A);free(B);free(C);free(tscs);tscs=NULL;A=B=C=NULL;
            exit(1);
        }
	ops = (long)2*(long)m*(long)n*(long)k; //FP operations in one dgemm run
        printf("SUMMARY of the test:\n");
	printf("FP operation count per dgemm call:%ld\n",ops);
        printf("Routine 1: %s\n",dgemmpath1);
        printf("Routine 2: %s\n",dgemmpath2);
	printf("Iter Seconds-routine1 GFLOPS-routine1 Seconds-routine2 GFLOPS-routine2\n");
        for (i=0; i<niters; ++i) {
	    walltime1 = tscs[2*i]/1000000; //wall time in sec (routine 1)
	    walltime2 = tscs[2*i+1]/1000000; //wall time in sec (routine 2)
	    printf(" %d       %f       %f       %f       %f\n",i+1,walltime1,(double)ops/walltime1/1.0e9,walltime2,(double)ops/walltime2/1.0e9);
	}
        free(A);free(B);free(C);free(tscs);tscs=NULL;A=B=C=NULL;
        return 0;
}
