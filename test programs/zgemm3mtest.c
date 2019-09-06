#include <stdio.h> // for printf(),...
#include <stdlib.h> // for rand(), malloc(), exit(), atoi(),...
#include <time.h> // for time() as seed
#include <math.h> // for fabs()
#include <dlfcn.h> // for dynamic linking of dgemm function to specified CPU/GPU libraries
#include "/opt/intel/mkl/include/mkl_vsl.h" // for MKL-based random number generator
#include <sys/time.h> // for timing of zgemm3m
#include <sys/sysinfo.h> // for determining available memory
#include <unistd.h> // for sleep()
//this program is for tests (about performances and outputs) of user-provided zgemm3m library against Intel MKL
/*an example of compilation command
gcc -fopenmp zgemm3mtest.c -Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -o zgemm3mtest
*/

int main(int argc, char* argv[]) // command line: ./zgemm3mtest [niter] [m] [n] [k] [upperbound] [lowerbound] [transa] [transb]
{
// variables blow for dynamic linking
        void (*zgemm3mroutine1)(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
        void zgemm3m(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
        void *handle1;
        char zgemm3mpath1[200],zgemm3mname1[18]; // get linking info from stdin
        char *DLERR;
// variables blow for main dgemm test
	double *A,*B,*C;
        double alpha[2]={1.0,0.0};
        double beta[2]={1.0,0.0};
	char trsa,trsb;

        long ops,availmem,occupmem,maxelem;
	int i,j,k,m,n,niters,seed,error,lda,ldb,ldc;
	double walltime1,walltime2,maxdif,tempdif,upperbound,lowerbound;
        VSLStreamStatePtr stream; // pointer to MKL_stream_state structure
        struct sysinfo s_info; // for getting memory info
        struct timeval starttime,endtime; // for linux timing
        double *tscs; //for storing time

// variables blow for testing of dynamic linking
        double a[16]={1.1,0.0,1.2,0.0,1.3,0.0,1.4,0.0,-1.7,0.0,-2.1,0.0,-4.1,0.0,0.9,0.0};
        double b[24]={-0.1,0.0,-0.2,0.0,-0.3,0.0,-0.4,0.0,-0.5,0.0,-0.6,0.0,-0.7,0.0,-0.8,0.0,-0.9,0.0,-1.0,0.0,-1.1,0.0,-1.2,0.0};
        double c[12]={1.78,0.0,-0.13,0.0,3.14,0.0,-0.69,0.0,4.5,0.0,-1.25,0.0}; //reference product
        double d[12]={0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
        int tempm=2;
        int tempn=3;
        int tempk=4;
        int templda=tempm;
        int templdb=tempk;
        int templdc=tempm;
        char temptrsa='N';
        char temptrsb='N';

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
        if (m < 3) m = 3;
	if (argc >= 4) {
		n = atoi(argv[3]);
	} else {
		n = 2000;
	}
        if (n < 3) n = 3;
	if (argc >= 5) {
		k = atoi(argv[4]);
	} else {
		k = 2000;
	}
        if (k < 3) k = 3;
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
                trsa = 'N'; lda = m;
        }else{
                lda = k;
                if(trsa=='T'||trsa=='t') trsa=='T';
                else trsa=='C';
        }
	if (argc >= 9) {
		trsb = *argv[8];
	} else {
		trsb = 'N';
	}
        if(trsb=='N' || trsb=='n'){
                trsb = 'N'; ldb = k;
        }else{
                ldb = n;
                if(trsb=='T'||trsb=='t') trsb=='T';
                else trsb=='C';
        }
        printf("A SIMPLE ZGEMM3M TESTING PROGRAM\n");
        printf("Matrix elements will be stored in column-major order.\n");
        printf("Transposition: matrix A: %c; matrix B: %c.\n",trsa,trsb);
        printf("ZGEMM3M will be called in this way:\n");
        printf("    <zgemm3m_name>(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc)\n");
        printf("Please make sure that the zgemm3m library you provided can respond normally to the call above\n\n");
// load the tested zgemm3m routine
        dlerror();
        printf("Enter your zgemm3m library path(path to the *.so file):");
        scanf("%s",zgemm3mpath1);
        handle1 = dlopen(zgemm3mpath1,RTLD_LAZY);
        DLERR = dlerror();
        if (DLERR) {  
            printf ("Error locating the library: %s\n",DLERR);  
            exit(1);  
        }
        printf("Enter the function name of zgemm3m in your library(e.g., zgemm3m):");
        scanf("%s",zgemm3mname1);
        zgemm3mroutine1 = dlsym(handle1,zgemm3mname1);
        DLERR = dlerror();
        if (DLERR) {
            printf ("Error locating zgemm3m function in your library: %s\n",DLERR);
            zgemm3mroutine1=NULL;dlclose(handle1);handle1=NULL;
            exit(1);
        }
        printf("Now test zgemm3m with small matrices, identical results indicate successful linking\n");
        (*zgemm3mroutine1)(&temptrsa, &temptrsb, &tempm, &tempn, &tempk, alpha, a, &templda, b, &templdb, beta, d, &templdc);
        printf("Reference product matrix:         %e  %e  %e  %e  %e  %e\n",c[0],c[2],c[4],c[6],c[8],c[10]);
        printf("Product matrix from your library: %e  %e  %e  %e  %e  %e\n\n",d[0],d[2],d[4],d[6],d[8],d[10]);

        sleep(5); // pause for viewing linking results
// check memory availability
        error = sysinfo(&s_info);
        if (error != 0){
            printf ("Cannot get memory info, now exit\n");
            zgemm3mroutine1=NULL;
            dlclose(handle1);handle1=NULL;
            exit(1);
        }
        availmem = (long)s_info.freeram*(long)s_info.mem_unit/(long)sizeof(double)/2; // available memory in OCTAWORDs
        occupmem = (long)m*(long)n+(long)m*(long)k+(long)n*(long)k+100000000;
        printf("Available memory in owords: %ld\nMemory required in owords: %ld\n",availmem,occupmem);//debug
        if (occupmem > availmem){
            lda = ldb = ldc = k = n = m = (int)sqrt(availmem/3-100000000);
            printf("Matrix dimensions reset to %d-%d-%d due to memory limitations\n",m,n,k);
        }
// allocate space for matrices and counter vector
	A = (double*) malloc(sizeof(double)*2*m*k);
	B = (double*) malloc(sizeof(double)*2*k*n);
	C = (double*) malloc(sizeof(double)*2*m*n);
	tscs = (double*) malloc(sizeof(double)*(2*niters));	// Counter vector
        if(A==NULL || B==NULL || C==NULL || tscs==NULL){
            printf("Memory allocation for arrays failed, now exit\n");
            if(A!=NULL) free(A);
            if(B!=NULL) free(B);
            if(C!=NULL) free(C);
            if(tscs!=NULL) free(tscs);
            tscs=NULL;A=B=C=NULL;
            zgemm3mroutine1=NULL;
            dlclose(handle1);handle1=NULL;
            exit(1);
        }
// print test information
        printf("Matrix dimensions lower than 3 will be reset to 3\n");
	printf("Dimensions of matrix A: %d * %d\n",m,k);
	printf("Dimensions of matrix B: %d * %d\n",k,n);
	printf("Number of iterations %d \n",niters);
        printf("DP Elements of A and B will be generated randomly at the start of every iteration in the range [ %e , %e )\n",lowerbound,upperbound);
        srand((unsigned)time(NULL));
        for (i=0; i<2*niters; ++i) tscs[i] = 0.0;
        printf("Now start ZGEMM3M iterations \n\n");
// **start ZGEMM3M-compare iterations here**
    for (i=0; i<niters; ++i) {
        if(niters < 100) printf("Iteration %d:\n",i+1);
// initialization of matrices
#pragma omp parallel for
	for (j=0; j<2*m*n ; j++) C[j] = 0.0;
//#pragma omp parallel for private(seed) private(stream)
        for (j=0; j<k; j++){  // Randomly generate double-precision numbers to fill the matrices(A and B); Use efficient MKL implementation (far more efficient than rand())
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,2*m,&A[j*2*m],lowerbound,upperbound);
            vslDeleteStream(&stream);
            seed = rand();
            vslNewStream(&stream,VSL_BRNG_MCG31,seed);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,2*n,&B[j*2*n],lowerbound,upperbound);
            vslDeleteStream(&stream);
        }
        if(niters < 100){
            printf("First 5 DP elements of matrix A: %e, %e, %e, %e, %e\n",A[0],A[1],A[2],A[3],A[4]);//DEBUG
            printf("First 5 DP elements of matrix B: %e, %e, %e, %e, %e\n",B[0],B[1],B[2],B[3],B[4]);//DEBUG
        }
// start first mult
	alpha[0]=4.0;alpha[1]=2.0;beta[0]=2.0;beta[1]=0.0;
        gettimeofday(&starttime,0);
        zgemm3m(&trsa, &trsb, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
        gettimeofday(&endtime,0);
        tscs[2*i] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100) printf("First 5 DP elements of product matrix (4+2i)AB: %e, %e, %e, %e, %e\n",C[0],C[1],C[2],C[3],C[4]);//DEBUG
// start second mult
	alpha[0]=-10.0;alpha[1]=-10.0;beta[0]=3.0;beta[1]=1.0;//beta=3+i;alpha=-(4+2i)(3+i)=-10-10i
        gettimeofday(&starttime,0);
        (*zgemm3mroutine1)(&trsa, &trsb, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
        gettimeofday(&endtime,0);
        tscs[2*i+1] = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec; //interval in usec
        if(niters < 100) printf("First 5 DP elements of matrix (-10-10i)AB+(3+i)C: %e, %e, %e, %e, %e\n",C[0],C[1],C[2],C[3],C[4]);//DEBUG
// compare matrices C and [0]
        maxdif = 0.00;maxelem = 0;
        for (j=0; j<2*m*n ; j++){
            tempdif = fabs(C[j]);
	    if(tempdif > maxdif) {maxdif = tempdif;maxelem = j;}
        }
        printf("Max abs of DP elements in the final matrix in iteration %d : element no. %ld: %e\n\n",i+1,maxelem,maxdif);
    }
// **end of iterations, print calculated FLOPS of routine 1 and 2 in each iteration**
        zgemm3mroutine1=NULL;
        dlclose(handle1);
        DLERR = dlerror();
        handle1=NULL;
        if (DLERR) {
            printf ("Error in closing libraries:%s\n",DLERR);
            free(A);free(B);free(C);free(tscs);tscs=NULL;A=B=C=NULL;
            exit(1);
        }
	ops = (long)2*(long)m*(long)n*(long)k; //FP operations in one zgemm run
        printf("SUMMARY of the test:\n");
	printf("Equivalent FP operation count per zgemm call:%ld\n",ops);
        printf("lib1: %s.\n",zgemm3mpath1);
        printf("MKL library: 2018, linked with libgomp, 32-bit integer interface.\n");
	printf("Iter\tSeconds-MKL2018\tGFLOPS-MKL2018\tSeconds-lib1\tGFLOPS-lib1\n");
        for (i=0; i<niters; ++i) {
	    walltime1 = tscs[2*i]/1000000; //wall time in sec (routine 1)
	    walltime2 = tscs[2*i+1]/1000000; //wall time in sec (routine 2)
	    printf(" %d\t  %f\t %f\t    %f\t %f\n",i+1,walltime1,(double)ops/walltime1/1.0e9,walltime2,(double)ops/walltime2/1.0e9);
	}
        free(A);free(B);free(C);free(tscs);tscs=NULL;A=B=C=NULL;
        return 0;
}
