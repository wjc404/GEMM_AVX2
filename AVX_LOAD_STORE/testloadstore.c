#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//compilation command: gcc -march=haswell testloadstore.S testloadstore.c -o testloadstore

extern void test_broadcast_d4(double *baseaddr, int64_t ops);
extern void test_broadcast_d8(double *baseaddr, int64_t ops);
extern void test_broadcast_d16(double *baseaddr, int64_t ops);
extern void test_maskload_d32(double *baseaddr, int64_t ops);
extern void test_loada_d32(double *baseaddr, int64_t ops);
extern void test_loadu_d32(double *baseaddr, int64_t ops);
extern void test_loada_d64(double *baseaddr, int64_t ops);
extern void test_loadu_d64(double *baseaddr, int64_t ops);
extern void test_storea_d32(double *baseaddr, int64_t ops);
extern void test_storeu_d32(double *baseaddr, int64_t ops);
extern void test_storea_d64(double *baseaddr, int64_t ops);
extern void test_storeu_d64(double *baseaddr, int64_t ops);
extern void test_maskstore_d32(double *baseaddr, int64_t ops);

int main(){
  struct timeval starttime,endtime;
  double usec_elapsed, gops_per_sec;
  const int64_t ops = 8589934592; // 2^33 operations per test
  double *array=(double *)aligned_alloc(4096,20480);

  //put data into L1 cache
  test_loada_d64(array,(int64_t)512);
  test_loada_d64(array+128,(int64_t)512);
  printf("All data are loaded into L1 cache.\n");

  gettimeofday(&starttime,0);
  test_loada_d64(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovapd loads (increment 64B): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_loadu_d64(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd loads (increment 64B, 256-bit aligned, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_loadu_d64(array+2,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd loads (increment 64B, 256-bit unaligned, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_loadu_d64(array+6,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd loads (increment 64B, cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storea_d64(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovapd stores (increment 64B): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storeu_d64(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd stores (increment 64B, 256-bit aligned, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storeu_d64(array+2,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd stores (increment 64B, 256-bit unaligned, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storeu_d64(array+6,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd stores (increment 64B, cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_loada_d32(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovapd loads (increment 32B): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_loadu_d32(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd loads (increment 32B, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_loadu_d32(array+6,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd loads (increment 32B, half-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_maskload_d32(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmaskmovpd loads (increment 32B, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_maskload_d32(array+6,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmaskmovpd loads (increment 32B, half-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storea_d32(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovapd stores (increment 32B): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storeu_d32(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd stores (increment 32B, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_storeu_d32(array+6,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmovupd stores (increment 32B, half-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_maskstore_d32(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmaskmovpd stores (increment 32B, no-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_maskstore_d32(array+6,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vmaskmovpd stores (increment 32B, half-cross-line): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_broadcast_d16(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vbroadcastf128 loads (increment 16B): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_broadcast_d8(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vbroadcastsd loads (increment 8B): %e\n", gops_per_sec);

  gettimeofday(&starttime,0);
  test_broadcast_d4(array,ops);
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  gops_per_sec = (double)ops / usec_elapsed / 1000;
  printf("GOPs per second for vbroadcastss loads (increment 4B): %e\n", gops_per_sec);

  free(array);array=NULL;
  return 0;
}
