#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
//gcc -fopenmp dgemm_gflops.c dgemm_gflops.S -o dgemm_gflops
extern void dgemmrun(double *ablk,double *bblk);
int main(){
  double *ablk=(double *)aligned_alloc(64,768);
  double *bblk=(double *)aligned_alloc(64,256);
  struct timeval starttime,endtime;
  int count;double usec_elapsed,flop;
  for(count=0;count<96;count++) ablk[count]=1.0;
  for(count=0;count<8;count++) bblk[4*count+0]=bblk[4*count+1]=bblk[4*count+2]=bblk[4*count+3]=(double)(count%2)-0.5;
  gettimeofday(&starttime,0);
  #pragma omp parallel
  {
    dgemmrun(ablk,bblk);
  }
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  flop = omp_get_max_threads() * (double)LOOPNUM * (double)768;
  printf("MAX dgemm GFLOPS of your machine: %f\n",flop/usec_elapsed/(double)1000);
  free(ablk);ablk=NULL;
  free(bblk);bblk=NULL;
  return 0;
}
