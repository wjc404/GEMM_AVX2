# include <stdio.h>
# include <stdlib.h>
# include <sys/time.h>
# include <omp.h>
# include <math.h>
//compilation command: gcc -fopenmp ping-pong.c ping-pong.S -o ping-pong
extern void pingpongexec(int64_t *readaddr,int64_t *writeaddr,int64_t finvalue);
int main(){
  const int64_t finvalue = (int64_t)16777216;
  struct timeval starttime,endtime;
  double usec_elapsed,nsec_delay;
  __attribute__ ((aligned (64))) int64_t workarr[16];
  workarr[0]=(int64_t)0;
  workarr[8]=(int64_t)1;
  gettimeofday(&starttime,0);
  omp_set_num_threads(2);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if(tid) pingpongexec(&workarr[0],&workarr[8],finvalue);
    else pingpongexec(&workarr[8],&workarr[0],finvalue);
  }
  gettimeofday(&endtime,0);
  usec_elapsed = 1000000*(endtime.tv_sec - starttime.tv_sec) + endtime.tv_usec - starttime.tv_usec;
  nsec_delay = usec_elapsed / (double) finvalue * (double) 1000;
  printf("%d ns.\n", (int)(nsec_delay+0.5));
  return 0;
}

