
/* m = 4, ymm0 for alpha, ymm1-3 for tmp, ymm4-15 for acc */
#define KERNEL_k1m4n1 "vbroadcastsd (%1),%%ymm1; addq $8,%1; vfmadd231pd (%0),%%ymm1,%%ymm4; addq $32,%0;"
#define unit_acc_m4n2(b_off,c1_no,c2_no,...)\
  "vbroadcastf128 "#b_off"("#__VA_ARGS__"),%%ymm3; vfmadd231pd %%ymm1,%%ymm3,%%ymm"#c1_no"; vfmadd231pd %%ymm2,%%ymm3,%%ymm"#c2_no";"
#define KERNEL_h_k1m4n2 "vmovddup (%0),%%ymm1; vmovddup 8(%0),%%ymm2; addq $32,%0;" unit_acc_m4n2(0,4,5,%1)
#define KERNEL_k1m4n2 KERNEL_h_k1m4n2 "addq $16,%1;"
#define KERNEL_h_k1m4n4 KERNEL_h_k1m4n2 unit_acc_m4n2(0,6,7,%1,%%r12,2)
#define KERNEL_k1m4n4 KERNEL_h_k1m4n4 "addq $16,%1;"
#define KERNEL_k1m4n6 KERNEL_h_k1m4n4 unit_acc_m4n2(0,8,9,%1,%%r12,4) "addq $16,%1;"
#define KERNEL_h_k1m4n8 KERNEL_k1m4n6 unit_acc_m4n2(0,10,11,%%r15)
#define KERNEL_k1m4n8 KERNEL_h_k1m4n8 "addq $16,%%r15;"
#define KERNEL_h_k1m4n10 KERNEL_h_k1m4n8 unit_acc_m4n2(0,12,13,%%r15,%%r12,2)
#define KERNEL_k1m4n10 KERNEL_h_k1m4n10 "addq $16,%%r15;"
#define KERNEL_k1m4n12 KERNEL_h_k1m4n10 unit_acc_m4n2(0,14,15,%%r15,%%r12,4) "addq $16,%%r15;"
#define KERNEL_k2m4n12 \
  "vmovddup (%0),%%ymm1; vmovddup 8(%0),%%ymm2; prefetcht0 384(%0);"\
  unit_acc_m4n2(0,4,5,%1) unit_acc_m4n2(0,6,7,%1,%%r12,2) unit_acc_m4n2(0,8,9,%1,%%r12,4)\
  unit_acc_m4n2(0,10,11,%%r15) unit_acc_m4n2(0,12,13,%%r15,%%r12,2) unit_acc_m4n2(0,14,15,%%r15,%%r12,4)\
  "vmovddup 32(%0),%%ymm1; vmovddup 40(%0),%%ymm2; addq $64,%0;"\
  unit_acc_m4n2(16,4,5,%1) unit_acc_m4n2(16,6,7,%1,%%r12,2) unit_acc_m4n2(16,8,9,%1,%%r12,4) "addq $32,%1;"\
  unit_acc_m4n2(16,10,11,%%r15) unit_acc_m4n2(16,12,13,%%r15,%%r12,2) unit_acc_m4n2(16,14,15,%%r15,%%r12,4) "addq $32,%%r15;"
#define SAVE_m4n1 "vfmadd213pd (%2),%%ymm0,%%ymm4; vmovupd %%ymm4,(%2); addq $32,%2;"
#define unit_save_m4n2(c1_no,c2_no)\
  "vunpcklpd %%ymm"#c2_no",%%ymm"#c1_no",%%ymm1; vfmadd213pd (%3),%%ymm0,%%ymm1; vmovupd %%ymm1,(%3);"\
  "vunpckhpd %%ymm"#c2_no",%%ymm"#c1_no",%%ymm2; vfmadd213pd (%3,%4,1),%%ymm0,%%ymm2; vmovupd %%ymm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m4n2 "movq %2,%3; addq $32,%2;" unit_save_m4n2(4,5)
#define SAVE_m4n4 SAVE_m4n2 unit_save_m4n2(6,7)
#define SAVE_m4n6 SAVE_m4n4 unit_save_m4n2(8,9)
#define SAVE_m4n8 SAVE_m4n6 unit_save_m4n2(10,11)
#define SAVE_m4n10 SAVE_m4n8 unit_save_m4n2(12,13)
#define SAVE_m4n12 SAVE_m4n10 unit_save_m4n2(14,15)
#define INIT_m4n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define unit_init_m4n2(c1_no,c2_no) "vpxor %%ymm"#c1_no",%%ymm"#c1_no",%%ymm"#c1_no"; vpxor %%ymm"#c2_no",%%ymm"#c2_no",%%ymm"#c2_no";"
#define INIT_m4n2 unit_init_m4n2(4,5)
#define INIT_m4n4 INIT_m4n2 unit_init_m4n2(6,7)
#define INIT_m4n6 INIT_m4n4 unit_init_m4n2(8,9)
#define INIT_m4n8 INIT_m4n6 unit_init_m4n2(10,11)
#define INIT_m4n10 INIT_m4n8 unit_init_m4n2(12,13)
#define INIT_m4n12 INIT_m4n10 unit_init_m4n2(14,15)

/* m = 2, xmm0 for alpha, xmm1-3 for tmp, xmm4-15 for acc */
#define KERNEL_k1m2n1 "vmovddup (%1),%%xmm1; addq $8,%1; vfmadd231pd (%0),%%xmm1,%%xmm4; addq $16,%0;"
#define unit_acc_m2n2(b_off,c1_no,c2_no,...)\
  "vmovupd "#b_off"("#__VA_ARGS__"),%%xmm3; vfmadd231pd %%xmm1,%%xmm3,%%xmm"#c1_no"; vfmadd231pd %%xmm2,%%xmm3,%%xmm"#c2_no";"
#define KERNEL_h_k1m2n2 "vmovddup (%0),%%xmm1; vmovddup 8(%0),%%xmm2; addq $16,%0;" unit_acc_m2n2(0,4,5,%1)
#define KERNEL_k1m2n2 KERNEL_h_k1m2n2 "addq $16,%1;"
#define KERNEL_h_k1m2n4 KERNEL_h_k1m2n2 unit_acc_m2n2(0,6,7,%1,%%r12,2)
#define KERNEL_k1m2n4 KERNEL_h_k1m2n4 "addq $16,%1;"
#define KERNEL_k1m2n6 KERNEL_h_k1m2n4 unit_acc_m2n2(0,8,9,%1,%%r12,4) "addq $16,%1;"
#define KERNEL_h_k1m2n8 KERNEL_k1m2n6 unit_acc_m2n2(0,10,11,%%r15)
#define KERNEL_k1m2n8 KERNEL_h_k1m2n8 "addq $16,%%r15;"
#define KERNEL_h_k1m2n10 KERNEL_h_k1m2n8 unit_acc_m2n2(0,12,13,%%r15,%%r12,2)
#define KERNEL_k1m2n10 KERNEL_h_k1m2n10 "addq $16,%%r15;"
#define KERNEL_k1m2n12 KERNEL_h_k1m2n10 unit_acc_m2n2(0,14,15,%%r15,%%r12,4) "addq $16,%%r15;"
#define SAVE_m2n1 "vfmadd213pd (%2),%%xmm0,%%xmm4; vmovupd %%xmm4,(%2); addq $16,%2;"
#define unit_save_m2n2(c1_no,c2_no)\
  "vunpcklpd %%xmm"#c2_no",%%xmm"#c1_no",%%xmm1; vfmadd213pd (%3),%%xmm0,%%xmm1; vmovupd %%xmm1,(%3);"\
  "vunpckhpd %%xmm"#c2_no",%%xmm"#c1_no",%%xmm2; vfmadd213pd (%3,%4,1),%%xmm0,%%xmm2; vmovupd %%xmm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m2n2 "movq %2,%3; addq $16,%2;" unit_save_m2n2(4,5)
#define SAVE_m2n4 SAVE_m2n2 unit_save_m2n2(6,7)
#define SAVE_m2n6 SAVE_m2n4 unit_save_m2n2(8,9)
#define SAVE_m2n8 SAVE_m2n6 unit_save_m2n2(10,11)
#define SAVE_m2n10 SAVE_m2n8 unit_save_m2n2(12,13)
#define SAVE_m2n12 SAVE_m2n10 unit_save_m2n2(14,15)
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define unit_init_m2n2(c1_no,c2_no) "vpxor %%xmm"#c1_no",%%xmm"#c1_no",%%xmm"#c1_no"; vpxor %%xmm"#c2_no",%%xmm"#c2_no",%%xmm"#c2_no";"
#define INIT_m2n2 unit_init_m2n2(4,5)
#define INIT_m2n4 INIT_m2n2 unit_init_m2n2(6,7)
#define INIT_m2n6 INIT_m2n4 unit_init_m2n2(8,9)
#define INIT_m2n8 INIT_m2n6 unit_init_m2n2(10,11)
#define INIT_m2n10 INIT_m2n8 unit_init_m2n2(12,13)
#define INIT_m2n12 INIT_m2n10 unit_init_m2n2(14,15)

/* m = 1, xmm0 for alpha, xmm1-3 for tmp, xmm4-9 for acc */
#define KERNEL_k1m1n1 "vmovsd (%1),%%xmm1; addq $8,%1; vfmadd231sd (%0),%%xmm1,%%xmm4; addq $8,%0;"
#define KERNEL_h_k1m1n2 "vmovddup (%0),%%xmm1; addq $8,%0; vfmadd231pd (%1),%%xmm1,%%xmm4;"
#define KERNEL_k1m1n2 KERNEL_h_k1m1n2 "addq $16,%1;"
#define KERNEL_h_k1m1n4 KERNEL_h_k1m1n2 "vfmadd231pd (%1,%%r12,2),%%xmm1,%%xmm5;"
#define KERNEL_k1m1n4 KERNEL_h_k1m1n4 "addq $16,%1;"
#define KERNEL_k1m1n6 KERNEL_h_k1m1n4 "vfmadd231pd (%1,%%r12,4),%%xmm1,%%xmm6; addq $16,%1;"
#define KERNEL_h_k1m1n8 KERNEL_k1m1n6 "vfmadd231pd (%%r15),%%xmm1,%%xmm7;"
#define KERNEL_k1m1n8 KERNEL_h_k1m1n8 "addq $16,%%r15;"
#define KERNEL_h_k1m1n10 KERNEL_h_k1m1n8 "vfmadd231pd (%%r15,%%r12,2),%%xmm1,%%xmm8;"
#define KERNEL_k1m1n10 KERNEL_h_k1m1n10 "addq $16,%%r15;"
#define KERNEL_k1m1n12 KERNEL_h_k1m1n10 "vfmadd231pd (%%r15,%%r12,4),%%xmm1,%%xmm9; addq $16,%%r15;"
#define SAVE_m1n1 "vfmadd213sd (%2),%%xmm0,%%xmm4; vmovsd %%xmm4,(%2); addq $8,%2;"
#define unit_save_m1n2(c1_no)\
  "vmovsd (%3),%%xmm1; vmovhpd (%3,%4,1),%%xmm1,%%xmm1; vfmadd231pd %%xmm"#c1_no",%%xmm0,%%xmm1;"\
  "vmovsd %%xmm1,(%3); vmovhpd %%xmm1,(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m1n2 "movq %2,%3; addq $8,%2;" unit_save_m1n2(4)
#define SAVE_m1n4 SAVE_m1n2 unit_save_m1n2(5)
#define SAVE_m1n6 SAVE_m1n4 unit_save_m1n2(6)
#define SAVE_m1n8 SAVE_m1n6 unit_save_m1n2(7)
#define SAVE_m1n10 SAVE_m1n8 unit_save_m1n2(8)
#define SAVE_m1n12 SAVE_m1n10 unit_save_m1n2(9)
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m1n2 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m1n4 INIT_m1n2 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m1n6 INIT_m1n4 "vpxor %%xmm6,%%xmm6,%%xmm6;"
#define INIT_m1n8 INIT_m1n6 "vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m1n10 INIT_m1n8 "vpxor %%xmm8,%%xmm8,%%xmm8;"
#define INIT_m1n12 INIT_m1n10 "vpxor %%xmm9,%%xmm9,%%xmm9;"

#define COMPUTE_SIMPLE(mdim,ndim)\
  INIT_m##mdim##n##ndim "movq %%r13,%5; movq %%r14,%1; leaq (%1,%%r12,4),%%r15; leaq (%%r15,%%r12,2),%%r15;"\
  "testq %5,%5; jz 5"#mdim"5"#ndim"2f;"\
  "5"#mdim"5"#ndim"1:\n\t"\
  KERNEL_k1m##mdim##n##ndim "decq %5; jnz 5"#mdim"5"#ndim"1b;"\
  "5"#mdim"5"#ndim"2:\n\t"\
  SAVE_m##mdim##n##ndim
#define COMPUTE_m4n1 COMPUTE_SIMPLE(4,1)
#define COMPUTE_m4n2 COMPUTE_SIMPLE(4,2)
#define COMPUTE_m4n4 COMPUTE_SIMPLE(4,4)
#define COMPUTE_m4n6 COMPUTE_SIMPLE(4,6)
#define COMPUTE_m4n8 COMPUTE_SIMPLE(4,8)
#define COMPUTE_m4n10 COMPUTE_SIMPLE(4,10)
#define COMPUTE_m4n12 \
  INIT_m4n12 "movq %%r13,%5; movq %%r14,%1; leaq (%1,%%r12,4),%%r15; leaq (%%r15,%%r12,2),%%r15; movq %2,%3;"\
  "cmpq $16,%5; jb 545122f; movq $16,%5;"\
  "545121:\n\t"\
  KERNEL_k2m4n12 "testq $8,%5; movq $62,%%r10; cmovnz %4,%%r10;"\
  KERNEL_k2m4n12 "prefetcht1 (%3); subq $31,%3; addq %%r10,%3;"\
  KERNEL_k2m4n12 "prefetcht1 (%6); addq $16,%6; cmpq $200,%5; cmoveq %2,%3;"\
  KERNEL_k2m4n12 "addq $8,%5; cmpq %5,%%r13; jnb 545121b;"\
  "movq %2,%3; negq %5; leaq 16(%%r13,%5,1),%5;"\
  "545122:\n\t"\
  "testq %5,%5; jz 545124f;"\
  "545123:\n\t"\
  "prefetcht0 (%3); prefetcht0 31(%3); prefetcht0 (%3,%4,4); prefetcht0 31(%3,%4,4);"\
  KERNEL_k1m4n12 "addq %4,%3; decq %5; jnz 545123b;"\
  "545124:\n\t"\
  "prefetcht0 (%%r14); prefetcht0 64(%%r14);" SAVE_m4n12

#define COMPUTE(ndim) {\
  b_pref = b_ptr + ndim * K;\
  __asm__ __volatile__ (\
    "vbroadcastsd %7,%%ymm0; movq %8,%%r13; movq %%r13,%%r12; salq $3,%%r12; movq %1,%%r14; movq %9,%%r11;"\
    "cmpq $4,%%r11; jb "#ndim"33102f;"\
    #ndim"33101:\n\t"\
    COMPUTE_m4n##ndim "subq $4,%%r11; cmpq $4,%%r11; jnb "#ndim"33101b;"\
    #ndim"33102:\n\t"\
    "cmpq $2,%%r11; jb "#ndim"33103f;"\
    COMPUTE_SIMPLE(2,ndim) "subq $2,%%r11;"\
    #ndim"33103:\n\t"\
    "testq %%r11,%%r11; jz "#ndim"33104f;"\
    COMPUTE_SIMPLE(1,ndim) "subq $1,%%r11;"\
    #ndim"33104:\n\t"\
    "movq %%r14,%1; vzeroupper;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_bytes),"+r"(k_cnt),"+r"(b_pref):"m"(ALPHA),"m"(K),"m"(M)\
  :"r10","r11","r12","r13","r14","r15","cc","memory","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7"\
  ,"xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15");\
  a_ptr -= M * K; b_ptr += ndim * K; c_ptr += ldc * ndim - M;\
}

//#include "common.h"
#define BLASLONG int //debug
#include <stdint.h>
int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, double alpha, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, BLASLONG ldc){
    if(m==0||n==0||k==0||alpha==0.0) return 0;
    int64_t ldc_bytes = (int64_t)ldc * sizeof(double);
    double ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k, k_cnt = 0;
    BLASLONG n_count = n;
    double *a_ptr = A,*b_ptr = B,*c_ptr = C,*c_tmp = C,*b_pref = B;
    for(;n_count>11;n_count-=12) COMPUTE(12)
    for(;n_count>9;n_count-=10) COMPUTE(10)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>5;n_count-=6) COMPUTE(6)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#define BLOCKDIM_K 288
#define BLOCKDIM_M 288
static void dgemm_tcopy_4(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    double *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 4;
      for(count_first=dim_first;count_first>3;count_first-=4){
        _mm256_storeu_pd(todst,_mm256_loadu_pd(tosrc));
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        _mm_storeu_pd(todst,_mm_loadu_pd(tosrc));
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void dgemm_ncopy_4(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4;
    double *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc_inc=4*lead_dim-dim_first;
    for(count_second=dim_second;count_second>3;count_second-=4){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst+=4;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
    }
    tosrc_inc-=2*lead_dim;
    for(;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
static void dgemm_tcopy_2(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    double *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 2;
      for(count_first=dim_first;count_first>1;count_first-=2){
        _mm_storeu_pd(todst,_mm_loadu_pd(tosrc));
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void dgemm_ncopy_2(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2;
    double *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;
    tosrc_inc=2*lead_dim-dim_first;
    for(count_second=dim_second;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
static void SCALE_MULT(double *dat,double *sca, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0 || (*sca)==1.0) return;
    double scale = *sca; double *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale; current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
//gcc -march=haswell --shared -fPIC -O2 dgemm_kernel_4x2_haswell.c -o dgemm_4x2.so
void dgemm_(char *transa,char *transb,BLASLONG *m,BLASLONG *n,BLASLONG *k,double *alpha,double *a,BLASLONG *lda,double *b,BLASLONG *ldb,double *beta,double *c,BLASLONG *ldc){
    if((*m)==0||(*n)==0) return;
    if((*beta)!=1.0) SCALE_MULT(c,beta,*ldc,*m,*n);
    if((*alpha)==0.0||(*k)==0) return;
/* start main calculation here */
    double *b_buffer = (double *)aligned_alloc(64,BLOCKDIM_K*(*n)*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,BLOCKDIM_K*BLOCKDIM_M*sizeof(double));
    double *a_current_pos,*b_current_pos;
    BLASLONG m_count,n_count,k_count,k_subdim,m_subdim;
    b_current_pos = b;
    for(k_count=0;k_count<(*k);k_count+=BLOCKDIM_K){
      k_subdim = (*k)-k_count;
      if(k_subdim > BLOCKDIM_K) k_subdim = BLOCKDIM_K;
      if(NOTRANSB) { dgemm_ncopy_2(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { dgemm_tcopy_2(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { dgemm_tcopy_4(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { dgemm_ncopy_4(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        CNAME(m_subdim,*n,k_subdim,*alpha,a_buffer,b_buffer,c+m_count,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}
