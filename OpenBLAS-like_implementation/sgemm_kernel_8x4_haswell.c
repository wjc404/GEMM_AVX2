/* %0 = "+r"(a_pointer), %1 = "+r"(b_pointer), %2 = "+r"(c_pointer), %3 = "+r"(ldc_in_bytes), %4 for k_count, %5 for c_store, %6 = &alpha, %7 = m_count, %8 = b_pref */
/* r11 = m, r12 = k << 4(const), r13 = k(const), r14 = b_head_pos(const)*/

//recommended settings: GEMM_P = 384, GEMM_Q = 256.

/* m = 8 *//* ymm0 for alpha, ymm1-ymm3 for temporary use, ymm4-ymm15 for accumulators */
#define KERNEL_k1m8n1 \
    "vmovups (%0),%%ymm1; addq $32,%0;"\
    "vbroadcastss (%1),%%ymm2; vfmadd231ps %%ymm1,%%ymm2,%%ymm4;"\
    "addq $4,%1;"
#define KERNEL_h_k1m8n2 \
    "vmovsldup (%0),%%ymm1; vmovshdup (%0),%%ymm2; addq $32,%0;"\
    "vbroadcastsd (%1),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,%%ymm4; vfmadd231ps %%ymm2,%%ymm3,%%ymm5;"
#define KERNEL_k1m8n2 KERNEL_h_k1m8n2 "addq $8,%1;"
#define KERNEL_h_k1m8n4 \
    KERNEL_h_k1m8n2 "vbroadcastsd 8(%1),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,%%ymm6; vfmadd231ps %%ymm2,%%ymm3,%%ymm7;"
#define KERNEL_k1m8n4 KERNEL_h_k1m8n4 "addq $16,%1;"
#define unit_kernel_k1m8n4(c1,c2,c3,c4,...) \
    "vbroadcastsd  ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,"#c1"; vfmadd231ps %%ymm2,%%ymm3,"#c2";"\
    "vbroadcastsd 8("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm1,%%ymm3,"#c3"; vfmadd231ps %%ymm2,%%ymm3,"#c4";"
#define KERNEL_h_k1m8n8 KERNEL_h_k1m8n4 unit_kernel_k1m8n4(%%ymm8,%%ymm9,%%ymm10,%%ymm11,%1,%%r12,1)
#define KERNEL_k1m8n8 KERNEL_h_k1m8n8 "addq $16,%1;"
#define KERNEL_h_k1m8n12 KERNEL_h_k1m8n8 unit_kernel_k1m8n4(%%ymm12,%%ymm13,%%ymm14,%%ymm15,%1,%%r12,2)
#define KERNEL_k1m8n12 KERNEL_h_k1m8n12 "addq $16,%1;"
#define INIT_m8n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define INIT_m8n2 INIT_m8n1 "vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m8n4 INIT_m8n2 "vpxor %%ymm6,%%ymm6,%%ymm6;vpxor %%ymm7,%%ymm7,%%ymm7;"
#define unit_init_m8n4(c1,c2,c3,c4) \
    "vpxor "#c1","#c1","#c1";vpxor "#c2","#c2","#c2";vpxor "#c3","#c3","#c3";vpxor "#c4","#c4","#c4";"
#define INIT_m8n8  INIT_m8n4 unit_init_m8n4(%%ymm8,%%ymm9,%%ymm10,%%ymm11)
#define INIT_m8n12 INIT_m8n8 unit_init_m8n4(%%ymm12,%%ymm13,%%ymm14,%%ymm15)
#define SAVE_m8n1 "vfmadd213ps (%2),%%ymm0,%%ymm4; vmovups %%ymm4,(%2);"
#define unit_save_m8n2(c1,c2) \
    "vunpcklps "#c2","#c1",%%ymm2; vunpckhps "#c2","#c1",%%ymm3; vunpcklpd %%ymm3,%%ymm2,"#c1"; vunpckhpd %%ymm3,%%ymm2,"#c2";"\
    "vfmadd213ps (%5),%%ymm0,"#c1"; vmovups "#c1",(%5);"\
    "vfmadd213ps (%5,%3,1),%%ymm0,"#c2"; vmovups "#c2",(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_m8n2 "movq %2,%5;" unit_save_m8n2(%%ymm4,%%ymm5)
#define SAVE_m8n4  SAVE_m8n2  unit_save_m8n2(%%ymm6,%%ymm7)
#define SAVE_m8n8  SAVE_m8n4  unit_save_m8n2(%%ymm8,%%ymm9)   unit_save_m8n2(%%ymm10,%%ymm11)
#define SAVE_m8n12 SAVE_m8n8  unit_save_m8n2(%%ymm12,%%ymm13) unit_save_m8n2(%%ymm14,%%ymm15)
#define COMPUTE_m8(ndim) \
    INIT_m8n##ndim\
    "movq %%r13,%4; movq %%r14,%1; movq %2,%5;"\
    "cmpq $24,%4; jb "#ndim"882f;"\
    #ndim"881:\n\t"\
    "prefetcht0 512(%0);" KERNEL_k1m8n##ndim KERNEL_k1m8n##ndim\
    "prefetcht0 512(%0);" KERNEL_k1m8n##ndim KERNEL_k1m8n##ndim\
    "prefetcht1 (%5); prefetcht1 31(%5); addq %3,%5;"\
    "prefetcht0 512(%0);" KERNEL_k1m8n##ndim KERNEL_k1m8n##ndim\
    "prefetcht0 512(%0);" KERNEL_k1m8n##ndim KERNEL_k1m8n##ndim\
    "prefetcht1 (%8); addq $16,%8;"\
    "subq $8,%4; cmpq $24,%4; jnb "#ndim"881b;"\
    "movq %2,%5;"\
    #ndim"882:\n\t"\
    "testq %4,%4; jz "#ndim"883f;"\
    "prefetcht0 (%5); prefetcht0 31(%5); addq %3,%5;"\
    KERNEL_k1m8n##ndim\
    "decq %4; jmp "#ndim"882b;"\
    #ndim"883:\n\t"\
    "prefetcht0 (%%r14); prefetcht0 64(%%r14);"\
    SAVE_m8n##ndim "addq $32,%2;"

/* m = 4 *//* xmm0 for alpha, xmm1-xmm3 for temporary use, xmm4-xmm15 for accumulators */
#define KERNEL_k1m4n1 \
    "vmovups (%0),%%xmm1; addq $16,%0;"\
    "vbroadcastss (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,%1;"
#define KERNEL_h_k1m4n2 \
    "vmovsldup (%0),%%xmm1; vmovshdup (%0),%%xmm2; addq $16,%0;"\
    "vmovddup (%1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm4; vfmadd231ps %%xmm2,%%xmm3,%%xmm5;"
#define KERNEL_k1m4n2 KERNEL_h_k1m4n2 "addq $8,%1;"
#define KERNEL_h_k1m4n4 \
    KERNEL_h_k1m4n2 "vmovddup 8(%1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm6; vfmadd231ps %%xmm2,%%xmm3,%%xmm7;"
#define KERNEL_k1m4n4 KERNEL_h_k1m4n4 "addq $16,%1;"
#define unit_kernel_k1m4n4(c1,c2,c3,c4,...) \
    "vmovddup  ("#__VA_ARGS__"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,"#c1"; vfmadd231ps %%xmm2,%%xmm3,"#c2";"\
    "vmovddup 8("#__VA_ARGS__"),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,"#c3"; vfmadd231ps %%xmm2,%%xmm3,"#c4";"
#define KERNEL_h_k1m4n8 KERNEL_h_k1m4n4 unit_kernel_k1m4n4(%%xmm8,%%xmm9,%%xmm10,%%xmm11,%1,%%r12,1)
#define KERNEL_k1m4n8 KERNEL_h_k1m4n8 "addq $16,%1;"
#define KERNEL_h_k1m4n12 KERNEL_h_k1m4n8 unit_kernel_k1m4n4(%%xmm12,%%xmm13,%%xmm14,%%xmm15,%1,%%r12,2)
#define KERNEL_k1m4n12 KERNEL_h_k1m4n12 "addq $16,%1;"
#define INIT_m4n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m4n2 INIT_m4n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m4n4 INIT_m4n2 "vpxor %%xmm6,%%xmm6,%%xmm6;vpxor %%xmm7,%%xmm7,%%xmm7;"
#define unit_init_m4n4(c1,c2,c3,c4) \
    "vpxor "#c1","#c1","#c1";vpxor "#c2","#c2","#c2";vpxor "#c3","#c3","#c3";vpxor "#c4","#c4","#c4";"
#define INIT_m4n8  INIT_m4n4 unit_init_m4n4(%%xmm8,%%xmm9,%%xmm10,%%xmm11)
#define INIT_m4n12 INIT_m4n8 unit_init_m4n4(%%xmm12,%%xmm13,%%xmm14,%%xmm15)
#define SAVE_m4n1 \
    "vfmadd213ps (%2),%%xmm0,%%xmm4; vmovups %%xmm4,(%2);"
#define unit_save_m4n2(c1,c2) \
    "vunpcklps "#c2","#c1",%%xmm2; vunpckhps "#c2","#c1",%%xmm3; vunpcklpd %%xmm3,%%xmm2,"#c1"; vunpckhpd %%xmm3,%%xmm2,"#c2";"\
    "vfmadd213ps (%5),%%xmm0,"#c1"; vmovups "#c1",(%5);"\
    "vfmadd213ps (%5,%3,1),%%xmm0,"#c2"; vmovups "#c2",(%5,%3,1);"\
    "leaq (%5,%3,2),%5;"
#define SAVE_m4n2 "movq %2,%5;" unit_save_m4n2(%%xmm4,%%xmm5)
#define SAVE_m4n4  SAVE_m4n2  unit_save_m4n2(%%xmm6,%%xmm7)
#define SAVE_m4n8  SAVE_m4n4  unit_save_m4n2(%%xmm8,%%xmm9)   unit_save_m4n2(%%xmm10,%%xmm11)
#define SAVE_m4n12 SAVE_m4n8  unit_save_m4n2(%%xmm12,%%xmm13) unit_save_m4n2(%%xmm14,%%xmm15)
#define COMPUTE_m4(ndim) \
    INIT_m4n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim"442:\n\t"\
    "testq %4,%4; jz "#ndim"443f;"\
    KERNEL_k1m4n##ndim\
    "decq %4; jmp "#ndim"442b;"\
    #ndim"443:\n\t"\
    SAVE_m4n##ndim "addq $16,%2;"

/* m = 2 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm9 for accumulators */
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m2n1 \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "addq $4,%1;"
#define SAVE_m2n1 \
    "vmovsd (%2),%%xmm1; vfmadd213ps %%xmm1,%%xmm0,%%xmm4; vmovsd %%xmm4,(%2);"
#define INIT_m2n2 INIT_m2n1 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define KERNEL_k1m2n2 \
    "vmovsd (%0),%%xmm1; addq $8,%0;"\
    "vbroadcastss  (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm4;"\
    "vbroadcastss 4(%1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm5;"\
    "addq $8,%1;"
#define SAVE_m2n2 SAVE_m2n1 \
    "vmovsd (%2,%3,1),%%xmm1; vfmadd213ps %%xmm1,%%xmm0,%%xmm5; vmovsd %%xmm5,(%2,%3,1);"
#define INIT_m2n4  INIT_m2n2
#define INIT_m2n8  INIT_m2n4 "vpxor %%xmm6,%%xmm6,%%xmm6; vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m2n12 INIT_m2n8 "vpxor %%xmm8,%%xmm8,%%xmm8; vpxor %%xmm9,%%xmm9,%%xmm9;"
#define KERNEL_k1m2n4 \
    "vmovups (%1),%%xmm3; addq $16,%1;"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "vbroadcastss 4(%0),%%xmm2; vfmadd231ps %%xmm3,%%xmm2,%%xmm5;"\
    "addq $8,%0;"
#define KERNEL_k1m2n8 \
    "vmovups (%1),%%xmm3; vmovups (%1,%%r12,1),%%xmm2; addq $16,%1;"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4; vfmadd231ps %%xmm2,%%xmm1,%%xmm6;"\
    "vbroadcastss 4(%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm5; vfmadd231ps %%xmm2,%%xmm1,%%xmm7;"\
    "addq $8,%0;"
#define KERNEL_k1m2n12 \
    "vmovups (%1),%%xmm3; vmovups (%1,%%r12,1),%%xmm2; vmovups (%1,%%r12,2),%%xmm1; addq $16,%1;"\
    "vbroadcastss  (%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm4; vfmadd231ps %%xmm2,%%xmm10,%%xmm6; vfmadd231ps %%xmm1,%%xmm10,%%xmm8;"\
    "vbroadcastss 4(%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm5; vfmadd231ps %%xmm2,%%xmm10,%%xmm7; vfmadd231ps %%xmm1,%%xmm10,%%xmm9;"\
    "addq $8,%0;"
#define unit_save_m2n4(c1,c2) \
    "vunpcklps "#c2","#c1",%%xmm1; vunpckhps "#c2","#c1",%%xmm2;"\
    "vmovsd (%5),%%xmm3; vmovhpd (%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm1;"\
    "vmovsd %%xmm1,(%5); vmovhpd %%xmm1,(%5,%3,1); leaq (%5,%3,2),%5;"\
    "vmovsd (%5),%%xmm3; vmovhpd (%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm2;"\
    "vmovsd %%xmm2,(%5); vmovhpd %%xmm2,(%5,%3,1); leaq (%5,%3,2),%5;"
#define SAVE_m2n4 "movq %2,%5;" unit_save_m2n4(%%xmm4,%%xmm5)
#define SAVE_m2n8   SAVE_m2n4    unit_save_m2n4(%%xmm6,%%xmm7)
#define SAVE_m2n12  SAVE_m2n8   unit_save_m2n4(%%xmm8,%%xmm9)
#define COMPUTE_m2(ndim) \
    INIT_m2n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim"222:\n\t"\
    "testq %4,%4; jz "#ndim"223f;"\
    KERNEL_k1m2n##ndim\
    "decq %4; jmp "#ndim"222b;"\
    #ndim"223:\n\t"\
    SAVE_m2n##ndim "addq $8,%2;"

/* m = 1 *//* xmm0 for alpha, xmm1-xmm3 and xmm10 for temporary use, xmm4-xmm6 for accumulators */
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define KERNEL_k1m1n1 \
    "vmovss (%1),%%xmm3; addq $4,%1;"\
    "vmovss (%0),%%xmm1; vfmadd231ss %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_m1n1 \
    "vfmadd213ss (%2),%%xmm0,%%xmm4; vmovss %%xmm4,(%2);"
#define INIT_m1n2 INIT_m1n1
#define KERNEL_k1m1n2 \
    "vmovsd (%1),%%xmm3; addq $8,%1;"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define SAVE_m1n2 \
    "vmovss (%2),%%xmm3; vinsertps $16,(%2,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm4;"\
    "vmovss %%xmm4,(%2); vextractps $1,%%xmm4,(%2,%3,1);"
#define INIT_m1n4  INIT_m1n2
#define INIT_m1n8  INIT_m1n4 "vpxor %%xmm5,%%xmm5,%%xmm5;"
#define INIT_m1n12 INIT_m1n8 "vpxor %%xmm6,%%xmm6,%%xmm6;"
#define KERNEL_k1m1n4 \
    "vmovups (%1),%%xmm3; addq $16,%1;"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4;"\
    "addq $4,%0;"
#define KERNEL_k1m1n8 \
    "vmovups (%1),%%xmm3; vmovups (%1,%%r12,1),%%xmm2; addq $16,%1;"\
    "vbroadcastss  (%0),%%xmm1; vfmadd231ps %%xmm3,%%xmm1,%%xmm4; vfmadd231ps %%xmm2,%%xmm1,%%xmm5;"\
    "addq $4,%0;"
#define KERNEL_k1m1n12 \
    "vmovups (%1),%%xmm3; vmovups (%1,%%r12,1),%%xmm2; vmovups (%1,%%r12,2),%%xmm1; addq $16,%1;"\
    "vbroadcastss  (%0),%%xmm10; vfmadd231ps %%xmm3,%%xmm10,%%xmm4; vfmadd231ps %%xmm2,%%xmm10,%%xmm5; vfmadd231ps %%xmm1,%%xmm10,%%xmm6;"\
    "addq $4,%0;"
#define unit_save_m1n4(c1) \
    "vpxor %%xmm10,%%xmm10,%%xmm10; vmovsd "#c1",%%xmm10,%%xmm2; vmovhlps "#c1",%%xmm10,%%xmm1;"\
    "vmovss (%5),%%xmm3; vinsertps $16,(%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm2;"\
    "vmovss %%xmm2,(%5); vextractps $1,%%xmm2,(%5,%3,1); leaq (%5,%3,2),%5;"\
    "vmovss (%5),%%xmm3; vinsertps $16,(%5,%3,1),%%xmm3,%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm1;"\
    "vmovss %%xmm1,(%5); vextractps $1,%%xmm1,(%5,%3,1); leaq (%5,%3,2),%5;"
#define SAVE_m1n4 "movq %2,%5;" unit_save_m1n4(%%xmm4)
#define SAVE_m1n8  SAVE_m1n4    unit_save_m1n4(%%xmm5)
#define SAVE_m1n12 SAVE_m1n8    unit_save_m1n4(%%xmm6)
#define COMPUTE_m1(ndim) \
    INIT_m1n##ndim\
    "movq %%r13,%4; movq %%r14,%1;"\
    #ndim"112:\n\t"\
    "testq %4,%4; jz "#ndim"113f;"\
    KERNEL_k1m1n##ndim\
    "decq %4; jmp "#ndim"112b;"\
    #ndim"113:\n\t"\
    SAVE_m1n##ndim "addq $4,%2;"

#define COMPUTE(ndim) {\
    next_b = b_pointer + ndim * K;\
    __asm__ __volatile__(\
    "vbroadcastss (%6),%%ymm0;"\
    "movq %4,%%r13; movq %4,%%r12; salq $4,%%r12; movq %1,%%r14; movq %7,%%r11;"\
    "cmpq $8,%7;jb 33101"#ndim"f;"\
    "33109"#ndim":\n\t"\
    COMPUTE_m8(ndim)\
    "subq $8,%7;cmpq $8,%7;jnb 33109"#ndim"b;"\
    "33101"#ndim":\n\t"\
    "cmpq $4,%7;jb 33103"#ndim"f;"\
    COMPUTE_m4(ndim)\
    "subq $4,%7;"\
    "33103"#ndim":\n\t"\
    "cmpq $2,%7;jb 33104"#ndim"f;"\
    COMPUTE_m2(ndim)\
    "subq $2,%7;"\
    "33104"#ndim":\n\t"\
    "testq %7,%7;jz 33105"#ndim"f;"\
    COMPUTE_m1(ndim)\
    "33105"#ndim":\n\t"\
    "movq %%r13,%4; movq %%r14,%1; movq %%r11,%7; vzeroupper;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(ldc_in_bytes),"+r"(K),"+r"(ctemp),"+r"(const_val),"+r"(M),"+r"(next_b)\
    ::"r11","r12","r13","r14",\
    "ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15","cc","memory");\
    a_pointer -= M * K; b_pointer += ndim * K; c_pointer += (LDC * ndim - M);\
}

//#include "common.h"
//#include <stdint.h>
#include <stdio.h>//debug
#include <stdlib.h>//debug
#define BLASLONG int//debug
int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, float alpha, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, BLASLONG LDC)
{
    if(m==0||n==0||k==0||alpha==0.0) return 0;
    int64_t ldc_in_bytes = (int64_t)LDC * sizeof(float);
    float constval = alpha;
    float *const_val=&constval;
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    float *a_pointer = A,*b_pointer = B,*c_pointer = C,*ctemp = C,*next_b = B;
    for(;n_count>11;n_count-=12) COMPUTE(12)
    for(;n_count>7;n_count-=8) COMPUTE(8)
    for(;n_count>3;n_count-=4) COMPUTE(4)
    for(;n_count>1;n_count-=2) COMPUTE(2)
    if(n_count>0) COMPUTE(1)
    return 0;
}

/* test zone */
static void sgemm_tcopy_4(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    float *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 4;
      for(count_first=dim_first;count_first>3;count_first-=4){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void sgemm_tcopy_8(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    float *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 8;
      for(count_first=dim_first;count_first>7;count_first-=8){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        todst[4]=tosrc[4];todst[5]=tosrc[5];todst[6]=tosrc[6];todst[7]=tosrc[7];
        tosrc+=8;todst+=8*dim_second;
      }
      todst -= count_second * 4;
      for(;count_first>3;count_first-=4){
        todst[0]=tosrc[0];todst[1]=tosrc[1];todst[2]=tosrc[2];todst[3]=tosrc[3];
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void sgemm_ncopy_4(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    float *tosrc1,*tosrc2,*tosrc3,*tosrc4;
    float *todst=dst;
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
static void sgemm_ncopy_8(float *src, float *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    float *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8;
    float *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc5=tosrc4+lead_dim;tosrc6=tosrc5+lead_dim;tosrc7=tosrc6+lead_dim;tosrc8=tosrc7+lead_dim;
    tosrc_inc=8*lead_dim-dim_first;
    for(count_second=dim_second;count_second>7;count_second-=8){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst+=8;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
    }
    tosrc_inc-=4*lead_dim;
    for(;count_second>3;count_second-=4){
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
static void SCALE_MULT(float *dat,float *sca, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0 || (*sca)==(float)1.0) return;
    float scale = *sca; float *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale; current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
#define BLOCKDIM_K 256 //GEMM_Q in OpenBLAS
#define BLOCKDIM_M 384 //GEMM_P in OpenBLAS
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
//gcc -march=haswell --shared -fPIC -O2 sgemm_kernel_8x4_haswell.c -o sgemm.so
void sgemm_(char *transa,char *transb,BLASLONG *m,BLASLONG *n,BLASLONG *k,float *alpha,float *a,BLASLONG *lda,float *b,BLASLONG *ldb,float *beta,float *c,BLASLONG *ldc){
    if((*m)==0||(*n)==0) return;
    if((*beta)!=1.0) SCALE_MULT(c,beta,*ldc,*m,*n);
    if((*alpha)==0.0||(*k)==0) return;
/* start main calculation here */
    float *b_buffer = (float *)aligned_alloc(64,BLOCKDIM_K*(*n)*sizeof(float));
    float *a_buffer = (float *)aligned_alloc(4096,BLOCKDIM_K*BLOCKDIM_M*sizeof(float));
    float *a_current_pos,*b_current_pos;
    BLASLONG m_count,n_count,k_count,k_subdim,m_subdim;
    b_current_pos = b;
    for(k_count=0;k_count<(*k);k_count+=BLOCKDIM_K){
      k_subdim = (*k)-k_count;
      if(k_subdim > BLOCKDIM_K) k_subdim = BLOCKDIM_K;
      if(NOTRANSB) { sgemm_ncopy_4(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { sgemm_tcopy_4(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { sgemm_tcopy_8(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { sgemm_ncopy_8(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        CNAME(m_subdim,*n,k_subdim,*alpha,a_buffer,b_buffer,c+m_count,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}