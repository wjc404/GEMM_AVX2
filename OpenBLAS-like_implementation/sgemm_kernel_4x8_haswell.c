/* %0 = a_ptr, %1 = b_ptr, %2 = c_ptr, %3 = ldc(bytes), %4 = c_tmp, %5 = k_counter, %6 = m_counter, %7 = &alpha, %8 = b_pref */
/* r11 = m; r12 = k << 5, r13 = k, r14 = b_head_ptr */
/* recommended settings: GEMM_Q = 256, GEMM_P = 384 */

#define GENERAL_INIT_ASM "vbroadcastss (%7),%%ymm0; movq %5,%%r13; movq %5,%%r12; salq $5,%%r12; movq %6,%%r11; movq %1,%%r14;"
#define GENERAL_RECOVER_ASM "movq %%r11,%6; movq %%r13,%5; movq %%r14,%1; vzeroupper;"

/* PART 1 : n=8/16/24: ymm0 for alpha, ymm1-ymm3 temporary use, ymm4-ymm15 accumulators */
/* c_block row_major z_partition; ymm1 = a_up, ymm2 = a_lo */
#define unit_acc_k1m4n8(c1,c2,c3,c4,...) \
  "vmovsldup ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm3,%%ymm1,"#c1"; vfmadd231ps %%ymm3,%%ymm2,"#c3";"\
  "vmovshdup ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm3,%%ymm1,"#c2"; vfmadd231ps %%ymm3,%%ymm2,"#c4";"
#define KERNEL_h_k1m4n8 \
  "vbroadcastsd (%0),%%ymm1; vbroadcastsd 8(%0),%%ymm2; addq $16,%0;"\
  unit_acc_k1m4n8(%%ymm4,%%ymm5,%%ymm6,%%ymm7,%1)
#define KERNEL_t_k1m4n8 KERNEL_h_k1m4n8 "addq $32,%1;"
#define KERNEL_h_k1m4n16 KERNEL_h_k1m4n8 unit_acc_k1m4n8(%%ymm8,%%ymm9,%%ymm10,%%ymm11,%1,%%r12,1)
#define KERNEL_t_k1m4n16 KERNEL_h_k1m4n16 "addq $32,%1;"
#define KERNEL_h_k1m4n24 KERNEL_h_k1m4n16 unit_acc_k1m4n8(%%ymm12,%%ymm13,%%ymm14,%%ymm15,%1,%%r12,2)
#define KERNEL_t_k1m4n24 KERNEL_h_k1m4n24 "addq $32,%1;"
#define unit_init_m4n8(c1,c2,c3,c4) \
  "vpxor "#c1","#c1","#c1"; vpxor "#c2","#c2","#c2"; vpxor "#c3","#c3","#c3"; vpxor "#c4","#c4","#c4";"
#define INIT_m4n8 unit_init_m4n8(%%ymm4,%%ymm5,%%ymm6,%%ymm7)
#define INIT_m4n16 INIT_m4n8 unit_init_m4n8(%%ymm8,%%ymm9,%%ymm10,%%ymm11)
#define INIT_m4n24 INIT_m4n16 unit_init_m4n8(%%ymm12,%%ymm13,%%ymm14,%%ymm15)
#define unit_save_m4n8(c1,c2,c3,c4) \
  "vunpcklpd "#c3","#c1",%%ymm1; vmovups (%4),%%xmm2; vinsertf128 $1,(%4,%3,4),%%ymm2,%%ymm2;"\
  "vfmadd213ps %%ymm2,%%ymm0,%%ymm1; vmovups %%xmm1,(%4); vextractf128 $1,%%ymm1,(%4,%3,4); addq %3,%4;"\
  "vunpcklpd "#c4","#c2",%%ymm1; vmovups (%4),%%xmm2; vinsertf128 $1,(%4,%3,4),%%ymm2,%%ymm2;"\
  "vfmadd213ps %%ymm2,%%ymm0,%%ymm1; vmovups %%xmm1,(%4); vextractf128 $1,%%ymm1,(%4,%3,4); addq %3,%4;"\
  "vunpckhpd "#c3","#c1",%%ymm1; vmovups (%4),%%xmm2; vinsertf128 $1,(%4,%3,4),%%ymm2,%%ymm2;"\
  "vfmadd213ps %%ymm2,%%ymm0,%%ymm1; vmovups %%xmm1,(%4); vextractf128 $1,%%ymm1,(%4,%3,4); addq %3,%4;"\
  "vunpckhpd "#c4","#c2",%%ymm1; vmovups (%4),%%xmm2; vinsertf128 $1,(%4,%3,4),%%ymm2,%%ymm2;"\
  "vfmadd213ps %%ymm2,%%ymm0,%%ymm1; vmovups %%xmm1,(%4); vextractf128 $1,%%ymm1,(%4,%3,4); addq %3,%4;"\
  "leaq (%4,%3,4),%4;"
#define SAVE_m4n8 "movq %2,%4; addq $16,%2;" unit_save_m4n8(%%ymm4,%%ymm5,%%ymm6,%%ymm7)
#define SAVE_m4n16 SAVE_m4n8 unit_save_m4n8(%%ymm8,%%ymm9,%%ymm10,%%ymm11)
#define SAVE_m4n24 SAVE_m4n16 unit_save_m4n8(%%ymm12,%%ymm13,%%ymm14,%%ymm15)
/* c_block row_major; ymm1 = a_up, ymm2 = a_lo */
#define unit_acc_k1m2n8(c1,c2,...) \
  "vmovups ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm3,%%ymm1,"#c1"; vfmadd231ps %%ymm3,%%ymm2,"#c2";"
#define KERNEL_h_k1m2n8 \
  "vbroadcastss (%0),%%ymm1; vbroadcastss 4(%0),%%ymm2; addq $8,%0;"\
  unit_acc_k1m2n8(%%ymm4,%%ymm5,%1)
#define KERNEL_t_k1m2n8 KERNEL_h_k1m2n8 "addq $32,%1;"
#define KERNEL_h_k1m2n16 KERNEL_h_k1m2n8 unit_acc_k1m2n8(%%ymm6,%%ymm7,%1,%%r12,1)
#define KERNEL_t_k1m2n16 KERNEL_h_k1m2n16 "addq $32,%1;"
#define KERNEL_h_k1m2n24 KERNEL_h_k1m2n16 unit_acc_k1m2n8(%%ymm8,%%ymm9,%1,%%r12,2)
#define KERNEL_t_k1m2n24 KERNEL_h_k1m2n24 "addq $32,%1;"
#define unit_init_m2n8(c1,c2) "vpxor "#c1","#c1","#c1"; vpxor "#c2","#c2","#c2";"
#define INIT_m2n8 unit_init_m2n8(%%ymm4,%%ymm5)
#define INIT_m2n16 INIT_m2n8 unit_init_m2n8(%%ymm6,%%ymm7)
#define INIT_m2n24 INIT_m2n16 unit_init_m2n8(%%ymm8,%%ymm9)
#define unit_save_m2n8(c1,c2) \
  "vunpcklps "#c2","#c1",%%ymm1; vmovsd (%4),%%xmm2; vmovhpd (%4,%3,1),%%xmm2,%%xmm2;"\
  "vfmadd231ps %%xmm0,%%xmm1,%%xmm2; vmovsd %%xmm2,(%4); vmovhpd %%xmm2,(%4,%3,1); leaq (%4,%3,2),%4;"\
  "vunpckhps "#c2","#c1",%%ymm3; vmovsd (%4),%%xmm2; vmovhpd (%4,%3,1),%%xmm2,%%xmm2;"\
  "vfmadd231ps %%xmm0,%%xmm3,%%xmm2; vmovsd %%xmm2,(%4); vmovhpd %%xmm2,(%4,%3,1); leaq (%4,%3,2),%4;"\
  "vextractf128 $1,%%ymm1,%%xmm1; vmovsd (%4),%%xmm2; vmovhpd (%4,%3,1),%%xmm2,%%xmm2;"\
  "vfmadd231ps %%xmm0,%%xmm1,%%xmm2; vmovsd %%xmm2,(%4); vmovhpd %%xmm2,(%4,%3,1); leaq (%4,%3,2),%4;"\
  "vextractf128 $1,%%ymm3,%%xmm3; vmovsd (%4),%%xmm2; vmovhpd (%4,%3,1),%%xmm2,%%xmm2;"\
  "vfmadd231ps %%xmm0,%%xmm3,%%xmm2; vmovsd %%xmm2,(%4); vmovhpd %%xmm2,(%4,%3,1); leaq (%4,%3,2),%4;"
#define SAVE_m2n8 "movq %2,%4; addq $8,%2;" unit_save_m2n8(%%ymm4,%%ymm5)
#define SAVE_m2n16 SAVE_m2n8 unit_save_m2n8(%%ymm6,%%ymm7)
#define SAVE_m2n24 SAVE_m2n16 unit_save_m2n8(%%ymm8,%%ymm9)
#define unit_acc_k1m1n8(c1,...) "vmovups ("#__VA_ARGS__"),%%ymm3; vfmadd231ps %%ymm3,%%ymm1,"#c1";"
#define KERNEL_h_k1m1n8 "vbroadcastss (%0),%%ymm1; addq $4,%0;" unit_acc_k1m1n8(%%ymm4,%1)
#define KERNEL_t_k1m1n8 KERNEL_h_k1m1n8 "addq $32,%1;"
#define KERNEL_h_k1m1n16 KERNEL_h_k1m1n8 unit_acc_k1m1n8(%%ymm5,%1,%%r12,1)
#define KERNEL_t_k1m1n16 KERNEL_h_k1m1n16 "addq $32,%1;"
#define KERNEL_h_k1m1n24 KERNEL_h_k1m1n16 unit_acc_k1m1n8(%%ymm6,%1,%%r12,2)
#define KERNEL_t_k1m1n24 KERNEL_h_k1m1n24 "addq $32,%1;"
#define INIT_m1n8 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define INIT_m1n16 INIT_m1n8 "vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m1n24 INIT_m1n16 "vpxor %%ymm6,%%ymm6,%%ymm6;"
#define unit_save_m1n8(c1_no) \
  "vextractf128 $1,%%ymm"#c1_no",%%xmm3;"\
  "vmovss (%4),%%xmm2; vinsertps $16,(%4,%3,1),%%xmm2,%%xmm2;"\
  "vinsertps $32,(%4,%3,2),%%xmm2,%%xmm2; addq %3,%4; vinsertps $48,(%4,%3,2),%%xmm2,%%xmm2; subq %3,%4;"\
  "vfmadd231ps %%xmm"#c1_no",%%xmm0,%%xmm2; vmovss %%xmm2,(%4); vextractps $1,%%xmm2,(%4,%3,1);"\
  "leaq (%4,%3,2),%4; vextractps $2,%%xmm2,(%4); vextractps $3,%%xmm2,(%4,%3,1); leaq (%4,%3,2),%4;"\
  "vmovss (%4),%%xmm2; vinsertps $16,(%4,%3,1),%%xmm2,%%xmm2;"\
  "vinsertps $32,(%4,%3,2),%%xmm2,%%xmm2; addq %3,%4; vinsertps $48,(%4,%3,2),%%xmm2,%%xmm2; subq %3,%4;"\
  "vfmadd231ps %%xmm3,%%xmm0,%%xmm2; vmovss %%xmm2,(%4); vextractps $1,%%xmm2,(%4,%3,1);"\
  "leaq (%4,%3,2),%4; vextractps $2,%%xmm2,(%4); vextractps $3,%%xmm2,(%4,%3,1); leaq (%4,%3,2),%4;"
#define SAVE_m1n8 "movq %2,%4; addq $4,%2;" unit_save_m1n8(4)
#define SAVE_m1n16 SAVE_m1n8 unit_save_m1n8(5)
#define SAVE_m1n24 SAVE_m1n16 unit_save_m1n8(6)

/* PART 2 : n=4/2/1: xmm0 for alpha, xmm1-xmm9 temporary use, xmm12-xmm15 accumulators */
/* c_block column_major z_partition */
#define KERNEL_t_k1m4n4 \
  "vmovddup (%1),%%xmm1; vmovddup 8(%1),%%xmm2; addq $16,%1;"\
  "vmovsldup (%0),%%xmm3; vfmadd231ps %%xmm3,%%xmm1,%%xmm12; vfmadd231ps %%xmm3,%%xmm2,%%xmm14;"\
  "vmovshdup (%0),%%xmm3; vfmadd231ps %%xmm3,%%xmm1,%%xmm13; vfmadd231ps %%xmm3,%%xmm2,%%xmm15;"\
  "addq $16,%0;"
#define INIT_m4n4 \
  "vpxor %%xmm12,%%xmm12,%%xmm12; vpxor %%xmm13,%%xmm13,%%xmm13;"\
  "vpxor %%xmm14,%%xmm14,%%xmm14; vpxor %%xmm15,%%xmm15,%%xmm15;"
#define SAVE_m4n4 "movq %2,%4; addq $16,%2;"\
  "vunpcklps %%xmm13,%%xmm12,%%xmm1; vunpckhps %%xmm13,%%xmm12,%%xmm2;"\
  "vunpcklpd %%xmm2,%%xmm1,%%xmm3; vunpckhpd %%xmm2,%%xmm1,%%xmm4;"\
  "vfmadd213ps (%4),%%xmm0,%%xmm3; vfmadd213ps (%4,%3,1),%%xmm0,%%xmm4;"\
  "vmovups %%xmm3,(%4); vmovups %%xmm4,(%4,%3,1); leaq (%4,%3,2),%4;"\
  "vunpcklps %%xmm15,%%xmm14,%%xmm1; vunpckhps %%xmm15,%%xmm14,%%xmm2;"\
  "vunpcklpd %%xmm2,%%xmm1,%%xmm3; vunpckhpd %%xmm2,%%xmm1,%%xmm4;"\
  "vfmadd213ps (%4),%%xmm0,%%xmm3; vfmadd213ps (%4,%3,1),%%xmm0,%%xmm4;"\
  "vmovups %%xmm3,(%4); vmovups %%xmm4,(%4,%3,1);"
/* c_block column_major */
#define KERNEL_t_k1m4n1 \
  "vbroadcastss (%1),%%xmm1; addq $4,%1;"\
  "vfmadd231ps (%0),%%xmm1,%%xmm12; addq $16,%0;"
#define KERNEL_t_k1m4n2 \
  "vmovups (%0),%%xmm1; addq $16,%0;"\
  "vbroadcastss (%1),%%xmm2; vfmadd231ps %%xmm2,%%xmm1,%%xmm12;"\
  "vbroadcastss 4(%1),%%xmm3; vfmadd231ps %%xmm3,%%xmm1,%%xmm13;"\
  "addq $8,%1;"
#define INIT_m4n1 "vpxor %%xmm12,%%xmm12,%%xmm12;"
#define INIT_m4n2 INIT_m4n1 "vpxor %%xmm13,%%xmm13,%%xmm13;"
#define SAVE_m4n1 "vfmadd213ps (%2),%%xmm0,%%xmm12; vmovups %%xmm12,(%2); addq $16,%2;"
#define SAVE_m4n2 "vfmadd213ps (%2),%%xmm0,%%xmm12; vmovups %%xmm12,(%2);"\
  "vfmadd213ps (%2,%3,1),%%xmm0,%%xmm13; vmovups %%xmm13,(%2,%3,1); addq $16,%2;"
#define KERNEL_t_k1m1n1 \
  "vmovss (%0),%%xmm1; addq $4,%0;"\
  "vfmadd231ss (%1),%%xmm1,%%xmm12; addq $4,%1;"
#define KERNEL_t_k1m2n1 \
  "vmovsd (%0),%%xmm1; addq $8,%0;"\
  "vbroadcastss (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm12; addq $4,%1;"
#define KERNEL_t_k1m2n2 \
  "vmovsd (%0),%%xmm1; addq $8,%0;"\
  "vbroadcastss (%1),%%xmm2; vfmadd231ps %%xmm1,%%xmm2,%%xmm12;"\
  "vbroadcastss 4(%1),%%xmm3; vfmadd231ps %%xmm1,%%xmm3,%%xmm13; addq $8,%1;"
#define INIT_m1n1 "vpxor %%xmm12,%%xmm12,%%xmm12;"
#define INIT_m2n1 INIT_m1n1
#define INIT_m2n2 INIT_m2n1 "vpxor %%xmm13,%%xmm13,%%xmm13;"
#define SAVE_m1n1 "vfmadd213ss (%2),%%xmm0,%%xmm12; vmovss %%xmm12,(%2); addq $4,%2;"
#define SAVE_m2n1 "vmovsd (%2),%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm12; vmovsd %%xmm12,(%2); addq $8,%2;"
#define SAVE_m2n2 "vmovsd (%2),%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm12; vmovsd %%xmm12,(%2);"\
  "vmovsd (%2,%3,1),%%xmm3; vfmadd213ps %%xmm3,%%xmm0,%%xmm13; vmovsd %%xmm13,(%2,%3,1); addq $8,%2;"
/* c_block row_major */
#define KERNEL_t_k1m1n2 \
  "vbroadcastss (%0),%%xmm1; addq $4,%0;"\
  "vmovsd (%1),%%xmm2; vfmadd231ps %%xmm2,%%xmm1,%%xmm12; addq $8,%1;"
#define INIT_m1n2 "vpxor %%xmm12,%%xmm12,%%xmm12;"
#define SAVE_m1n2 \
  "vmovss (%2),%%xmm2; vinsertps $16,(%2,%3,1),%%xmm2,%%xmm2;"\
  "vfmadd213ps %%xmm2,%%xmm0,%%xmm12; vmovss %%xmm12,(%2); vextractps $1,%%xmm12,(%2,%3,1); addq $4,%2;"
#define KERNEL_t_k1m1n4 \
  "vbroadcastss (%0),%%xmm1; addq $4,%0;"\
  "vfmadd231ps (%1),%%xmm1,%%xmm12; addq $16,%1;"
#define INIT_m1n4 "vpxor %%xmm12,%%xmm12,%%xmm12;"
#define SAVE_m1n4 "movq %2,%4; addq $4,%2;"\
  "vmovss (%4),%%xmm2; vinsertps $16,(%4,%3,1),%%xmm2,%%xmm2;"\
  "vinsertps $32,(%4,%3,2),%%xmm2,%%xmm2; addq %3,%4; vinsertps $48,(%4,%3,2),%%xmm2,%%xmm2; subq %3,%4;"\
  "vfmadd231ps %%xmm12,%%xmm0,%%xmm2; vmovss %%xmm2,(%4); vextractps $1,%%xmm2,(%4,%3,1);"\
  "leaq (%4,%3,2),%4; vextractps $2,%%xmm2,(%4); vextractps $3,%%xmm2,(%4,%3,1);"
/* c_block row_major z_partition */
#define KERNEL_t_k1m2n4 \
  "vmovddup (%0),%%xmm1; addq $8,%0;"\
  "vmovsldup (%1),%%xmm2; vfmadd231ps %%xmm2,%%xmm1,%%xmm12;"\
  "vmovshdup (%1),%%xmm3; vfmadd231ps %%xmm3,%%xmm1,%%xmm13; addq $16,%1;"
#define INIT_m2n4 "vpxor %%xmm12,%%xmm12,%%xmm12; vpxor %%xmm13,%%xmm13,%%xmm13;"
#define SAVE_m2n4 "movq %2,%4; addq $8,%2;"\
  "vmovsd (%4),%%xmm2; vmovhpd (%4,%3,2),%%xmm2,%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm12;"\
  "vmovsd %%xmm12,(%4); vmovhpd %%xmm12,(%4,%3,2); addq %3,%4;"\
  "vmovsd (%4),%%xmm2; vmovhpd (%4,%3,2),%%xmm2,%%xmm2; vfmadd213ps %%xmm2,%%xmm0,%%xmm13;"\
  "vmovsd %%xmm13,(%4); vmovhpd %%xmm13,(%4,%3,2);"

#define COMPUTE_m4(ndim) \
  "movq %%r13,%5; movq %%r14,%1; movq %2,%4;" INIT_m4n##ndim\
  "cmpq $32,%5; jb "#ndim"441f; "#ndim"440:\n\t"\
  KERNEL_t_k1m4n##ndim KERNEL_t_k1m4n##ndim\
  KERNEL_t_k1m4n##ndim KERNEL_t_k1m4n##ndim\
  "prefetcht1 (%4); prefetcht1 15(%4); addq %3,%4;"\
  KERNEL_t_k1m4n##ndim KERNEL_t_k1m4n##ndim\
  KERNEL_t_k1m4n##ndim KERNEL_t_k1m4n##ndim\
  "prefetcht1 (%8); addq $16,%8;"\
  "subq $8,%5; cmpq $32,%5; jnb "#ndim"440b;"\
  "movq %2,%4; "#ndim"441:\n\t"\
  "testq %5,%5; jz "#ndim"442f;"\
  "prefetcht0 (%4); prefetcht0 15(%4); addq %3,%4;"\
  KERNEL_t_k1m4n##ndim "decq %5; jmp "#ndim"441b;"\
  #ndim"442:\n\t"\
  SAVE_m4n##ndim

#define COMPUTE_m2(ndim) \
  "movq %%r13,%5; movq %%r14,%1;" INIT_m2n##ndim\
  #ndim"221:\n\t"\
  "testq %5,%5; jz "#ndim"222f;"\
  KERNEL_t_k1m2n##ndim "decq %5; jmp "#ndim"221b;"\
  #ndim"222:\n\t"\
  SAVE_m2n##ndim

#define COMPUTE_m1(ndim) \
  "movq %%r13,%5; movq %%r14,%1;" INIT_m1n##ndim\
  #ndim"111:\n\t"\
  "testq %5,%5; jz "#ndim"112f;"\
  KERNEL_t_k1m1n##ndim "decq %5; jmp "#ndim"111b;"\
  #ndim"112:\n\t"\
  SAVE_m1n##ndim

#define COMPUTE(ndim) {\
  b_pref=b_ptr+ndim*ldc;\
  __asm__ __volatile__(\
  GENERAL_INIT_ASM\
  "cmpq $4,%6; jb 99301f;"\
  "99300:\n\t"\
  COMPUTE_m4(ndim) "subq $4,%6; cmpq $4,%6; jnb 99300b;"\
  "99301:\n\t"\
  "cmpq $2,%6; jb 99302f;"\
  COMPUTE_m2(ndim) "subq $2,%6;"\
  "99302:\n\t"\
  "testq %6,%6; jz 99303f;"\
  COMPUTE_m1(ndim)\
  "99303:\n\t"\
  GENERAL_RECOVER_ASM\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(ldc_in_bytes),"+r"(c_tmp),"+r"(K),"+r"(M),"+r"(alp),"+r"(b_pref)\
  ::"ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7",\
  "ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15",\
  "r11","r12","r13","r14","cc","memory");\
  a_ptr-=M*K; b_ptr+=K*ndim; c_ptr+=ldc*ndim-M;\
}

//#include "common.h"
//#include <stdint.h>
#include <stdio.h>//debug
#include <stdlib.h>//debug
#define BLASLONG int//debug
int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, float alpha, float * __restrict__ A, float * __restrict__ B, float * __restrict__ C, BLASLONG ldc)
{
    if(m==0||n==0||k==0||alpha==(float)0.0) return 0;
    int64_t ldc_in_bytes = (int64_t)ldc * sizeof(float); float ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    float *a_ptr = A,*b_ptr = B,*c_ptr = C,*c_tmp = C,*alp = &ALPHA,*b_pref = B;
    for(;n_count>23;n_count-=24) COMPUTE(24)
    for(;n_count>15;n_count-=16) COMPUTE(16)
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
//gcc -march=haswell --shared -fPIC -O2 sgemm_kernel_4x8_haswell.c -o sgemm.so
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
      if(NOTRANSB) { sgemm_ncopy_8(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { sgemm_tcopy_8(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { sgemm_tcopy_4(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { sgemm_ncopy_4(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        CNAME(m_subdim,*n,k_subdim,*alpha,a_buffer,b_buffer,c+m_count,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}
