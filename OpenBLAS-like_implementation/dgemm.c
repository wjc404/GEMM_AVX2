
//ymm0-ymm3(xmm0-xmm3) for temporary use, ymm4-ymm15(xmm4-xmm15) for accumulators.
//%0 -> a; %1 -> b; %2 -> c; %3 -> alpha; %4 = ldc(bytes); 
//cases with n=8,12 requires r12 for efficient addressing. r12 = k << 5;

/* c_block zigzag partition */
#define KERNEL_k2m4n12 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; prefetcht0 512(%0);"\
    "prefetcht0 128(%1); prefetcht0 128(%1,%%r12,1); prefetcht0 128(%1,%%r12,2);"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128   (%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 16(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "vbroadcastf128   (%1,%%r12,2),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm2,%%ymm13;"\
    "vbroadcastf128 16(%1,%%r12,2),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"\
    "vmovupd 32(%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $64,%0;"\
    "vbroadcastf128 32(%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 48(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128 32(%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 48(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "vbroadcastf128 32(%1,%%r12,2),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm2,%%ymm13;"\
    "vbroadcastf128 48(%1,%%r12,2),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"\
    "addq $64,%1;"

#define KERNEL_k1m4n12 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128   (%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 16(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "vbroadcastf128   (%1,%%r12,2),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm2,%%ymm13;"\
    "vbroadcastf128 16(%1,%%r12,2),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"\
    "addq $32,%1;"

#define unit_save_m4n2(c1,c2) \
    "vunpcklpd "#c2","#c1",%%ymm2; vunpckhpd "#c1","#c2",%%ymm3;"\
    "vfmadd213pd (%2),%%ymm0,%%ymm2; vfmadd213pd (%2,%4,1),%%ymm0,%%ymm3;"\
    "vmovupd %%ymm2,(%2); vmovupd %%ymm3,(%2,%4,1);"\
    "prefetcht1 56(%2); prefetcht1 56(%2,%4,1);"

#define SAVE_m4n12 \
    "vbroadcastsd (%3),%%ymm0;"\
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm6,%%ymm7)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm8,%%ymm9)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm10,%%ymm11)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm12,%%ymm13)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm14,%%ymm15)\
    "salq $1,%4;subq %4,%2;salq $2,%4;subq %4,%2;sarq $3,%4;addq $32,%2;"

#define KERNEL_k1m4n8 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128   (%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 16(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "addq $32,%1;"

#define KERNEL_k2m4n8 KERNEL_k1m4n8 KERNEL_k1m4n8

#define SAVE_m4n8 \
    "vbroadcastsd (%3),%%ymm0;"\
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm6,%%ymm7)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm8,%%ymm9)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm10,%%ymm11)\
    "salq $1,%4;subq %4,%2;salq $1,%4;subq %4,%2;sarq $2,%4;addq $32,%2;"

#define KERNEL_k1m4n4 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "addq $32,%1;"

#define KERNEL_k2m4n4 KERNEL_k1m4n4 KERNEL_k1m4n4

#define SAVE_m4n4 \
    "vbroadcastsd (%3),%%ymm0;"\
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m4n2(%%ymm6,%%ymm7)\
    "subq %4,%2;subq %4,%2;addq $32,%2;"

#define KERNEL_k1m4n2 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "addq $16,%1;"

#define KERNEL_k2m4n2 KERNEL_k1m4n2 KERNEL_k1m4n2

#define SAVE_m4n2 \
    "vbroadcastsd (%3),%%ymm0;"\
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "addq $32,%2;"

#define INIT_m4n2 "vpxor %%ymm4,%%ymm4,%%ymm4; vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m4n4 INIT_m4n2 "vpxor %%ymm6,%%ymm6,%%ymm6; vpxor %%ymm7,%%ymm7,%%ymm7;"
#define INIT_m4n8 INIT_m4n4 "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9; vpxor %%ymm10,%%ymm10,%%ymm10; vpxor %%ymm11,%%ymm11,%%ymm11;"
#define INIT_m4n12 INIT_m4n8 "vpxor %%ymm12,%%ymm12,%%ymm12; vpxor %%ymm13,%%ymm13,%%ymm13; vpxor %%ymm14,%%ymm14,%%ymm14; vpxor %%ymm15,%%ymm15,%%ymm15;"

/* c_block column_major */
#define INIT_m4n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"

#define KERNEL_k1m4n1 \
    "vmovupd   (%0),%%ymm0; addq $32,%0;"\
    "vbroadcastsd (%1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4; addq $8,%1;"

#define KERNEL_k2m4n1 KERNEL_k1m4n1 KERNEL_k1m4n1

#define SAVE_m4n1 \
    "vbroadcastsd (%3),%%ymm0;"\
    "vfmadd213pd (%2),%%ymm0,%%ymm4; vmovupd %%ymm4,(%2);"\
    "addq $32,%2;"

#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"

#define KERNEL_k1m2n1 \
    "vmovupd   (%0),%%xmm0; addq $16,%0;"\
    "vmovddup  (%1),%%xmm2; vfmadd231pd %%xmm0,%%xmm2,%%xmm4; addq $8,%1;"

#define SAVE_m2n1 \
    "vmovddup (%3),%%xmm0;"\
    "vfmadd213pd (%2),%%xmm0,%%xmm4; vmovupd %%xmm4,(%2);"\
    "addq $16,%2;"

#define INIT_m2n2 "vpxor %%xmm4,%%xmm4,%%xmm4; vpxor %%xmm5,%%xmm5,%%xmm5;"

#define KERNEL_k1m2n2 \
    "vmovupd   (%0),%%xmm0; addq $16,%0;"\
    "vmovddup  (%1),%%xmm2; vfmadd231pd %%xmm0,%%xmm2,%%xmm4;"\
    "vmovddup 8(%1),%%xmm3; vfmadd231pd %%xmm0,%%xmm3,%%xmm5; addq $16,%1;"

#define SAVE_m2n2 \
    "vmovddup (%3),%%xmm0;"\
    "vfmadd213pd (%2),%%xmm0,%%xmm4; vmovupd %%xmm4,(%2);"\
    "vfmadd213pd (%2,%4,1),%%xmm0,%%xmm5; vmovupd %%xmm5,(%2,%4,1);"\
    "addq $16,%2;"

#define INIT_m1n1 INIT_m2n1

#define KERNEL_k1m1n1 \
    "vmovsd    (%0),%%xmm0; addq $8,%0;"\
    "vmovsd    (%1),%%xmm2; vfmadd231sd %%xmm0,%%xmm2,%%xmm4; addq $8,%1;"

#define SAVE_m1n1 \
    "vmovsd (%3),%%xmm0;"\
    "vfmadd213sd (%2),%%xmm0,%%xmm4; vmovsd %%xmm4,(%2);"\
    "addq $8,%2;"

/* c_block row_major */
#define INIT_m1n2 INIT_m2n1

#define KERNEL_k1m1n2 \
    "vmovupd   (%1),%%xmm0; addq $16,%1;"\
    "vmovddup  (%0),%%xmm2; vfmadd231pd %%xmm0,%%xmm2,%%xmm4; addq $8,%0;"

#define unit_save_m1n2(xmm1) \
    "vmovsd (%2),%%xmm2; vmovhpd (%2,%4,1),%%xmm2,%%xmm2;"\
    "vfmadd231pd %%"#xmm1",%%xmm0,%%xmm2;"\
    "vmovsd %%xmm2,(%2); vmovhpd %%xmm2,(%2,%4,1);"

#define SAVE_m1n2 \
    "vmovddup (%3),%%xmm0;"\
    unit_save_m1n2(xmm4)\
    "addq $8,%2;"

#define INIT_m1n4 INIT_m4n1

#define KERNEL_k1m1n4 \
    "vmovupd   (%1),%%ymm0; addq $32,%1;"\
    "vbroadcastsd (%0),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4; addq $8,%0;"

#define SAVE_m1n4 \
    "vmovddup (%3),%%xmm0;"\
    "vextractf128 $1,%%ymm4,%%xmm3;"\
    unit_save_m1n2(xmm4)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m1n2(xmm3)\
    "subq %4,%2;subq %4,%2;addq $8,%2;"

#define INIT_m1n8 INIT_m4n2

#define KERNEL_k1m1n8 \
    "vmovupd   (%1),%%ymm0; vmovupd (%1,%%r12,1),%%ymm1; addq $32,%1;"\
    "vbroadcastsd (%0),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4; vfmadd231pd %%ymm1,%%ymm2,%%ymm5; addq $8,%0;"

#define SAVE_m1n8 \
    "vmovddup (%3),%%xmm0;"\
    "vextractf128 $1,%%ymm4,%%xmm3;"\
    unit_save_m1n2(xmm4)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m1n2(xmm3)\
    "leaq (%2,%4,2),%2;"\
    "vextractf128 $1,%%ymm5,%%xmm3;"\
    unit_save_m1n2(xmm5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m1n2(xmm3)\
    "salq $1,%4;subq %4,%2;salq $1,%4;subq %4,%2;sarq $2,%4;addq $8,%2;"

#define INIT_m1n12 INIT_m1n8 "vpxor %%ymm6,%%ymm6,%%ymm6;"

#define KERNEL_k1m1n12 \
    "vmovupd   (%1),%%ymm0; vmovupd (%1,%%r12,1),%%ymm1; vmovupd (%1,%%r12,2),%%ymm3; addq $32,%1;"\
    "vbroadcastsd (%0),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4; vfmadd231pd %%ymm1,%%ymm2,%%ymm5; vfmadd231pd %%ymm3,%%ymm2,%%ymm6; addq $8,%0;"

#define SAVE_m1n12 \
    "vmovddup (%3),%%xmm0;"\
    "vextractf128 $1,%%ymm4,%%xmm3;"\
    unit_save_m1n2(xmm4)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m1n2(xmm3)\
    "leaq (%2,%4,2),%2;"\
    "vextractf128 $1,%%ymm5,%%xmm3;"\
    unit_save_m1n2(xmm5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m1n2(xmm3)\
    "leaq (%2,%4,2),%2;"\
    "vextractf128 $1,%%ymm6,%%xmm3;"\
    unit_save_m1n2(xmm6)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m1n2(xmm3)\
    "salq $1,%4;subq %4,%2;salq $2,%4;subq %4,%2;sarq $3,%4;addq $8,%2;"

#define unit_save_m2n4(c1,c2) \
    "vunpcklpd %%ymm"#c2",%%ymm"#c1",%%ymm2; vunpckhpd %%ymm"#c2",%%ymm"#c1",%%ymm3;"\
    "vextractf128 $1,%%ymm2,%%xmm"#c1"; vextractf128 $1,%%ymm3,%%xmm"#c2";"\
    "vfmadd213pd (%2),%%xmm0,%%xmm2; vfmadd213pd (%2,%4,1),%%xmm0,%%xmm3;"\
    "vmovupd %%xmm2,(%2); vmovupd %%xmm3,(%2,%4,1);"\
    "leaq (%2,%4,2),%2;"\
    "vfmadd213pd (%2),%%xmm0,%%xmm"#c1"; vfmadd213pd (%2,%4,1),%%xmm0,%%xmm"#c2";"\
    "vmovupd %%xmm"#c1",(%2); vmovupd %%xmm"#c2",(%2,%4,1);"\

#define INIT_m2n4 "vpxor %%ymm4,%%ymm4,%%ymm4; vpxor %%ymm5,%%ymm5,%%ymm5;"

#define KERNEL_k1m2n4 \
    "vmovupd   (%1),%%ymm0; addq $32,%1;"\
    "vbroadcastsd (%0),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm4;"\
    "vbroadcastsd 8(%0),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm5; addq $16,%0;"

#define SAVE_m2n4 \
    "vmovddup (%3),%%xmm0;"\
    unit_save_m2n4(4,5)\
    "subq %4,%2;subq %4,%2;addq $16,%2;"

#define INIT_m2n8 INIT_m2n4 "vpxor %%ymm6,%%ymm6,%%ymm6; vpxor %%ymm7,%%ymm7,%%ymm7;"

#define KERNEL_k1m2n8 \
    "vmovupd   (%1),%%ymm0; vmovupd (%1,%%r12,1),%%ymm1; addq $32,%1;"\
    "vbroadcastsd (%0),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm4; vfmadd231pd %%ymm1,%%ymm3,%%ymm6;"\
    "vbroadcastsd 8(%0),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm5; vfmadd231pd %%ymm1,%%ymm3,%%ymm7; addq $16,%0;"

#define SAVE_m2n8 \
    "vmovddup (%3),%%xmm0;"\
    unit_save_m2n4(4,5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m2n4(6,7)\
    "salq $1,%4;subq %4,%2;salq $1,%4;subq %4,%2;sarq $2,%4;addq $16,%2;"

#define INIT_m2n12 INIT_m2n8 "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9;"

#define KERNEL_k1m2n12 \
    "vmovupd   (%1),%%ymm0; vmovupd (%1,%%r12,1),%%ymm1; vmovupd (%1,%%r12,2),%%ymm2; addq $32,%1;"\
    "vbroadcastsd (%0),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm4; vfmadd231pd %%ymm1,%%ymm3,%%ymm6; vfmadd231pd %%ymm2,%%ymm3,%%ymm8;"\
    "vbroadcastsd 8(%0),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm5; vfmadd231pd %%ymm1,%%ymm3,%%ymm7; vfmadd231pd %%ymm2,%%ymm3,%%ymm9; addq $16,%0;"

#define SAVE_m2n12 \
    "vmovddup (%3),%%xmm0;"\
    unit_save_m2n4(4,5)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m2n4(6,7)\
    "leaq (%2,%4,2),%2;"\
    unit_save_m2n4(8,9)\
    "salq $1,%4;subq %4,%2;salq $2,%4;subq %4,%2;sarq $3,%4;addq $16,%2;"

#define KERNEL_m4(ndim) \
    "movq %2,%7;cmpq $24,%5;jb 74"#ndim"1f;"\
    "74"#ndim"0:\n\t"\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    "prefetcht1 (%7); prefetcht1 31(%7); addq %4,%7;"\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    "prefetcht1 (%8); addq $10,%8;"\
    "subq $12,%5;cmpq $24,%5;jnb 74"#ndim"0b;"\
    "movq %2,%7;"\
    "74"#ndim"1:\n\t"\
    "cmpq $1,%5;jb 74"#ndim"2f;"\
    "prefetcht0 (%7); prefetcht0 31(%7); addq %4,%7;"\
    KERNEL_k1m4n##ndim\
    "decq %5;jmp 74"#ndim"1b;"\
    "74"#ndim"2:\n\t"\
    "movq %%r13,%5; movq %%r11,%1;"\
    "prefetcht0 (%1);prefetcht0 64(%1);"

#define KERNEL_m2(ndim) \
    "72"#ndim"1:\n\t"\
    "cmpq $1,%5;jb 72"#ndim"2f;"\
    KERNEL_k1m2n##ndim\
    "decq %5;jmp 72"#ndim"1b;"\
    "72"#ndim"2:\n\t"\
    "movq %%r13,%5; movq %%r11,%1;"

#define KERNEL_m1(ndim) \
    "71"#ndim"1:\n\t"\
    "cmpq $1,%5;jb 71"#ndim"2f;"\
    KERNEL_k1m1n##ndim\
    "decq %5;jmp 71"#ndim"1b;"\
    "71"#ndim"2:\n\t"\
    "movq %%r13,%5; movq %%r11,%1;"

//%0 -> a; %1 -> b; %2 -> c; %3 -> alpha; %4 = ldc(bytes); %5 = k_count, %6 = m_count, %7 = c_pref, %8 = b_pref;
//cases with n=8,12 requires r12 for efficient addressing. r12 = k << 5; r13 for k, r14 for m, r15 for a_head_address; r11 for b_head_address;
#define COMPUTE(ndim) {\
    b_pref = b_pointer + ndim * K;\
    __asm__ __volatile__(\
    "movq %1,%%r11; movq %5,%%r13; movq %6,%%r14; movq %0,%%r15; movq %5,%%r12; salq $5,%%r12;"\
    "cmpq $4,%6;jb "#ndim"01f;"\
    #ndim"00:\n\t"\
    INIT_m4n##ndim\
    KERNEL_m4(ndim)\
    SAVE_m4n##ndim\
    "subq $4,%6;cmpq $4,%6;jnb "#ndim"00b;"\
    #ndim"01:\n\t"\
    "cmpq $2,%6;jb "#ndim"02f;"\
    INIT_m2n##ndim\
    KERNEL_m2(ndim)\
    SAVE_m2n##ndim\
    "subq $2,%6;"\
    #ndim"02:\n\t"\
    "cmpq $1,%6;jb "#ndim"03f;"\
    INIT_m1n##ndim\
    KERNEL_m1(ndim)\
    SAVE_m1n##ndim\
    #ndim"03:\n\t"\
    "movq %%r14,%6;salq $3,%%r14;subq %%r14,%2;movq %%r15,%0;vzeroupper;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(ALPHA),"+r"(ldc_in_bytes),"+r"(K),"+r"(M),"+r"(c_pref),"+r"(b_pref):\
    :"xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","r11","r12","r13","r14","r15","cc","memory");\
    b_pointer += K * ndim; c_pointer += ldc * ndim;\
}

#define BLASLONG int//temp
#define CNAME KERNELFUNC//temp
#include <stdio.h>//temp
#include <stdlib.h>//temp
#include <immintrin.h>//temp

//#include "common.h"
//#include <stdint.h>
int __attribute__ ((noinline)) CNAME(BLASLONG m, BLASLONG n, BLASLONG k, double alpha, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, BLASLONG ldc){
    if(m==0 || n==0 || k==0 || alpha == 0.0) return 0; double AA = alpha; double *ALPHA = &AA;
    int64_t ldc_in_bytes = (int64_t)ldc * sizeof(double), M = (int64_t)m, K = (int64_t)k;
    double *a_pointer = A, *b_pointer = B, *c_pointer = C, *c_pref = C,*b_pref = B;
    BLASLONG ndim_count = n;
    for(;ndim_count>11;ndim_count-=12) COMPUTE(12)
    for(;ndim_count>7;ndim_count-=8) COMPUTE(8)
    for(;ndim_count>3;ndim_count-=4) COMPUTE(4)
    for(;ndim_count>1;ndim_count-=2) COMPUTE(2)
    if(ndim_count>0) COMPUTE(1)
    return 0;
}

/* test zone */
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
#define BLOCKDIM_K 240 //GEMM_Q in OpenBLAS
#define BLOCKDIM_M 512 //GEMM_P in OpenBLAS
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
//gcc -march=haswell --shared -fPIC -O2 dgemm.c -o dgemm.so
void dgemm_(char *transa,char *transb,BLASLONG *m,BLASLONG *n,BLASLONG *k,double *alpha,double *a,BLASLONG *lda,double *b,BLASLONG *ldb,double *beta,double *c,BLASLONG *ldc){
    if((*m)==0||(*n)==0) return;
    if((*beta)!=1.0) SCALE_MULT(c,beta,*ldc,*m,*n);
    if((*alpha)==0.0||(*k)==0) return;
/* start main calculation here */
    //if((*m)==91 && (*n)==45 && (*k)==31) c[101]*=2.0;
    double *b_buffer = (double *)aligned_alloc(64,BLOCKDIM_K*(*n)*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,BLOCKDIM_K*BLOCKDIM_M*sizeof(double));
    double *a_current_pos,*b_current_pos;
    BLASLONG m_count,n_count,k_count,k_subdim,m_subdim;
    b_current_pos = b;
    for(k_count=0;k_count<(*k);k_count+=BLOCKDIM_K){
      k_subdim = (*k)-k_count;
      if(k_subdim > BLOCKDIM_K) k_subdim = BLOCKDIM_K;
      if(NOTRANSB) { dgemm_ncopy_4(b_current_pos,b_buffer,*ldb,k_subdim,*n); b_current_pos += BLOCKDIM_K; }
      else { dgemm_tcopy_4(b_current_pos,b_buffer,*ldb,*n,k_subdim); b_current_pos += (int64_t)(*ldb) * BLOCKDIM_K; }
      if(NOTRANSA) a_current_pos = a + (int64_t)k_count * (int64_t)(*lda);
      else a_current_pos = a + k_count;
      for(m_count=0;m_count<(*m);m_count+=BLOCKDIM_M){
        m_subdim = (*m)-m_count;
        if(m_subdim > BLOCKDIM_M) m_subdim = BLOCKDIM_M;
        if(NOTRANSA) { dgemm_tcopy_4(a_current_pos,a_buffer,*lda,m_subdim,k_subdim); a_current_pos += BLOCKDIM_M; }
        else { dgemm_ncopy_4(a_current_pos,a_buffer,*lda,k_subdim,m_subdim); a_current_pos += (int64_t)(*lda) * BLOCKDIM_M; }
        KERNELFUNC(m_subdim,*n,k_subdim,*alpha,a_buffer,b_buffer,c+m_count,*ldc);
      }
    }
    free(a_buffer);a_buffer=NULL;
    free(b_buffer);b_buffer=NULL;
}
/* debug zone */
//gcc -march=haswell dgemm.c -o test
/*
int main(){
    char transa='N',transb='N';
    BLASLONG m,n,k,lda,ldb,ldc; m=n=k=lda=ldb=ldc=12;
    double alpha = 1.0, beta = 1.0;
    double A[m*k],B[k*n],C[n*m];
    BLASLONG count,count2;
    for(count=0;count<m*k;count++) A[count]=5.0+(double)(count%2);
    for(count=0;count<k*n;count++) B[count]=20.0+1.0*(count/ldb%8);
    for(count=0;count<m*n;count++) C[count]=0.0;
    dgemm_(&transa,&transb,&m,&n,&k,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
    for(count=0;count<m;count++){
      printf("Row %d\t:",count+1);
      for(count2=0;count2<n;count2++){
        printf(" %.1f",C[count2*ldc+count]);
      }
      putchar('\n');
    }
    return 0;
}
*/
