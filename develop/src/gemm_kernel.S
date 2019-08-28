#define A0	%rdi //ablk pointer
#define B0	%rsi //bblk pointer
#define CL      %r14 //cload pointer
#define CS      %r15 //cstore pointer
#define LDC     %rcx //ldc * sizeof(float)
#define AL      %rax //aload pointer
#define CIP  -8(%rsp)//cstartpos
//BlkDimK=BlkDimN=256
#ifdef DOUBLE
 #define VEC_BROAD vbroadcastsd
 #define VEC_FMA231 vfmadd231pd
 #define VEC_ADD vaddpd
 #define MASKMOV vmaskmovpd
 #define SIZE 8
#else
 #define VEC_BROAD vbroadcastss
 #define VEC_FMA231 vfmadd231ps
 #define VEC_ADD vaddps
 #define MASKMOV vmaskmovps
 #define SIZE 4
#endif

.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm5
    VEC_FMA231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm8
    VEC_FMA231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm10
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_FMA231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm13
    VEC_FMA231 %ymm0,%ymm2,%ymm14
    VEC_FMA231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_f Aoff,Boff,delta,deltb
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    addq $\delta,A0
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm5
    VEC_FMA231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm8
    VEC_FMA231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm10
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_FMA231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    addq $\deltb,B0
    VEC_FMA231 %ymm0,%ymm1,%ymm13
    VEC_FMA231 %ymm0,%ymm2,%ymm14
    VEC_FMA231 %ymm0,%ymm3,%ymm15
.endm
//A_PREF=256 in MakeFile
.macro KERNEL_8
    prefetcht0 A_PREF(A0)
    prefetcht0 A_PREF+64(A0)
    KERNEL_1 0,0
    prefetcht0 96*SIZE(B0)
    prefetcht0 A_PREF+128(A0)
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PREF+192(A0)
    prefetcht0 A_PREF+256(A0)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PREF+320(A0)
#ifdef DOUBLE
    prefetcht0 104*SIZE(B0)
#endif
    KERNEL_1 288,12*SIZE
    incq %r11
    prefetcht0 A_PREF+384(A0)
    prefetcht0 A_PREF+448(A0)
    KERNEL_1 384,16*SIZE
    prefetcht0 112*SIZE(B0)
    prefetcht0 A_PREF+512(A0)
    KERNEL_1 480,20*SIZE
    prefetcht0 A_PREF+576(A0)
    prefetcht0 A_PREF+640(A0)
    KERNEL_1 576,24*SIZE
    prefetcht0 A_PREF+704(A0)
#ifdef DOUBLE
    prefetcht0 120*SIZE(B0)
#endif
    KERNEL_f 672,28*SIZE,768,32*SIZE
.endm

.macro KERNEL_8_F
    prefetcht0 A_PREF(A0)
    prefetcht0 A_PREF+64(A0)
    KERNEL_1 0,0
    prefetcht0 96*SIZE(B0)
    prefetcht0 A_PREF+128(A0)
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PREF+192(A0)
    prefetcht0 A_PREF+256(A0)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PREF+320(A0)
#ifdef DOUBLE
    prefetcht0 104*SIZE(B0)
#endif
    KERNEL_1 288,12*SIZE
    prefetcht0 A_PREF+384(A0)
#if A_PREF > 319
    prefetcht0 A_PREF+448(A0,%r9,1)
#else
    prefetcht0 A_PREF+448(A0)
#endif
    KERNEL_1 384,16*SIZE
    prefetcht0 112*SIZE(B0)
#if A_PREF > 255
    prefetcht0 A_PREF+512(A0,%r9,1)
#else
    prefetcht0 A_PREF+512(A0)
#endif
    KERNEL_1 480,20*SIZE
#if A_PREF > 191
    prefetcht0 A_PREF+576(A0,%r9,1)
#else
    prefetcht0 A_PREF+576(A0)
#endif
#if A_PREF > 127
    prefetcht0 A_PREF+640(A0,%r9,1)
#else
    prefetcht0 A_PREF+640(A0)
#endif
    KERNEL_1 576,24*SIZE
    prefetcht0 A_PREF+704(A0,%r9,1)
#ifdef DOUBLE
    prefetcht0 120*SIZE(B0)
#endif
    KERNEL_f 672,28*SIZE,768,32*SIZE
.endm

.macro SHIFTYMM
    vmovaps %ymm7,%ymm4
    vmovaps %ymm8,%ymm5
    vmovaps %ymm9,%ymm6
    vmovaps %ymm10,%ymm7
    vmovaps %ymm11,%ymm8
    vmovaps %ymm12,%ymm9
    vmovaps %ymm13,%ymm10
    vmovaps %ymm14,%ymm11
    vmovaps %ymm15,%ymm12
.endm

.macro PREFcol1_t0 src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 95(\src)
.endm

.macro PREFcol1_t1 src
    prefetcht1 (\src)
    prefetcht1 64(\src)
    prefetcht1 95(\src)
.endm

.macro load_mask maskaddr,r1,r2,r3
    vmovups (\maskaddr),\r1
    vmovups 32(\maskaddr),\r2
    vmovups 64(\maskaddr),\r3
.endm

.macro CLEAR r1,r2,r3
    vpxor \r1,\r1,\r1
    vpxor \r2,\r2,\r2
    vpxor \r3,\r3,\r3
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR %ymm13,%ymm14,%ymm15
.endm

.macro STORECBLK_1col
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    VEC_ADD 64(CS),%ymm6,%ymm6
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1col_irregm StoreMask
    load_mask \StoreMask,%ymm1,%ymm2,%ymm3
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm4,%ymm0,%ymm4
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm5,%ymm0,%ymm5
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm6,%ymm0,%ymm6
    MASKMOV %ymm4,%ymm1,(CS)
    MASKMOV %ymm5,%ymm2,32(CS)
    MASKMOV %ymm6,%ymm3,64(CS)
    addq LDC,CS
.endm

.macro INIT_C_3col
    CLEAR %ymm7,%ymm8,%ymm9
    CLEAR %ymm10,%ymm11,%ymm12
    CLEAR %ymm13,%ymm14,%ymm15
.endm

.macro FIN_C_3col
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    VEC_ADD 64(CS),%ymm6,%ymm6
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm7,%ymm7
    VEC_ADD 32(CS),%ymm8,%ymm8
    VEC_ADD 64(CS),%ymm9,%ymm9
    vmovups %ymm7,(CS)
    vmovups %ymm8,32(CS)
    vmovups %ymm9,64(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm10,%ymm10
    VEC_ADD 32(CS),%ymm11,%ymm11
    VEC_ADD 64(CS),%ymm12,%ymm12
    vmovups %ymm10,(CS)
    vmovups %ymm11,32(CS)
    vmovups %ymm12,64(CS)
.endm

.macro FIN_C_3col_irregm
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm7,%ymm0,%ymm7
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm8,%ymm0,%ymm8
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm9,%ymm0,%ymm9
    MASKMOV %ymm7,%ymm1,(CS)
    MASKMOV %ymm8,%ymm2,32(CS)
    MASKMOV %ymm9,%ymm3,64(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm10,%ymm0,%ymm10
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm11,%ymm0,%ymm11
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm12,%ymm0,%ymm12
    MASKMOV %ymm10,%ymm1,(CS)
    MASKMOV %ymm11,%ymm2,32(CS)
    MASKMOV %ymm12,%ymm3,64(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm1,%ymm0
    VEC_ADD %ymm13,%ymm0,%ymm13
    MASKMOV 32(CS),%ymm2,%ymm0
    VEC_ADD %ymm14,%ymm0,%ymm14
    MASKMOV 64(CS),%ymm3,%ymm0
    VEC_ADD %ymm15,%ymm0,%ymm15
    MASKMOV %ymm13,%ymm1,(CS)
    MASKMOV %ymm14,%ymm2,32(CS)
    MASKMOV %ymm15,%ymm3,64(CS)
.endm

.macro SETMASKm//use stack to store mask integer array
#ifdef DOUBLE
    xorq %rax,%rax
    subq %r8,%rax
    addq $11,%rax
    movq %rax,-8(%rsp)
    decq %rax
    movq %rax,-16(%rsp)
    decq %rax
    movq %rax,-24(%rsp)
    decq %rax
    movq %rax,-32(%rsp)
    decq %rax
    movq %rax,-40(%rsp)
    decq %rax
    movq %rax,-48(%rsp)
    decq %rax
    movq %rax,-56(%rsp)
    decq %rax
    movq %rax,-64(%rsp)
    decq %rax
    movq %rax,-72(%rsp)
    decq %rax
    movq %rax,-80(%rsp)
    decq %rax
    movq %rax,-88(%rsp)
    decq %rax
    movq %rax,-96(%rsp)
    leaq -96(%rsp),%rax
#else
    xorl %eax,%eax
    subl %r8d,%eax
    addl $23,%eax
    movl %eax,-4(%rsp)
    decl %eax
    movl %eax,-8(%rsp)
    decl %eax
    movl %eax,-12(%rsp)
    decl %eax
    movl %eax,-16(%rsp)
    decl %eax
    movl %eax,-20(%rsp)
    decl %eax
    movl %eax,-24(%rsp)
    decl %eax
    movl %eax,-28(%rsp)
    decl %eax
    movl %eax,-32(%rsp)
    decl %eax
    movl %eax,-36(%rsp)
    decl %eax
    movl %eax,-40(%rsp)
    decl %eax
    movl %eax,-44(%rsp)
    decl %eax
    movl %eax,-48(%rsp)
    decl %eax
    movl %eax,-52(%rsp)
    decl %eax
    movl %eax,-56(%rsp)
    decl %eax
    movl %eax,-60(%rsp)
    decl %eax
    movl %eax,-64(%rsp)
    decl %eax
    movl %eax,-68(%rsp)
    decl %eax
    movl %eax,-72(%rsp)
    decl %eax
    movl %eax,-76(%rsp)
    decl %eax
    movl %eax,-80(%rsp)
    decl %eax
    movl %eax,-84(%rsp)
    decl %eax
    movl %eax,-88(%rsp)
    decl %eax
    movl %eax,-92(%rsp)
    decl %eax
    movl %eax,-96(%rsp)
    leaq -96(%rsp),%rax
#endif
.endm

.macro SET_LDC
#ifdef DOUBLE
    salq $3,LDC
#else
    salq $2,LDC
#endif
.endm

.section .text
//enter the function gemmblkregccc, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc
.globl gemmblkregccc
.type gemmblkregccc,@function
gemmblkregccc:

    push %r15
    push %r14
    push %r12
    movq %rdx,CIP
    movq %rdi,AL
    addq $49152,AL //point to (prefetch) next ablk zone of abuffer; delta=48kB
    movslq %ecx,LDC
    SET_LDC
    movq CIP,CS

    INIT_C_3col
    movq $0xa000000000000000,%r10 //ablk pointer increment table = (0xa000,0x0000,0x0000,0x0000)
    xorq %r12,%r12
.Louter_gemmblkregccc:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFcol1_t1 CS
    prefetcht2 95(CS,LDC,1) //TLB pref
    subq $96,AL
    prefetcht1 (AL)
    prefetcht1 64(AL)
    xorq %r11,%r11
.Linner_gemmblkregccc:
    KERNEL_8
    cmpq $7,%r11
    jb .Linner_gemmblkregccc
    PREFcol1_t0 CS
    KERNEL_8_F

    addq %r9,A0
    incq %r12
    STORECBLK_1col
    rorq $16,%r10
    cmpq $252,%r12
    jb .Louter_gemmblkregccc

    movq AL,%r9
    subq $384,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_gemmblkregccc_last:
    PREFcol1_t0 CS
    prefetcht1 (CL)
    prefetcht1 64(CL)
    prefetcht1 95(CL)
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_last:
    prefetcht0 A_PREF(A0)
    prefetcht0 A_PREF+64(A0)
    prefetcht1 (%r9)
    KERNEL_1 0,0
    prefetcht0 A_PREF+128(A0)
    prefetcht0 96*SIZE(B0)
    prefetcht1 64(%r9)
    incq %r11
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PREF+192(A0)
    prefetcht0 A_PREF+256(A0)
    prefetcht1 128(%r9)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PREF+320(A0)
#ifdef DOUBLE
    prefetcht0 104*SIZE(B0)
#endif
    prefetcht1 192(%r9)
    addq $256,%r9
    KERNEL_f 288,12*SIZE,384,16*SIZE
    cmpq $16,%r11
    jb .Linner_gemmblkregccc_last

    incq %r12
    STORECBLK_1col
    UPDATECBLK_1col
    cmpq $256,%r12
    jb .Louter_gemmblkregccc_last

    movq CIP,CS
    FIN_C_3col

    vzeroupper
    pop %r12
    pop %r14
    pop %r15
    retq

//enter the function gemmblktailccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim
.globl gemmblktailccc
.type gemmblktailccc,@function
gemmblktailccc:

    push %r15
    push %r14
    push %r12
    push %rdx //cstartpos
    movslq %ecx,LDC
    SET_LDC
    movslq %r8d,%r8 //mdim
    SETMASKm //generate mask integers. now rax point to the base element of mask integers, just like %2 in gemm.c
    add $8,%rsp //recover rsp so "CIP" can work normally
    movq CIP,CS
    INIT_C_3col
    xorq %r12,%r12
    movq $0x6000000000000000,%r10
.Louter_tail:
    UPDATECBLK_1col
    PREFcol1_t0 CS
    xorq %r11,%r11
.Linner_tail:
    KERNEL_8
    cmpq $8,%r11
    jb .Linner_tail

    STORECBLK_1col_irregm %rax
    incq %r12
    movswq %r10w,%r9
    subq %r9,A0
    ror $16,%r10
    cmpq $256,%r12
    jb .Louter_tail

    movq CIP,CS
    FIN_C_3col_irregm
    vzeroupper
    pop %r12
    pop %r14
    pop %r15
    retq

//enter the function timedelay
.globl timedelay
.type timedelay,@function
timedelay:
    xorq %r11,%r11
.Ltimedelay:
    incq %r11
    vhaddpd %ymm0,%ymm0,%ymm0
    cmpq $2000,%r11
    jb .Ltimedelay

    vzeroupper
    retq
