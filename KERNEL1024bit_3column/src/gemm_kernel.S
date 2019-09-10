#define A0      %rdi //ablk pointer
#define B0      %rsi //bblk pointer
#define CL      %r14 //cload pointer
#define CS      %r15 //cstore pointer
#define LDC     %rcx //ldc * sizeof(float)
#define AL      %rax //aload pointer
#define CIP  -8(%rsp)//cstartpos
#define AD      %r10 //A offset

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

#define NEXT_A_PREF_STEP (128*BlkDimK/BlkDimN) //in bytes

.macro KERNEL_1 Aoff,Boff
    VEC_BROAD \Boff(B0),%ymm1
    VEC_BROAD \Boff+SIZE(B0),%ymm2
    VEC_BROAD \Boff+2*SIZE(B0),%ymm3
    vmovaps \Aoff(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm8
    VEC_FMA231 %ymm0,%ymm3,%ymm12
    vmovaps \Aoff+32(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm5
    VEC_FMA231 %ymm0,%ymm2,%ymm9
    VEC_FMA231 %ymm0,%ymm3,%ymm13
    vmovaps \Aoff+64(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm6
    VEC_FMA231 %ymm0,%ymm2,%ymm10
    VEC_FMA231 %ymm0,%ymm3,%ymm14
    vmovaps \Aoff+96(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_FMA231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_f Aoff,Boff,delta,deltb
    VEC_BROAD \Boff(B0),%ymm1
    VEC_BROAD \Boff+SIZE(B0),%ymm2
    VEC_BROAD \Boff+2*SIZE(B0),%ymm3
    addq $\deltb,B0
    vmovaps \Aoff(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm8
    VEC_FMA231 %ymm0,%ymm3,%ymm12
    vmovaps \Aoff+32(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm5
    VEC_FMA231 %ymm0,%ymm2,%ymm9
    VEC_FMA231 %ymm0,%ymm3,%ymm13
    vmovaps \Aoff+64(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm6
    VEC_FMA231 %ymm0,%ymm2,%ymm10
    VEC_FMA231 %ymm0,%ymm3,%ymm14
    vmovaps \Aoff+96(A0),%ymm0
    addq $\delta,A0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_FMA231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_8 Arefpos,Areset //Arefpos=ablk_startpos+(BlkDimK-8)*128;Areset=(-BlkDimK)*128
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    KERNEL_1 128,3*SIZE
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    KERNEL_1 256,6*SIZE
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    incq %r11
    KERNEL_1 384,9*SIZE
    prefetcht0 A_PR_BYTE+512(A0)
    prefetcht0 A_PR_BYTE+576(A0)
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    KERNEL_1 512,12*SIZE
#if A_PR_BYTE > 383
    prefetcht0 A_PR_BYTE+640(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+640(A0)
#endif
#if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+704(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+704(A0)
#endif
    KERNEL_1 640,15*SIZE
#if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+768(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+768(A0)
#endif
#if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+832(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+832(A0)
#endif
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    KERNEL_1 768,18*SIZE
#if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+896(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+896(A0)
#endif
#if A_PR_BYTE > 63
    prefetcht0 A_PR_BYTE+960(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+960(A0)
#endif
    KERNEL_f 896,21*SIZE,1024,24*SIZE
.endm

.macro SHIFTYMM
    vmovaps %ymm8,%ymm4
    vmovaps %ymm9,%ymm5
    vmovaps %ymm10,%ymm6
    vmovaps %ymm11,%ymm7
    vmovaps %ymm12,%ymm8
    vmovaps %ymm13,%ymm9
    vmovaps %ymm14,%ymm10
    vmovaps %ymm15,%ymm11
.endm

.macro PREFm12 src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 127(\src)
.endm

.macro CLEAR r1,r2,r3,r4
    vpxor \r1,\r1,\r1
    vpxor \r2,\r2,\r2
    vpxor \r3,\r3,\r3
    vpxor \r4,\r4,\r4
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR %ymm12,%ymm13,%ymm14,%ymm15
.endm

.macro STORECBLK_1col
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    VEC_ADD 64(CS),%ymm6,%ymm6
    VEC_ADD 96(CS),%ymm7,%ymm7
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    vmovups %ymm7,96(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1col_irregm maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm4,%ymm2,%ymm4
    VEC_ADD %ymm5,%ymm3,%ymm5
    MASKMOV %ymm4,%ymm0,(CS)
    MASKMOV %ymm5,%ymm1,32(CS)
    vmovups 64(\maskpointer),%ymm0
    vmovups 96(\maskpointer),%ymm1
    MASKMOV 64(CS),%ymm0,%ymm2
    MASKMOV 96(CS),%ymm1,%ymm3
    VEC_ADD %ymm6,%ymm2,%ymm6
    VEC_ADD %ymm7,%ymm3,%ymm7
    MASKMOV %ymm6,%ymm0,64(CS)
    MASKMOV %ymm7,%ymm1,96(CS)
    addq LDC,CS
.endm

.macro INIT_C_2col
    CLEAR %ymm8,%ymm9,%ymm10,%ymm11
    CLEAR %ymm12,%ymm13,%ymm14,%ymm15
.endm

.macro FIN_C_2col
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    VEC_ADD 64(CS),%ymm6,%ymm6
    VEC_ADD 96(CS),%ymm7,%ymm7
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    vmovups %ymm7,96(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm8,%ymm8
    VEC_ADD 32(CS),%ymm9,%ymm9
    VEC_ADD 64(CS),%ymm10,%ymm10
    VEC_ADD 96(CS),%ymm11,%ymm11
    vmovups %ymm8,(CS)
    vmovups %ymm9,32(CS)
    vmovups %ymm10,64(CS)
    vmovups %ymm11,96(CS)
.endm

.macro FIN_C_2col_irregm maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm8,%ymm2,%ymm8
    VEC_ADD %ymm9,%ymm3,%ymm9
    MASKMOV %ymm8,%ymm0,(CS)
    MASKMOV %ymm9,%ymm1,32(CS)
    vmovups 64(\maskpointer),%ymm0
    vmovups 96(\maskpointer),%ymm1
    MASKMOV 64(CS),%ymm0,%ymm2
    MASKMOV 96(CS),%ymm1,%ymm3
    VEC_ADD %ymm10,%ymm2,%ymm10
    VEC_ADD %ymm11,%ymm3,%ymm11
    MASKMOV %ymm10,%ymm0,64(CS)
    MASKMOV %ymm11,%ymm1,96(CS)
    addq LDC,CS
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm12,%ymm2,%ymm12
    VEC_ADD %ymm13,%ymm3,%ymm13
    MASKMOV %ymm12,%ymm0,(CS)
    MASKMOV %ymm13,%ymm1,32(CS)
    vmovups 64(\maskpointer),%ymm0
    vmovups 96(\maskpointer),%ymm1
    MASKMOV 64(CS),%ymm0,%ymm2
    MASKMOV 96(CS),%ymm1,%ymm3
    VEC_ADD %ymm14,%ymm2,%ymm14
    VEC_ADD %ymm15,%ymm3,%ymm15
    MASKMOV %ymm14,%ymm0,64(CS)
    MASKMOV %ymm15,%ymm1,96(CS)
.endm

.macro SETMASKm//use stack to store mask integer array
#ifdef DOUBLE
    xorq %rax,%rax
    subq %r8,%rax
    addq $15,%rax
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
    decq %rax
    movq %rax,-104(%rsp)
    decq %rax
    movq %rax,-112(%rsp)
    decq %rax
    movq %rax,-120(%rsp)
    decq %rax
    movq %rax,-128(%rsp)
    leaq -128(%rsp),%rax
#else
    xorl %eax,%eax
    subl %r8d,%eax
    addl $31,%eax
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
    decl %eax
    movl %eax,-100(%rsp)
    decl %eax
    movl %eax,-104(%rsp)
    decl %eax
    movl %eax,-108(%rsp)
    decl %eax
    movl %eax,-112(%rsp)
    decl %eax
    movl %eax,-116(%rsp)
    decl %eax
    movl %eax,-120(%rsp)
    decl %eax
    movl %eax,-124(%rsp)
    decl %eax
    movl %eax,-128(%rsp)
    leaq -128(%rsp),%rax
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
    addq $256*BlkDimK,AL //point to (prefetch) next ablk zone of abuffer, start from the tail part
    movslq %ecx,LDC
    SET_LDC
    movq CIP,CS

    INIT_C_2col
    xorq %r12,%r12
    movq A0,%r9
    addq $(BlkDimK-8)*128,%r9 //Arefpos
    movq $(-BlkDimK)*128,%r8 //Areset
.Louter_gemmblkregccc:
    xorq AD,AD
    UPDATECBLK_1col
    PREFm12 CS
    subq $NEXT_A_PREF_STEP,AL
    prefetcht1 (AL)
# if NEXT_A_PREF_STEP > 64
    prefetcht1 64(AL)
# endif
# if NEXT_A_PREF_STEP > 128
    prefetcht1 128(AL)
# endif
# if NEXT_A_PREF_STEP > 192
    prefetcht1 192(AL)
# endif
# if NEXT_A_PREF_STEP > 256
    prefetcht1 256(AL)
# endif
    xorq %r11,%r11
.Linner_gemmblkregccc:
    KERNEL_8 %r9,%r8
    cmpq $BlkDimK/24,%r11
    jb .Linner_gemmblkregccc

    addq AD,A0
    incq %r12
    STORECBLK_1col
    cmpq $BlkDimN-3,%r12
    jb .Louter_gemmblkregccc

    movq AL,%r9
    subq $NEXT_A_PREF_STEP*3,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_gemmblkregccc_last:
    PREFm12 CS
    prefetcht1 (CL)
    prefetcht1 64(CL)
    prefetcht1 127(CL)
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_last:
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht1 (%r9)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 64(%r9)
    KERNEL_1 128,3*SIZE
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    prefetcht1 128(%r9)
    incq %r11
    KERNEL_1 256,6*SIZE
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 (B_PR_ELEM+6)*SIZE(B0)
    prefetcht1 192(%r9)
    addq $256,%r9
    KERNEL_f 384,9*SIZE,512,12*SIZE
    cmpq $BlkDimK/12,%r11
    jb .Linner_gemmblkregccc_last

    incq %r12
    STORECBLK_1col
    UPDATECBLK_1col
    cmpq $BlkDimN,%r12
    jb .Louter_gemmblkregccc_last

    movq CIP,CS
    FIN_C_2col

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
    SETMASKm //generate mask integers. now rax point to the base element of mask integers
    add $8,%rsp //recover rsp so "CIP" can work normally
    movq CIP,CS
    INIT_C_2col
    xorq %r12,%r12
    movq A0,%r9
    addq $(BlkDimK-8)*128,%r9 //Arefpos
    movq $(-BlkDimK)*128,%r8 //Areset
.Louter_tail:
    xorq AD,AD
    UPDATECBLK_1col
    PREFm12 CS
    xorq %r11,%r11
.Linner_tail:
    KERNEL_8 %r9,%r8
    cmpq $BlkDimK/24,%r11
    jb .Linner_tail

    addq AD,A0
    STORECBLK_1col_irregm %rax
    incq %r12
    cmpq $BlkDimN,%r12
    jb .Louter_tail

    movq CIP,CS
    FIN_C_2col_irregm %rax
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
    
