//definitions of external functions in gemm_driver.c
# include "../gemm_set_parameters.h"
# define A0      %rdi //ablk pointer
# define B0      %rsi //bblk pointer
# define CL      %r14 //cload pointer
# define CS      %r15 //cstore pointer
# define LDC     %rcx //ldc * sizeof(float)
# define AL      %rax //aload pointer
# define CIP  -8(%rsp)//cstartpos
# define AD      %r10 //A offset

# ifdef DOUBLE
 # define VEC_BROAD vbroadcastsd
 # define VEC_FMA231 vfmadd231pd
 # define VEC_ADD vaddpd
 # define MASKMOV vmaskmovpd
 # define SIZE 8
# else
 # define VEC_BROAD vbroadcastss
 # define VEC_FMA231 vfmadd231ps
 # define VEC_ADD vaddps
 # define MASKMOV vmaskmovps
 # define SIZE 4
# endif

# define BYTES_PER_COL (384/GEMM_UNROLL_N)
# define NEXT_A_PREF_STEP (BYTES_PER_COL*GEMM_BLOCK_DIM_K/GEMM_BLOCK_DIM_N) //in bytes

.macro PREF_1_col insn,src
    \insn (\src)
# if BYTES_PER_COL > 65
    \insn 64(\src)
# endif
# if BYTES_PER_COL > 129
    \insn 128(\src)
# endif
    \insn BYTES_PER_COL-1(\src)
.endm

.macro SET_LDC
# ifdef DOUBLE
    salq $3,LDC
# else
    salq $2,LDC
# endif
.endm

.macro SETMASKm_1vec offset
# ifdef DOUBLE
    decq %rax
    movq %rax,\offset-8(%rsp)
    decq %rax
    movq %rax,\offset-16(%rsp)
    decq %rax
    movq %rax,\offset-24(%rsp)
    decq %rax
    movq %rax,\offset-32(%rsp)
# else
    decl %eax
    movl %eax,\offset-4(%rsp)
    decl %eax
    movl %eax,\offset-8(%rsp)
    decl %eax
    movl %eax,\offset-12(%rsp)
    decl %eax
    movl %eax,\offset-16(%rsp)
    decl %eax
    movl %eax,\offset-20(%rsp)
    decl %eax
    movl %eax,\offset-24(%rsp)
    decl %eax
    movl %eax,\offset-28(%rsp)
    decl %eax
    movl %eax,\offset-32(%rsp)
# endif
.endm

.macro SETMASKm//use stack to store mask integer array
# ifdef DOUBLE
    xorq %rax,%rax
    subq %r8,%rax
    addq $GEMM_BLOCK_DIM_M,%rax
# else
    xorl %eax,%eax
    subl %r8d,%eax
    addl $GEMM_BLOCK_DIM_M,%eax
# endif
    SETMASKm_1vec 0
    SETMASKm_1vec -32
# if GEMM_UNROLL_N < 6
    SETMASKm_1vec -64
# endif
# if GEMM_UNROLL_N < 4
    SETMASKm_1vec -96
# endif
# if GEMM_UNROLL_N < 3
    SETMASKm_1vec -128
    SETMASKm_1vec -160
# endif
    leaq -BYTES_PER_COL(%rsp),%rax
.endm

# if GEMM_UNROLL_N == 2
 # include "gemm_kernel_unroll2.S"
# endif
# if GEMM_UNROLL_N == 3
 # include "gemm_kernel_unroll3.S"
# endif
# if GEMM_UNROLL_N == 4
 # include "gemm_kernel_unroll4.S"
# endif
# if GEMM_UNROLL_N == 6
 # include "gemm_kernel_unroll6.S"
# endif

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
    addq $BYTES_PER_COL*2*GEMM_BLOCK_DIM_K,AL //point to (prefetch) next ablk zone of abuffer, start from the tail part
    movslq %ecx,LDC
    SET_LDC
    movq CIP,CS

    INIT_C
    xorq %r12,%r12
    movq A0,%r9
    addq $(GEMM_BLOCK_DIM_K-8)*BYTES_PER_COL,%r9 //Arefpos
    movq $(-GEMM_BLOCK_DIM_K)*BYTES_PER_COL,%r8 //Areset
.Louter_gemmblkregccc:
    xorq AD,AD
    UPDATECBLK_1col
    PREF_1_col prefetcht0,CS
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
    cmpq $GEMM_LOOP_TIMES_K,%r11
    jb .Linner_gemmblkregccc

    addq AD,A0
    incq %r12
    STORECBLK_1col
    cmpq $GEMM_BLOCK_DIM_N-GEMM_UNROLL_N,%r12
    jb .Louter_gemmblkregccc

    movq A0,%r9
    addq $BYTES_PER_COL*GEMM_BLOCK_DIM_K,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_gemmblkregccc_last:
    PREF_1_col prefetcht0,CS
    PREF_1_col prefetcht1,CL
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_last:
    KERNEL_4 %r9
    cmpq $GEMM_LOOP_TIMES_K*2,%r11
    jb .Linner_gemmblkregccc_last

    incq %r12
    STORECBLK_1col
    UPDATECBLK_1col
    cmpq $GEMM_BLOCK_DIM_N,%r12
    jb .Louter_gemmblkregccc_last

    movq CIP,CS
    FIN_C

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
    INIT_C
    xorq %r12,%r12
    movq A0,%r9
    addq $(GEMM_BLOCK_DIM_K-8)*BYTES_PER_COL,%r9 //Arefpos
    movq $(-GEMM_BLOCK_DIM_K)*BYTES_PER_COL,%r8 //Areset
.Louter_tail:
    xorq AD,AD
    UPDATECBLK_1col
    PREF_1_col prefetcht0,CS
    xorq %r11,%r11
.Linner_tail:
    KERNEL_8 %r9,%r8
    cmpq $GEMM_LOOP_TIMES_K,%r11
    jb .Linner_tail

    addq AD,A0
    STORECBLK_1col_edgem %rax
    incq %r12
    cmpq $GEMM_BLOCK_DIM_N,%r12
    jb .Louter_tail

    movq CIP,CS
    FIN_C_edgem %rax
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
    
