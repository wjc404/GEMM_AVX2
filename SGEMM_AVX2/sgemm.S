#define A0	%rdi //ablk pointer
#define B0	%rsi //bblk pointer
#define CL      %r14 //cload pointer
#define CS      %r15 //cstore pointer
#define LDC     %r13 //ldc * sizeof(float)
#define LDA     %rcx //lda * sizeof(float)
#define AL      %rax //aload pointer
#define ABS     %rbx //ablk_store pointer
#define CIP  -8(%rsp)//cstartpos
//BlkDimM=24,BlkDimK=BlkDimN=256

.macro KERNELm24n4k1 Aoff,Boff
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    vbroadcastss \Boff(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm4
    vfmadd231ps %ymm0,%ymm2,%ymm5
    vfmadd231ps %ymm0,%ymm3,%ymm6
    vbroadcastss \Boff+4(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm7
    vfmadd231ps %ymm0,%ymm2,%ymm8
    vfmadd231ps %ymm0,%ymm3,%ymm9
    vbroadcastss \Boff+8(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm10
    vfmadd231ps %ymm0,%ymm2,%ymm11
    vfmadd231ps %ymm0,%ymm3,%ymm12
    vbroadcastss \Boff+12(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm13
    vfmadd231ps %ymm0,%ymm2,%ymm14
    vfmadd231ps %ymm0,%ymm3,%ymm15
.endm

.macro KERNELm24n4kf Aoff,Boff,delta,deltb
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    addq $\delta,A0
    vbroadcastss \Boff(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm4
    vfmadd231ps %ymm0,%ymm2,%ymm5
    vfmadd231ps %ymm0,%ymm3,%ymm6
    vbroadcastss \Boff+4(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm7
    vfmadd231ps %ymm0,%ymm2,%ymm8
    vfmadd231ps %ymm0,%ymm3,%ymm9
    vbroadcastss \Boff+8(B0),%ymm0
    vfmadd231ps %ymm0,%ymm1,%ymm10
    vfmadd231ps %ymm0,%ymm2,%ymm11
    vfmadd231ps %ymm0,%ymm3,%ymm12
    vbroadcastss \Boff+12(B0),%ymm0
    addq $\deltb,B0
    vfmadd231ps %ymm0,%ymm1,%ymm13
    vfmadd231ps %ymm0,%ymm2,%ymm14
    vfmadd231ps %ymm0,%ymm3,%ymm15
.endm

.macro KERNELm24n4k8
    prefetcht0 256(A0)
    prefetcht0 320(A0)
    KERNELm24n4k1 0,0
    prefetcht0 384(B0)
    prefetcht0 384(A0)
    KERNELm24n4k1 96,16
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    KERNELm24n4k1 192,32
    prefetcht0 576(A0)
    KERNELm24n4k1 288,48
    incq %r11
    prefetcht0 640(A0)
    prefetcht0 704(A0)
    KERNELm24n4k1 384,64
    prefetcht0 448(B0)
    prefetcht0 768(A0)
    KERNELm24n4k1 480,80
    prefetcht0 832(A0)
    prefetcht0 896(A0)
    KERNELm24n4k1 576,96
    prefetcht0 960(A0)
    KERNELm24n4kf 672,112,768,128
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

.macro load_mask maskaddr
    vmovups (\maskaddr),%ymm1
    vmovups 32(\maskaddr),%ymm2
    vmovups 64(\maskaddr),%ymm3
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    vmovups (CL),%ymm13
    vmovups 32(CL),%ymm14
    vmovups 64(CL),%ymm15
    addq LDC,CL
.endm

.macro UPDATECBLK_1col_irregm
    SHIFTYMM
    vmaskmovps (CL),%ymm1,%ymm13
    vmaskmovps 32(CL),%ymm2,%ymm14
    vmaskmovps 64(CL),%ymm3,%ymm15
    addq LDC,CL
.endm

.macro STORECBLK_1col
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1col_irregm StoreMask
    load_mask \StoreMask
    vmaskmovps %ymm4,%ymm1,(CS)
    vmaskmovps %ymm5,%ymm2,32(CS)
    vmaskmovps %ymm6,%ymm3,64(CS)
    addq LDC,CS
.endm

.macro INIT_C_3col
    vmovups (CL),%ymm7
    vmovups 32(CL),%ymm8
    vmovups 64(CL),%ymm9
    addq LDC,CL
    vmovups (CL),%ymm10
    vmovups 32(CL),%ymm11
    vmovups 64(CL),%ymm12
    addq LDC,CL
    vmovups (CL),%ymm13
    vmovups 32(CL),%ymm14
    vmovups 64(CL),%ymm15
    addq LDC,CL
.endm

.macro INIT_C_3col_irregm InitMask
    load_mask \InitMask
    vmaskmovps (CL),%ymm1,%ymm7
    vmaskmovps 32(CL),%ymm2,%ymm8
    vmaskmovps 64(CL),%ymm3,%ymm9
    addq LDC,CL
    vmaskmovps (CL),%ymm1,%ymm10
    vmaskmovps 32(CL),%ymm2,%ymm11
    vmaskmovps 64(CL),%ymm3,%ymm12
    addq LDC,CL
    vmaskmovps (CL),%ymm1,%ymm13
    vmaskmovps 32(CL),%ymm2,%ymm14
    vmaskmovps 64(CL),%ymm3,%ymm15
    addq LDC,CL
.endm

.macro FIN_C_3col
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
    vmovups %ymm7,(CS)
    vmovups %ymm8,32(CS)
    vmovups %ymm9,64(CS)
    addq LDC,CS
    vmovups %ymm10,(CS)
    vmovups %ymm11,32(CS)
    vmovups %ymm12,64(CS)
.endm

.macro FIN_C_3col_irregm
    vmaskmovps %ymm4,%ymm1,(CS)
    vmaskmovps %ymm5,%ymm2,32(CS)
    vmaskmovps %ymm6,%ymm3,64(CS)
    addq LDC,CS
    vmaskmovps %ymm7,%ymm1,(CS)
    vmaskmovps %ymm8,%ymm2,32(CS)
    vmaskmovps %ymm9,%ymm3,64(CS)
    addq LDC,CS
    vmaskmovps %ymm10,%ymm1,(CS)
    vmaskmovps %ymm11,%ymm2,32(CS)
    vmaskmovps %ymm12,%ymm3,64(CS)
.endm

.macro PREFm12 src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 92(\src)
.endm

.macro DECPUSH aaa
    decq \aaa
    push \aaa
.endm

.macro SETMASKm Temp,Mdim //use stack to store mask integer array
    xorq \Temp,\Temp
    subq \Mdim,\Temp
    addq $23,\Temp
    push \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    DECPUSH \Temp
    movq %rsp,\Temp //save base address of mask integer array
    addq $192,%rsp //recover rsp
.endm

.section .text
//enter the function sgemmblkregccc_ac, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc
.globl sgemmblkregccc
.type sgemmblkregccc,@function
sgemmblkregccc:

    push %r15
    push %r14
    push %r13
    push %r12
    movq %rdx,CIP
    movq %rdi,AL
    addq $49152,AL //point to (prefetch) next ablk zone of abuffer; delta=48kB
    movslq %ecx,LDC
    salq $2,LDC //sizeof(float)=2^2
    movq CIP,CL
    movq CIP,CS

    INIT_C_3col
    movq $0xa000000000000000,%r10 //ablk pointer increment table = (0xa000,0x0000,0x0000,0x0000)
    xorq %r12,%r12
.Louter_sgemmblkregccc:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFm12 CS
    PREFm12 CL
    subq $96,AL
    prefetcht1 (AL)
    prefetcht1 64(AL)
    xorq %r11,%r11
.Linner_sgemmblkregccc:
    KERNELm24n4k8
    cmpq $8,%r11
    jb .Linner_sgemmblkregccc

    addq %r9,A0
    prefetcht0 (A0)
    prefetcht0 64(A0)
    prefetcht0 128(A0)
    prefetcht0 192(A0)
    incq %r12
    STORECBLK_1col
    rorq $16,%r10
    cmpq $252,%r12
    jb .Louter_sgemmblkregccc

    movq AL,%r9
    subq $384,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_sgemmblkregccc_last:
    PREFm12 CL
    prefetcht1 152(CL)
    prefetcht1 188(CL)
    xorq %r11,%r11
.Linner_sgemmblkregccc_last:
    prefetcht0 256(A0)
    prefetcht0 320(A0)
    prefetcht1 (%r9)
    KERNELm24n4k1 0,0
    prefetcht0 384(A0)
    prefetcht0 384(B0)
    prefetcht1 64(%r9)
    incq %r11
    KERNELm24n4k1 96,16
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    prefetcht1 128(%r9)
    KERNELm24n4k1 192,32
    prefetcht0 576(A0)
    prefetcht1 192(%r9)
    addq $256,%r9
    KERNELm24n4kf 288,48,384,64
    cmpq $16,%r11
    jb .Linner_sgemmblkregccc_last

    incq %r12
    STORECBLK_1col
    PREFm12 CS
    UPDATECBLK_1col
    cmpq $256,%r12
    jb .Louter_sgemmblkregccc_last
    movq CIP,CS
    FIN_C_3col

    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function sgemmblktailccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim
.globl sgemmblktailccc
.type sgemmblktailccc,@function
sgemmblktailccc:

    push %r15
    push %r14
    push %r13
    push %r12
    push %rdx //cstartpos
    movslq %ecx,LDC
    salq $2,LDC
    movslq %r8d,%r8 //mdim
    SETMASKm %rax,%r8 //generate mask integers. now rax point to the base element of mask integers, just like %2 in DGEMM.c
    add $8,%rsp //recover rsp so "CIP" can work normally
    movq CIP,CL
    movq CIP,CS
    INIT_C_3col_irregm %rax
    xorq %r12,%r12
    movq $0x6000000000000000,%r10
.Louter_tail:
    UPDATECBLK_1col_irregm
    PREFm12 CS
    PREFm12 CL
    xorq %r11,%r11
.Linner_tail:
    KERNELm24n4k8
    cmpq $8,%r11
    jb .Linner_tail

    STORECBLK_1col_irregm %rax
    incq %r12
    movswq %r10w,%r9
    subq %r9,A0
    ror $16,%r10
    cmpq $252,%r12
    jb .Louter_tail
    UPDATECBLK_1col_irregm
    movq CIP,CL
.Louter_tail_last:
    PREFm12 CL
    xorq %r11,%r11
.Linner_tail_last:
    KERNELm24n4k1 0,0
    KERNELm24n4k1 96,16
    incq %r11
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    KERNELm24n4k1 192,32
    prefetcht0 576(A0)
    prefetcht0 416(B0)
    KERNELm24n4kf 288,48,384,64
    cmpq $16,%r11
    jb .Linner_tail_last

    STORECBLK_1col_irregm %rax
    incq %r12
    PREFm12 CS
    UPDATECBLK_1col_irregm
    cmpq $256,%r12
    jb .Louter_tail_last

    movq CIP,CS
    FIN_C_3col_irregm
    vzeroupper
    pop %r12
    pop %r13
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
