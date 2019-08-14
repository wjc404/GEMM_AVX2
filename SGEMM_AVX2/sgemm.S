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
    vxorps %ymm13,%ymm13,%ymm13
    vxorps %ymm14,%ymm14,%ymm14
    vxorps %ymm15,%ymm15,%ymm15
.endm

.macro STORECBLK_1col
    vaddps (CS),%ymm4,%ymm4
    vaddps 32(CS),%ymm5,%ymm5
    vaddps 64(CS),%ymm6,%ymm6
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1col_irregm StoreMask
    load_mask \StoreMask
    vmaskmovps (CS),%ymm1,%ymm0
    vaddps %ymm4,%ymm0,%ymm4
    vmaskmovps 32(CS),%ymm2,%ymm0
    vaddps %ymm5,%ymm0,%ymm5
    vmaskmovps 64(CS),%ymm3,%ymm0
    vaddps %ymm6,%ymm0,%ymm6
    vmaskmovps %ymm4,%ymm1,(CS)
    vmaskmovps %ymm5,%ymm2,32(CS)
    vmaskmovps %ymm6,%ymm3,64(CS)
    addq LDC,CS
.endm

.macro INIT_C_3col
    vxorps %ymm7,%ymm7,%ymm7
    vxorps %ymm8,%ymm8,%ymm8
    vxorps %ymm9,%ymm9,%ymm9
    vxorps %ymm10,%ymm10,%ymm10
    vxorps %ymm11,%ymm11,%ymm11
    vxorps %ymm12,%ymm12,%ymm12
    vxorps %ymm13,%ymm13,%ymm13
    vxorps %ymm14,%ymm14,%ymm14
    vxorps %ymm15,%ymm15,%ymm15
.endm

.macro FIN_C_3col
    vaddps (CS),%ymm4,%ymm4
    vaddps 32(CS),%ymm5,%ymm5
    vaddps 64(CS),%ymm6,%ymm6
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    addq LDC,CS
    vaddps (CS),%ymm7,%ymm7
    vaddps 32(CS),%ymm8,%ymm8
    vaddps 64(CS),%ymm9,%ymm9
    vmovups %ymm7,(CS)
    vmovups %ymm8,32(CS)
    vmovups %ymm9,64(CS)
    addq LDC,CS
    vaddps (CS),%ymm10,%ymm10
    vaddps 32(CS),%ymm11,%ymm11
    vaddps 64(CS),%ymm12,%ymm12
    vmovups %ymm10,(CS)
    vmovups %ymm11,32(CS)
    vmovups %ymm12,64(CS)
.endm

.macro FIN_C_3col_irregm
    vmaskmovps (CS),%ymm1,%ymm0
    vaddps %ymm7,%ymm0,%ymm7
    vmaskmovps 32(CS),%ymm2,%ymm0
    vaddps %ymm8,%ymm0,%ymm8
    vmaskmovps 64(CS),%ymm3,%ymm0
    vaddps %ymm9,%ymm0,%ymm9
    vmaskmovps %ymm7,%ymm1,(CS)
    vmaskmovps %ymm8,%ymm2,32(CS)
    vmaskmovps %ymm9,%ymm3,64(CS)
    addq LDC,CS
    vmaskmovps (CS),%ymm1,%ymm0
    vaddps %ymm10,%ymm0,%ymm10
    vmaskmovps 32(CS),%ymm2,%ymm0
    vaddps %ymm11,%ymm0,%ymm11
    vmaskmovps 64(CS),%ymm3,%ymm0
    vaddps %ymm12,%ymm0,%ymm12
    vmaskmovps %ymm10,%ymm1,(CS)
    vmaskmovps %ymm11,%ymm2,32(CS)
    vmaskmovps %ymm12,%ymm3,64(CS)
    addq LDC,CS
    vmaskmovps (CS),%ymm1,%ymm0
    vaddps %ymm13,%ymm0,%ymm13
    vmaskmovps 32(CS),%ymm2,%ymm0
    vaddps %ymm14,%ymm0,%ymm14
    vmaskmovps 64(CS),%ymm3,%ymm0
    vaddps %ymm15,%ymm0,%ymm15
    vmaskmovps %ymm13,%ymm1,(CS)
    vmaskmovps %ymm14,%ymm2,32(CS)
    vmaskmovps %ymm15,%ymm3,64(CS)
.endm

.macro PREFm12 src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 92(\src)
.endm

.macro SETMASKm//use stack to store mask integer array
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
.endm

.section .text
//enter the function sgemmblkregccc, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc
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
    movq CIP,CS

    INIT_C_3col
    movq $0xa000000000000000,%r10 //ablk pointer increment table = (0xa000,0x0000,0x0000,0x0000)
    xorq %r12,%r12
.Louter_sgemmblkregccc:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFm12 CS
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
    prefetcht1 (CL)
    prefetcht1 64(CL)
    prefetcht1 128(CL)
    prefetcht1 188(CL)
    addq LDC,CL
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
    SETMASKm
    add $8,%rsp //recover rsp so "CIP" can work normally
    movq CIP,CS
    INIT_C_3col
    xorq %r12,%r12
    movq $0x6000000000000000,%r10
.Louter_tail:
    UPDATECBLK_1col
    PREFm12 CS
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
    cmpq $256,%r12
    jb .Louter_tail

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

