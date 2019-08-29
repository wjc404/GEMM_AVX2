#define A0	%rdi //ablk pointer
#define B0	%rsi //bblk pointer

.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    vbroadcastsd \Boff(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm4
    vfmadd231pd %ymm0,%ymm2,%ymm5
    vfmadd231pd %ymm0,%ymm3,%ymm6
    vbroadcastsd \Boff+8(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm7
    vfmadd231pd %ymm0,%ymm2,%ymm8
    vfmadd231pd %ymm0,%ymm3,%ymm9
    vbroadcastsd \Boff+16(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm10
    vfmadd231pd %ymm0,%ymm2,%ymm11
    vfmadd231pd %ymm0,%ymm3,%ymm12
    vbroadcastsd \Boff+24(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm13
    vfmadd231pd %ymm0,%ymm2,%ymm14
    vfmadd231pd %ymm0,%ymm3,%ymm15
.endm

.section .text
//enter the function dgemmrun, rdi=ablk, rsi=bblk; double ablk[96],bblk[32]; ablk and bblk should be aligned to 64B
.globl dgemmrun
.type dgemmrun,@function
dgemmrun:

    xorq %r11,%r11
.L1:
    KERNEL_1 0,0
    KERNEL_1 96,32
    KERNEL_1 192,64
    KERNEL_1 288,96
    KERNEL_1 384,128
    KERNEL_1 480,160
    KERNEL_1 576,192
    incq %r11
    KERNEL_1 672,224
    cmpq $LOOPNUM,%r11
    jb .L1

    vzeroupper
    retq

