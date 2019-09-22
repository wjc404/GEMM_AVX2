//GEMM_UNROLL_N = 4
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

.macro KERNEL_4 nextablk
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht1 (\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 64(\nextablk)
    incq %r11
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht1 128(\nextablk)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PR_BYTE+320(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
#endif
    prefetcht1 192(\nextablk)
    addq $256,\nextablk
    KERNEL_f 288,12*SIZE,384,16*SIZE
.endm

.macro KERNEL_8 Arefpos,Areset //Arefpos=ablk_startpos+(GEMM_BLOCK_DIM_K-8)*96;Areset=(-GEMM_BLOCK_DIM_K)*96
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht0 A_PR_BYTE+128(A0)
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PR_BYTE+320(A0)
# ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
# endif
    KERNEL_1 288,12*SIZE
    incq %r11
# if A_PR_BYTE > 383
    prefetcht0 A_PR_BYTE+384(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+384(A0)
# endif
# if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+448(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+448(A0)
# endif
    KERNEL_1 384,16*SIZE
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
# if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+512(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+512(A0)
# endif
    KERNEL_1 480,20*SIZE
# if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+576(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+576(A0)
# endif
# if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+640(A0,AD,1)
# else
    prefetcht0 A_PR_BYTE+640(A0)
# endif
    KERNEL_1 576,24*SIZE
    prefetcht0 A_PR_BYTE+704(A0,AD,1)
# ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+24)*SIZE(B0)
# endif
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

.macro STORECBLK_1col_edgem maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    vmovups 64(\maskpointer),%ymm2
    MASKMOV (CS),%ymm0,%ymm3
    VEC_ADD %ymm4,%ymm3,%ymm4
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm5,%ymm3,%ymm5
    MASKMOV 64(CS),%ymm2,%ymm3
    VEC_ADD %ymm6,%ymm3,%ymm6
    MASKMOV %ymm4,%ymm0,(CS)
    MASKMOV %ymm5,%ymm1,32(CS)
    MASKMOV %ymm6,%ymm2,64(CS)
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR %ymm7,%ymm8,%ymm9
    CLEAR %ymm10,%ymm11,%ymm12
    CLEAR %ymm13,%ymm14,%ymm15
.endm

.macro FIN_C
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

.macro FIN_C_edgem maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    vmovups 64(\maskpointer),%ymm2
    MASKMOV (CS),%ymm0,%ymm3
    VEC_ADD %ymm7,%ymm3,%ymm7
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm8,%ymm3,%ymm8
    MASKMOV 64(CS),%ymm2,%ymm3
    VEC_ADD %ymm9,%ymm3,%ymm9
    MASKMOV %ymm7,%ymm0,(CS)
    MASKMOV %ymm8,%ymm1,32(CS)
    MASKMOV %ymm9,%ymm2,64(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm0,%ymm3
    VEC_ADD %ymm10,%ymm3,%ymm10
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm11,%ymm3,%ymm11
    MASKMOV 64(CS),%ymm2,%ymm3
    VEC_ADD %ymm12,%ymm3,%ymm12
    MASKMOV %ymm10,%ymm0,(CS)
    MASKMOV %ymm11,%ymm1,32(CS)
    MASKMOV %ymm12,%ymm2,64(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm0,%ymm3
    VEC_ADD %ymm13,%ymm3,%ymm13
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm14,%ymm3,%ymm14
    MASKMOV 64(CS),%ymm2,%ymm3
    VEC_ADD %ymm15,%ymm3,%ymm15
    MASKMOV %ymm13,%ymm0,(CS)
    MASKMOV %ymm14,%ymm1,32(CS)
    MASKMOV %ymm15,%ymm2,64(CS)
.endm

