//GEMM_UNROLL_N = 6
.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm5
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm6
    VEC_FMA231 %ymm0,%ymm2,%ymm7
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm8
    VEC_FMA231 %ymm0,%ymm2,%ymm9
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm10
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_BROAD \Boff+4*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm12
    VEC_FMA231 %ymm0,%ymm2,%ymm13
    VEC_BROAD \Boff+5*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm14
    VEC_FMA231 %ymm0,%ymm2,%ymm15
.endm

.macro KERNEL_f Aoff,Boff,delta,deltb
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    addq $\delta,A0
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm5
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm6
    VEC_FMA231 %ymm0,%ymm2,%ymm7
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm8
    VEC_FMA231 %ymm0,%ymm2,%ymm9
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm10
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_BROAD \Boff+4*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm12
    VEC_FMA231 %ymm0,%ymm2,%ymm13
    VEC_BROAD \Boff+5*SIZE(B0),%ymm0
    addq $\deltb,B0
    VEC_FMA231 %ymm0,%ymm1,%ymm14
    VEC_FMA231 %ymm0,%ymm2,%ymm15
.endm

.macro KERNEL_4 nextablk
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 (\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    prefetcht1 64(\nextablk)
    KERNEL_1 64,6*SIZE
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht1 128(\nextablk)
    incq %r11
    KERNEL_1 128,12*SIZE
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    prefetcht1 192(\nextablk)
    addq $256,\nextablk
    KERNEL_f 192,18*SIZE,256,24*SIZE
.endm

.macro KERNEL_8 Arefpos,Areset //Arefpos=ablk_startpos+(GEMM_BLOCK_DIM_K-8)*64;Areset=(-GEMM_BLOCK_DIM_K)*64
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    KERNEL_1 64,6*SIZE
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    KERNEL_1 128,12*SIZE
#if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+192(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+192(A0)
#endif
    incq %r11
    KERNEL_1 192,18*SIZE
#if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+256(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+256(A0)
#endif
    prefetcht0 (B_PR_ELEM+24)*SIZE(B0)
    KERNEL_1 256,24*SIZE
#if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+320(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+320(A0)
#endif
    prefetcht0 (B_PR_ELEM+32)*SIZE(B0)
    KERNEL_1 320,30*SIZE
#if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+384(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+384(A0)
#endif
    prefetcht0 (B_PR_ELEM+40)*SIZE(B0)
    KERNEL_1 384,36*SIZE
    prefetcht0 A_PR_BYTE+448(A0,AD,1)
    KERNEL_f 448,42*SIZE,512,48*SIZE
.endm

.macro SHIFTYMM
    vmovaps %ymm6,%ymm4
    vmovaps %ymm7,%ymm5
    vmovaps %ymm8,%ymm6
    vmovaps %ymm9,%ymm7
    vmovaps %ymm10,%ymm8
    vmovaps %ymm11,%ymm9
    vmovaps %ymm12,%ymm10
    vmovaps %ymm13,%ymm11
    vmovaps %ymm14,%ymm12
    vmovaps %ymm15,%ymm13
.endm

.macro CLEAR r1,r2
    vpxor \r1,\r1,\r1
    vpxor \r2,\r2,\r2
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR %ymm14,%ymm15
.endm

.macro STORECBLK_1col
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1col_edgem maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm4,%ymm2,%ymm4
    VEC_ADD %ymm5,%ymm3,%ymm5
    MASKMOV %ymm4,%ymm0,(CS)
    MASKMOV %ymm5,%ymm1,32(CS)
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR %ymm6,%ymm7
    CLEAR %ymm8,%ymm9
    CLEAR %ymm10,%ymm11
    CLEAR %ymm12,%ymm13
    CLEAR %ymm14,%ymm15
.endm

.macro FIN_C
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm6,%ymm6
    VEC_ADD 32(CS),%ymm7,%ymm7
    vmovups %ymm6,(CS)
    vmovups %ymm7,32(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm8,%ymm8
    VEC_ADD 32(CS),%ymm9,%ymm9
    vmovups %ymm8,(CS)
    vmovups %ymm9,32(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm10,%ymm10
    VEC_ADD 32(CS),%ymm11,%ymm11
    vmovups %ymm10,(CS)
    vmovups %ymm11,32(CS)
    addq LDC,CS
    VEC_ADD (CS),%ymm12,%ymm12
    VEC_ADD 32(CS),%ymm13,%ymm13
    vmovups %ymm12,(CS)
    vmovups %ymm13,32(CS)
.endm

.macro FIN_C_edgem maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm6,%ymm2,%ymm6
    VEC_ADD %ymm7,%ymm3,%ymm7
    MASKMOV %ymm6,%ymm0,(CS)
    MASKMOV %ymm7,%ymm1,32(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm8,%ymm2,%ymm8
    VEC_ADD %ymm9,%ymm3,%ymm9
    MASKMOV %ymm8,%ymm0,(CS)
    MASKMOV %ymm9,%ymm1,32(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm10,%ymm2,%ymm10
    VEC_ADD %ymm11,%ymm3,%ymm11
    MASKMOV %ymm10,%ymm0,(CS)
    MASKMOV %ymm11,%ymm1,32(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm12,%ymm2,%ymm12
    VEC_ADD %ymm13,%ymm3,%ymm13
    MASKMOV %ymm12,%ymm0,(CS)
    MASKMOV %ymm13,%ymm1,32(CS)
    addq LDC,CS
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm14,%ymm2,%ymm14
    VEC_ADD %ymm15,%ymm3,%ymm15
    MASKMOV %ymm14,%ymm0,(CS)
    MASKMOV %ymm15,%ymm1,32(CS)
.endm

