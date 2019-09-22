//GEMM_UNROLL_N = 2
.macro KERNEL_1 Aoff,Boff
    VEC_BROAD \Boff(B0),%ymm1
    VEC_BROAD \Boff+SIZE(B0),%ymm2
    vmovaps \Aoff(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm10
    vmovaps \Aoff+32(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm5
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    vmovaps \Aoff+64(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm6
    VEC_FMA231 %ymm0,%ymm2,%ymm12
    vmovaps \Aoff+96(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm13
    vmovaps \Aoff+128(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm8
    VEC_FMA231 %ymm0,%ymm2,%ymm14
    vmovaps \Aoff+160(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm9
    VEC_FMA231 %ymm0,%ymm2,%ymm15
.endm

.macro KERNEL_f Aoff,Boff,delta,deltb
    VEC_BROAD \Boff(B0),%ymm1
    VEC_BROAD \Boff+SIZE(B0),%ymm2
    addq $\deltb,B0
    vmovaps \Aoff(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm10
    vmovaps \Aoff+32(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm5
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    vmovaps \Aoff+64(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm6
    VEC_FMA231 %ymm0,%ymm2,%ymm12
    vmovaps \Aoff+96(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm13
    vmovaps \Aoff+128(A0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm8
    VEC_FMA231 %ymm0,%ymm2,%ymm14
    vmovaps \Aoff+160(A0),%ymm0
    addq $\delta,A0
    VEC_FMA231 %ymm0,%ymm1,%ymm9
    VEC_FMA231 %ymm0,%ymm2,%ymm15
.endm

.macro KERNEL_4 nextablk
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 (\nextablk)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    prefetcht1 64(\nextablk)
    KERNEL_1 192,2*SIZE
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 A_PR_BYTE+512(A0)
    prefetcht1 128(\nextablk)
    incq %r11
    KERNEL_1 384,4*SIZE
    prefetcht0 A_PR_BYTE+576(A0)
    prefetcht0 A_PR_BYTE+640(A0)
    prefetcht0 A_PR_BYTE+704(A0)
    prefetcht1 192(\nextablk)
    addq $256,\nextablk
    KERNEL_f 576,6*SIZE,768,8*SIZE
.endm

.macro KERNEL_8 Arefpos,Areset //Arefpos=ablk_startpos+(GEMM_BLOCK_DIM_K-8)*192;Areset=(-GEMM_BLOCK_DIM_K)*192
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    cmpq \Arefpos,A0
    cmoveq \Areset,AD
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht0 A_PR_BYTE+320(A0)
    KERNEL_1 192,2*SIZE
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    prefetcht0 A_PR_BYTE+512(A0)
    KERNEL_1 384,4*SIZE
    prefetcht0 A_PR_BYTE+576(A0)
    prefetcht0 A_PR_BYTE+640(A0)
    prefetcht0 A_PR_BYTE+704(A0)
    incq %r11
    KERNEL_1 576,6*SIZE
    prefetcht0 A_PR_BYTE+768(A0)
    prefetcht0 A_PR_BYTE+832(A0)
    prefetcht0 A_PR_BYTE+896(A0)
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
    KERNEL_1 768,8*SIZE
    prefetcht0 A_PR_BYTE+960(A0)
    prefetcht0 A_PR_BYTE+1024(A0)
#if A_PR_BYTE > 447
    prefetcht0 A_PR_BYTE+1088(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1088(A0)
#endif
    KERNEL_1 960,10*SIZE
#if A_PR_BYTE > 383
    prefetcht0 A_PR_BYTE+1152(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1152(A0)
#endif
#if A_PR_BYTE > 319
    prefetcht0 A_PR_BYTE+1216(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1216(A0)
#endif
#if A_PR_BYTE > 255
    prefetcht0 A_PR_BYTE+1280(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1280(A0)
#endif
    KERNEL_1 1152,12*SIZE
#if A_PR_BYTE > 191
    prefetcht0 A_PR_BYTE+1344(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1344(A0)
#endif
#if A_PR_BYTE > 127
    prefetcht0 A_PR_BYTE+1408(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1408(A0)
#endif
#if A_PR_BYTE > 63
    prefetcht0 A_PR_BYTE+1472(A0,AD,1)
#else
    prefetcht0 A_PR_BYTE+1472(A0)
#endif
    KERNEL_f 1344,14*SIZE,1536,16*SIZE
.endm

.macro SHIFTYMM
    vmovaps %ymm10,%ymm4
    vmovaps %ymm11,%ymm5
    vmovaps %ymm12,%ymm6
    vmovaps %ymm13,%ymm7
    vmovaps %ymm14,%ymm8
    vmovaps %ymm15,%ymm9
.endm

.macro CLEAR r1,r2,r3,r4,r5,r6
    vpxor \r1,\r1,\r1
    vpxor \r2,\r2,\r2
    vpxor \r3,\r3,\r3
    vpxor \r4,\r4,\r4
    vpxor \r5,\r5,\r5
    vpxor \r6,\r6,\r6
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR %ymm10,%ymm11,%ymm12,%ymm13,%ymm14,%ymm15
.endm

.macro STORECBLK_1col
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    VEC_ADD 64(CS),%ymm6,%ymm6
    VEC_ADD 96(CS),%ymm7,%ymm7
    VEC_ADD 128(CS),%ymm8,%ymm8
    VEC_ADD 160(CS),%ymm9,%ymm9
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    vmovups %ymm7,96(CS)
    vmovups %ymm8,128(CS)
    vmovups %ymm9,160(CS)
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
    vmovups 64(\maskpointer),%ymm0
    vmovups 96(\maskpointer),%ymm1
    MASKMOV 64(CS),%ymm0,%ymm2
    MASKMOV 96(CS),%ymm1,%ymm3
    VEC_ADD %ymm6,%ymm2,%ymm6
    VEC_ADD %ymm7,%ymm3,%ymm7
    MASKMOV %ymm6,%ymm0,64(CS)
    MASKMOV %ymm7,%ymm1,96(CS)
    vmovups 128(\maskpointer),%ymm0
    vmovups 160(\maskpointer),%ymm1
    MASKMOV 128(CS),%ymm0,%ymm2
    MASKMOV 160(CS),%ymm1,%ymm3
    VEC_ADD %ymm8,%ymm2,%ymm8
    VEC_ADD %ymm9,%ymm3,%ymm9
    MASKMOV %ymm8,%ymm0,128(CS)
    MASKMOV %ymm9,%ymm1,160(CS)
    addq LDC,CS
.endm

.macro INIT_C
    CLEAR %ymm10,%ymm11,%ymm12,%ymm13,%ymm14,%ymm15
.endm

.macro FIN_C
    VEC_ADD (CS),%ymm4,%ymm4
    VEC_ADD 32(CS),%ymm5,%ymm5
    VEC_ADD 64(CS),%ymm6,%ymm6
    VEC_ADD 96(CS),%ymm7,%ymm7
    VEC_ADD 128(CS),%ymm8,%ymm8
    VEC_ADD 160(CS),%ymm9,%ymm9
    vmovups %ymm4,(CS)
    vmovups %ymm5,32(CS)
    vmovups %ymm6,64(CS)
    vmovups %ymm7,96(CS)
    vmovups %ymm8,128(CS)
    vmovups %ymm9,160(CS)
.endm

.macro FIN_C_edgem maskpointer
    vmovups (\maskpointer),%ymm0
    vmovups 32(\maskpointer),%ymm1
    MASKMOV (CS),%ymm0,%ymm2
    MASKMOV 32(CS),%ymm1,%ymm3
    VEC_ADD %ymm10,%ymm2,%ymm10
    VEC_ADD %ymm11,%ymm3,%ymm11
    MASKMOV %ymm10,%ymm0,(CS)
    MASKMOV %ymm11,%ymm1,32(CS)
    vmovups 64(\maskpointer),%ymm0
    vmovups 96(\maskpointer),%ymm1
    MASKMOV 64(CS),%ymm0,%ymm2
    MASKMOV 96(CS),%ymm1,%ymm3
    VEC_ADD %ymm12,%ymm2,%ymm12
    VEC_ADD %ymm13,%ymm3,%ymm13
    MASKMOV %ymm12,%ymm0,64(CS)
    MASKMOV %ymm13,%ymm1,96(CS)
    vmovups 128(\maskpointer),%ymm0
    vmovups 160(\maskpointer),%ymm1
    MASKMOV 128(CS),%ymm0,%ymm2
    MASKMOV 160(CS),%ymm1,%ymm3
    VEC_ADD %ymm14,%ymm2,%ymm14
    VEC_ADD %ymm15,%ymm3,%ymm15
    MASKMOV %ymm14,%ymm0,128(CS)
    MASKMOV %ymm15,%ymm1,160(CS)
.endm

