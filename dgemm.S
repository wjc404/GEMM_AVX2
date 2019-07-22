#define A0	%rdi //ablk pointer
#define B0	%rsi //bblk pointer
#define CL      %r14 //cload pointer
#define CS      %r15 //cstore pointer
#define LDC     %r13 //ldc * sizeof(double)
#define LDA     %rcx //lda * sizeof(double)
#define AL      %rax //aload pointer
#define ABS     %rbx //ablk_store pointer
#define CIP  -8(%rsp)//cstartpos

//ablk: packed matrix block of a, max size: m=12, k=256, occu = 24kB (L1D - L2 Cache)
//bblk: packed matrix block of b, max size: k=256, n=256, occu = 512kB (L2 - L3 Cache)
//cblk: packed matrix block of c, stored in ymm4-ymm15 with m=12, n=4 (Registers <-> Main Memory)
//loads of ablk and bblk: processed by load functions, between block-matrix multiplication runs
//loads/stores of cblk: processed by block-matrix multiplication functions, on the fly
//the main part of calculation update cblk column by column, the edge part of calculation update it block by block
//the elements in bblk have been arranged in a special order (see below) so that accesses to bblk are continuous in the calculation part
/*
element_no._in_bblk  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 ...
k_coord._in_matrix_b 1  1  1  1  2  2  2  2  3  3  3  3  4  4  4  4  5  5  5  5  ...
n_coord._in_matrix_b 1  2  3  4  1  2  3  4  1  2  3  4  1  2  3  4  1  2  3  4  ...
*/

//macros for core calculation
.macro KERNELm12n1k1 //for irregn part
    vmovapd (A0),%ymm1
    vmovapd 32(A0),%ymm2
    vmovapd 64(A0),%ymm3
    addq $96,A0
    vbroadcastsd (B0),%ymm0
    addq $8,B0
    vfmadd231pd %ymm0,%ymm1,%ymm4
    vfmadd231pd %ymm0,%ymm2,%ymm5
    vfmadd231pd %ymm0,%ymm3,%ymm6
.endm

.macro KERNELm12n4k1 Aoff,Boff //for reg part
    vmovapd \Aoff(A0),%ymm1
    vmovapd \Aoff+32(A0),%ymm2
    vmovapd \Aoff+64(A0),%ymm3
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

.macro KERNELm12n4kf Aofff,Bofff,delta,deltb
    vmovapd \Aofff(A0),%ymm1
    vmovapd \Aofff+32(A0),%ymm2
    vmovapd \Aofff+64(A0),%ymm3
    addq $\delta,A0
    vbroadcastsd \Bofff(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm4
    vfmadd231pd %ymm0,%ymm2,%ymm5
    vfmadd231pd %ymm0,%ymm3,%ymm6
    vbroadcastsd \Bofff+8(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm7
    vfmadd231pd %ymm0,%ymm2,%ymm8
    vfmadd231pd %ymm0,%ymm3,%ymm9
    vbroadcastsd \Bofff+16(B0),%ymm0
    vfmadd231pd %ymm0,%ymm1,%ymm10
    vfmadd231pd %ymm0,%ymm2,%ymm11
    vfmadd231pd %ymm0,%ymm3,%ymm12
    vbroadcastsd \Bofff+24(B0),%ymm0
    addq $\deltb,B0
    vfmadd231pd %ymm0,%ymm1,%ymm13
    vfmadd231pd %ymm0,%ymm2,%ymm14
    vfmadd231pd %ymm0,%ymm3,%ymm15
.endm

.macro KERNELm12n4k2 Aof,Bof
    prefetcht0 \Aof+256(A0)
    prefetcht0 \Aof+320(A0)
    KERNELm12n4k1 \Aof,\Bof
    prefetcht0 \Aof+384(A0)
    prefetcht0 \Bof+768(B0)
    KERNELm12n4k1 96+\Aof,32+\Bof
.endm

.macro KERNELm12n4k8
    KERNELm12n4k2 0,0
    KERNELm12n4k2 192,64
    incq %r11
    KERNELm12n4k2 384,128
    prefetcht0 832(A0)
    prefetcht0 896(A0)
    KERNELm12n4k1 576,192
    prefetcht0 960(A0)
    prefetcht0 960(B0)
    KERNELm12n4kf 672,224,768,256
.endm

//macros dealing with load/store of C elements
.macro SHIFTYMM
    vmovapd %ymm7,%ymm4
    vmovapd %ymm8,%ymm5
    vmovapd %ymm9,%ymm6
    vmovapd %ymm10,%ymm7
    vmovapd %ymm11,%ymm8
    vmovapd %ymm12,%ymm9
    vmovapd %ymm13,%ymm10
    vmovapd %ymm14,%ymm11
    vmovapd %ymm15,%ymm12
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    vmovupd (CL),%ymm13
    vmovupd 32(CL),%ymm14
    vmovupd 64(CL),%ymm15
    addq LDC,CL
.endm

.macro STORECBLK_1col
    vmovupd %ymm4,(CS)
    vmovupd %ymm5,32(CS)
    vmovupd %ymm6,64(CS)
    addq LDC,CS
.endm

.macro INIT_C_3col
    vmovupd (CL),%ymm7
    vmovupd 32(CL),%ymm8
    vmovupd 64(CL),%ymm9
    addq LDC,CL
    vmovupd (CL),%ymm10
    vmovupd 32(CL),%ymm11
    vmovupd 64(CL),%ymm12
    addq LDC,CL
    vmovupd (CL),%ymm13
    vmovupd 32(CL),%ymm14
    vmovupd 64(CL),%ymm15
    addq LDC,CL
.endm

.macro FIN_C_3col
    vmovupd %ymm4,(CS)
    vmovupd %ymm5,32(CS)
    vmovupd %ymm6,64(CS)
    addq LDC,CS
    vmovupd %ymm7,(CS)
    vmovupd %ymm8,32(CS)
    vmovupd %ymm9,64(CS)
    addq LDC,CS
    vmovupd %ymm10,(CS)
    vmovupd %ymm11,32(CS)
    vmovupd %ymm12,64(CS)
.endm

.macro LOAD_C_4col_quarter iny1,iny2,iny3,clspointer,ldcm8 //iny1-iny3 are ymm registers, together receiving a column of c
    vmovupd (\clspointer),\iny1
    vmovupd 32(\clspointer),\iny2
    vmovupd 64(\clspointer),\iny3
    prefetcht0 (\clspointer,\ldcm8,4)
    prefetcht0 64(\clspointer,\ldcm8,4)
    prefetcht0 88(\clspointer,\ldcm8,4)
    addq \ldcm8,\clspointer
.endm

.macro STORE_C_4col_quarter ouy1,ouy2,ouy3,CLSPOINTER,LDCM8 //ouy1-ouy3 are ymm registers, together sending a column of c
    vmovupd \ouy1,(\CLSPOINTER)
    vmovupd \ouy2,32(\CLSPOINTER)
    vmovupd \ouy3,64(\CLSPOINTER)
    addq \LDCM8,\CLSPOINTER
.endm

.macro LOAD_C_m12n4 CPOINTER1,M8LDC
    LOAD_C_4col_quarter %ymm4,%ymm5,%ymm6,\CPOINTER1,\M8LDC
    LOAD_C_4col_quarter %ymm7,%ymm8,%ymm9,\CPOINTER1,\M8LDC
    LOAD_C_4col_quarter %ymm10,%ymm11,%ymm12,\CPOINTER1,\M8LDC
    LOAD_C_4col_quarter %ymm13,%ymm14,%ymm15,\CPOINTER1,\M8LDC
    shlq $2,\M8LDC
    subq \M8LDC,\CPOINTER1 //recover c pointer for successive store
    shrq $2,\M8LDC
.endm

.macro STORE_C_m12n4 CPOINTER2,m8ldc
    STORE_C_4col_quarter %ymm4,%ymm5,%ymm6,\CPOINTER2,\m8ldc
    STORE_C_4col_quarter %ymm7,%ymm8,%ymm9,\CPOINTER2,\m8ldc
    STORE_C_4col_quarter %ymm10,%ymm11,%ymm12,\CPOINTER2,\m8ldc
    STORE_C_4col_quarter %ymm13,%ymm14,%ymm15,\CPOINTER2,\m8ldc
.endm

.macro MULT_C_m12n1 REG1,REG2,REG3 //C MULT BETA
    vmulpd %ymm0,\REG1,\REG1
    vmulpd %ymm0,\REG2,\REG2
    vmulpd %ymm0,\REG3,\REG3
.endm

.macro MULT_C_m12n4 //C MULT BETA
    MULT_C_m12n1 %ymm4,%ymm5,%ymm6
    MULT_C_m12n1 %ymm7,%ymm8,%ymm9
    MULT_C_m12n1 %ymm10,%ymm11,%ymm12
    MULT_C_m12n1 %ymm13,%ymm14,%ymm15
.endm

.macro LOAD_C_4col_Quarter_mmask inY1,inY2,inY3,Clspointer,ldcM8 //inY1-inY3 are ymm registers, together receiving a column of c, ymm1-ymm3 as load masks
    vmaskmovpd (\Clspointer),%ymm1,\inY1
    vmaskmovpd 32(\Clspointer),%ymm2,\inY2
    vmaskmovpd 64(\Clspointer),%ymm3,\inY3
    prefetcht0 (\Clspointer,\ldcM8,4)
    prefetcht0 64(\Clspointer,\ldcM8,4)
    prefetcht0 88(\Clspointer,\ldcM8,4)
    addq \ldcM8,\Clspointer
.endm

.macro STORE_C_4col_Quarter_mmask ouY1,ouY2,ouY3,cLSPOINTER,LDCm8 //ouY1-ouY3 are ymm registers, together sending a column of c, ymm1-ymm3 as store masks
    vmaskmovpd \ouY1,%ymm1,(\cLSPOINTER)
    vmaskmovpd \ouY2,%ymm2,32(\cLSPOINTER)
    vmaskmovpd \ouY3,%ymm3,64(\cLSPOINTER)
    addq \LDCm8,\cLSPOINTER
.endm

.macro LOAD_C_M12N4_mmask CPOINTER3,M8LdC
    LOAD_C_4col_Quarter_mmask %ymm4,%ymm5,%ymm6,\CPOINTER3,\M8LdC
    LOAD_C_4col_Quarter_mmask %ymm7,%ymm8,%ymm9,\CPOINTER3,\M8LdC
    LOAD_C_4col_Quarter_mmask %ymm10,%ymm11,%ymm12,\CPOINTER3,\M8LdC
    LOAD_C_4col_Quarter_mmask %ymm13,%ymm14,%ymm15,\CPOINTER3,\M8LdC
    shlq $2,\M8LdC
    subq \M8LdC,\CPOINTER3 //recover c pointer for successive store
    shrq $2,\M8LdC
.endm

.macro STORE_C_M12N4_mmask CPOINTER4,m8lDc
    STORE_C_4col_Quarter_mmask %ymm4,%ymm5,%ymm6,\CPOINTER4,\m8lDc
    STORE_C_4col_Quarter_mmask %ymm7,%ymm8,%ymm9,\CPOINTER4,\m8lDc
    STORE_C_4col_Quarter_mmask %ymm10,%ymm11,%ymm12,\CPOINTER4,\m8lDc
    STORE_C_4col_Quarter_mmask %ymm13,%ymm14,%ymm15,\CPOINTER4,\m8lDc
.endm

.macro load_mask maskaddr
    vmovdqu (\maskaddr),%ymm1
    vmovdqu 32(\maskaddr),%ymm2
    vmovdqu 64(\maskaddr),%ymm3
.endm

.macro UPDATECBLK_1col_irregm
    SHIFTYMM
    vmaskmovpd (CL),%ymm1,%ymm13
    vmaskmovpd 32(CL),%ymm2,%ymm14
    vmaskmovpd 64(CL),%ymm3,%ymm15
    addq LDC,CL
.endm

.macro STORECBLK_1col_irregm StoreMask
    load_mask \StoreMask
    STORE_C_4col_Quarter_mmask %ymm4,%ymm5,%ymm6,CS,LDC
.endm

.macro INIT_C_3col_irregm InitMask
    load_mask \InitMask
    vmaskmovpd (CL),%ymm1,%ymm7
    vmaskmovpd 32(CL),%ymm2,%ymm8
    vmaskmovpd 64(CL),%ymm3,%ymm9
    addq LDC,CL
    vmaskmovpd (CL),%ymm1,%ymm10
    vmaskmovpd 32(CL),%ymm2,%ymm11
    vmaskmovpd 64(CL),%ymm3,%ymm12
    addq LDC,CL
    vmaskmovpd (CL),%ymm1,%ymm13
    vmaskmovpd 32(CL),%ymm2,%ymm14
    vmaskmovpd 64(CL),%ymm3,%ymm15
    addq LDC,CL
.endm

.macro FIN_C_3col_irregm
    STORE_C_4col_Quarter_mmask %ymm4,%ymm5,%ymm6,CS,LDC
    STORE_C_4col_Quarter_mmask %ymm7,%ymm8,%ymm9,CS,LDC
    vmaskmovpd %ymm10,%ymm1,(CS)
    vmaskmovpd %ymm11,%ymm2,32(CS)
    vmaskmovpd %ymm12,%ymm3,64(CS)
.endm

//miscellanous
.macro PREFm12 src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 88(\src)
.endm

.macro PREFam3k4 aroffset
    prefetcht1 \aroffset(AL)
    prefetcht1 \aroffset(AL,LDA,1)
    prefetcht1 \aroffset(AL,LDA,2)
.endm

.macro load4line_1rreg_a line1addr,line3addr,linediff,deltaddr,msklin1,msklin2,msklin3,msklin4
    vmaskmovpd (\line1addr),\msklin1,%ymm0
    vmaskmovpd (\line1addr,\linediff,1),\msklin2,%ymm1
    vmaskmovpd (\line3addr),\msklin3,%ymm2
    vmaskmovpd (\line3addr,\linediff,1),\msklin4,%ymm3
    addq \deltaddr,\line1addr
    addq \deltaddr,\line3addr
.endm

.macro LOAD_A_1col apointer,ablkpointer,delta
    vmovupd (\apointer),%ymm4
    vmovupd 32(\apointer),%ymm5
    vmovupd 64(\apointer),%ymm6
    addq \delta,\apointer
    vmovapd %ymm4,(\ablkpointer)
    vmovapd %ymm5,32(\ablkpointer)
    vmovapd %ymm6,64(\ablkpointer)
    addq $96,\ablkpointer
.endm

.macro LOAD_A_1col_mask apointert,ablkpointert,deltat
    vmaskmovpd (\apointert),%ymm1,%ymm4
    vmaskmovpd 32(\apointert),%ymm2,%ymm5
    vmaskmovpd 64(\apointert),%ymm3,%ymm6
    addq \deltat,\apointert
    vmovapd %ymm4,(\ablkpointert)
    vmovapd %ymm5,32(\ablkpointert)
    vmovapd %ymm6,64(\ablkpointert)
    addq $96,\ablkpointert
.endm

.macro load_mask_irrk irrkmaskbase
    vbroadcastsd (\irrkmaskbase),%ymm8
    vbroadcastsd 8(\irrkmaskbase),%ymm9
    vbroadcastsd 16(\irrkmaskbase),%ymm10
    vbroadcastsd 24(\irrkmaskbase),%ymm11
    addq $32,\irrkmaskbase
.endm

.macro storeresult_tr44 rs1,rs2,rs3,rs4,incpointer,destpointer
    vmovapd \rs1,(\destpointer)
    vmovapd \rs2,96(\destpointer)
    vmovapd \rs3,192(\destpointer)
    vmovapd \rs4,288(\destpointer)
    addq \incpointer,\destpointer
.endm

.macro DECPUSH aaa
    decq \aaa
    push \aaa
.endm

.macro SETMASKk tEMP,kTAIL //use stack to store mask integer array; kTAIL = 0/1/2/3
    xorq \tEMP,\tEMP
    subq \kTAIL,\tEMP
    addq $3,\tEMP
    push \tEMP
    DECPUSH \tEMP
    DECPUSH \tEMP
    DECPUSH \tEMP
    movq %rsp,\tEMP //save base address of mask integer array
    addq $32,%rsp //recover rsp
.endm

.macro SETMASKm Temp,Mdim //use stack to store mask integer array
    xorq \Temp,\Temp
    subq \Mdim,\Temp
    addq $11,\Temp
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
    movq %rsp,\Temp //save base address of mask integer array
    addq $96,%rsp //recover rsp
.endm

.macro TR44 r1,r2,r3,r4,c1,c2,c3,c4,i1,i2,i3,i4 //transpose 4x4; all are ymm registers; r=row, c=column, i=intermediate
    vperm2f128 $32,\r3,\r1,\i1
    vperm2f128 $32,\r4,\r2,\i2
    vperm2f128 $49,\r3,\r1,\i3
    vperm2f128 $49,\r4,\r2,\i4
    vunpcklpd \i2,\i1,\c1
    vunpckhpd \i2,\i1,\c2
    vunpcklpd \i4,\i3,\c3
    vunpckhpd \i4,\i3,\c4
.endm

.macro TRANS4x4 line1,lda,line3,col //TR44 + data transfer; all are registers
    prefetcht0 384(\col)
    prefetcht0 480(\col)
    prefetcht0 576(\col)
    prefetcht0 672(\col)
    vmovupd (\line1),%ymm0
    vmovupd (\line1,\lda,1),%ymm1
    vmovupd (\line3),%ymm2
    vmovupd (\line3,\lda,1),%ymm3
    addq $32,\line1
    addq $32,\line3
    TR44 %ymm0,%ymm1,%ymm2,%ymm3,%ymm8,%ymm9,%ymm10,%ymm11,%ymm4,%ymm5,%ymm6,%ymm7
    storeresult_tr44 %ymm8,%ymm9,%ymm10,%ymm11,$384,\col
.endm

.macro TRANS4x4MULB bline1,bline2,bline3,bline4,wri,ymmbroadsd //multalpha + TR44 + data transfer; all are registers
    prefetcht0 (\wri)
    prefetcht0 64(\wri)
    vmovupd (\bline1),%ymm0
    vmovupd (\bline2),%ymm1
    vmovupd (\bline3),%ymm2
    vmovupd (\bline4),%ymm3
    addq $32,\bline1
    addq $32,\bline2
    addq $32,\bline3
    addq $32,\bline4
    vmulpd %ymm0,\ymmbroadsd,%ymm8
    vmulpd %ymm1,\ymmbroadsd,%ymm9
    vmulpd %ymm2,\ymmbroadsd,%ymm10
    vmulpd %ymm3,\ymmbroadsd,%ymm11
    TR44 %ymm8,%ymm9,%ymm10,%ymm11,%ymm8,%ymm9,%ymm10,%ymm11,%ymm4,%ymm5,%ymm6,%ymm7
    vmovapd %ymm8,(\wri)
    vmovapd %ymm9,32(\wri)
    vmovapd %ymm10,64(\wri)
    vmovapd %ymm11,96(\wri)
    addq $128,\wri
.endm

.macro COPY4x4MULB BLINE1,BLINE2,BLINE3,BLINE4,WRI,YMMBROADSD //multalpha + data transfer; all are registers
    prefetcht0 8192(\WRI)
    prefetcht0 8256(\WRI)
    vmovupd (\BLINE1),%ymm0
    vmovupd (\BLINE2),%ymm1
    vmovupd (\BLINE3),%ymm2
    vmovupd (\BLINE4),%ymm3
    addq $32,\BLINE1
    addq $32,\BLINE2
    addq $32,\BLINE3
    addq $32,\BLINE4
    vmulpd %ymm0,\YMMBROADSD,%ymm8
    vmulpd %ymm1,\YMMBROADSD,%ymm9
    vmulpd %ymm2,\YMMBROADSD,%ymm10
    vmulpd %ymm3,\YMMBROADSD,%ymm11
    vmovapd %ymm8,(\WRI)
    vmovapd %ymm9,32(\WRI)
    vmovapd %ymm10,64(\WRI)
    vmovapd %ymm11,96(\WRI)
    addq $8192,\WRI
.endm

.macro COPYex4MULB eBLINE1,eBLINE2,eBLINE3,eBLINE4,eWRI,eYMMBROADSD,eBMASK,eDELTBLINE
    vmaskmovpd (\eBLINE1),\eBMASK,%ymm0
    vmaskmovpd (\eBLINE2),\eBMASK,%ymm1
    vmaskmovpd (\eBLINE3),\eBMASK,%ymm2
    vmaskmovpd (\eBLINE4),\eBMASK,%ymm3
    addq \eDELTBLINE,\eBLINE1
    addq \eDELTBLINE,\eBLINE2
    addq \eDELTBLINE,\eBLINE3
    addq \eDELTBLINE,\eBLINE4
    vmulpd %ymm0,\eYMMBROADSD,%ymm8
    vmulpd %ymm1,\eYMMBROADSD,%ymm9
    vmulpd %ymm2,\eYMMBROADSD,%ymm10
    vmulpd %ymm3,\eYMMBROADSD,%ymm11
    vmaskmovpd %ymm8,\eBMASK,(\eWRI)
    vmaskmovpd %ymm9,\eBMASK,32(\eWRI)
    vmaskmovpd %ymm10,\eBMASK,64(\eWRI)
    vmaskmovpd %ymm11,\eBMASK,96(\eWRI)
.endm

.section .text
//enter the function load_reg_a_r, rdi=astartpos, rsi=ablk, edx=lda
//rdi and r10 for A pointer, rsi for ablk pointer, rdx for LDA, r11 for loop counter, r15 for outer loop counter
.globl load_reg_a_r
.type load_reg_a_r,@function
load_reg_a_r:
    push %r15
    movslq %edx,%rdx
    shlq $3,%rdx
    leaq (%rdi,%rdx,2),%r10
    xorq %r15,%r15
.Ltransa0:
    xorq %r11,%r11
.Ltransa1:
    incq %r11
    prefetcht0 64(%rdi)
    prefetcht0 64(%r10)
    TRANS4x4 %rdi,%rdx,%r10,%rsi
    prefetcht0 64(%rdi,%rdx,1)
    prefetcht0 64(%r10,%rdx,1)
    TRANS4x4 %rdi,%rdx,%r10,%rsi
    cmpq $32,%r11
    jb .Ltransa1

    incq %r15
    leaq -2048(%rdi,%rdx,4),%rdi
    leaq -2048(%r10,%rdx,4),%r10
    subq $24544,%rsi
    cmpq $3,%r15
    jb .Ltransa0

    vzeroupper
    pop %r15
    retq

//enter the function load_irregk_a_r, rdi=astartpos, rsi=ablk, edx=lda, ecx=kdim
//rdi and r10 for A pointer, rsi for ablk pointer, rdx for LDA, rcx for kdim/4, r11 for loop counter, r15 for outer loop counter, r14 for kdim%4; later r13 and r14 also for reseting pointers
.globl load_irregk_a_r
.type load_irregk_a_r,@function
load_irregk_a_r:
    push %r15
    push %r14
    push %r13
    movslq %edx,%rdx
    shlq $3,%rdx
    movslq %ecx,%rcx
    movq %rcx,%r14
    andq $0x0000000000000003,%r14
    shrq $2,%rcx
    SETMASKk %r11,%r14
    vmovdqu (%r11),%ymm15 //ymm15 store the mask for loading k-tail
    movq %rcx,%r14
    shlq $5,%r14
    negq %r14
    leaq (%r14,%rdx,4),%r14 //r14 now for reset delta of a_matrix pointers; r14 = sizeof(double) * (4 * lda - kdim / 4 * 4)
    movq %rcx,%r13
    shlq $7,%r13
    negq %r13
    leaq 32(%r13,%r13,2),%r13 //r13 now for reset delta of ablk pointer; r13 = sizeof(double) * (4 - kdim / 4 * 4 * 12)
    leaq (%rdi,%rdx,2),%r10
    xorq %r15,%r15
.Ltransika0:
    cmpq $1,%rcx // with jb: if "l > r" then jump
    jb .Ltransika2 //if kdim < 4, the loop content of Ltransika1 (e.g. TRANS4x4) should never be executed.

    xorq %r11,%r11
.Ltransika1:
    incq %r11
    TRANS4x4 %rdi,%rdx,%r10,%rsi
    cmpq %rcx,%r11
    jb .Ltransika1

.Ltransika2: //this is not a loop header
    incq %r15
    load4line_1rreg_a %rdi,%r10,%rdx,%r14,%ymm15,%ymm15,%ymm15,%ymm15 //load rows into ymm0-ymm3 with k-masks
    TR44 %ymm0,%ymm1,%ymm2,%ymm3,%ymm8,%ymm9,%ymm10,%ymm11,%ymm4,%ymm5,%ymm6,%ymm7 //transpose k-tail
    TR44 %ymm15,%ymm15,%ymm15,%ymm15,%ymm0,%ymm1,%ymm2,%ymm3,%ymm4,%ymm5,%ymm6,%ymm7 //transpose row-based masks to column-based ones!
    vmaskmovpd %ymm8,%ymm0,(%rsi)
    vmaskmovpd %ymm9,%ymm1,96(%rsi)
    vmaskmovpd %ymm10,%ymm2,192(%rsi)
    vmaskmovpd %ymm11,%ymm3,288(%rsi)
    addq %r13,%rsi
    cmpq $3,%r15
    jb .Ltransika0

    vzeroupper
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function load_irreg_a_r, rdi=astartpos, rsi=ablk, edx=lda, ecx=mdim, r8d=kdim
//rdi and r10 for A pointer, rsi for ablk pointer, rdx for LDA, rcx for mdim and later m-mask address, r8 for kdim/4, r11 for loop counter and temporal storage, r15 for outer loop counter, r14 for kdim%4; later r13 and r14 also for reseting pointers
.globl load_irreg_a_r
.type load_irreg_a_r,@function
load_irreg_a_r:
    push %r15
    push %r14
    push %r13
    movslq %edx,%rdx
    shlq $3,%rdx
    movslq %r8d,%r8
    movq %r8,%r14
    andq $0x0000000000000003,%r14
    shrq $2,%r8
    SETMASKk %r11,%r14
    vmovdqu (%r11),%ymm15 //ymm15 store the mask for loading k-tail (k-mask)
    movslq %ecx,%rcx
    SETMASKm %r11,%rcx
    movq %r11,%rcx //from now on rcx is for base pointer of irregm masks
    movq %r8,%r14
    shlq $5,%r14
    negq %r14
    leaq (%r14,%rdx,4),%r14 //r14 now for reset delta of a_matrix pointers; r14 = sizeof(double) * (4 * lda - kdim / 4 * 4)
    movq %r8,%r13
    shlq $7,%r13
    negq %r13
    leaq 32(%r13,%r13,2),%r13 //r13 now for reset delta of ablk pointer; r13 = sizeof(double) * (4 - kdim / 4 * 4 * 12)
    leaq (%rdi,%rdx,2),%r10
    xorq %r15,%r15
.Ltransirra0:
    load_mask_irrk %rcx //ymm8-ymm11 for m-masks
    cmpq $1,%r8 // with jb: if "l > r" then jump
    jb .Ltransirra2 //if kdim < 4, the loop content of Ltransirra1 should never be executed.

    xorq %r11,%r11
.Ltransirra1:
    incq %r11
    load4line_1rreg_a %rdi,%r10,%rdx,$32,%ymm8,%ymm9,%ymm10,%ymm11
    TR44 %ymm0,%ymm1,%ymm2,%ymm3,%ymm0,%ymm1,%ymm2,%ymm3,%ymm4,%ymm5,%ymm6,%ymm7
    storeresult_tr44 %ymm0,%ymm1,%ymm2,%ymm3,$384,%rsi
    cmpq %r8,%r11
    jb .Ltransirra1

.Ltransirra2: //this is not a loop header
    incq %r15
    vandpd %ymm15,%ymm8,%ymm8
    vandpd %ymm15,%ymm9,%ymm9
    vandpd %ymm15,%ymm10,%ymm10
    vandpd %ymm15,%ymm11,%ymm11 //modify m-masks by k-mask
    load4line_1rreg_a %rdi,%r10,%rdx,%r14,%ymm8,%ymm9,%ymm10,%ymm11
    TR44 %ymm0,%ymm1,%ymm2,%ymm3,%ymm0,%ymm1,%ymm2,%ymm3,%ymm4,%ymm5,%ymm6,%ymm7 //transpose k-tail
    TR44 %ymm8,%ymm9,%ymm10,%ymm11,%ymm8,%ymm9,%ymm10,%ymm11,%ymm4,%ymm5,%ymm6,%ymm7 //transpose row-based masks to column-based ones!
    vmaskmovpd %ymm0,%ymm8,(%rsi)
    vmaskmovpd %ymm1,%ymm9,96(%rsi)
    vmaskmovpd %ymm2,%ymm10,192(%rsi)
    vmaskmovpd %ymm3,%ymm11,288(%rsi)
    addq %r13,%rsi
    cmpq $3,%r15
    jb .Ltransirra0

    vzeroupper
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function load_tail_a_r, rdi=astartpos, rsi=ablk, edx=lda, ecx=mdim
//rdi and r14 for A pointer, rsi for ablk pointer, rdx for lda*8, r11 and r15 for loop counter, rcx for mdim, r10 for mask base address
.globl load_tail_a_r
.type load_tail_a_r,@function
load_tail_a_r:
    push %r15
    push %r14
    movslq %edx,%rdx
    shlq $3,%rdx
    movslq %ecx,%rcx
    SETMASKm %r10,%rcx
    leaq (%rdi,%rdx,2),%r14
    xorq %r15,%r15
.Ltransat0:
    xorq %r11,%r11
    load_mask_irrk %r10 //ymm8-ymm11 for irregk masks
.Ltransat1:
    incq %r11
    load4line_1rreg_a %rdi,%r14,%rdx,$32,%ymm8,%ymm9,%ymm10,%ymm11
    TR44 %ymm0,%ymm1,%ymm2,%ymm3,%ymm0,%ymm1,%ymm2,%ymm3,%ymm4,%ymm5,%ymm6,%ymm7
    storeresult_tr44 %ymm0,%ymm1,%ymm2,%ymm3,$384,%rsi
    cmpq $64,%r11
    jb .Ltransat1

    incq %r15
    leaq -2048(%rdi,%rdx,4),%rdi
    leaq -2048(%r14,%rdx,4),%r14
    subq $24544,%rsi
    cmpq $3,%r15
    jb .Ltransat0

    vzeroupper
    pop %r14
    pop %r15
    retq

//enter the function load_reg_a_c, rdi=astartpos, rsi=ablk, edx=lda
//rdi for A pointer, rsi for ablk pointer, rdx for LDA, r11 for loop counter
.globl load_reg_a_c
.type load_reg_a_c,@function
load_reg_a_c:
    movslq %edx,%rdx
    shlq $3,%rdx
    xorq %r11,%r11
.Lloadac:
    incq %r11
    LOAD_A_1col %rdi,%rsi,%rdx
    cmpq $256,%r11
    jb .Lloadac

    vzeroupper
    retq

//enter the function load_irregk_a_c, rdi=astartpos, rsi=ablk, edx=lda, ecx=kdim
//rdi for A pointer, rsi for ablk pointer, rdx for LDA, r11 for loop counter, rcx for kdim
.globl load_irregk_a_c
.type load_irregk_a_c,@function
load_irregk_a_c:
    movslq %ecx,%rcx
    movslq %edx,%rdx
    shlq $3,%rdx
    xorq %r11,%r11
.Lloadikac:
    incq %r11
    LOAD_A_1col %rdi,%rsi,%rdx
    cmpq %rcx,%r11
    jb .Lloadikac

    vzeroupper
    retq

//enter the function load_irreg_a_c, rdi=astartpos, rsi=ablk, edx=lda, ecx=mdim, r8d=kdim
//rdi for A pointer, rsi for ablk pointer, rdx for lda*8, r11 for loop counter, rcx for mdim, r8 for kdim, r10 for mask base address
.globl load_irreg_a_c
.type load_irreg_a_c,@function
load_irreg_a_c:
    movslq %ecx,%rcx //mdim
    movslq %r8d,%r8  //kdim
    movslq %edx,%rdx
    shlq $3,%rdx //lda*8
    SETMASKm %r10,%rcx
    load_mask %r10 //set up ymm1-ymm3 for masks
    xorq %r11,%r11
.Lloadirrac:
    incq %r11
    LOAD_A_1col_mask %rdi,%rsi,%rdx
    cmpq %r8,%r11
    jb .Lloadirrac

    vzeroupper
    retq

//enter the function load_tail_a_c, rdi=astartpos, rsi=ablk, edx=lda, ecx=mdim
//rdi for A pointer, rsi for ablk pointer, rdx for LDA, r11 for loop counter, rcx for mdim, r10 for mask base address
.globl load_tail_a_c
.type load_tail_a_c,@function
load_tail_a_c:
    movslq %edx,%rdx
    shlq $3,%rdx
    movslq %ecx,%rcx
    SETMASKm %r10,%rcx
    load_mask %r10
    xorq %r11,%r11
.Lloadat:
    incq %r11
    LOAD_A_1col_mask %rdi,%rsi,%rdx
    cmpq $256,%r11
    jb .Lloadat

    vzeroupper
    retq

//enter the function load_reg_b_c, rdi=bstartpos, rsi=bblk, edx=ldb, rcx=alpha<double *>
//rdi,r10,r12 and r13 for b pointer; rsi for bblk pointer; r11 and rcx for loop counter; rdx for ldb*8; rax for ldb*BlkDimN*8
.globl load_reg_b_c
.type load_reg_b_c,@function
load_reg_b_c:
    push %r13
    push %r12
    vbroadcastsd (%rcx),%ymm15 //alpha vector
    movslq %edx,%rdx
    shlq $3,%rdx
    movq %rdx,%rax
    shlq $8,%rax
    leaq (%rdi,%rdx,1),%r10
    leaq (%r10,%rdx,1),%r12
    leaq (%r12,%rdx,1),%r13
    xorq %rcx,%rcx //bcol*4 counter
.Lloadregbcouter:
    incq %rcx
    xorq %r11,%r11 //brow*4 counter
.Lloadregbcinner1:
    incq %r11
    TRANS4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $16,%r11
    jb .Lloadregbcinner1

    addq %rdx,%rdi
    addq %rdx,%r10
    addq %rdx,%r12
    addq %rdx,%r13
    cmpq $64,%rcx
    jb .Lloadregbcinner2
    subq %rax,%r13
.Lloadregbcinner2:
    incq %r11
    TRANS4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $32,%r11
    jb .Lloadregbcinner2

    addq %rdx,%rdi
    addq %rdx,%r10
    addq %rdx,%r12
    addq %rdx,%r13
    cmpq $64,%rcx
    jb .Lloadregbcinner3
    subq %rax,%r12
.Lloadregbcinner3:
    incq %r11
    TRANS4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $48,%r11
    jb .Lloadregbcinner3

    addq %rdx,%rdi
    addq %rdx,%r10
    addq %rdx,%r12
    addq %rdx,%r13
    cmpq $64,%rcx
    jb .Lloadregbcinner4
    subq %rax,%r10
.Lloadregbcinner4:
    incq %r11
    TRANS4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $64,%r11
    jb .Lloadregbcinner4

    leaq -2048(%rdi,%rdx,1),%rdi
    leaq -2048(%r10,%rdx,1),%r10
    leaq -2048(%r12,%rdx,1),%r12
    leaq -2048(%r13,%rdx,1),%r13
    cmpq $64,%rcx
    jb .Lloadregbcouter

    vzeroupper
    pop %r12
    pop %r13
    retq

//enter the function load_reg_b_r, rdi=bstartpos, rsi=bblk, edx=ldb, rcx=alpha<double *>
//rdi,r10,r12 and r13 for b pointer; rsi for bblk pointer; r11 and rcx for loop counter; rdx for ldb*8; rax for bblk address
//ymm12-ymm14 for masks, ymm0-ymm3 for inputs, ymm8-ymm11 for outputs, ymm15 for alpha vector
.globl load_reg_b_r
.type load_reg_b_r,@function
load_reg_b_r:
    push %r13
    push %r12
    movq %rsi,%rax
    vbroadcastsd (%rcx),%ymm15 //alpha vector
    movslq %edx,%rdx
    shlq $3,%rdx
    xorq %r11,%r11
    push %r11
    push %r11
    push %r11
    movq $-1,%r11
    push %r11
    push %r11
    push %r11
    vmovdqu (%rsp),%ymm14 //ymm14 = mask3 = load/store the lowest 3 doubles (in packed 4 doubles)
    vmovdqu 8(%rsp),%ymm13 //ymm13 = mask2 = load/store the lowest 2 doubles (in packed 4 doubles)
    vmovdqu 16(%rsp),%ymm12 //ymm12 = mask1 = load/store the lowest double (in packed 4 doubles)
    addq $48,%rsp //recover stack pointer
    leaq (%rdi,%rdx,1),%r10
    leaq (%r10,%rdx,1),%r12
    leaq (%r12,%rdx,1),%r13
    xorq %rcx,%rcx //brow*4 counter
.Lloadregbrouter1:
    leaq (%rax,%rcx,8),%rsi
    xorq %r11,%r11 //bcol counter
.Lloadregbrinner1:
    incq %r11
    COPY4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $64,%r11
    jb .Lloadregbrinner1

    addq $16,%rcx
    leaq -2048(%rdi,%rdx,4),%rdi
    leaq -2048(%r10,%rdx,4),%r10
    leaq -2048(%r12,%rdx,4),%r12
    leaq -2048(%r13,%rdx,4),%r13
    cmpq $256,%rcx
    jb .Lloadregbrouter1

.Lloadregbrouter2:
    leaq 516120(%rax,%rcx,8),%rsi
    COPYex4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15,%ymm12,$8
    leaq (%rax,%rcx,8),%rsi
    movq $1,%r11
.Lloadregbrinner2:
    incq %r11
    COPY4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $64,%r11
    jb .Lloadregbrinner2

    addq $16,%rcx
    COPYex4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15,%ymm14,$24
    leaq -2048(%rdi,%rdx,4),%rdi
    leaq -2048(%r10,%rdx,4),%r10
    leaq -2048(%r12,%rdx,4),%r12
    leaq -2048(%r13,%rdx,4),%r13
    cmpq $512,%rcx
    jb .Lloadregbrouter2

.Lloadregbrouter3:
    leaq 516112(%rax,%rcx,8),%rsi
    COPYex4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15,%ymm13,$16
    leaq (%rax,%rcx,8),%rsi
    movq $1,%r11
.Lloadregbrinner3:
    incq %r11
    COPY4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $64,%r11
    jb .Lloadregbrinner3

    addq $16,%rcx
    COPYex4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15,%ymm13,$16
    leaq -2048(%rdi,%rdx,4),%rdi
    leaq -2048(%r10,%rdx,4),%r10
    leaq -2048(%r12,%rdx,4),%r12
    leaq -2048(%r13,%rdx,4),%r13
    cmpq $768,%rcx
    jb .Lloadregbrouter3

.Lloadregbrouter4:
    leaq 516104(%rax,%rcx,8),%rsi
    COPYex4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15,%ymm14,$24
    leaq (%rax,%rcx,8),%rsi
    movq $1,%r11
.Lloadregbrinner4:
    incq %r11
    COPY4x4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15
    cmpq $64,%r11
    jb .Lloadregbrinner4

    addq $16,%rcx
    COPYex4MULB %rdi,%r10,%r12,%r13,%rsi,%ymm15,%ymm12,$8
    leaq -2048(%rdi,%rdx,4),%rdi
    leaq -2048(%r10,%rdx,4),%r10
    leaq -2048(%r12,%rdx,4),%r12
    leaq -2048(%r13,%rdx,4),%r13
    cmpq $1024,%rcx
    jb .Lloadregbrouter4

    vzeroupper
    pop %r12
    pop %r13
    retq

//The dgemm-calc codes blow use rdi and rsi as block matrix pointers, use r13-r15 for load/store of C elements, use r11 and r12 as loop counters, use ymm0-ymm3 for temporary load of matrix (A&B) elements, use ymm4-ymm15 for C elements, use r9 and r10 for update of ablk pointer (r9 also for prefetcht1-ablknext), use rax,rbx and rcx for ablk preload (also rdx when transa=y), rax also for building masks in dgemmblktailccc

//enter the function dgemmblkregccc_ac, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc
.globl dgemmblkregccc
.type dgemmblkregccc,@function
dgemmblkregccc:

    push %r15
    push %r14
    push %r13
    push %r12
    movq %rdx,CIP
    movq %rdi,AL
    addq $49152,AL //point to (prefetch) next ablk zone of abuffer
    movslq %ecx,LDC
    shlq $3,LDC
    movq CIP,CL
    movq CIP,CS

    INIT_C_3col
    movq $0xa000000000000000,%r10 //ablk pointer increment table = (0xa000,0x0000,0x0000,0x0000)
    xorq %r12,%r12
.Louter_dgemmblkregccc:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFm12 CS
    PREFm12 CL
    subq $96,AL
    prefetcht1 (AL)
    prefetcht1 64(AL)
    xorq %r11,%r11
.Linner_dgemmblkregccc:
    KERNELm12n4k8
    cmpq $8,%r11
    jb .Linner_dgemmblkregccc

    addq %r9,A0
    prefetcht0 (A0)
    prefetcht0 64(A0)
    prefetcht0 128(A0)
    prefetcht0 192(A0)
    incq %r12
    STORECBLK_1col
    rorq $16,%r10
    cmpq $252,%r12
    jb .Louter_dgemmblkregccc

    movq AL,%r9
    subq $384,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_dgemmblkregccc_last:
    PREFm12 CL
    prefetcht1 152(CL)
    prefetcht1 184(CL)
    xorq %r11,%r11
.Linner_dgemmblkregccc_last:
    prefetcht0 256(A0)
    prefetcht0 320(A0)
    prefetcht1 (%r9)
    KERNELm12n4k1 0,0
    prefetcht0 384(A0)
    prefetcht0 768(B0)
    prefetcht1 64(%r9)
    incq %r11
    KERNELm12n4k1 96,32
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    prefetcht1 128(%r9)
    KERNELm12n4k1 192,64
    prefetcht0 576(A0)
    prefetcht0 832(B0)
    prefetcht1 192(%r9)
    addq $256,%r9
    KERNELm12n4kf 288,96,384,128
    cmpq $16,%r11
    jb .Linner_dgemmblkregccc_last

    incq %r12
    STORECBLK_1col
    PREFm12 CS
    UPDATECBLK_1col
    cmpq $256,%r12
    jb .Louter_dgemmblkregccc_last
    movq CIP,CS
    FIN_C_3col

    vzeroupper
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function dgemmblkregccc_ac, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8=aprefpos, r9d=lda, stack1=nextablk
.globl dgemmblkregccc_ac
.type dgemmblkregccc_ac,@function
dgemmblkregccc_ac:

    push %r15
    push %r14
    push %r13
    push %r12
    push %rbx
    movq 48(%rsp),ABS
    movq %rdx,CIP
    movq %r8,AL
    movslq %ecx,LDC
    shlq $3,LDC
    movslq %r9d,LDA
    shlq $3,LDA
    movq CIP,CL
    movq CIP,CS

    INIT_C_3col
    movq $0xa000000000000000,%r10 //ablk pointer increment table = (0xa000,0x0000,0x0000,0x0000)
    xorq %r12,%r12
.Louter_ac:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFm12 CS
    PREFm12 CL
    PREFm12 AL
    prefetcht0 (ABS)
    prefetcht0 64(ABS)
    xorq %r11,%r11
.Linner_ac:
    KERNELm12n4k8
    cmpq $8,%r11
    jb .Linner_ac

    addq %r9,A0
    prefetcht0 (A0)
    prefetcht0 64(A0)
    prefetcht0 128(A0)
    prefetcht0 192(A0)
    incq %r12
    STORECBLK_1col
    LOAD_A_1col AL,ABS,LDA
    rorq $16,%r10
    cmpq $252,%r12
    jb .Louter_ac

    movq ABS,%r9
    subq $24192,%r9
    UPDATECBLK_1col
    movq CIP,CL
.Louter_ac_last:
    PREFm12 AL
    prefetcht0 (ABS)
    prefetcht0 64(ABS)
    PREFm12 CL
    prefetcht1 152(CL)
    prefetcht1 184(CL)
    xorq %r11,%r11
.Linner_ac_last:
    prefetcht0 256(A0)
    prefetcht0 320(A0)
    prefetcht1 (%r9)
    KERNELm12n4k1 0,0
    prefetcht0 384(A0)
    prefetcht0 768(B0)
    prefetcht1 64(%r9)
    incq %r11
    KERNELm12n4k1 96,32
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    prefetcht1 128(%r9)
    KERNELm12n4k1 192,64
    prefetcht0 576(A0)
    prefetcht0 832(B0)
    prefetcht1 192(%r9)
    addq $256,%r9
    KERNELm12n4kf 288,96,384,128
    cmpq $16,%r11
    jb .Linner_ac_last

    incq %r12
    STORECBLK_1col
    LOAD_A_1col AL,ABS,LDA
    PREFm12 CS
    UPDATECBLK_1col
    cmpq $256,%r12
    jb .Louter_ac_last
    movq CIP,CS
    FIN_C_3col

    vzeroupper
    pop %rbx
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function dgemmblkregccc_ar, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8=aprefpos, r9d=lda
.globl dgemmblkregccc_ar
.type dgemmblkregccc_ar,@function
dgemmblkregccc_ar:

    push %r15
    push %r14
    push %r13
    push %r12
    push %rbx
    movq %rdx,CIP
    movq %r8,AL
    movslq %ecx,LDC
    shlq $3,LDC
    movslq %r9d,LDA
    shlq $3,LDA
    movq CIP,CL
    movq CIP,CS

    INIT_C_3col
    movq $0xa000000000000000,%r10 //ablk increment table = (0xa000,0x0000,0x0000,0x0000)
    addq $16,AL
    movq LDA,%rbx
    shlq $1,%rbx
    movq LDA,%rdx
    leaq (%rbx,%rbx,2),%rbx
    leaq (%rdx,%rdx,2),%rdx
    xorq %r12,%r12
.Louter_ar:
    UPDATECBLK_1col
    PREFm12 CS
    PREFm12 CL
    PREFam3k4 0
    movq %rdx,%r9
    addq %r9,AL
    addq $8,AL
    movq %rbx,%rdx
    movq %r9,%rbx
    negq %rbx
    xorq %r11,%r11
.Linner_ar:
    KERNELm12n4k8
    cmpq $8,%r11
    jb .Linner_ar

    movswq %r10w,%r9
    addq %r9,A0
    prefetcht0 (A0)
    prefetcht0 64(A0)
    prefetcht0 128(A0)
    prefetcht0 192(A0)
    rorq $16,%r10
    incq %r12
    STORECBLK_1col
    cmpq $252,%r12
    jb .Louter_ar

    subq $0x7f0,AL //point apref to aprefpos
    UPDATECBLK_1col
    movq CIP,CL
.Louter_ar_last:
    PREFam3k4 0
    PREFam3k4 0x7f8
    PREFm12 CL
    prefetcht1 152(CL)
    prefetcht1 184(CL)
    xorq %r11,%r11
.Linner_ar_last:
    prefetcht0 256(A0)
    prefetcht0 320(A0)
    addq $64,AL
    KERNELm12n4k1 0,0
    prefetcht0 384(A0)
    prefetcht0 768(B0)
    prefetcht1 (AL)
    incq %r11
    KERNELm12n4k1 96,32
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    prefetcht1 (AL,LDA,1)
    KERNELm12n4k1 192,64
    prefetcht0 576(A0)
    prefetcht0 832(B0)
    prefetcht1 (AL,LDA,2)
    KERNELm12n4kf 288,96,384,128
    cmpq $16,%r11
    jb .Linner_ar_last

    incq %r12
    subq $1024,AL
    leaq (AL,LDA,2),AL
    addq LDA,AL
    STORECBLK_1col
    PREFm12 CS
    UPDATECBLK_1col
    cmpq $256,%r12
    jb .Louter_ar_last

    movq CIP,CS
    FIN_C_3col

    vzeroupper
    pop %rbx
    pop %r12
    pop %r13
    pop %r14
    pop %r15
    retq

//enter the function dgemmblktailccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim
.globl dgemmblktailccc
.type dgemmblktailccc,@function
dgemmblktailccc:

    push %r15
    push %r14
    push %r13
    push %r12
    push %rdx //cstartpos
    movslq %ecx,LDC
    shlq $3,LDC
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
    KERNELm12n4k8
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
    KERNELm12n4k2 0,0
    incq %r11
    prefetcht0 448(A0)
    prefetcht0 512(A0)
    KERNELm12n4k1 192,64
    prefetcht0 576(A0)
    prefetcht0 832(B0)
    KERNELm12n4kf 288,96,384,128
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

//The dgemm-calc codes blow use rdi and rsi as block matrix pointers, use rdx and rcx for load/store of C elements, use r11 and r10 as loop counters, use ymm0-ymm3 for temporary load of matrix (A&B) elements, use ymm4-ymm15 for C elements

//enter the function dgemmblkirregkccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=kdim, r9=&beta
.globl dgemmblkirregkccc
.type dgemmblkirregkccc,@function
dgemmblkirregkccc:
    push %r15
    movslq %ecx,%rcx
    shlq $3,%rcx //rcx = m8ldc
    movslq %r8d,%r8 //r8 = kdim
    xorq %r10,%r10 //c_column*4 counter
    movq A0,%r15 //save ablk base address
.Ldgemmirrkouter:
    movq %r15,A0 //reset ablk pointer
    LOAD_C_m12n4 %rdx,%rcx //load cblk into ymm4-ymm15 with prefetch of next cblk
    vbroadcastsd (%r9),%ymm0
    MULT_C_m12n4 //mult cblk in ymm4-ymm15 with ymm0 (beta vector)
    xorq %r11,%r11 //k counter
.Ldgemmirrkinner:
    incq %r11
    prefetcht0 768(B0)
    KERNELm12n4kf 0,0,96,32
    cmpq %r8,%r11
    jb .Ldgemmirrkinner

    incq %r10
    STORE_C_m12n4 %rdx,%rcx //store ymm4-ymm15 to cblk
    cmpq $64,%r10
    jb .Ldgemmirrkouter

    vzeroupper
    pop %r15
    retq

//enter the function dgemmblkirregnccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=ndim
.globl dgemmblkirregnccc
.type dgemmblkirregnccc,@function
dgemmblkirregnccc:
    push %r15
    movslq %ecx,%rcx
    shlq $3,%rcx //rcx = m8ldc
    movslq %r8d,%r8
    xorq %r10,%r10 //c_column counter
    movq A0,%r15 //save ablk base address
    cmpq $4,%r8
    jb .Ldgemmirrnts //if ndim < 4, skip calccblkm12n4 part.
    subq $3,%r8 //r8 = ndim - 3
.Ldgemmirrnouter:
    movq %r15,A0 //reset ablk pointer
    LOAD_C_m12n4 %rdx,%rcx //load cblk into ymm4-ymm15 with prefetch of next cblk
    xorq %r11,%r11 //k counter
.Ldgemmirrninner:
    KERNELm12n4k8
    cmpq $32,%r11
    jb .Ldgemmirrninner

    addq $4,%r10
    STORE_C_m12n4 %rdx,%rcx //store ymm4-ymm15 to cblk
    cmpq %r8,%r10
    jb .Ldgemmirrnouter

    addq $3,%r8 //r8 = ndim
.Ldgemmirrnts:
    cmpq %r8,%r10 //cmp-jz: if "l==r" than jump; if ccol_count==ndim than skip the rest ccol calculation
    jz .Ldgemmirrntt
.Ldgemmirrntouter:
    movq %r15,A0 //reset ablk pointer
    vmovupd (%rdx),%ymm4
    vmovupd 32(%rdx),%ymm5
    vmovupd 64(%rdx),%ymm6
    xorq %r11,%r11
.Ldgemmirrntinner:
    incq %r11
    KERNELm12n1k1
    cmpq $256,%r11
    jb .Ldgemmirrntinner

    incq %r10
    vmovupd %ymm4,(%rdx)
    vmovupd %ymm5,32(%rdx)
    vmovupd %ymm6,64(%rdx)
    addq %rcx,%rdx
    cmpq %r8,%r10
    jb .Ldgemmirrntouter

.Ldgemmirrntt:
    vzeroupper
    pop %r15
    retq

//enter the function dgemmblkirregccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim, r9d=ndim, stack1 = kdim, stack2 = &beta
.globl dgemmblkirregccc
.type dgemmblkirregccc,@function
dgemmblkirregccc:
    push %r15
    push %r14
    push %r13
    movslq %ecx,%rcx
    shlq $3,%rcx //rcx = m8ldc
    movslq %r8d,%r8 //r8 = mdim
    movslq %r9d,%r9 //r9 = ndim
    movslq 32(%rsp),%r14 //r14 = kdim
    movq 40(%rsp),%r13 //r13 = &beta
    SETMASKm %r11,%r8
    movq %r11,%r8 // now r8 is the base pointer of m masks
    load_mask %r8 // load m mask into ymm1-ymm3
    xorq %r10,%r10 //c_column counter
    movq A0,%r15 //save ablk base address
    cmpq $4,%r9
    jb .Ldgemmirr3ts //if ndim < 4, skip calccblkm12n4 part.
    subq $3,%r9 //r9 = ndim - 3
.Ldgemmirr3outer:
    movq %r15,A0 //reset ablk pointer
    LOAD_C_M12N4_mmask %rdx,%rcx //load cblk into ymm4-ymm15 with prefetch of next cblk
    vbroadcastsd (%r13),%ymm0
    MULT_C_m12n4 //mult cblk in ymm4-ymm15 with ymm0 (beta vector)
    xorq %r11,%r11 //k counter
.Ldgemmirr3inner:
    incq %r11
    prefetcht0 768(B0)
    KERNELm12n4kf 0,0,96,32
    cmpq %r14,%r11
    jb .Ldgemmirr3inner

    load_mask %r8 // load m mask into ymm1-ymm3
    addq $4,%r10
    STORE_C_M12N4_mmask %rdx,%rcx //store ymm4-ymm15 to cblk
    cmpq %r9,%r10
    jb .Ldgemmirr3outer

    addq $3,%r9 //r9 = ndim
.Ldgemmirr3ts:
    vmovaps %ymm1,%ymm13 //transfer mmasks to ymm13-ymm15
    vmovaps %ymm2,%ymm14
    vmovaps %ymm3,%ymm15
    cmpq %r9,%r10 //cmp-jz: if "l==r" than jump; if ccol_count==ndim than skip the rest ccol calculation
    jz .Ldgemmirr3tt
.Ldgemmirr3touter:
    movq %r15,A0 //reset ablk pointer
    vmaskmovpd (%rdx),%ymm13,%ymm4
    vmaskmovpd 32(%rdx),%ymm14,%ymm5
    vmaskmovpd 64(%rdx),%ymm15,%ymm6
    vbroadcastsd (%r13),%ymm0
    MULT_C_m12n1 %ymm4,%ymm5,%ymm6//mult ccolumn in ymm4-ymm6 with ymm0 (beta vector)
    xorq %r11,%r11
.Ldgemmirr3tinner:
    incq %r11
    KERNELm12n1k1
    cmpq %r14,%r11
    jb .Ldgemmirr3tinner

    incq %r10
    vmaskmovpd %ymm4,%ymm13,(%rdx)
    vmaskmovpd %ymm5,%ymm14,32(%rdx)
    vmaskmovpd %ymm6,%ymm15,64(%rdx)
    addq %rcx,%rdx
    cmpq %r9,%r10
    jb .Ldgemmirr3touter

.Ldgemmirr3tt:
    vzeroupper
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

