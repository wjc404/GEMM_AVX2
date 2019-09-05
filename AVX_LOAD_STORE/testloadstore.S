.macro LOADU_16_D32 offset1
  vmovupd \offset1(%r10),%ymm0
  vmovupd \offset1+32(%r10),%ymm1
  vmovupd \offset1+64(%r10),%ymm2
  vmovupd \offset1+96(%r10),%ymm3
  vmovupd \offset1+128(%r10),%ymm4
  vmovupd \offset1+160(%r10),%ymm5
  vmovupd \offset1+192(%r10),%ymm6
  vmovupd \offset1+224(%r10),%ymm7
  vmovupd \offset1+256(%r10),%ymm8
  vmovupd \offset1+288(%r10),%ymm9
  vmovupd \offset1+320(%r10),%ymm10
  vmovupd \offset1+352(%r10),%ymm11
  vmovupd \offset1+384(%r10),%ymm12
  vmovupd \offset1+416(%r10),%ymm13
  vmovupd \offset1+448(%r10),%ymm14
  vmovupd \offset1+480(%r10),%ymm15
.endm

.macro LOADU_16_D64 offset2
  vmovupd \offset2(%r10),%ymm0
  vmovupd \offset2+64(%r10),%ymm1
  vmovupd \offset2+128(%r10),%ymm2
  vmovupd \offset2+192(%r10),%ymm3
  vmovupd \offset2+256(%r10),%ymm4
  vmovupd \offset2+320(%r10),%ymm5
  vmovupd \offset2+384(%r10),%ymm6
  vmovupd \offset2+448(%r10),%ymm7
  vmovupd \offset2+512(%r10),%ymm8
  vmovupd \offset2+576(%r10),%ymm9
  vmovupd \offset2+640(%r10),%ymm10
  vmovupd \offset2+704(%r10),%ymm11
  vmovupd \offset2+768(%r10),%ymm12
  vmovupd \offset2+832(%r10),%ymm13
  vmovupd \offset2+896(%r10),%ymm14
  vmovupd \offset2+960(%r10),%ymm15
.endm

.macro LOADA_16_D32 offset3
  vmovapd \offset3(%r10),%ymm0
  vmovapd \offset3+32(%r10),%ymm1
  vmovapd \offset3+64(%r10),%ymm2
  vmovapd \offset3+96(%r10),%ymm3
  vmovapd \offset3+128(%r10),%ymm4
  vmovapd \offset3+160(%r10),%ymm5
  vmovapd \offset3+192(%r10),%ymm6
  vmovapd \offset3+224(%r10),%ymm7
  vmovapd \offset3+256(%r10),%ymm8
  vmovapd \offset3+288(%r10),%ymm9
  vmovapd \offset3+320(%r10),%ymm10
  vmovapd \offset3+352(%r10),%ymm11
  vmovapd \offset3+384(%r10),%ymm12
  vmovapd \offset3+416(%r10),%ymm13
  vmovapd \offset3+448(%r10),%ymm14
  vmovapd \offset3+480(%r10),%ymm15
.endm

.macro LOADA_16_D64 offset4
  vmovapd \offset4(%r10),%ymm0
  vmovapd \offset4+64(%r10),%ymm1
  vmovapd \offset4+128(%r10),%ymm2
  vmovapd \offset4+192(%r10),%ymm3
  vmovapd \offset4+256(%r10),%ymm4
  vmovapd \offset4+320(%r10),%ymm5
  vmovapd \offset4+384(%r10),%ymm6
  vmovapd \offset4+448(%r10),%ymm7
  vmovapd \offset4+512(%r10),%ymm8
  vmovapd \offset4+576(%r10),%ymm9
  vmovapd \offset4+640(%r10),%ymm10
  vmovapd \offset4+704(%r10),%ymm11
  vmovapd \offset4+768(%r10),%ymm12
  vmovapd \offset4+832(%r10),%ymm13
  vmovapd \offset4+896(%r10),%ymm14
  vmovapd \offset4+960(%r10),%ymm15
.endm

.macro BROADCAST_16_D4 offset
  vbroadcastss \offset(%r10),%ymm0
  vbroadcastss \offset+4(%r10),%ymm0
  vbroadcastss \offset+8(%r10),%ymm0
  vbroadcastss \offset+12(%r10),%ymm0
  vbroadcastss \offset+16(%r10),%ymm0
  vbroadcastss \offset+20(%r10),%ymm0
  vbroadcastss \offset+24(%r10),%ymm0
  vbroadcastss \offset+28(%r10),%ymm0
  vbroadcastss \offset+32(%r10),%ymm0
  vbroadcastss \offset+36(%r10),%ymm0
  vbroadcastss \offset+40(%r10),%ymm0
  vbroadcastss \offset+44(%r10),%ymm0
  vbroadcastss \offset+48(%r10),%ymm0
  vbroadcastss \offset+52(%r10),%ymm0
  vbroadcastss \offset+56(%r10),%ymm0
  vbroadcastss \offset+60(%r10),%ymm0
.endm

.macro BROADCAST_16_D8 offset5
  vbroadcastsd \offset5(%r10),%ymm0
  vbroadcastsd \offset5+8(%r10),%ymm1
  vbroadcastsd \offset5+16(%r10),%ymm2
  vbroadcastsd \offset5+24(%r10),%ymm3
  vbroadcastsd \offset5+32(%r10),%ymm4
  vbroadcastsd \offset5+40(%r10),%ymm5
  vbroadcastsd \offset5+48(%r10),%ymm6
  vbroadcastsd \offset5+56(%r10),%ymm7
  vbroadcastsd \offset5+64(%r10),%ymm8
  vbroadcastsd \offset5+72(%r10),%ymm9
  vbroadcastsd \offset5+80(%r10),%ymm10
  vbroadcastsd \offset5+88(%r10),%ymm11
  vbroadcastsd \offset5+96(%r10),%ymm12
  vbroadcastsd \offset5+104(%r10),%ymm13
  vbroadcastsd \offset5+112(%r10),%ymm14
  vbroadcastsd \offset5+120(%r10),%ymm15
.endm

.macro BROADCAST_16_D16 offset
  vbroadcastf128 \offset(%r10),%ymm0
  vbroadcastf128 \offset+16(%r10),%ymm0
  vbroadcastf128 \offset+32(%r10),%ymm0
  vbroadcastf128 \offset+48(%r10),%ymm0
  vbroadcastf128 \offset+64(%r10),%ymm0
  vbroadcastf128 \offset+80(%r10),%ymm0
  vbroadcastf128 \offset+96(%r10),%ymm0
  vbroadcastf128 \offset+112(%r10),%ymm0
  vbroadcastf128 \offset+128(%r10),%ymm0
  vbroadcastf128 \offset+144(%r10),%ymm0
  vbroadcastf128 \offset+160(%r10),%ymm0
  vbroadcastf128 \offset+176(%r10),%ymm0
  vbroadcastf128 \offset+192(%r10),%ymm0
  vbroadcastf128 \offset+208(%r10),%ymm0
  vbroadcastf128 \offset+224(%r10),%ymm0
  vbroadcastf128 \offset+240(%r10),%ymm0
.endm

.macro MASKLOAD_16_D32 offset6
  vmaskmovpd \offset6(%r10),%ymm0,%ymm0
  vmaskmovpd \offset6+32(%r10),%ymm1,%ymm1
  vmaskmovpd \offset6+64(%r10),%ymm2,%ymm2
  vmaskmovpd \offset6+96(%r10),%ymm3,%ymm3
  vmaskmovpd \offset6+128(%r10),%ymm4,%ymm4
  vmaskmovpd \offset6+160(%r10),%ymm5,%ymm5
  vmaskmovpd \offset6+192(%r10),%ymm6,%ymm6
  vmaskmovpd \offset6+224(%r10),%ymm7,%ymm7
  vmaskmovpd \offset6+256(%r10),%ymm8,%ymm8
  vmaskmovpd \offset6+288(%r10),%ymm9,%ymm9
  vmaskmovpd \offset6+320(%r10),%ymm10,%ymm10
  vmaskmovpd \offset6+352(%r10),%ymm11,%ymm11
  vmaskmovpd \offset6+384(%r10),%ymm12,%ymm12
  vmaskmovpd \offset6+416(%r10),%ymm13,%ymm13
  vmaskmovpd \offset6+448(%r10),%ymm14,%ymm14
  vmaskmovpd \offset6+480(%r10),%ymm15,%ymm15
.endm

.macro STOREU_16_D32 OFFSET1
  vmovupd %ymm0,\OFFSET1(%r10)
  vmovupd %ymm1,\OFFSET1+32(%r10)
  vmovupd %ymm2,\OFFSET1+64(%r10)
  vmovupd %ymm3,\OFFSET1+96(%r10)
  vmovupd %ymm4,\OFFSET1+128(%r10)
  vmovupd %ymm5,\OFFSET1+160(%r10)
  vmovupd %ymm6,\OFFSET1+192(%r10)
  vmovupd %ymm7,\OFFSET1+224(%r10)
  vmovupd %ymm8,\OFFSET1+256(%r10)
  vmovupd %ymm9,\OFFSET1+288(%r10)
  vmovupd %ymm10,\OFFSET1+320(%r10)
  vmovupd %ymm11,\OFFSET1+352(%r10)
  vmovupd %ymm12,\OFFSET1+384(%r10)
  vmovupd %ymm13,\OFFSET1+416(%r10)
  vmovupd %ymm14,\OFFSET1+448(%r10)
  vmovupd %ymm15,\OFFSET1+480(%r10)
.endm

.macro STOREU_16_D64 OFFSET2
  vmovupd %ymm0,\OFFSET2(%r10)
  vmovupd %ymm1,\OFFSET2+64(%r10)
  vmovupd %ymm2,\OFFSET2+128(%r10)
  vmovupd %ymm3,\OFFSET2+192(%r10)
  vmovupd %ymm4,\OFFSET2+256(%r10)
  vmovupd %ymm5,\OFFSET2+320(%r10)
  vmovupd %ymm6,\OFFSET2+384(%r10)
  vmovupd %ymm7,\OFFSET2+448(%r10)
  vmovupd %ymm8,\OFFSET2+512(%r10)
  vmovupd %ymm9,\OFFSET2+576(%r10)
  vmovupd %ymm10,\OFFSET2+640(%r10)
  vmovupd %ymm11,\OFFSET2+704(%r10)
  vmovupd %ymm12,\OFFSET2+768(%r10)
  vmovupd %ymm13,\OFFSET2+832(%r10)
  vmovupd %ymm14,\OFFSET2+896(%r10)
  vmovupd %ymm15,\OFFSET2+960(%r10)
.endm

.macro STOREA_16_D32 OFFSET3
  vmovapd %ymm0,\OFFSET3(%r10)
  vmovapd %ymm1,\OFFSET3+32(%r10)
  vmovapd %ymm2,\OFFSET3+64(%r10)
  vmovapd %ymm3,\OFFSET3+96(%r10)
  vmovapd %ymm4,\OFFSET3+128(%r10)
  vmovapd %ymm5,\OFFSET3+160(%r10)
  vmovapd %ymm6,\OFFSET3+192(%r10)
  vmovapd %ymm7,\OFFSET3+224(%r10)
  vmovapd %ymm8,\OFFSET3+256(%r10)
  vmovapd %ymm9,\OFFSET3+288(%r10)
  vmovapd %ymm10,\OFFSET3+320(%r10)
  vmovapd %ymm11,\OFFSET3+352(%r10)
  vmovapd %ymm12,\OFFSET3+384(%r10)
  vmovapd %ymm13,\OFFSET3+416(%r10)
  vmovapd %ymm14,\OFFSET3+448(%r10)
  vmovapd %ymm15,\OFFSET3+480(%r10)
.endm

.macro STOREA_16_D64 OFFSET4
  vmovapd %ymm0,\OFFSET4(%r10)
  vmovapd %ymm1,\OFFSET4+64(%r10)
  vmovapd %ymm2,\OFFSET4+128(%r10)
  vmovapd %ymm3,\OFFSET4+192(%r10)
  vmovapd %ymm4,\OFFSET4+256(%r10)
  vmovapd %ymm5,\OFFSET4+320(%r10)
  vmovapd %ymm6,\OFFSET4+384(%r10)
  vmovapd %ymm7,\OFFSET4+448(%r10)
  vmovapd %ymm8,\OFFSET4+512(%r10)
  vmovapd %ymm9,\OFFSET4+576(%r10)
  vmovapd %ymm10,\OFFSET4+640(%r10)
  vmovapd %ymm11,\OFFSET4+704(%r10)
  vmovapd %ymm12,\OFFSET4+768(%r10)
  vmovapd %ymm13,\OFFSET4+832(%r10)
  vmovapd %ymm14,\OFFSET4+896(%r10)
  vmovapd %ymm15,\OFFSET4+960(%r10)
.endm

.macro MASKSTORE_16_D32 OFFSET6
  vmaskmovpd %ymm0,%ymm0,\OFFSET6(%r10)
  vmaskmovpd %ymm1,%ymm1,\OFFSET6+32(%r10)
  vmaskmovpd %ymm2,%ymm2,\OFFSET6+64(%r10)
  vmaskmovpd %ymm3,%ymm3,\OFFSET6+96(%r10)
  vmaskmovpd %ymm4,%ymm4,\OFFSET6+128(%r10)
  vmaskmovpd %ymm5,%ymm5,\OFFSET6+160(%r10)
  vmaskmovpd %ymm6,%ymm6,\OFFSET6+192(%r10)
  vmaskmovpd %ymm7,%ymm7,\OFFSET6+224(%r10)
  vmaskmovpd %ymm8,%ymm8,\OFFSET6+256(%r10)
  vmaskmovpd %ymm9,%ymm9,\OFFSET6+288(%r10)
  vmaskmovpd %ymm10,%ymm10,\OFFSET6+320(%r10)
  vmaskmovpd %ymm11,%ymm11,\OFFSET6+352(%r10)
  vmaskmovpd %ymm12,%ymm12,\OFFSET6+384(%r10)
  vmaskmovpd %ymm13,%ymm13,\OFFSET6+416(%r10)
  vmaskmovpd %ymm14,%ymm14,\OFFSET6+448(%r10)
  vmaskmovpd %ymm15,%ymm15,\OFFSET6+480(%r10)
.endm

.section .text
//input_rdi=base_address_of_double_array,input_rsi=num_of_ops(int64)
.globl test_broadcast_d4
.type test_broadcast_d4,@function
test_broadcast_d4:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_broadcast_d4:
  incq %r11
  BROADCAST_16_D4 0
  BROADCAST_16_D4 64
  BROADCAST_16_D4 128
  BROADCAST_16_D4 192
  BROADCAST_16_D4 256
  BROADCAST_16_D4 320
  BROADCAST_16_D4 384
  BROADCAST_16_D4 448
  BROADCAST_16_D4 512
  BROADCAST_16_D4 576
  BROADCAST_16_D4 640
  BROADCAST_16_D4 704
  BROADCAST_16_D4 768
  BROADCAST_16_D4 832
  BROADCAST_16_D4 896
  BROADCAST_16_D4 960
  cmpq %rsi,%r11
  jb .Ltest_broadcast_d4
  vzeroupper
  retq

.section .text
//input_rdi=base_address_of_double_array,input_rsi=num_of_ops(int64)
.globl test_broadcast_d8
.type test_broadcast_d8,@function
test_broadcast_d8:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_broadcast_d8:
  incq %r11
  BROADCAST_16_D8 0
  BROADCAST_16_D8 128
  BROADCAST_16_D8 256
  BROADCAST_16_D8 384
  BROADCAST_16_D8 512
  BROADCAST_16_D8 640
  BROADCAST_16_D8 768
  BROADCAST_16_D8 896
  BROADCAST_16_D8 1024
  BROADCAST_16_D8 1152
  BROADCAST_16_D8 1280
  BROADCAST_16_D8 1408
  BROADCAST_16_D8 1536
  BROADCAST_16_D8 1664
  BROADCAST_16_D8 1792
  BROADCAST_16_D8 1920
  cmpq %rsi,%r11
  jb .Ltest_broadcast_d8
  vzeroupper
  retq

.section .text
//input_rdi=base_address_of_double_array,input_rsi=num_of_ops(int64)
.globl test_broadcast_d16
.type test_broadcast_d16,@function
test_broadcast_d16:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_broadcast_d16:
  incq %r11
  BROADCAST_16_D16 0
  BROADCAST_16_D16 256
  BROADCAST_16_D16 512
  BROADCAST_16_D16 768
  BROADCAST_16_D16 1024
  BROADCAST_16_D16 1280
  BROADCAST_16_D16 1536
  BROADCAST_16_D16 1792
  BROADCAST_16_D16 2048
  BROADCAST_16_D16 2304
  BROADCAST_16_D16 2560
  BROADCAST_16_D16 2816
  BROADCAST_16_D16 3072
  BROADCAST_16_D16 3328
  BROADCAST_16_D16 3584
  BROADCAST_16_D16 3840
  cmpq %rsi,%r11
  jb .Ltest_broadcast_d16
  vzeroupper
  retq

.globl test_maskload_d32
.type test_maskload_d32,@function
test_maskload_d32:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_maskload_d32:
  incq %r11
  MASKLOAD_16_D32 0
  MASKLOAD_16_D32 512
  MASKLOAD_16_D32 1024
  MASKLOAD_16_D32 1536
  MASKLOAD_16_D32 2048
  MASKLOAD_16_D32 2560
  MASKLOAD_16_D32 3072
  MASKLOAD_16_D32 3584
  MASKLOAD_16_D32 4096
  MASKLOAD_16_D32 4608
  MASKLOAD_16_D32 5120
  MASKLOAD_16_D32 5632
  MASKLOAD_16_D32 6144
  MASKLOAD_16_D32 6656
  MASKLOAD_16_D32 7168
  MASKLOAD_16_D32 7680
  cmpq %rsi,%r11
  jb .Ltest_maskload_d32
  vzeroupper
  retq

.globl test_loada_d32
.type test_loada_d32,@function
test_loada_d32:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_loada_d32:
  incq %r11
  LOADA_16_D32 0
  LOADA_16_D32 512
  LOADA_16_D32 1024
  LOADA_16_D32 1536
  LOADA_16_D32 2048
  LOADA_16_D32 2560
  LOADA_16_D32 3072
  LOADA_16_D32 3584
  LOADA_16_D32 4096
  LOADA_16_D32 4608
  LOADA_16_D32 5120
  LOADA_16_D32 5632
  LOADA_16_D32 6144
  LOADA_16_D32 6656
  LOADA_16_D32 7168
  LOADA_16_D32 7680
  cmpq %rsi,%r11
  jb .Ltest_loada_d32
  vzeroupper
  retq

.globl test_loadu_d32
.type test_loadu_d32,@function
test_loadu_d32:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_loadu_d32:
  incq %r11
  LOADU_16_D32 0
  LOADU_16_D32 512
  LOADU_16_D32 1024
  LOADU_16_D32 1536
  LOADU_16_D32 2048
  LOADU_16_D32 2560
  LOADU_16_D32 3072
  LOADU_16_D32 3584
  LOADU_16_D32 4096
  LOADU_16_D32 4608
  LOADU_16_D32 5120
  LOADU_16_D32 5632
  LOADU_16_D32 6144
  LOADU_16_D32 6656
  LOADU_16_D32 7168
  LOADU_16_D32 7680
  cmpq %rsi,%r11
  jb .Ltest_loadu_d32
  vzeroupper
  retq

.globl test_loada_d64
.type test_loada_d64,@function
test_loada_d64:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_loada_d64:
  incq %r11
  LOADA_16_D64 0
  LOADA_16_D64 1024
  LOADA_16_D64 2048
  LOADA_16_D64 3072
  LOADA_16_D64 4096
  LOADA_16_D64 5120
  LOADA_16_D64 6144
  LOADA_16_D64 7168
  LOADA_16_D64 8192
  LOADA_16_D64 9216
  LOADA_16_D64 10240
  LOADA_16_D64 11264
  LOADA_16_D64 12288
  LOADA_16_D64 13312
  LOADA_16_D64 14336
  LOADA_16_D64 15360
  cmpq %rsi,%r11
  jb .Ltest_loada_d64
  vzeroupper
  retq

.globl test_loadu_d64
.type test_loadu_d64,@function
test_loadu_d64:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_loadu_d64:
  incq %r11
  LOADU_16_D64 0
  LOADU_16_D64 1024
  LOADU_16_D64 2048
  LOADU_16_D64 3072
  LOADU_16_D64 4096
  LOADU_16_D64 5120
  LOADU_16_D64 6144
  LOADU_16_D64 7168
  LOADU_16_D64 8192
  LOADU_16_D64 9216
  LOADU_16_D64 10240
  LOADU_16_D64 11264
  LOADU_16_D64 12288
  LOADU_16_D64 13312
  LOADU_16_D64 14336
  LOADU_16_D64 15360
  cmpq %rsi,%r11
  jb .Ltest_loadu_d64
  vzeroupper
  retq

.globl test_storea_d32
.type test_storea_d32,@function
test_storea_d32:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_storea_d32:
  incq %r11
  STOREA_16_D32 0
  STOREA_16_D32 512
  STOREA_16_D32 1024
  STOREA_16_D32 1536
  STOREA_16_D32 2048
  STOREA_16_D32 2560
  STOREA_16_D32 3072
  STOREA_16_D32 3584
  STOREA_16_D32 4096
  STOREA_16_D32 4608
  STOREA_16_D32 5120
  STOREA_16_D32 5632
  STOREA_16_D32 6144
  STOREA_16_D32 6656
  STOREA_16_D32 7168
  STOREA_16_D32 7680
  cmpq %rsi,%r11
  jb .Ltest_storea_d32
  vzeroupper
  retq

.globl test_storeu_d32
.type test_storeu_d32,@function
test_storeu_d32:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_storeu_d32:
  incq %r11
  STOREU_16_D32 0
  STOREU_16_D32 512
  STOREU_16_D32 1024
  STOREU_16_D32 1536
  STOREU_16_D32 2048
  STOREU_16_D32 2560
  STOREU_16_D32 3072
  STOREU_16_D32 3584
  STOREU_16_D32 4096
  STOREU_16_D32 4608
  STOREU_16_D32 5120
  STOREU_16_D32 5632
  STOREU_16_D32 6144
  STOREU_16_D32 6656
  STOREU_16_D32 7168
  STOREU_16_D32 7680
  cmpq %rsi,%r11
  jb .Ltest_storeu_d32
  vzeroupper
  retq

.globl test_storea_d64
.type test_storea_d64,@function
test_storea_d64:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_storea_d64:
  incq %r11
  STOREA_16_D64 0
  STOREA_16_D64 1024
  STOREA_16_D64 2048
  STOREA_16_D64 3072
  STOREA_16_D64 4096
  STOREA_16_D64 5120
  STOREA_16_D64 6144
  STOREA_16_D64 7168
  STOREA_16_D64 8192
  STOREA_16_D64 9216
  STOREA_16_D64 10240
  STOREA_16_D64 11264
  STOREA_16_D64 12288
  STOREA_16_D64 13312
  STOREA_16_D64 14336
  STOREA_16_D64 15360
  cmpq %rsi,%r11
  jb .Ltest_storea_d64
  vzeroupper
  retq

.globl test_storeu_d64
.type test_storeu_d64,@function
test_storeu_d64:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_storeu_d64:
  incq %r11
  STOREU_16_D64 0
  STOREU_16_D64 1024
  STOREU_16_D64 2048
  STOREU_16_D64 3072
  STOREU_16_D64 4096
  STOREU_16_D64 5120
  STOREU_16_D64 6144
  STOREU_16_D64 7168
  STOREU_16_D64 8192
  STOREU_16_D64 9216
  STOREU_16_D64 10240
  STOREU_16_D64 11264
  STOREU_16_D64 12288
  STOREU_16_D64 13312
  STOREU_16_D64 14336
  STOREU_16_D64 15360
  cmpq %rsi,%r11
  jb .Ltest_storeu_d64
  vzeroupper
  retq

.globl test_maskstore_d32
.type test_maskstore_d32,@function
test_maskstore_d32:
  shrq $8,%rsi
  xorq %r11,%r11
  movq %rdi,%r10
.Ltest_maskstore_d32:
  incq %r11
  MASKSTORE_16_D32 0
  MASKSTORE_16_D32 512
  MASKSTORE_16_D32 1024
  MASKSTORE_16_D32 1536
  MASKSTORE_16_D32 2048
  MASKSTORE_16_D32 2560
  MASKSTORE_16_D32 3072
  MASKSTORE_16_D32 3584
  MASKSTORE_16_D32 4096
  MASKSTORE_16_D32 4608
  MASKSTORE_16_D32 5120
  MASKSTORE_16_D32 5632
  MASKSTORE_16_D32 6144
  MASKSTORE_16_D32 6656
  MASKSTORE_16_D32 7168
  MASKSTORE_16_D32 7680
  cmpq %rsi,%r11
  jb .Ltest_maskstore_d32
  vzeroupper
  retq
