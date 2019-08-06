.macro DelayChain r64
  incq \r64
  incq \r64
  incq \r64
  decq \r64
  decq \r64
  decq \r64
.endm

//enter the function ping-pong-exec, rdi=readaddr, rsi=writeaddr, rdx=finvalue
.globl pingpongexec
.type pingpongexec,@function
pingpongexec:
  movq $1,%r11
.Lloop:
  movq (%rdi),%r10
  DelayChain %r10
  cmpq %r11,%r10
  jb .Lloop
  incq %r10
  movq %r10,%r11
  movq %r10,(%rsi)
  cmpq %rdx,%r10
  jb .Lloop
  retq

