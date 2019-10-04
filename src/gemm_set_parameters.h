//this file should be included by gemm_driver.c and gemm_kernel.S

//read tuning parameters
# ifdef DOUBLE
 # include "../dgemm_tune.h"
# else
 # include "../sgemm_tune.h"
# endif

//GEMM_UNROLL_N: 2,3,4 or 6
# if GEMM_UNROLL_N < 2
 # undef GEMM_UNROLL_N
 # define GEMM_UNROLL_N 2
# endif
# if GEMM_UNROLL_N == 5
 # undef GEMM_UNROLL_N
 # define GEMM_UNROLL_N 4
# endif
# if GEMM_UNROLL_N > 6
 # undef GEMM_UNROLL_N
 # define GEMM_UNROLL_N 6
# endif

//restrict other parameters
# if GEMM_LOOP_TIMES_N < 1
 # undef GEMM_LOOP_TIMES_N
 # define GEMM_LOOP_TIMES_N 1
# endif
# if GEMM_LOOP_TIMES_N > 200
 # undef GEMM_LOOP_TIMES_N
 # define GEMM_LOOP_TIMES_N 200
# endif
# if GEMM_LOOP_TIMES_K < 1
 # undef GEMM_LOOP_TIMES_K
 # define GEMM_LOOP_TIMES_K 1
# endif
# if GEMM_LOOP_TIMES_K > 32
 # undef GEMM_LOOP_TIMES_K
 # define GEMM_LOOP_TIMES_K 32
# endif
# if PREF_CYCLES_PACKED_A < 1
 # undef PREF_CYCLES_PACKED_A
 # define PREF_CYCLES_PACKED_A 16
# endif
# if PREF_CYCLES_PACKED_B < 1
 # undef PREF_CYCLES_PACKED_B
 # define PREF_CYCLES_PACKED_B 64
# endif

//setting prefetch parameters, assuming 2x256bit FMA units per core
# define A_PR_BYTE (PREF_CYCLES_PACKED_A*64/GEMM_UNROLL_N)
# define B_PR_ELEM (PREF_CYCLES_PACKED_B*GEMM_UNROLL_N/6)

//setting common block dimensions
# define GEMM_BLOCK_DIM_N (GEMM_LOOP_TIMES_N*GEMM_UNROLL_N)
# define GEMM_BLOCK_DIM_K (GEMM_LOOP_TIMES_K*8*GEMM_UNROLL_N) //GEMM_UNROLL_LOOP_K = 8 in current implementation
# ifdef DOUBLE
 # define GEMM_BLOCK_DIM_M (48/GEMM_UNROLL_N) //12 ymm accumulators for current implementation, i.e. 48 doubles
# else
 # define GEMM_BLOCK_DIM_M (96/GEMM_UNROLL_N) //12 ymm accumulators for current implementation, i.e. 96 floats
# endif
