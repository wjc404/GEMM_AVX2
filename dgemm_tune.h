// The size of packed A matrix is 3kB * GEMM_LOOP_TIMES_K, which should not exceed the capacity of L1 cache.
// The value of PREF_CYCLES_PACKED_A should be greater than the latency of L2 cache (in cycles).
// The value of PREF_CYCLES_PACKED_B should be greater than the latency of L3 cache (in cycles).

// Parameters tuned on AMD R7-3700X
# define GEMM_UNROLL_N 6
# define GEMM_LOOP_TIMES_N 16
# define GEMM_LOOP_TIMES_K 6
# define PREF_CYCLES_PACKED_A 18
# define PREF_CYCLES_PACKED_B 72

/* //Parameters tuned on i9 9900K 
# define GEMM_UNROLL_N 4
# define GEMM_LOOP_TIMES_N 64
# define GEMM_LOOP_TIMES_K 8
# define PREF_CYCLES_PACKED_A 16
# define PREF_CYCLES_PACKED_B 96
*/

/*
# define GEMM_UNROLL_N 3
# define GEMM_LOOP_TIMES_N 32
# define GEMM_LOOP_TIMES_K 8
# define PREF_CYCLES_PACKED_A 12
# define PREF_CYCLES_PACKED_B 64
*/

/*
# define GEMM_UNROLL_N 2
# define GEMM_LOOP_TIMES_N 96
# define GEMM_LOOP_TIMES_K 8
# define PREF_CYCLES_PACKED_A 12
# define PREF_CYCLES_PACKED_B 64
*/
