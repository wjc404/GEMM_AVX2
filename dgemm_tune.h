// The size of packed A matrix is 3kB * GEMM_LOOP_TIMES_K, which should not succeeds the capacity of L1 cache.
// The value of PREF_CYCLES_PACKED_A should be greater than the latency of L2 cache (in cycles).
// The value of PREF_CYCLES_PACKED_B should be greater than the latency of L3 cache (in cycles).

# define GEMM_UNROLL_N 6
# define GEMM_LOOP_TIMES_N 16
# define GEMM_LOOP_TIMES_K 6
# define PREF_CYCLES_PACKED_A 24
# define PREF_CYCLES_PACKED_B 96

//# define GEMM_UNROLL_N 3
//# define GEMM_LOOP_TIMES_N 32
//# define GEMM_LOOP_TIMES_K 8
//# define PREF_CYCLES_PACKED_A 12
//# define PREF_CYCLES_PACKED_B 96
