# Introduction:

This directory contains heavily-optimized AVX2 DGEMM and SGEMM codes dealing with large matrices (dimension: 3000~40000).
Function interface: FORTRAN, 32-bit integer.
The performance can be tuned via 4 parameters in Makefile: BlkDimK, BlkDimN, A_PR_BYTE, B_PR_ELEM.

# Parameters:

BlkDimK: the dimension K of packed matrix A and B, should be exactly divisible by 128. The dimension M of packed A is fixed to 96 bytes in current implementation. To place packed A in L1 cache, careful adjustment of this parameter is needed. Please note that the required memory bandwidth of accessing matrix C is inversely proportional to BlkDimK. Recommended setting: 0.4~0.8*(L1_size_in_bytes/96)

BlkDimN: the dimension N of packed matrix B, should be exactly divisible by 16. This parameter and BlkDimK control the size of packed matrix B, determining where packed B sits (L2/L3 cache).

A_PR_BYTE: the distance of prefetch from packed A, in bytes. As some elements of A could be evicted accidentally from L1 at runtime, the prefetch mechanism ensures that they come back to L1 in time before being accessed. So this parameter should be set according to the latency of L2 cache. For CPUs with 2 256-bit FMA units per core, the current implementation reads 16 bytes per cycle from packed A, the recommended setting is 1.5~2*(L2_latency_in_cycles*16).

B_PR_ELEM: the distance of prefetch from packed B, in elements(floats for SGEMM, doubles for DGEMM). For CPUs with 2 256-bit FMA units per core, the current implementation reads 2/3 element per cycle from packed B, the recommended setting is 1.5~3*(latency_of_the_cache_holding_packed_B_in_cycles*2/3).


# Tuned parameters for some processors:

Ryzen 7 3700X:
BlkDimK=256, A_PR_BYTE=256, B_PR_ELEM=24; 
BlkDimN: 256 for SGEMM.so, 192 for DGEMM.so
  
Core i9 9900K:
BlkDimK=256, A_PR_BYTE=256, B_PR_ELEM=64, BlkDimN=256
