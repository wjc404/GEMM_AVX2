# Introduction

This directory contains AVX2 SGEMM and DGEMM codes based on 1024bitx3 kernel (SGEMM: 32x3; DGEMM: 16x3). The 1-thread performances are almost identical to previous implementations on 768bitx4 kernel.

# Parameters in Makefile

BlkDimK (the K-dim of packed A and B matrices) and BlkDimN (the N-dim of packed B matrix) should be exactly divisible by 96.

A_PR_BYTE and B_PR_ELEM control the prefetch distances of packed A and B.

# Tuned parameters

i9-9900K: BlkDimK=BlkDimN=192, A_PR_BYTE=256, B_PR_ELEM=48.

r7-3700X: BlkDimK=192, A_PR_BYTE=256, B_PR_ELEM=48; BlkDimN=192 for SGEMM, 96 for DGEMM. 

