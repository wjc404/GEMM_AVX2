# Introduction

This folder contains AVX2 SGEMM and DGEMM codes based on 512bitx6column kernel (8x6 for DGEMM, 16x6 for SGEMM). The 1-thread performances are close to those from 768bitx4column kernel.

# Parameters in Makefile

BlkDimK should be exactly divisible by 48.

BlkDimN should be exactly divisible by 48.

A_PR_BYTE and B_PR_ELEM control prefetch distances.
