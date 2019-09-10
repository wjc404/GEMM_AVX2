# Introduction

This directory contains AVX2 SGEMM and DGEMM codes based on 1024bitx3 kernel (SGEMM: 32x3; DGEMM: 16x3). The 1-thread performances are close to previous implementations on 768bitx4 kernel.

# Parameters in Makefile

BlkDimK (the K-dim of packed A and B matrices) and BlkDimN (the N-dim of packed B matrix) should be exactly divisible by 96.

A_PR_BYTE and B_PR_ELEM control the prefetch distances of packed A and B.
