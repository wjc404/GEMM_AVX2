# Introduction

This directory contains AVX2 SGEMM and DGEMM codes based on 512bitx6column kernel (8x6 for SGEMM, 16x6 for DGEMM). The performances are roughly 9/10 of those from 768bitx4column kernel.

# Parameters in Makefile

BlkDimK should be exactly divisible by 192.

BlkDimN should be exactly divisible by 48.

A_PR_BYTE and B_PR_ELEM control prefetch distances.
