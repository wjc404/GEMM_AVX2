CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O2

CONFIG_FILES = dgemm_tune.h sgemm_tune.h src/gemm_set_parameters.h
KERNEL_PREFIX = src/kernel/gemm_kernel
KERNEL_MACROS = $(KERNEL_PREFIX)_unroll2.S $(KERNEL_PREFIX)_unroll3.S $(KERNEL_PREFIX)_unroll4.S $(KERNEL_PREFIX)_unroll6.S
EDGE_PREFIX = src/edge_kernel/gemm_edge_kernel
EDGE_KERNELS = $(EDGE_PREFIX).c $(EDGE_PREFIX)_unroll2.h $(EDGE_PREFIX)_unroll3.h $(EDGE_PREFIX)_unroll4.h $(EDGE_PREFIX)_unroll6.h
SRCFILE = $(KERNEL_PREFIX).S src/gemm_driver.c
INCFILE = src/gemm_copy.c $(EDGE_KERNELS) $(KERNEL_MACROS)

default: DGEMM.so SGEMM.so

DGEMM.so: $(SRCFILE) $(INCFILE) $(CONFIG_FILES)
	$(CC) -DDOUBLE $(CCFLAGS) $(SRCFILE) -o $@
  
SGEMM.so: $(SRCFILE) $(INCFILE) $(CONFIG_FILES)
	$(CC) $(CCFLAGS) $(SRCFILE) -o $@

clean:
	rm -f *GEMM.so
  
