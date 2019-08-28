CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O2
SRCFILE = src/gemm_kernel.S src/gemm_driver.c
INCFILE = src/gemm_kernel_irreg.c src/gemm_copy.c

default: DGEMM.so SGEMM.so

DGEMM.so: $(SRCFILE) $(INCFILE)
	$(CC) -DDOUBLE -DA_PREF=256 $(CCFLAGS) $(SRCFILE) -o $@
  
SGEMM.so: $(SRCFILE) $(INCFILE)
	$(CC) -DA_PREF=256 $(CCFLAGS) $(SRCFILE) -o $@

clean:
	rm -f *GEMM.so
