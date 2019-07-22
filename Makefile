CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O3

default: DGEMM.so DGEMM_LARGEMEM.so

DGEMM.so: dgemm.c dgemm.S
	$(CC) $(CCFLAGS) $^ -o $@

DGEMM_LARGEMEM.so: dgemm_largemem.c dgemm.S
	$(CC) $(CCFLAGS) $^ -o $@

clean:
	rm -f DGEMM.so DGEMM_LARGEMEM.so

