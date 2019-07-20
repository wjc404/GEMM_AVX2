CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O3

DGEMM.so: dgemm.c dgemm.S
	$(CC) $(CCFLAGS) $^ -o $@

clean:
	rm -f DGEMM.so

