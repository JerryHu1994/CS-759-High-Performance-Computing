# Warnings
WFLAGS	:= -Wall -Wextra

# Optimization and architecture
OPT		:= -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
CUDA_LIB_PATH := /usr/local/cuda/lib64

.DEFAULT_GOAL := all

EXEC := problem1 problem2

all : Makefile $(EXEC)

problem1: problem1.cpp Makefile
	module load openmpi/2.1.1;mpicxx $(CXXSTD) $(WFLAGS) $(OPT) -fopenmp -o $@ $<

problem2: problem2.cpp problem2.cu problem2.hu Makefile 
	module load openmpi/2.1.1;module load cuda;nvcc -c -Xcompiler "$(CXXSTD) $(WFLAGS) $(OPT)" -o problem2_cuda.o problem2.cu -ccbin $(BIN) 
	module load openmpi/2.1.1;mpicxx -c $(CXXSTD) $(WFLAGS) $(OPT) -fopenmp -o problem2.o problem2.cpp
	module load openmpi/2.1.1;module load cuda;mpicxx problem2_cuda.o problem2.o -L$(CUDA_LIB_PATH) -lcudart -fopenmp -o problem2

.PHONY: clean
clean:
	@ rm -f $(EXEC) *.o
