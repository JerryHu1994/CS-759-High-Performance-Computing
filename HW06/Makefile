# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3
ARCH   	:= -march=native

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := -lm -fopenmp
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
.DEFAULT_GOAL := all

EXEC := problem1A problem1B problem2 problem3
.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

all : Makefile $(EXEC)  


%.o : %.c Makefile
	@ echo Compiling $<...
	@ $(CC) $(CCSTD) $(WFLAGS) -Wno-unused-result $(OPT) $(CFLAGS) $(LDFLAGS) -c $< -o $@

problem1A: problem1A.c
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

problem1B: problem1B.c
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

problem2: problem2.cu
	module load cuda;nvcc -o problem2 $(OPT) problem2.cu -ccbin $(BIN)

problem3: problem3.cu
	module load cuda;nvcc -o problem3 $(OPT) problem3.cu -ccbin $(BIN)

# TODO: add targets for building executables


.PHONY: clean
clean:
	rm -f *.o *.exe *.dat $(EXEC)
