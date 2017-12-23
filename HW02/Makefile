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
LDFLAGS := 

CFLAGS	:= -lm -D_POSIX_C_SOURCE=199309L

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

EXEC = problem1a.exe problem1b.exe problem1c.exe problem2a.exe problem2b.exe problem3a.exe problem3b.exe problem3c.exe
OBJS = problem1a.o problem1b.o problem1c.o problem2a.o problem2b.o problem3a.o problem3b.o problem3c.o

all : Makefile $(EXEC)

%.o : %.c Makefile
	@ echo Compiling $<...
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	$(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(ARCH) $(CXXFLAGS) -c $< -o $@

# TODO: add targets for building executables
problem1a.exe : problem1a.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem1b.exe : problem1b.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem1c.exe : problem1c.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem2a.exe : problem2a.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem2b.exe : problem2b.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem3a.exe : problem3a.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem3b.exe : problem3b.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem3c.exe : problem3c.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
.PHONY: clean
clean:
	@ rm -f $(EXEC) $(OBJS) *.txt *.dat *.out 
