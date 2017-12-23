# Warnings
WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT		:= -O3

# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++11

# Linker options
LDOPT 	:= $(OPT)
LDFLAGS := 

# Names of executables to create
EXEC	:= problem1.exe problem2.exe

.DEFAULT_GOAL := all

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : ARCH :=
debug : $(EXEC)

EXEC = problem1.exe problem2.exe problem3.exe
all : Makefile $(EXEC)

%.o : %.c Makefile
	@ echo Compiling $<...
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	$(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(ARCH) $(CXXFLAGS) -c $< -o $@

# TODO: add targets for building executables
problem1.exe : problem1.cpp
	$(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(ARCH) $(CXXFLAGS) -o $@ $<
problem2.exe : problem2.c
	$(CC) $(CCSTD) $(WFLAGS) $(OPT) $(ARCH) $(CFLAGS) -o $@ $<
problem3.exe : problem3.cpp
	$(CXX) $(CXXSTD) $(WFLAGS) $(ARCH) $(CXXFLAGS) -g -o $@ $<

OBJS = problem1.o problem2.o problem3.o

.PHONY: clean
clean:
	@ rm -f *.o *.exe
