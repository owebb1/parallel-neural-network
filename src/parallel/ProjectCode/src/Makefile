#
# make: build buffer manager executable test program(s)
#       for our testing
#
# make clean: clean up all .o and executables built by this makefile
#

# path to top-level SwatDB directory (relative to this one)
SRCDIR = ./
INDIR = ./

# compiler
CC = mpic++

# compiler flags for test code build
CFLAGS = -g -std=c++11 -Werror -Wall -w -lstdc++

# include directories
INCLUDES = -I. -I$(INDIR)

# libraries needed to link in
LIBS = -lm -pthread

SRCS = neuralnet.cpp


# suffix replacement rule
OBJS = neuralnetparallel

# generic makefile
.PHONY: clean 

all: neuralnetparallel

neuralnetparallel: $(SRCS)
	$(CC) $(CFLAGS) $(INCLUDES) -o neuralnetparallel $(SRCS) $(LIBS)

# automatic variables: $< is the name of the prerequiste of the rule
# (.cpp file),  and $@ is name of target of the rule (.o file)
.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o $(OBJS) 
