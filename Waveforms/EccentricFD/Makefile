FC = gcc

ifeq ($(shell hostname), ekcosm)
	GSLPATH ?= /home/ekli/lib/gsl-2.2
else ifeq ($(shell hostname), ek)
	GSLPATH ?= /home/ekli/.local/lib/gsl-2.5
else ifeq ($(shell hostname), ekili-tq)
	GSLPATH ?= /opt/gsl-2.7.1
endif

GSLLIB ?= $(GSLPATH)/lib
GSLINC ?= $(GSLPATH)/include

SRC = ./source
LIB ?= ./lib

#FLAGS = -Wall -fpic -g -fopenmp -lm -I$(GSLINC) -I./include
FLAGS = -O3 -std=c99 -Wall -fpic -g -lm -I$(GSLINC) -I./
FFLAGS = -L$(GSLLIB) -lgsl -lgslcblas -L$(LIB)

OBJ_utils = $(LIB)/InspiralEccentricFDBasic.o \
			$(LIB)/InspiralOptimizedCoefficientsEccentricityFD.o \
			$(LIB)/InspiralEccentricFD.o

.PHONY: clean

default: main

all: main outlib

main: outlib $(OBJ_utils) $(LIB)/main.o
	$(FC) $(FLAGS) $(FFLAGS) $(OBJ_utils) $(LIB)/main.o -o $@

libEccFD: $(OBJ_utils)
	$(FC) -shared -fPIC $(FFLAGS) $(OBJ_utils) -o $@.so

$(LIB)/%.o: $(SRC)/%.c
	$(FC) $(FLAGS) $(FFLAGS) -c $(SRC)/$*.c -o $(LIB)/$*.o

outlib:
	@mkdir -p $(LIB)

clean:
	-rm -rf $(LIB)/*.o 
	-rm -rf $(LIB)/*.so
	-rm -rf *.so
	-rm -rf *.o
