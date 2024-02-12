# Makefile
CXX = g++
#CXX = clang++

CXXFLAGS += -g -fPIC $(shell root-config --cflags) -I.
#CXXFLAGS += -O3
CXXFLAGS += -I$(RATROOT)/include
LDFLAGS += -lstdc++ $(shell root-config --glibs)
LDFLAGS += -L$(RATROOT)/lib -lRATEvent -lrat 

convert: convert.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

all: convert

clean:
	rm -f *.o convert
