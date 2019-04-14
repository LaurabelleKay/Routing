EXE = test
FLAGS = -Wall

all: test.cu main.cpp
	nvcc router.cu main.cpp -o $(EXE)
