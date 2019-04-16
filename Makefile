EXE = test
FLAGS = -Wall
SFMLPATH = "D:\Laurabelle\Documents\Applications\SFML-2.5.1-windows-vc15-64-bit\SFML-2.5.1\include"
SFMLLIB = "D:\Laurabelle\Documents\Applications\SFML-2.5.1-windows-vc15-64-bit\SFML-2.5.1\lib"

all: router.cu main.cpp
	nvcc -I $(SFMLPATH) -L $(SFMLLIB) -lsfml-graphics -lsfml-window -lsfml-system router.cu main.cpp display.cpp -o $(EXE)