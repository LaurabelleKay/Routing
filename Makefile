EXE = router
EXE2 = routerDisplay
FLAGS = -Wall
SFMLPATH = "\Documents\Applications\SFML-2.5.1-windows-vc15-64-bit\SFML-2.5.1\include"
SFMLLIB = "\Documents\Applications\SFML-2.5.1-windows-vc15-64-bit\SFML-2.5.1\lib"

all: router.cu main.cpp
	nvcc router.cu main.cpp -o $(EXE)

display: router.cpp main.cpp display.cpp
	nvcc -D DISPLAY -I $(SFMLPATH) -L $(SFMLLIB) -lsfml-graphics -lsfml-window -lsfml-system router.cu main.cpp display.cpp -o $(EXE2)
