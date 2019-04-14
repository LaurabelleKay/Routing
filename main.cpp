#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <set>

#include "main.h"
#include "router.h"

using namespace std;

int main(int argc, char **argv)
{
    int gridx, gridy; //Grid sizes
    int numCells;
    int numWires;
    int algorithm;
    int maxPins = 0;
    char *filename;
    ifstream infile;

    if(argc < 3)
    {
        cout << "Invalid number of arguments" << endl;
        filename = "benchmarks/wavy.txt";
        algorithm = 0;
    }
    else
    {
        filename = argv[1];
    }
    

    
    infile.open(filename);
    if(!infile)
    {
        printf("Unable to open file\n");
        exit(1);
    }
    printf("File opened\n");

    infile >> gridx;
    infile >> gridy;
    infile >> numCells;

    /*string algo = argv[2];

    if (algo.compare("LM") == 0 || algo.compare("Lee Moore") == 0)
    {
        algorithm = 0;
    }
    else if (algo.compare("A") == 0 || algo.compare("A*") == 0)
    {
        algorithm = 1;
    }
    else
    {
        cout << "Invalid Algorithm\n";
        exit(-1);
    }*/

    printf("Reading info...\n");
    //Read in cell information
    int **cells = new int *[numCells];
    for (int i = 0; i < numCells; i++)
    {
        cells[i] = new int[2];
        infile >> cells[i][0];
        infile >> cells[i][1];
    }

    infile >> numWires;
    Wire *W = new Wire[numWires];
    Point **points = new Point *[gridx];

    for (int i = 0; i < numWires; i++)
    {
        infile >> W[i].numPins;
        maxPins = maxPins < W[i].numPins ? W[i].numPins : maxPins;
        W[i].colour = i + 4;
        W[i].pins = new int *[W[i].numPins];
        W[i].found = new bool[W[i].numPins];
        for (int j = 0; j < W[i].numPins; j++)
        {
            W[i].pins[j] = new int[2];
            infile >> W[i].pins[j][0];
            infile >> W[i].pins[j][1];
        }
    }
    
    printf("Init...\n");
    init(points, W, gridx, gridy, numWires, numCells, cells); //Initialise search space

    int **edges = spanningTree(W, numWires, maxPins);
    return 0;
}

void init(Point **points, Wire *W, int gridx, int gridy, int numWires, int numCells, int **cells)
{
    for (int i = 0; i < gridx; i++)
    {
        points[i] = new Point[gridy];
        for (int j = 0; j < gridy; j++)
        {
            points[i][j].x = i;
            points[i][j].y = j;
            points[i][j].dist = 0;
            points[i][j].target = false;
            points[i][j].expanded = false;
            points[i][j].obstructed = false;
            points[i][j].obstructedBy = -2;
            points[i][j].prev = NULL;
            points[i][j].f = 1E06;
        }
    }
    int x, y, np;
    //Set sinks and pins as obstructions
    for (int i = 0; i < numWires; i++)
    {
        np = W[i].numPins;
        W[i].counter = new int[numWires];
        for (int j = 0; j < np; j++)
        {
            x = W[i].pins[j][0];
            y = W[i].pins[j][1];
            points[x][y].obstructed = true;
            points[x][y].obstructedBy = i;
        }
        for (int j = 0; j < numWires; j++)
        {
            W[i].counter[j] = 0;
        }
    }

    for (int i = 0; i < numCells; i++)
    {
        x = cells[i][0];
        y = cells[i][1];
        points[x][y].obstructed = true;
        points[x][y].obstructedBy = -1;
    }
}