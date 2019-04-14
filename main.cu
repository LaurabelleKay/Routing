#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include "common.h"

using namespace std;

void init(Point **points, Wire *W, int gridx, int gridy, int numWires, int numCells, int **cells);

int main(int argc, char **argv)
{
    int gridx, gridy; //Grid sizes
    int numCells;
    int numWires;
    int algorithm;

    ifstream infile;
    infile.open(argv[1]);
    printf("Here\n");
    if(!infile)
    {
        printf("Here\n");
        cerr << "Unable to open file";
        exit(1);
    }
    printf("Here\n");
    infile >> gridx;
    infile >> gridy;
    infile >> numCells;

    string algo = argv[2];

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
        printf("ERROR");
        exit(-1);
    }

    //Read in cell information
    int **cells = new int*[numCells];
    for(int i = 0; i < numCells; i++)
    {
        cells[i] = new int[2];
        infile >> cells[i][0];
        infile >> cells[i][1];
    }

    /*vector<vector<int>> cells(numCells);
    for (int i = 0; i < numCells; i++)
    {
        cells[i] = vector<int>(2);
        infile >> cells[i][0];
        infile >> cells[i][1];
    }*/

    infile >> numWires;
    Wire *W = new Wire[numWires];
    Point **points = new Point*[gridx];

    for(int i = 0; i < numWires; i++)
    {
        infile >> W[i].numPins;
        W[i].colour = i + 4;
        W[i].pins = new int*[W[i].numPins];
        W[i].found = new bool[W[i].numPins];
        for(int j = 0; j < W[i].numPins; j++)
        {
            W[i].pins[j] = new int[2];
            infile >> W[i].pins[j][0];
            infile >> W[i].pins[j][1];
        }
    }

    /*vector<Wire> W(numWires);
    vector<vector<Point>> points(gridx);*/

    //Read in wire information
    /*for (int i = 0; i < numWires; i++)
    {
        infile >> W[i].numPins;
        W[i].colour = i + 4; //Colour offset to start colours form blue
        W[i].pins = vector<vector<int>>(W[i].numPins);
        W[i].found = vector<bool>(W[i].numPins);
        for (int j = 0; j < W[i].numPins; j++)
        {
            W[i].pins[j] = vector<int>(2);
            infile >> W[i].pins[j][0];
            infile >> W[i].pins[j][1];
        }
    }*/

    init(points, W, gridx, gridy, numWires, numCells, cells); //Initialise search space

    return 0;
}

__global__ void spanningTree(Wire *W)
{
    int id = threadIdx.x;
    int **adj = new int*[W[id].numPins];

    //Create an adjaceny matrix
    for(int i = 0; i < W[id].numPins; i++)
    {
        adj[i] = new int[W[id].numPins];
        for(int j = 0; j < W[id].numPins; j++)
        {
            // ? Could possibly launch another kernel from here to do this in parallel
           adj[i][j] =  abs(W[id].pins[i][0] - W[id].pins[j][0]) - abs(W[id].pins[i][1] - W[id].pins[j][1]);
        }
    }

    int numEdges = 0; // Counts the number of edges
    int *set = calloc(W[id].numPins, sizeof(int));
    sel[0] = true;

    int x, yl
    while(numEdges < W[id].numPins)
    {
        int min = 1E09;
        x = 0; y = 0;

        for(int i = 0; i < W[i].numPins; i++)
        {
            //If this pin is in our set
            if(set)
            {
                for(int j = 0; j < W[id].numPins; j++)
                {
                    //Compare with any pins not in the set, where an edge exists
                    if(!set[j] && adj[i][j])
                    {
                        //Find the smallest edge
                        if(adj[i][j] < min)
                        {
                            min = adj[i][j];
                            x = j;
                            y = k;
                        }
                    }
                }
            }
        }

        set[y] = true;
        numEdges++;
    }
}

void init(Point **points, Wire *W, int gridx, int gridy, int numWires, int numCells, int **cells)
{
    for(int i = 0; i < gridx; i++)
    {
        points[i] = new Point[gridy];
        for(int j = 0; j < gridy; j++)
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
//Initialises the search space
/*void init(vector<vector<Point>> &points, vector<Wire> &W, int gridx, int gridy, int numWires, int numCells, vector<vector<int>> cells)
{
    for (int i = 0; i < gridx; i++)
    {
        points[i] = vector<Point>(gridy);
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
        W[i].counter = vector<int>(numWires);
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
}*/