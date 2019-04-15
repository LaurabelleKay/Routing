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
//#include "display.h"

using namespace std;

#define SFML_STATIC

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
        filename = "benchmarks/kuma.txt";
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
        W[i].r = rand() % 255;
        W[i].g = rand() % 255;
        W[i].b = rand() % 255;
        W[i].pins = new int *[W[i].numPins];
        W[i].found = new bool[W[i].numPins];
        for (int j = 0; j < W[i].numPins; j++)
        {
            W[i].pins[j] = new int[2];
            infile >> W[i].pins[j][0];
            infile >> W[i].pins[j][1];
        }
    }
    
    printf("Initialising Search Space...\n");

    init(points, W, gridx, gridy, numWires, numCells, cells); //Initialise search space

    //drawGrid(gridx, gridy, points, W);

    printf("Spanning Tree...\n");

    vector<vector<pair<int, int>>>edges(numWires);
    int numEdges = spanningTree(W, numWires, edges);

    priority_queue<pair<int, int>>PQ;

    vector<BoundingBox> BB(numWires);
    for(int i = 0; i < numWires; i++)
    {
        BB[i] = boundingBox(W[i]);
        PQ.push(make_pair(BB[i].area, i));
    }

    vector<vector<int>> dependencyList(numWires);
    vector<bool> done(numWires);

    BoundingBox B;

    while(!PQ.empty())
    {
        B = BB[PQ.top().second];
        done[PQ.top().second] = true;
        for(int i = 0; i < numWires; i++)
        {
            if(i != PQ.top().second && !done[i])
            {
                if(hasOverlap(B, BB[i]))
                {
                    dependencyList[i].push_back(PQ.top().second);
                }
            }
        }
        PQ.pop();
    }

    int count = 0;
    vector<int> routeList;
    for(int i = 0; i < numWires; i++)
    {
        if(dependencyList[i].empty())
        {
            routeList.push_back(i);
            count ++;
        }
    }

    printf("Concurrent: %d\n", count);
   
   schedule(points, W, edges, routeList, BB, gridx, gridy, numWires);

    return 0;
}

bool hasOverlap(BoundingBox a, BoundingBox b)
{
    if(a.minx < b.maxx && a.maxx > b.minx && a.miny < b.maxy && a.maxy > b.miny)
    {
        return true;
    }
    return false;
}

//Calculate the bounding for for every pin in this net
BoundingBox boundingBox(Wire W)
{
    int maxX = -1E06;
    int minX = 1E06;
    int maxY = -1E06;
    int minY = 1E06;

    int mxxi = -1;
    int mnxi = -1;
    int mxyi = -1;
    int mnyi = -1;

    for(int i = 0; i < W.numPins; i++)
    {
        mnxi = W.pins[i][0] < minX ? i : mnxi;
        minX = W.pins[i][0] < minX ? W.pins[i][0] : minX;

        mxxi = W.pins[i][0] > maxX ? i : mxxi;
        maxX = W.pins[i][0] > maxX ? W.pins[i][0] : maxX;

        mnyi = W.pins[i][1] < minY ? i : mnyi;
        minY = W.pins[i][1] < minY ? W.pins[i][1] : minY;

        mxyi = W.pins[i][1] > maxY ? i : mxyi;
        maxY = W.pins[i][1] > maxY ? W.pins[i][1] : maxY;
    }

    BoundingBox BB;
    BB.maxx = maxX;
    BB.maxy = maxY;
    BB.minx = minX;
    BB.miny = minY;

    BB.area = (maxX - minX) * (maxY - minY);

    return BB;
}

int spanningTree(Wire *W, int numWires, vector<vector<pair<int, int>>> &edges)
{
    int count = 0;

    set<int> pins;
    vector<vector<vector<int>>> adj(numWires);

    for (int i = 0; i < numWires; i++)
    {
        adj[i] = vector<vector<int>>(W[i].numPins);
        for (int j = 0; j < W[i].numPins; j++)
        {
            adj[i][j] = vector<int>(W[i].numPins);
            for (int k = 0; k < W[i].numPins; k++)
            {
                adj[i][j][k] = abs(W[i].pins[j][0] - W[i].pins[k][0]) + abs(W[i].pins[j][1] - W[i].pins[k][1]);
            }
        }
    }

    for (int i = 0; i < numWires; i++)
    {
        vector<vector<int>> Adj = adj[i];
        int numEdge = 0;
        vector<bool> sel(W[i].numPins);
        sel[0] = true;

        int x, y;
        while (numEdge < W[i].numPins - 1)
        {
            int min = 1E09;
            x = 0;
            y = 0;

            for (int j = 0; j < W[i].numPins; j++)
            {
                if (sel[j])
                {
                    for (int k = 0; k < W[i].numPins; k++)
                    {
                        if (!sel[k] && Adj[j][k])
                        {
                            if (min > Adj[j][k])
                            {
                                min = Adj[j][k];
                                x = j;
                                y = k;
                            }
                        }
                    }
                }
            }
            edges[i].push_back(make_pair(x, y));
            sel[y] = true;
            numEdge++;
            count++;
        }
    }

    return count;
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