#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <set>
#include <SFML/Graphics.hpp>

#include "main.h"
#include "router.h"

using namespace std;

#define SFML_STATIC


int main(int argc, char **argv)
{
    /*sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");
    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.display();
    }*/

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
    
    printf("Initialising Search Space...\n");

    init(points, W, gridx, gridy, numWires, numCells, cells); //Initialise search space

    printf("Spanning Tree...\n");

    vector<vector<pair<int, int>>>edges(numWires);
    spanningTree(W, numWires, edges);

    priority_queue<pair<float, int>>PQ;

    vector<BoundingBox> BB(numWires);
    for(int i = 0; i < numWires; i++)
    {
        BB[i] = boundingBox(W[i]);
        PQ.push(make_pair(BB[i].area, i));
    }

    vector<vector<int>> dependencyList(numWires);
    BoundingBox B;

    while(!PQ.empty())
    {
        B = BB[PQ.top().second];
        for(int i = 0; i < numWires; i++)
        {
            if(i != PQ.top().second)
            {
                if(hasOverlap(B, BB[i]))
                {
                    dependencyList[i].push_back(PQ.top().second);
                }
            }
        }
        PQ.pop();
    }

    for (int i = 0; i < numWires; i++)
    {
        PQ.push(make_pair(BB[i].area, i));
    }

    //? find overlaps
    //launch kernels
    return 0;
}

bool hasOverlap(BoundingBox a, BoundingBox b)
{
    if(a.minx > b.maxx || b.maxx > a.minx)
    {
        return false;
    }
    if(a.miny < b.maxy || b.miny < a.maxy)
    {
        return false;
    }

    return true;
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
    
    printf("maxx: %d, minx: %d, maxy: %d, miny: %d\n", maxX, minX, maxY, minY);

    BB.area = (maxX - minX) * (maxY - minY);

    return BB;
}

void spanningTree(Wire *W, int numWires, vector<vector<pair<int, int>>> &edges)
{
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
        }
    }
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