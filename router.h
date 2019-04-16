#ifndef R_H
#define R_H

#include "common.h"
using namespace::std;

void gridToGraph(Point **points, int *graph, int gridx, int gridy);
void ripUpReroute(
    int numWires,
    int *graph,
    int gridx,
    int gridy,
    vector<int> unsuccessful,
    Wire *W,
    vector<vector<pair<int, int>>> edges,
    vector<int> &reRoute,
    vector<BoundingBox> BB,
    vector<vector<int>> &dependencyList);
int expand(int rTop, int rBottom, int rRight, int rLeft, int numWires, int wireIndex, int *graph, int gridx, int gridy);
void ripUp(int wireIndex, int numPins, int **pins, int *graph, int gridx, int gridy);
void schedule(
    Point **points,
    Wire *W,
    vector<vector<pair<int, int>>> edges,
    vector<vector<int>> dependencyList,
    vector<int> routeList,
    vector<BoundingBox> BB,
    int gridx,
    int gridy,
    int numEdges,
    int numWires);

#endif