#ifndef R_H
#define R_H

#include "common.h"
using namespace::std;

void gridToGraph(Point **points, int *graph, int gridx, int gridy);
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