#ifndef C_H
#define C_H

#include<vector>

struct BoundingBox
{
    int maxx;
    int minx;
    int maxy;
    int miny;
    int area;
};

struct Point
{
    int x;
    int y;

    int dist;

    int f, g, h;

    bool target;
    bool expanded;
    bool obstructed;
    int obstructedBy;

    struct Point *prev;
};

struct Wire
{
    int numPins;

    int **pins;
    bool *found;
    //vector<vector<int>> pins;
    //vector<int> found;

    int colour;
    int r, g, b;
    int weight;

    //A counter for the wires encountered by this wire
    int *counter;
   // vector<int> counter;

    //Keep track of if this wire has been ripped, and by whom
    int *rippedBy;
    //vector<int> rippedBy;

    int routed;

    //vector<Point*> route;
};

#endif