#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <set>
#include <utility>
#include <string>
#include <queue>
#include <algorithm>

#ifdef LIN
#include "display.h"
#include "tester.h"
#endif

#include "common.h"
#include "Router.h"
//#include "tester.h"
using namespace std;

bool final;

int main(int argc, char **argv)
{
    int gridx, gridy;
    int numCells;
    int numWires;
    int algorithm;

    if (argc < 3)
    {
        cout << "Invalid number of arguments\n";
        exit(-1);
    }

    ifstream infile;
    infile.open(argv[1]);
    if (!infile)
    {
        cerr << "Unable to open file";
        exit(1);
    }

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
        exit(-1);
    }

    //Read in cell information
    vector<vector<int>> cells(numCells);
    for (int i = 0; i < numCells; i++)
    {
        cells[i] = vector<int>(2);
        infile >> cells[i][0];
        infile >> cells[i][1];
    }

    infile >> numWires;
    vector<Wire> W(numWires);
    vector<vector<Point>> points(gridx);

    //Read in wire information
    for (int i = 0; i < numWires; i++)
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
    }

    init(points, W, gridx, gridy, numWires, numCells, cells);

    priority_queue<wPair, vector<wPair>, greater<wPair>> PQ; //min heap queue ordered by wire weight

    //Determine weights and insert into priority queue
    for (int i = 0; i < numWires; i++)
    {
        W[i].weight = calculateWeight(W[i], numWires);
        PQ.push(make_pair(W[i].weight, i));
    }

    int index;
    int ret;
    pair<int, int> top;
    int success = 0;
    int attempts = 0;
    vector<int> bestWeights(numWires);


    queue<int> unrouted;
    queue<int> backup; 

    final = true;
    for (int i = 0; i < numWires; i++)
    {
        top = PQ.top();
        index = top.second;

        if (algorithm == 0)
        {
            ret = LM(index, gridx, gridy, W, points);
        }
        else if (algorithm == 1)
        {
            ret = aStar(index, gridx, gridy, W, points);
        }
        else
        {
            cout << "Error\n";
        }

        if (ret == 0)
        {
            W[index].routed = true;
            success++;
        }
        else
        {
            W[index].routed = false;
            unrouted.push(index);
        }


        reset(points, gridx, gridy);
        PQ.pop();
    }
    
    if (success != numWires)
    {
        //Keep ripping up and re-routing while there are unrouted nets
        while(!unrouted.empty())
        {
            reset(points, gridx, gridy);
            index = unrouted.front();

            //Expand from the source
            expand(index, gridx, gridy, 0, W[index], points);

            //Expand from each unreached pin
            for (int j = 1; j < W[index].numPins; j++)
            {
                if (W[index].found[j] == false)
                {
                    expand(index, gridx, gridy, j, W[index], points);
                }
            }

            //Find the wire that causes the most obstructions
            int maxC = 0;
            int maxJ = 0;
            for (int j = 0; j < numWires; j++)
            {
                //Don't include the current wire when searching for the maximum
                auto it = find( W[index].rippedBy.begin(), W[index].rippedBy.end(), j ); //Check if wire(index) was previously ripped by wire(j)
                
                //If it hasn't been ripped, consider wire(j) for ripping, prevents a loop of 2 wires ripping each other up
                if(it == W[index].rippedBy.end())
                {
                    if (W[index].counter[j] > maxC && j != index)
                    {
                        maxC = W[index].counter[j];
                        maxJ = j;
                    }
                } 
            }

            cout << "Rip up: " << maxJ << endl;
            ripUp(W[maxJ], gridx, gridy); //Rip up the blocking wire
            W[maxJ].rippedBy.push_back(index);

            //If blocking wire was previously routed, it now isn't, add it to the queue
            if(W[maxJ].routed == true)
            {
                W[maxJ].routed = false;
                unrouted.push(maxJ);
            }

            cout << "For: " << index << endl;
            ripUp(W[index], gridx, gridy); //We need to also disard the path for this wire

            //reroute this net
            cout << "Reroute: " << index << endl;
            if (algorithm == 0)
            {
                ret = LM(index, gridx, gridy, W, points);
            }
            else if (algorithm == 1)
            {
                ret = aStar(index, gridx, gridy, W, points);
            }
            if(ret == 0)
            {
                W[index].routed = true;
                unrouted.pop();
            }
            //unrouted.pop();

            //Now attempt ot re-route any other un-routed (including the ripped up) nets
            backup = unrouted; //First backup our queue
            while(!unrouted.empty())
            {
                reset(points, gridx, gridy);
                int ind = unrouted.front();
                cout << "RR: " << ind << endl;
                if (algorithm == 0)
                {
                    ret = LM(ind, gridx, gridy, W, points);
                }
                else if (algorithm == 1)
                {
                    ret = aStar(ind, gridx, gridy, W, points);
                }
                if(ret == 0)
                {
                    backup.pop();
                    W[ind].routed = true;
                }
                unrouted.pop();
            }

            unrouted = backup; //Any that are successfully routed are removed, and those which weren't are still in the queue

            //We don't want to immediately reroute this wire after attempting to reroute earlier
            if(W[index].routed == false)
            {
                //unrouted.push(index);
            }
        }

        /*for (int i = 0; i < numWires; i++)
        {
            if (W[i].routed == false)
            {
                //Expand from the source
                expand(i, gridx, gridy, 0, W[i], points);

                //Expand from each unreached pin
                for (int j = 1; j < W[i].numPins; j++)
                {
                    if (W[i].found[j] == false)
                    {
                        expand(i, gridx, gridy, j, W[i], points);
                    }
                }

                //Find the wire that causes the most obstructions
                int maxC = 0;
                int maxJ = 0;
                for (int j = 0; j < numWires; j++)
                {
                    if (W[i].counter[j] > maxC && j != i)
                    {
                        maxC = W[i].counter[j];
                        maxJ = j;
                    }
                }

                cout << "Rip up: " << maxJ << endl;
                ripUp(W[maxJ], gridx, gridy); //Rip up the blocking wire
                W[maxJ].routed = false;
                cout << "Rip up: " << i << endl;
                ripUp(W[i], gridx, gridy);    // We need to also disard the path for this wire

                //reroute this net
                cout << "Reroute: " << i << endl;
                if (algorithm == 0)
                {
                    ret = LM(i, gridx, gridy, W, points);
                }
                else if (algorithm == 1)
                {
                    ret = aStar(i, gridx, gridy, W, points);
                }
                reset(points, gridx, gridy);

                //reroute the ripped up net
                cout << "Reroute: " << maxJ << endl;
                if (algorithm == 0)
                {
                    ret = LM(maxJ, gridx, gridy, W, points);
                }
                else if (algorithm == 1)
                {
                    ret = aStar(maxJ, gridx, gridy, W, points);
                }
            }
        }*/
    }

    success = 0;
    for(int i = 0; i < numWires; i++)
    {
        if(W[i].routed == true)
        {
            success ++;
        }

    }


    cout << success << "/" << numWires << "\n";
    cout << "Routing done\n";

    return 0;
}

//Initialises the search space
void init(vector<vector<Point>> &points, vector<Wire> &W, int gridx, int gridy, int numWires, int numCells, vector<vector<int>> cells)
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
}

/*Calculates the weight for net ordering, less dispersed nets with a small 
number of pins are routed before very scattered nets with a large number of pins*/
int calculateWeight(Wire W, int nw)
{
    int np = W.numPins;
    int d = 0;

    for (int i = 1; i < np; i++)
    {
        d += distanceTo(W.pins[0], W.pins[i]);
    }
    return -d;
}

//Calculates the nearest target pin to a point in the grid
Point nearest(Point a, vector<Point> dests, vector<bool> skip)
{
    int d = 1E06;
    Point near;
    for (unsigned int i = 0; i < dests.size(); i++)
    {
        if (skip[i] == true)
        {
            continue;
        }
        if (d > distanceTo(a, dests[i]))
        {
            d = distanceTo(a, dests[i]);
            near = dests[i];
        }
    }
    return near;
}

//Routes the wire specified by 'index' using the Lee Moore Algorithm
int LM(int index, int gx, int gy, vector<Wire> &W, vector<vector<Point>> &points)
{
    int n = W[index].numPins;
    int targets = n - 1;
    int found = 0;
    int sx = W[index].pins[0][0];
    int sy = W[index].pins[0][1];
    queue<Point> Q;
    Q.push(points[sx][sy]); //FIFO queue
    points[sx][sy].expanded = true;
    Point current;
    bool stop = false;
    int x, y, np;
    np = W[index].numPins;

    for (int i = 0; i < np; i++)
    {
        x = W[index].pins[i][0];
        y = W[index].pins[i][1];
        points[x][y].obstructed = false;
    }

    while (!Q.empty())
    {
        current = Q.front();
        for (int i = 1; i < n; i++)
        {
            if (W[index].pins[i][0] == current.x && W[index].pins[i][1] == current.y)
            {
                found++;
                W[index].found[i] = true;
                trace(points, W[index], targets, current.x, current.y, gx, gy, index);
                if (found == targets)
                {
                    stop = true;
                }
                break;
            }
        }
        if (stop)
        {
            break;
        }

        if (current.y > 0)
        {
            if (points[current.x][current.y - 1].obstructed == false && points[current.x][current.y - 1].expanded == false)
            {
                points[current.x][current.y - 1].dist = points[current.x][current.y].dist + 1;
                points[current.x][current.y - 1].expanded = true;
                points[current.x][current.y - 1].prev = &points[current.x][current.y];
                Q.push(points[current.x][current.y - 1]);
            }
        }

        if (current.y < gy - 1)
        {
            if (points[current.x][current.y + 1].obstructed == false && points[current.x][current.y + 1].expanded == false)
            {
                points[current.x][current.y + 1].dist += points[current.x][current.y].dist + 1;
                points[current.x][current.y + 1].expanded = true;
                points[current.x][current.y + 1].prev = &points[current.x][current.y];
                Q.push(points[current.x][current.y + 1]);
            }
        }

        if (current.x < gx - 1)
        {
            if (points[current.x + 1][current.y].obstructed == false && points[current.x + 1][current.y].expanded == false)
            {
                points[current.x + 1][current.y].dist += points[current.x][current.y].dist + 1;
                points[current.x + 1][current.y].expanded = true;
                points[current.x + 1][current.y].prev = &points[current.x][current.y];
                Q.push(points[current.x + 1][current.y]);
            }
        }

        if (current.x > 0)
        {
            if (points[current.x - 1][current.y].obstructed == false && points[current.x - 1][current.y].expanded == false)
            {
                points[current.x - 1][current.y].dist += points[current.x][current.y].dist + 1;
                points[current.x - 1][current.y].expanded = true;
                points[current.x - 1][current.y].prev = &points[current.x][current.y];
                Q.push(points[current.x - 1][current.y]);
            }
        }
        Q.pop();
    }
    if (found != targets)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

//Routes the wire specified by 'index' using the A* algorithm
int aStar(int index, int gx, int gy, vector<Wire> &W, vector<vector<Point>> &points)
{
    int n = W[index].numPins;
    int targets = n - 1;
    int found = 0;
    int sx = W[index].pins[0][0];
    int sy = W[index].pins[0][1];
    Point current;

    int nT = W[index].numPins - 1;
    vector<Point> dests(nT); //Holds each target pin for the current wire

    int x, y, h, f, g;

    for (int i = 0; i < nT; i++)
    {
        x = W[index].pins[i + 1][0];
        y = W[index].pins[i + 1][1];
        dests[i] = points[x][y];
    }

    int np = W[index].numPins;

    //Set the source and sinks of this pin as not obstructed to allow for expansion
    for (int i = 0; i < np; i++)
    {
        x = W[index].pins[i][0];
        y = W[index].pins[i][1];
        points[x][y].obstructed = false;
    }

    bool stop = false;
    priority_queue<Point, vector<Point>, CompareF> PQ; //min heap priority queue, sorted by f

    vector<bool> skip(targets);
    for (int i = 1; i < n; i++)
    {
        skip[i] = false;
    }

    /*while (!PQ.empty())
    {
        PQ.pop();
    }*/

    points[sx][sy].f = 0;
    points[sx][sy].expanded = true;
    PQ.push(points[sx][sy]);

    Point near;
    while (!PQ.empty())
    {

        current = PQ.top();

        //Determine if this point is a target
        for (int i = 1; i < n; i++)
        {
            if (W[index].pins[i][0] == current.x && W[index].pins[i][1] == current.y)
            {
                found++;
                W[index].found[i] = true;
                skip[i - 1] = true; //Skip this destination in 'nearest' function
                trace(points, W[index], targets, current.x, current.y, gx, gy, index);
                if (found == targets)
                {
                    stop = true;
                }
                break;
            }
        }
        if (stop)
        {
            break;
        }

        if (current.y > 0) //Boundary Check
        {
            //Determine if the point is obstructed by a cell or another wire
            if ((points[current.x][current.y - 1].obstructedBy == -2 || points[current.x][current.y - 1].obstructedBy == index) && points[current.x][current.y - 1].expanded == false)
            {
                g = points[current.x][current.y].dist + 1;
                near = nearest(points[current.x][current.y - 1], dests, skip);
                h = distanceTo(points[current.x][current.y - 1], near);
                f = h + g;

                points[current.x][current.y - 1].dist = g;
                points[current.x][current.y - 1].f = f;
                points[current.x][current.y - 1].expanded = true;
                points[current.x][current.y - 1].prev = &points[current.x][current.y];
                PQ.push(points[current.x][current.y - 1]);
            }
        }

        if (current.y < gy - 1)
        {
            if ((points[current.x][current.y + 1].obstructedBy == -2 || points[current.x][current.y + 1].obstructedBy == index) && points[current.x][current.y + 1].expanded == false)
            {
                g = points[current.x][current.y].dist + 1;
                h = distanceTo(points[current.x][current.y + 1], nearest(points[current.x][current.y + 1], dests, skip));
                f = h + g;

                points[current.x][current.y + 1].dist = g;
                points[current.x][current.y + 1].f = f;
                points[current.x][current.y + 1].expanded = true;
                points[current.x][current.y + 1].prev = &points[current.x][current.y];
                PQ.push(points[current.x][current.y + 1]);
            }
        }

        if (current.x < gx - 1)
        {
            if ((points[current.x + 1][current.y].obstructedBy == -2 || points[current.x + 1][current.y].obstructedBy == index) && points[current.x + 1][current.y].expanded == false)
            {
                g = points[current.x][current.y].dist + 1;
                h = distanceTo(points[current.x + 1][current.y], nearest(points[current.x + 1][current.y], dests, skip));
                f = h + g;

                points[current.x + 1][current.y].dist = g;
                points[current.x + 1][current.y].f = f;
                points[current.x + 1][current.y].expanded = true;
                points[current.x + 1][current.y].prev = &points[current.x][current.y];
                PQ.push(points[current.x + 1][current.y]);
            }
        }

        if (current.x > 0)
        {
            if ((points[current.x - 1][current.y].obstructedBy == -2 || points[current.x - 1][current.y].obstructedBy == index) && points[current.x - 1][current.y].expanded == false)
            {
                g = points[current.x][current.y].dist + 1;
                h = distanceTo(points[current.x - 1][current.y], nearest(points[current.x - 1][current.y], dests, skip));
                f = h + g;

                points[current.x - 1][current.y].dist = g;
                points[current.x - 1][current.y].f = f;
                points[current.x - 1][current.y].expanded = true;
                points[current.x - 1][current.y].prev = &points[current.x][current.y];
                PQ.push(points[current.x - 1][current.y]);
            }
        }
        PQ.pop();
    }

    if (found != targets)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

//Calculates the manhattan distance between 2 points
int distanceTo(Point a, Point b)
{
    int d = abs(a.x - b.x) + abs(a.y - b.y);
    return d;
}

//Calculates the manhattan distance between 2 pins
int distanceTo(vector<int> a, vector<int> b)
{
    int d = abs(a[0] - b[0]) + abs(a[1] - b[1]);
    return d;
}

/*Traces the route back from the target sink to the source. If this is the final
 run, displays the trace on the GUI */
void trace(vector<vector<Point>> &points, Wire &W, int targets, int x, int y, int gx, int gy, int index)
{
    int d;
    vector<Point> R;
    Point current;
    current = points[x][y];
    d = current.dist;
    R = vector<Point>(d);
    vector<Point *> route(d);
    int i = 0;

    points[current.x][current.y].obstructed = true;
    points[current.x][current.y].obstructedBy = index;
    while (1)
    {
        current = *points[current.x][current.y].prev;
        R[i] = points[current.x][current.y];
        route[i] = &points[current.x][current.y];
        points[current.x][current.y].obstructed = true; //This wire is now an obstruction
        points[current.x][current.y].obstructedBy = index;

        if (current.prev == NULL)
        {
            break;
        }
        i++;
    }

    //W.route = vector<Point *>(d);
    W.route.insert(W.route.end(), route.begin(), route.end());
#ifdef LIN
    if (final)
    {
        drawRoute(R, gx, gy, index);
    }

#endif
}

//Resets the grid in preparation for another search by a different wire
void reset(vector<vector<Point>> &points, int gx, int gy)
{
    for (int i = 0; i < gx; i++)
    {
        for (int j = 0; j < gy; j++)
        {
            points[i][j].expanded = false;
            points[i][j].dist = 0;
            points[i][j].f = 0;
            points[i][j].prev = NULL;
        }
    }
}

/*Performs an expansion of the wire specified by 'index' from the point 
  specified by 'pin' using the Lee Moore  algorithm until the queue is empty.
  Keeps track of which wires are encountered during the expansion*/
void expand(int index, int gx, int gy, int pin, Wire &W, vector<vector<Point>> points)
{

    int sx = W.pins[pin][0];
    int sy = W.pins[pin][1];
    Point current;
    queue<Point> Q;
    Q.push(points[sx][sy]);
    points[sx][sy].expanded = true;

    while (!Q.empty())
    {
        current = Q.front();
        if (current.y > 0)
        {
            if ((points[current.x][current.y - 1].obstructedBy == -2 || points[current.x][current.y - 1].obstructedBy == index) && points[current.x][current.y - 1].expanded == false)
            {
                points[current.x][current.y - 1].dist = points[current.x][current.y].dist + 1;
                points[current.x][current.y - 1].expanded = true;
                points[current.x][current.y - 1].prev = &points[current.x][current.y];
                Q.push(points[current.x][current.y - 1]);
            }
            else if (points[current.x][current.y - 1].obstructedBy >= 0 && points[current.x][current.y - 1].obstructedBy != index)
            {
                W.counter[points[current.x][current.y - 1].obstructedBy]++;
            }
        }
        if (current.y < gy - 1)
        {
            if ((points[current.x][current.y + 1].obstructedBy == -2 || points[current.x][current.y + 1].obstructedBy == index) && points[current.x][current.y + 1].expanded == false)
            {
                points[current.x][current.y + 1].expanded = true;
                points[current.x][current.y + 1].prev = &points[current.x][current.y];
                Q.push(points[current.x][current.y + 1]);
            }
            else if (points[current.x][current.y + 1].obstructedBy >= 0 && points[current.x][current.y + 1].obstructedBy != index)
            {
                W.counter[points[current.x][current.y + 1].obstructedBy]++;
            }
        }

        if (current.x < gx - 1)
        {
            if ((points[current.x + 1][current.y].obstructedBy == -2 || points[current.x + 1][current.y].obstructedBy == index) && points[current.x + 1][current.y].expanded == false)
            {
                points[current.x + 1][current.y].expanded = true;
                points[current.x + 1][current.y].prev = &points[current.x][current.y];
                Q.push(points[current.x + 1][current.y]);
            }
            else if (points[current.x + 1][current.y].obstructedBy >= 0 && points[current.x + 1][current.y].obstructedBy != index)
            {
                W.counter[points[current.x + 1][current.y].obstructedBy]++;
            }
        }

        if (current.x > 0)
        {
            if ((points[current.x - 1][current.y].obstructedBy == -2 || points[current.x - 1][current.y].obstructedBy == index) && points[current.x - 1][current.y].expanded == false)
            {
                points[current.x - 1][current.y].expanded = true;
                points[current.x - 1][current.y].prev = &points[current.x][current.y];
                Q.push(points[current.x - 1][current.y]);
            }
            else if (points[current.x - 1][current.y].obstructedBy >= 0 && points[current.x - 1][current.y].obstructedBy != index)
            {
                W.counter[points[current.x - 1][current.y].obstructedBy]++;
            }
        }
        Q.pop();
    }
}

//Follow the route for this wire and remove its obstrucitons from the grid
void ripUp(Wire &W, int gx, int gy)
{
    Point *current;
    //cout << "Rip up: " << W. << endl;
    for (unsigned int i = 0; i < W.route.size(); i++)
    {
        current = W.route[i];
        current->obstructed = false;
        current->obstructedBy = -2;
    }
#ifdef LIN
    removeRoute(W.route, gx, gy);
#endif
    W.route.clear();
}