#include "common.h"

using namespace std;

void init(Point **points, Wire *W, int gridx, int gridy, int numWires, int numCells, int **cells);
void spanningTree(Wire *W, int numWires, vector<vector<pair<int, int>>> &edges);
BoundingBox boundingBox(Wire W);
bool hasOverlap(BoundingBox a, BoundingBox b);