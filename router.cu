#include <stdlib.h>
#include <vector>

#include "router.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define MAX_SHM 1024
#define index(i, j) (MAX_SHM * i + j)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void LeeMoore(
   int sourcex, //! An array of N where N is number of regions
   int sourcey, 
   int sinkx,
   int sinky,
   int *graph,
   int *route
)
{
   int bid = blockIdx.x;
   int x = threadIdx.x;
   int y = threadIdx.y;

   __shared__ int frontier[MAX_SHM];
   __shared__ int costs[MAX_SHM];
   __shared__ int tempCosts[MAX_SHM];
   __shared__ int done;

   //TODO: Get the region min & max points
   //TODO: Use min & max to get offsets
   //TODO: load into shm
   
   while(!done)
   {
      if(frontier[index(x, y)])
      {
         frontier[index(x, y)] = 0;

         //TODO: for all neighbours
         // if (graph[n] ! obsructed)
         //    cost = costs[x, y] + 1
         //    if cost < tempCosts[n]
         //       tempCost[n] = cost
      }
      __syncthreads();

      //if(cost[id] > temp[id])
         //cost[id] = temp[id]
         //frontier[id] = true
         //done = 0
   }
}

void schedule(
   Point **points, 
   Wire *W, 
   vector<vector<pair<int,int>>>edges,
   vector<int> routeList,
   vector<BoundingBox> BB,
   int gridx,
   int gridy,
   int numWires)
{
   int *graph;

   //TODO: Work out texture memory stuff if there's time
   gpuErrchk(cudaMallocManaged(&graph, gridx * gridy * sizeof(int)));
   
   gridToGraph(points, graph, gridx, gridy);

   for(unsigned int i = 0; i < routeList.size(); i++)
   {

   }
   
}

void gridToGraph(Point **points, int *graph, int gridx, int gridy)
{
   for(int i = 0; i < gridx; i++)
   {
      for(int j = 0; j < gridy; j++)
      {
         if(points[i][j].obstructed == true)
         {
            graph[gridy * i + j] = 1;
         }
         else
         {
            graph[gridy * i + j] = -1;
         }
      }
   }
}