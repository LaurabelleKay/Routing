#include <stdlib.h>
#include <vector>

#include "router.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#define MAX_SHM 1024
#define EMPTY -1
#define TRUE 1
#define FALSE 0

#define index(k, i, j) ((k) * (i) + (j))
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))


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
void leeMoore(
   int srcx, 
   int srcy, 
   int snkx,
   int snky,
   int rTop,
   int rBottom, 
   int rLeft,
   int rRight,
   int gridx, 
   int gridy,
   int *graph
   //int *route
)
{
   int bid = blockIdx.x;
   int x = threadIdx.x;
   int y = threadIdx.y;

   /*int srcx = sourcex[bid];
   int snkx = sinkx[bid];
   int srcy = sourcey[bid];
   int snky = sinky[bid];*/

   __shared__ int frontier[MAX_SHM];
   __shared__ int costs[MAX_SHM];
   __shared__ int tempCosts[MAX_SHM];
   __shared__ int done;

   //int rTop, int rBottom, int rLeft, int rRight; //Region boundaries

   int cost;

   //Make sure the region isn't out of bounds of the grid
   /*rRight = max(srcx, snkx) + 2;
   rRight = rRight >= gridx ? (gridx - 1) : rRight; 

   rLeft = min(srcx, snkx) - 2;
   rLeft = rLeft < 0 ? 0 : rLeft;

   rTop = min(srcy, nky) - 2;
   rTop = rTop < 0 ? 0 : rTop;

   rBottom = max(srcy, snky) + 2;
   rBottom = rBotton >= gridy ? (gridy - 1) : rBottom;*/

   int xg, yg; //Locations on the graph for this thread to read data from 
   xg = rLeft + x;
   yg = rTop + y;

   int dimx = rRight - rLeft;
   int dimy = rBottom - rTop;

   done = 0; //FIXME: Don't think this will work!
   int count = 0;

   //printf("%d\n", index(dimx, x, y));
   
   costs[index(dimy, x, y)] = 1000;
   tempCosts[index(dimy, x, y)] = 1E04;
   frontier[index(dimy, x, y)] = false;

   //Set the source as the frontier, and its cost as 0
   if(srcx - rLeft == x && srcy - rTop == y)
   {
      printf("%d - %d == %d | %d - %d == %d\n", srcx, rLeft, x, srcy, rTop, y);
      frontier[index(dimy, x, y)] = TRUE;
      costs[index(dimy, x, y)] = 0;
      tempCosts[index(dimy, x, y)] = 0;
   }

   //FIXME: The sink is set as an obstruction!!
   if(snkx - rLeft == x && snky - rTop == y)
   {
     printf("Sink reachable!\n");
   }

   while(count++ < 10000)
   {
      if(frontier[index(dimy, x, y)])
      {
         frontier[index(dimy, x, y)] = FALSE;

         

         //Assess top neighbour
         if(yg > rTop)
         {
            if(graph[gridy * xg + (yg - 1)] == EMPTY) //Check for an obstruction
            {
               cost = costs[index(dimy, x, y)] + 1;
               //We only want to replace the cost if it's lower
               if(cost < tempCosts[index(dimy, x, y - 1)])
               {
                  tempCosts[index(dimy, x, y - 1)] = cost;
               }

            }
         }

         //Assess bottom neighbour
         if(yg < rBottom)
         {
            if(graph[gridy * xg + (yg + 1)] == EMPTY)
            {
               cost = costs[index(dimy, x, y)] + 1;
               if(cost < tempCosts[index(dimy, x, y + 1)])
               {
                  tempCosts[index(dimy, x, y + 1)] = cost;
               }

            }
         }

         //Assess left neighbour
         if(xg < rRight)
         {
            if(graph[gridy * (xg + 1) + yg] == EMPTY)
            {
               cost = costs[index(dimy, x, y)] + 1;
               if(cost < tempCosts[index(dimy, x + 1, y)])
               {
                  tempCosts[index(dimy, x + 1, y)] = cost;
               }

            }
         }

         //Assess right neighbour
         if(xg > rLeft)
         {
            if(graph[gridy * (xg - 1) + yg] == EMPTY)
            {
               cost = costs[index(dimy, x, y)] + 1;
               if(cost < tempCosts[index(dimy, x - 1, y)])
               {
                  tempCosts[index(dimy, x - 1, y)] = cost;
               }

            }
         }
      }

      __syncthreads();
      
      if(costs[index(dimy, x, y)] > tempCosts[index(dimy, x, y)])
      {
         costs[index(dimy, x, y)] = tempCosts[index(dimy, x, y)];
         frontier[index(dimy, x, y)] = TRUE;
      }
      tempCosts[index(dimy, x, y)] = costs[index(dimy, x, y)];

       __syncthreads();
   }

   __syncthreads();
   //printf("%d\n", done);
}

void schedule(
   Point **points, 
   Wire *W, 
   vector<vector<pair<int,int>>>edges,
   vector<int> routeList,
   vector<BoundingBox> BB,
   int gridx,
   int gridy,
   int numEdges)
{
   int *graph;

   //TODO: Work out texture memory stuff if there's time
   gpuErrchk(cudaMallocManaged(&graph, gridx * gridy * sizeof(int)));
   
   gridToGraph(points, graph, gridx, gridy);

  /* vector<int> sourcex;
   vector<int> sourcey;
   vector<int> sinkx;
   vector<int> sinky;

   int val;

   for(unsigned int i = 0; i < routeList.size(); i++)
   {
      for(unsigned int j = 0; j < edges[i].size(); j++)
      {
         val = W[i].pins[edges[i].first][0];
         sourcex.push_back(val);

         val = W[i].pins[edges[i].second][0];
         sinkx.push_back(val);

         val = W[i].pins[edges[i].first][1];
         sourcey.push_back(val);

         val = W[i].pins[edges[i].second][1];
         sinky.push_back(val);
      }
   }*/

   int srcx, srcy, snkx, snky;

   cudaStream_t *streams = (cudaStream_t *)malloc(numEdges * sizeof(cudaStream_t));

   int rTop; int rBottom; int rLeft; int rRight; //Region boundaries

   int s = 0;

    for(unsigned int i = 0; i < 1/*routeList.size()*/; i++)
   {
      for(unsigned int j = 0; j < 1/*edges[i].size()*/; j++)
      {
         srcx = W[i].pins[edges[i][j].first][0];
         snkx = W[i].pins[edges[i][j].second][0];
         srcy = W[i].pins[edges[i][j].first][1];
         snky = W[i].pins[edges[i][j].second][1];

         rRight = max(srcx, snkx) + 2;
         rRight = rRight >= gridx ? (gridx - 1) : rRight; 

         rLeft = min(srcx, snkx) - 2;
         rLeft = rLeft < 0 ? 0 : rLeft;

         rTop = min(srcy, snky) - 2;
         rTop = rTop < 0 ? 0 : rTop;

         rBottom = max(srcy, snky) + 2;
         rBottom = rBottom >= gridy ? (gridy - 1) : rBottom;

         int dimx = rRight - rLeft;
         int dimy = rBottom - rTop;

         gpuErrchk(cudaStreamCreate(&(streams[s])));
         
         dim3 dimBlock(dimx, dimy);
         printf("rTop: %d, rBottom: %d, rLeft: %d, rRight: %d\n", rTop, rBottom, rLeft, rRight);
         printf("dx: %d, dy: %d\n", dimx, dimy);
         printf("src: (%d, %d)  snk: (%d, %d)\n", srcx, srcy, snkx, snky);

         //TODO: figure out how to store the route
         //TODO: Also need to return if the routing was successful
         leeMoore<<<1, dimBlock, 0, streams[s++]>>>(srcx, srcy, snkx, snky, 
                                                   rTop, rBottom, rLeft, rRight, 
                                                   gridx, gridy, graph);

         gpuErrchk(cudaPeekAtLastError());
      }
   }

   gpuErrchk(cudaDeviceSynchronize());

   /*int *srcx = &sourcex[0];
   int *srcy = &sourcey[0];
   int *snkx = &sinkx[0];
   int *snky = &snky[0];

   
   //Make sure the region isn't out of bounds of the grid
   rRight = max(srcx, snkx) + 2;
   rRight = rRight >= gridx ? (gridx - 1) : rRight; 

   rLeft = min(srcx, snkx) - 2;
   rLeft = rLeft < 0 ? 0 : rLeft;

   rTop = min(srcy, nky) - 2;
   rTop = rTop < 0 ? 0 : rTop;

   rBottom = max(srcy, snky) + 2;
   rBottom = rBotton >= gridy ? (gridy - 1) : rBottom;*/
   
}

void gridToGraph(Point **points, int *graph, int gridx, int gridy)
{
   for(int i = 0; i < gridx; i++)
   {
      for(int j = 0; j < gridy; j++)
      {
         if(points[i][j].obstructed == true)
         {
            graph[gridy * i + j] = 0;
         }
         else
         {
            graph[gridy * i + j] = EMPTY;
         }
      }
   }
}