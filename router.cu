#include <stdlib.h>
#include <vector>

#include "router.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "display.h"

#define MAX_SHM 1024
#define EMPTY -2
#define TRUE 1
#define FALSE 0
#define EXPANDED -1

//#define index(k, i, j) ((k) * (i) + (j))
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
   int wire,
   int *graph
   //int *route
)
{
   int bid = blockIdx.x;
   int x = threadIdx.x;
   int y = threadIdx.y;

   __shared__ int frontier[MAX_SHM];
   __shared__ int costs[MAX_SHM];
   //__shared__ int tempCosts[MAX_SHM];
   __shared__ int from1[MAX_SHM];
   __shared__ int from2[MAX_SHM];
   __shared__ int done;

   //int rTop, int rBottom, int rLeft, int rRight; //Region boundaries

   int cost;

   int xg, yg; //Locations on the graph for this thread to read data from 
   xg = rLeft + x;
   yg = rTop + y;

   //printf("[%d][%d] = %d\n", xg, yg, graph[gridy * xg + yg]);

   int dimx = rRight - rLeft;
   int dimy = rBottom - rTop;

   if(x == 0 && y == 0)
   {
      for(int i = 0; i < dimx; i++)
      {
         for(int j = 0; j < dimy; j++)
         {
            printf("%d\t", graph[gridy * (rLeft + i) + (rBottom - j)]);
         }
         printf("\n");
      }
   }
   __syncthreads();

   //printf("[%d][%d] maps to [%d][%d]\n", x, y, xg, yg);

   done = 0; //FIXME: Don't think this will work!
   int count = 0;
   
   costs[dimy * x + y] = 1000;
   frontier[dimy * x + y] = FALSE;

   //Set the source as the frontier, and its cost as 0
   if(xg == srcx && yg == srcy)
   {
      frontier[dimy * x + y] = TRUE;
      costs[dimy * x + y] = 0;
   }

   from1[dimy * (x + 1) + y] = -30;
   from2[dimy * (x + 1) + y] = -30;

   //Source to sink propagation
   while(count++ < 50)
   {
      if(frontier[dimy * x + y] == TRUE)
      {
         frontier[dimy * x + y] = EXPANDED;
         //printf("[%d][%d]([%d][%d]) frontier expanded. Cost: %d\n", x, y, xg, yg, costs[dimy * x + y]);
         //printf("[%d][%d] from [%d][%d]\n", x, y, from1[dimy * x + y], from2[dimy * x + y]);
         done = 1;

         if(xg == snkx && yg == snky)
         {
            printf("Sink found! Cost is: %d\n", costs[dimy * x + y]);
         }

         //Assess top neighbour
         if(yg > 0)
         {
            if((graph[gridy * xg + (yg - 1)] == -2 || graph[gridy * xg + (yg - 1)] == wire) && frontier[dimy * x  + (y - 1)] != EXPANDED) //Check for an obstruction
            {
               cost = costs[dimy * x + y] + 1;
               costs[dimy * x  + (y - 1)] = cost;
               frontier[dimy * x  + (y - 1)] = TRUE;
               from1[dimy * x  + (y - 1)] = x;
               from2[dimy * x  + (y - 1)] = y;
               //printf("[%d][%d]([%d][%d]) expands [%d][%d]([%d][%d])\n", x, y, xg, yg, x, y - 1, xg, yg - 1);
            }
         }

         //Assess bottom neighbour
         if(yg < dimy - 1)
         {
            if((graph[gridy * xg + (yg + 1)] == -2 || graph[gridy * xg + (yg + 1)] == wire) && frontier[dimy * x  + (y  + 1)] != EXPANDED)
            {
               cost = costs[dimy * x + y] + 1;
               costs[dimy * x  + (y  + 1)] = cost;
               frontier[dimy * x  + (y  + 1)] = TRUE;
               from1[dimy * x  + (y  + 1)] = x;
               from2[dimy * x  + (y  + 1)] = y;
               //printf("[%d][%d]([%d][%d]) expands [%d][%d]([%d][%d])\n", x, y, xg, yg, x, y + 1, xg, yg + 1);
            }
         }

         //Assess left neighbour
         if(xg < dimx - 1)
         {
            if((graph[gridy * (xg + 1) + yg] == -2 || graph[gridy * (xg + 1) + yg] == wire) && frontier[dimy * (x + 1) + y] != EXPANDED)
            {
               cost = costs[dimy * x + y] + 1;
               costs[dimy * (x + 1) + y] = cost;
               frontier[dimy * (x + 1) + y] = TRUE;
               from1[dimy * (x + 1) + y] = x;
               from2[dimy * (x + 1) + y] = y;
               //printf("[%d][%d]([%d][%d]) expands [%d][%d]([%d][%d])\n", x, y, xg, yg, x + 1, y, xg + 1, yg);
            }  
         }

         //Assess right neighbour
         if(x > 0)
         {
            if((graph[gridy * (xg - 1) + yg] == -2 || graph[gridy * (xg - 1) + yg] == wire) && frontier[dimy * (x - 1) + y] != EXPANDED)
            {
               cost = costs[dimy * x + y] + 1;
               costs[dimy * (x - 1) + y] = cost;
               frontier[dimy * (x - 1) + y] = TRUE;
               from1[dimy * (x - 1) + y] = x;
               from2[dimy * (x - 1) + y] = y;
               //printf("[%d][%d]([%d][%d]) expands [%d][%d]([%d][%d])\n", x, y, xg, yg, x - 1, y, xg - 1, yg);
            }
         }

      }
      __syncthreads();
   }
   done = 0;
   //Sink to source route tracing
   if(snkx - rLeft == x && snky - rTop == y)
   {
      frontier[dimy * x + y] = 1;
   }
   while(!done)
   {
      if(frontier[dimy * x + y])
      {
         frontier[dimy * x + y] = FALSE;

         graph[gridy * xg + yg] = wire; //Obstruct this cell in the grid
         if(srcx - rLeft == x && srcy - rTop == y)
         {
            done = 1;
            break;
         }
         
         //Assess top neighbour
         if(y > 0)
         {
            if(costs[dimy * x  + (y - 1)] == costs[dimy * x + y] - 1)
            {
               frontier[dimy * x  + (y - 1)] = 1;
               continue;
            }
         }

         //Assess bottom neighbour
         if(y < dimy)
         {
            if(costs[dimy * x  + (y  + 1)] == costs[dimy * x + y] - 1)
            {
               frontier[dimy * x  + (y  + 1)] = 1;
               continue;
            }
         }

         //Assess left neighbour
         if(x < dimx)
         {
            if(costs[dimy * (x + 1) + y] == costs[dimy * x + y] - 1)
            {
               frontier[dimy * (x + 1) + y] = 1;
               continue;
            }
         }

         //Assess right neighbour
         if(x > 0)
         {
            if(costs[dimy * (x - 1) + y] == costs[dimy * x + y] - 1)
            {
               frontier[dimy * (x - 1) + y] = 1;
               continue;
            }
         }
      }
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
   int numEdges)
{
   int *graph;

   //TODO: Work out texture memory stuff if there's time
   gpuErrchk(cudaMallocManaged(&graph, gridx * gridy * sizeof(int)));
   
   gridToGraph(points, graph, gridx, gridy);

   drawGrid(gridx, gridy, graph, W);

   int srcx, srcy, snkx, snky;

   cudaStream_t *streams = (cudaStream_t *)malloc(numEdges * sizeof(cudaStream_t));

   int rTop; int rBottom; int rLeft; int rRight; //Region boundaries

   int s = 0;

    for(unsigned int i = 0; i < 1/*routeList.size()*/; i++)
   {
      int ind = routeList[i];
      for(unsigned int j = 0; j < 1/*edges[i].size()*/; j++)
      {
         srcx = W[ind].pins[edges[ind][j].first][0];
         snkx = W[ind].pins[edges[ind][j].second][0];
         srcy = W[ind].pins[edges[ind][j].first][1];
         snky = W[ind].pins[edges[ind][j].second][1];

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
                                                   gridx, gridy, ind, graph);

         gpuErrchk(cudaPeekAtLastError());
      }
   }

   gpuErrchk(cudaDeviceSynchronize());  
   drawGrid(gridx, gridy, graph, W); 
}

void gridToGraph(Point **points, int *graph, int gridx, int gridy)
{
   for(int i = 0; i < gridx; i++)
   {
      for(int j = 0; j < gridy; j++)
      {
         graph[gridy * i + j] = points[i][j].obstructedBy;
      }
   }
}