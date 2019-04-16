#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <cctype>

#include "router.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#ifdef DISPLAY
#include "display.h"
#endif

#define MAX_SHM 1024
#define EMPTY -2
#define TRUE 1
#define FALSE 0
#define EXPANDED -1
#define MAX_ATTEMPTS 10

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

bool toDelete(int n)
{
   return (n >= 4096);
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
   int edgeIndex,
   int *success,
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
   if(x == 0 & y == 0)
   {
      success[edgeIndex] = 0;
   }

   __shared__ int foundSink;
   if(x == 0 && y == 0)
   {
      foundSink = 0;
   }
   
   //Source to sink propagation
   while(count++ < (dimx * dimy))
   {
      __syncthreads();
      if(foundSink == 1)
      {
         break;
      }
      
      if(frontier[dimy * x + y] == TRUE)
      {
         frontier[dimy * x + y] = EXPANDED;
         //printf("[%d][%d]([%d][%d]) frontier expanded. Cost: %d\n", x, y, xg, yg, costs[dimy * x + y]);
         //printf("[%d][%d] from [%d][%d]\n", x, y, from1[dimy * x + y], from2[dimy * x + y]);
         //atomicAdd(&done, 1);

         if(xg == snkx && yg == snky)
         {
            printf("Sink found! Cost is: %d\n", costs[dimy * x + y]);
            foundSink = 1;
            done = 1;
            atomicAdd(&done, 1E06);
            //printf("Done is now: %d\n", done);
         }

         //Assess top neighbour
         if(y > 0)
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
         if(y < dimy - 1)
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
         if(x < dimx - 1)
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
   }

   __syncthreads();

   //return;
   done = 0;

   //Sink to source route tracing
   int xx, yy;
   if(xg == snkx && yg == snky)
   { 
      if(foundSink == 1)
      {    
         success[edgeIndex] = 1;
         xx = x;
         yy = y;
         int found = 0;
         while(!found)
         {
            if(rLeft + xx == srcx && rTop + yy == srcy)
            {
               found = 1;
               break;
            }
         
            graph[gridy * (rLeft + xx) + (rTop + yy)] = wire;

            //Assess top neighbour
            if(yy > 0)
            {
               if(costs[dimy * xx  + (yy - 1)] == costs[dimy * xx + yy] - 1)
               {
                  yy = yy - 1;
                  continue;
               }
            }

            //Assess bottom neighbour
            if(y < dimy)
            {
               if(costs[dimy * xx  + (yy  + 1)] == costs[dimy * xx + yy] - 1)
               {
                  yy = yy + 1;
                  continue;
               }
            }

            //Assess left neighbour
            if(x < dimx)
            {
               if(costs[dimy * (xx + 1) + yy] == costs[dimy * xx + yy] - 1)
               {
                  xx = xx + 1;
                  continue;
               }
            }

            //Assess right neighbour
            if(x > 0)
            {
               if(costs[dimy * (xx - 1) + yy] == costs[dimy * xx + yy] - 1)
               {
                  xx = xx - 1;
                  continue;
               }
            }
         }
      }
   }
   __syncthreads();
   //printf("All here?\n");
   //TODO: Needs to return if the oruting was successful or not
}

void schedule(
   Point **points, 
   Wire *W, 
   vector<vector<pair<int,int>>>edges,
   vector<vector<int>> dependencyList,
   vector<int> routeList,
   vector<BoundingBox> BB,
   int gridx,
   int gridy,
   int numEdges,
   int numWires)
{
   int *graph;

   //TODO: Work out texture memory stuff if there's time
   gpuErrchk(cudaMallocManaged(&graph, gridx * gridy * sizeof(int)));
   
   gridToGraph(points, graph, gridx, gridy);

   #ifdef DISPLAY
   drawGrid(gridx, gridy, graph, W);
   #endif

   int srcx, srcy, snkx, snky;
   int rTop; int rBottom; int rLeft; int rRight; //Region boundaries
   int s;

   

   int numSuccessful = 0;
   int attempts = 0;

   //Attempt to route all nets
   while(!routeList.empty())
   {
      s = 0;
      cudaStream_t *streams = (cudaStream_t *)malloc(numEdges * sizeof(cudaStream_t)); 

      int *success;
      gpuErrchk(cudaMallocManaged(&success, numEdges * sizeof(int)));

      int edgeIndex = 0;

      for(unsigned int i = 0; i < routeList.size(); i++)
      {
         int ind = routeList[i];
         for(unsigned int j = 0; j < edges[ind].size(); j++)
         {
            
            //Get source and sinx coordinates for this edge
            srcx = W[ind].pins[edges[ind][j].first][0];
            snkx = W[ind].pins[edges[ind][j].second][0];
            srcy = W[ind].pins[edges[ind][j].first][1];
            snky = W[ind].pins[edges[ind][j].second][1];

            //Use the source and sink to create a search region
            rRight = max(srcx, snkx) + 2;
            rRight = rRight >= gridx ? (gridx - 1) : rRight; 

            rLeft = min(srcx, snkx) - 2;
            rLeft = rLeft < 0 ? 0 : rLeft;

            rTop = min(srcy, snky) - 2;
            rTop = rTop < 0 ? 0 : rTop;

            rBottom = max(srcy, snky) + 2;
            rBottom = rBottom >= gridy ? (gridy - 1) : rBottom;

            //Calculate block dimensions for the kernel base don region size
            int dimx = rRight - rLeft;
            int dimy = rBottom - rTop;

            gpuErrchk(cudaStreamCreate(&(streams[s])));
            
            dim3 dimBlock(dimx, dimy);
            printf("rTop: %d, rBottom: %d, rLeft: %d, rRight: %d\n", rTop, rBottom, rLeft, rRight);
            printf("dx: %d, dy: %d\n", dimx, dimy);
            printf("src: (%d, %d)  snk: (%d, %d)\n", srcx, srcy, snkx, snky);

            leeMoore<<<1, dimBlock, 0, streams[s++]>>>(srcx, srcy, snkx, snky, 
                                                      rTop, rBottom, rLeft, rRight, 
                                                      gridx, gridy, ind, 
                                                      edgeIndex, success, graph);

            gpuErrchk(cudaPeekAtLastError());
            edgeIndex++;
         }
      }

      gpuErrchk(cudaDeviceSynchronize());  

      for(int i = 0; i < s; i++)
      {
         gpuErrchk(cudaStreamDestroy((streams[i])));
      }

      free(streams);

      #ifdef DISPLAY
      drawGrid(gridx, gridy, graph, W); 
      #endif

      vector<int> successful(numWires);
      vector<int> unsucessful;
      vector<int> done(numWires);

      edgeIndex = 0;
      for(unsigned int i = 0; i < routeList.size(); i++)
      {
         ind = routeList[i];
         done[routeList[i]] = 1;
         successful[i] = 1;
         for(unsigned int j = 0; j < edges[ind].size(); j++)
         {
            if(success[edgeIndex] == 0)
            {
               W[i].found[edges[ind][j].second] = -1; //The sink for this edge hasn't been found
               successful[i] = 0; //If any edges are unsucessful, the wire was unsuccessful
               unsuccessful.push_back(i);
               edgeIndex++;
            }
         }
      }

      free(success);

      vector<vector< std::vector<int>::iterator >> its(dependencyList.size());
      std::vector<int>::iterator it;
      for(unsigned int i = 0; i < dependencyList.size(); i++)
      {
         for(unsigned int j = 0; j < routeList.size(); j++)
         {
            it = std::find(dependencyList[i].begin(), dependencyList[i].end(), routeList[j]);
            if(it != dependencyList[i].end())
            {
               dependencyList[i][*it] = 4096;
            }
         }
      }

      
      for(unsigned int i = 0; i < dependencyList.size(); i++)
      {
         if(dependencyList[i].empty())
         {
            continue;
         }
         dependencyList[i].erase(std::remove_if(dependencyList[i].begin(), dependencyList[i].end(), toDelete), dependencyList[i].end());
      }
      
      routeList.clear();
      numEdges = 0;

      int count = 0;
      for(unsigned int i = 0; i < dependencyList.size(); i++)
      {
         if(dependencyList[i].empty() && done[i] != 1)
         {
               routeList.push_back(i);
               numEdges += edges[i].size();
               count ++;
         }
      }

      printf("concurrent: %d\n", count);
   }

   if(numSuccessful != numWires)
   {
      printf("Not all routed!");
   }
  
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

void graphToGrid(Point **points, int *graph, int gridx, int gridy)
{
   for(int i = 0; i < gridx; i++)
   {
      for(int j = 0; j < gridy; j++)
      {
         points[i][j].obstructedBy = graph[gridy * i + j];
      }
   }
}