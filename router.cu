#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <queue>
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
            //printf("Sink found! Cost is: %d\n", costs[dimy * x + y]);
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

bool overlap(BoundingBox a, BoundingBox b)
{
    if((a.minx - 5) < (b.maxx + 5) && (a.maxx + 5) > (b.minx - 5)
     && (a.miny - 5) < (b.maxy + 5) && (a.maxy + 5) > (b.miny - 5))
    {
        return true;
    }
    return false;
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

   gpuErrchk(cudaMallocManaged(&graph, gridx * gridy * sizeof(int)));
   
   gridToGraph(points, graph, gridx, gridy);

   #ifdef DISPLAY
   drawGrid(gridx, gridy, graph, W);
   #endif

   int srcx, srcy, snkx, snky;
   int rTop; int rBottom; int rLeft; int rRight; //Region boundaries
   int s;

   vector<pair<int, int>> cpuRoute; //Store any nets than need to be routed on the cpu
   vector<int> successful(numWires); 
   vector<int> unsuccessful;
   vector<int> done(numWires);
   vector<int> reRoute;

   int numSuccessful = 0;
   int *success;
   int edgeIndex = 0;
   int attempts = 0;
   int regionBoundary = 2;

   while(numSuccessful < numWires && attempts < 2)
   {
      attempts++;
      //regionBoundary = attempts > 1 ? regionBoundary * 2 : regionBoundary;
       
      while(!routeList.empty())
      {
         s = 0;
         cudaStream_t *streams = (cudaStream_t *)malloc(numEdges * sizeof(cudaStream_t)); 
         gpuErrchk(cudaMallocManaged(&success, numEdges * sizeof(int)));

         edgeIndex = 0;
         for(unsigned int i = 0; i < routeList.size(); i++)
         {
            int ind = routeList[i];
            int area = BB[ind].area;

            for(unsigned int j = 0; j < edges[ind].size(); j++)
            {
               //Get source and sinx coordinates for this edge
               srcx = W[ind].pins[edges[ind][j].first][0];
               snkx = W[ind].pins[edges[ind][j].second][0];
               srcy = W[ind].pins[edges[ind][j].first][1];
               snky = W[ind].pins[edges[ind][j].second][1];

               //Use the source and sink to create a search region
               rRight = BB[ind].maxx + 5;
               rRight = rRight > gridx ? gridx - 1 : rRight;

               rLeft = BB[ind].minx - 5;
               rLeft = rLeft < 0 ? 0 : rLeft;

               rTop = BB[ind].miny - 5;
               rTop = rTop < 0 ? 0 : rTop;

               rBottom = BB[ind].maxy + 5;
               rBottom = rBottom > gridy ? gridy - 1 : rBottom;

               //Calculate block dimensions for the kernel base don region size
               int dimx = rRight - rLeft;
               int dimy = rBottom - rTop;

               gpuErrchk(cudaStreamCreate(&(streams[s])));
               
               dim3 dimBlock(dimx, dimy);
               if(dimx * dimy <= 1024)
               {
                  leeMoore<<<1, dimBlock, 0, streams[s++]>>>(srcx, srcy, snkx, snky, 
                                                         rTop, rBottom, rLeft, rRight, 
                                                         gridx, gridy, ind, 
                                                         edgeIndex, success, graph);
                  gpuErrchk(cudaPeekAtLastError());
               }
               else
               {
                  cpuRoute.push_back(make_pair(ind, edgeIndex));
               }
               edgeIndex++;
            }
         }

        gpuErrchk(cudaDeviceSynchronize()); 

         for(unsigned int i = 0; i < cpuRoute.size(); i++)
         {
            int ind = cpuRoute[i].first;
            edgeIndex = cpuRoute[i].second;
            
               srcx = W[ind].pins[edges[ind][i].first][0];
               snkx = W[ind].pins[edges[ind][i].second][0];
               srcy = W[ind].pins[edges[ind][i].first][1];
               snky = W[ind].pins[edges[ind][i].second][1];

               rRight = BB[ind].maxx + 5;
               rRight = rRight > gridx ? gridx - 1 : rRight;

               rLeft = BB[ind].minx - 5;
               rLeft = rLeft < 0 ? 0 : rLeft;

               rTop = BB[ind].miny - 5;
               rTop = rTop < 0 ? 0 : rTop;

               rBottom = BB[ind].maxy + 5;
               rBottom = rBottom > gridy ? gridy - 1 : rBottom;

               int dimx = rRight - rLeft;
               int dimy = rBottom - rTop;

               int ret  = LM(graph, gridx, gridy, rTop, rBottom, rRight, rLeft, srcx, srcy, snkx, snky, ind, edgeIndex);
               success[edgeIndex] = ret;

               //CPU Route
            
         }
         cpuRoute.clear();
         
         for(int i = 0; i < s; i++)
         {
            gpuErrchk(cudaStreamDestroy((streams[i])));
         }

         free(streams);
         
         edgeIndex = 0;
         for(unsigned int i = 0; i < routeList.size(); i++)
         {
            int ind = routeList[i];
            done[ind] = 1;
            successful[ind] = 1;
            for(unsigned int j = 0; j < edges[ind].size(); j++)
            {
               if(successful[ind] == 0)
               {
                  edgeIndex++; //Don't check any more edges, but we still need our index to increment
                  continue;
               }

               if(success[edgeIndex] == 0)
               {
                  W[ind].found[edges[ind][j].second] = -1; //The sink for this edge hasn't been found
                  successful[ind] = 0; //If any edges are unsucessful, the wire was unsuccessful
                  unsuccessful.push_back(ind);
               }
               edgeIndex++;
            }
         }
         
         gpuErrchk(cudaFree(success));

         vector<vector< std::vector<int>::iterator >> its(dependencyList.size());
         std::vector<int>::iterator it;
         for(unsigned int i = 0; i < dependencyList.size(); i++)
         {
            for(unsigned int j = 0; j < routeList.size(); j++)
            {
               it = std::find(dependencyList[i].begin(), dependencyList[i].end(), routeList[j]);
               if(it != dependencyList[i].end())
               {
                  int index = std::distance(dependencyList[i].begin(), it);
                  dependencyList[i][index] = 4096;
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
               if(i >= numWires)
               {
                  break;
               }
               routeList.push_back(i);
               numEdges += edges[i].size();
               count ++;
            }
         }
      }

      numSuccessful = 0;
      for(int i = 0; i < numWires; i++)
      {
         done[i] = 0;
         if(successful[i] == 1)
         {
            numSuccessful++;
         }
      }

      printf("successful: %d\n", numSuccessful);

      if(numSuccessful != numWires)
      {
         ripUpReroute(numWires, graph, gridx, gridy, unsuccessful, W, edges, reRoute, BB, dependencyList, &numSuccessful);
         int count = 0;

         dependencyList.clear();
         dependencyList = vector<vector<int>>(reRoute.size());
         BoundingBox B;
         for(unsigned int i = 0; i < reRoute.size(); i++)
         {
            if(reRoute[i] != -1)
            {
               B = BB[reRoute[i]];
               for(unsigned int j = 0; j < reRoute.size(); j++)
               {
                  if(i != j && reRoute[j] != -1)
                  {
                     if(overlap(B, BB[j]))
                     {
                        dependencyList[i].push_back(i);
                     }
                  }
               }
               reRoute[i] = -1;
            }
         }
         
         numEdges = 0;
         routeList.clear();
         for(int i = 0; i < numWires; i++)
         {
            if(dependencyList[i].empty())
            {
               routeList.push_back(i);
               numEdges += edges[i].size();
               count ++;
            }
         }
         unsuccessful.clear();
      }
      
      #ifdef DISPLAY
      //drawGrid(gridx, gridy, graph, W); 
      #endif
   }

   printf("Total attempts: %d\n", attempts);
   #ifdef DISPLAY
      drawGrid(gridx, gridy, graph, W); 
   #endif
}

int LM(
   int *graph,
   int gridx,
   int gridy,
   int rTop,
   int rBottom,
   int rRight,
   int rLeft,
   int srcx,
   int srcy,
   int snkx,
   int snky, 
   int wire,
   int edgeIndex
)
{
   vector<vector<int>> frontier(gridx);
   {
      for(int i = 0; i < gridx; i++)
      {
         frontier[i] = vector<int>(gridy);
      }
   }

   vector<vector<int>> costs(gridx);
   for(int i = 0; i < gridx; i++)
   {
      costs[i] = vector<int>(gridy, 1E06);
   }

   vector<vector<pair<int, int>>> from(gridx);
   for(int i = 0; i < gridy; i++)
   {

   }

   int done;
   int foundSink = 0;
   int count = 0;

   queue<pair<int, int>> Q;

   Q.push(make_pair(srcx, srcy));
   printf("rt: %d, rb %d, rl %d, rr %d\n", rTop, rBottom, rLeft, rRight);

   costs[srcx][srcy] = 0;
   frontier[srcx][srcy] = 1;

   int x, y;

   pair<int, int> current;

   while(!Q.empty())
   {
      current = Q.front();
      x = current.first;
      y = current.second;

      if(x == snkx && y == snky)
      {
         foundSink = 1;
         break;
      }

      //Assess top neighbour
      if(y > rTop)
      {
         if((graph[gridy * x + (y - 1)] == -2 || graph[gridy * x + (y - 1)] == wire) && frontier[x][y - 1] != 1)
         {
            costs[x][y - 1] = costs[x][y] + 1;
            frontier[x][y - 1] = 1;
            Q.push(make_pair(x, y - 1));
         }
      }

      if(y < rBottom - 1)
      {
         if((graph[gridy * x + (y + 1)] == -2 || graph[gridy * x + (y + 1)] == wire) && frontier[x][y + 1] != 1)
         {
            costs[x][y + 1] = costs[x][y] + 1;
            frontier[x][y + 1] = 1;
            Q.push(make_pair(x, y + 1));
         }
      }

      if(x > rLeft)
      {
         if((graph[gridy * (x - 1) + y] == -2 || graph[gridy * (x - 1) + y] == wire) && frontier[x - 1][y] != 1)
         {
            costs[x - 1][y] = costs[x][y] + 1;
            frontier[x - 1][y] = 1;
            Q.push(make_pair(x - 1, y));
         }
      }

      if(x < rRight - 1)
      {
         if((graph[gridy * (x + 1) + y] == -2 || graph[gridy * (x + 1) + y] == wire) && frontier[x + 1][y] != 1)
         {
            costs[x + 1][y] = costs[x][y] + 1;
            frontier[x + 1][y] = 1;
            Q.push(make_pair(x + 1, y));
         }
      }
      Q.pop();
   }

   while(!Q.empty())
   {
      Q.pop();
   }

   int success = 0;

   for(int i = 0; i < gridx; i++)
   {
        for(int j = 0; j < gridy; j++)
        {
           frontier[i][j] = 0;
        }
   }

   if(foundSink == 1)
   {
      success = 1;
      x = snkx;
      y = snky;
      int found = 0;
      while(!found)
      {
         //printf("%d, %d\n", x, y);
         if(x == srcx && y == srcy)
         {
            found = 1;
            break;
         }

         graph[gridy * x + y] = wire;

         if(y > rTop)
         {
            if(costs[x][y - 1] == costs[x][y] - 1)
            {
               y = y - 1;
               continue;
            }
         }

         if(y < rBottom)
         {
            if(costs[x][y + 1] == costs[x][y] - 1)
            {
               y = y + 1;
               continue;
            }
         }

         if(x > rLeft)
         {
            if(costs[x - 1][y] == costs[x][y] - 1)
            {
               x = x - 1;
               continue;
            }
         }

         if(x < rRight)
         {
            if(costs[x + 1][y] == costs[x][y] - 1)
            {
               x = x + 1;
               continue;
            }
         }
      }

   }
   return success;
}

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
   vector<vector<int>> &dependencyList,
   int *numSuccessful
)
{
   //vector<int> reRoute;
   int rTop, rBottom, rLeft, rRight;
   int srcx, srcy, snkx, snky;

   for(unsigned int i = 0; i < unsuccessful.size(); i++)
   {
      int ind = unsuccessful[i];
      for(unsigned int j = 0; j < W[ind].found.size(); j++)
      {
         if(W[ind].found[j] == -1)
         {
            for(int k = 0; k < edges[ind].size(); k++)
            {
               if(edges[ind][k].second == j)
               {
                  //Get source and sinx coordinates for this edge
                  srcx = W[ind].pins[edges[ind][k].first][0];
                  snkx = W[ind].pins[edges[ind][k].second][0];
                  srcy = W[ind].pins[edges[ind][k].first][1];
                  snky = W[ind].pins[edges[ind][k].second][1];

                  //Use the source and sink to create a search region
                  rRight = max(srcx, snkx) + 2;
                  rRight = rRight >= gridx ? (gridx - 1) : rRight; 

                  rLeft = min(srcx, snkx) - 2;
                  rLeft = rLeft < 0 ? 0 : rLeft;

                  rTop = min(srcy, snky) - 2;
                  rTop = rTop < 0 ? 0 : rTop;

                  rBottom = max(srcy, snky) + 2;
                  rBottom = rBottom >= gridy ? (gridy - 1) : rBottom;

                  int blockingWire = expand(rTop, rBottom, rRight, rLeft, numWires, ind, graph, gridx, gridy);
                  printf("Blocking wire for %d: %d\n", ind, blockingWire);

                  ripUp(blockingWire, W[blockingWire].numPins, W[blockingWire].pins, graph, gridx, gridy);
                  reRoute.push_back(ind);

                  if(blockingWire >= 0)
                  {
                     ripUp(ind, W[ind].numPins, W[ind].pins, graph, gridx, gridy);
                     reRoute.push_back(blockingWire);
                  }   
               }
            }
         }
      }
   }   
}

int expand(int rTop, int rBottom, int rRight, int rLeft, int numWires, int wireIndex, int *graph, int gridx, int gridy)
{
   vector<int> counter(numWires);
   for(int i = rLeft; i < rRight; i++)
   {
      for(int j = rTop; j < rBottom; j++)
      {
         if(graph[gridy * i + j] >= 0)
         {
            counter[graph[gridy * i + j]] ++;
         }
      }
   }

   int max = -5;
   int maxi = -1;
   for(int i = 0; i < numWires; i++)
   {
      if(i != wireIndex && counter[i] > max)
      {
         max = counter[i];
         maxi = i;
      }
   }
   return maxi;
}

void ripUp(int wireIndex, int numPins, int **pins, int *graph, int gridx, int gridy)
{

   //Set any cells that are obstructed by that wire to empty
   for(int i = 0; i < gridx; i++)
   {
      for(int j = 0; j < gridy; j++)
      {
         if(graph[gridy * i + j] == wireIndex)
         {
            graph[gridy * i + j] = EMPTY;
         }
      }
   }

   int x, y;

   //Set the pins to be obstructed
   for(int i = 0; i < numPins; i++)
   {
      x = pins[i][0];
      y = pins[i][1];
      graph[gridy * x + y] = wireIndex;
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