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
)
{
   int x = threadIdx.x;
   int y = threadIdx.y;

   __shared__ int frontier[MAX_SHM];
   __shared__ int costs[MAX_SHM];
   __shared__ int done;

   int cost;

   int xg, yg; //Locations on the graph for this thread to read data from 
   xg = rLeft + x;
   yg = rTop + y;

   int dimx = rRight - rLeft;
   int dimy = rBottom - rTop;

   done = 0; 
   int count = 0;
   
   costs[dimy * x + y] = 1000;
   frontier[dimy * x + y] = FALSE;

   //Set the source as the frontier, and its cost as 0
   if(xg == srcx && yg == srcy)
   {
      frontier[dimy * x + y] = TRUE;
      costs[dimy * x + y] = 0;
   }

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

         if(xg == snkx && yg == snky)
         {
            foundSink = 1;
            done = 1;
            atomicAdd(&done, 1E06);
         }

         //Assess top neighbour
         if(y > 0)
         {
            //If the area is not obsructed, or is just obstructed by the same wire and it hasn't already been expanded, expand this area
            if((graph[gridy * xg + (yg - 1)] == -2 || graph[gridy * xg + (yg - 1)] == wire) && frontier[dimy * x  + (y - 1)] != EXPANDED) //Check for an obstruction
            {
               cost = costs[dimy * x + y] + 1;
               costs[dimy * x  + (y - 1)] = cost;
               frontier[dimy * x  + (y - 1)] = TRUE;
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
            }
         }

      }
   }

   __syncthreads();

   done = 0;

   //Sink to source route tracing, only the sink thread performs this
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
               //If they're 1 cost lower, go in that direction
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

   while(numSuccessful < numWires && attempts < 2)
   {
      attempts++;
       
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

               //Launch the kernel only if hte region will fit
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
                  cpuRoute.push_back(make_pair(ind, edgeIndex)); //Add to CPU's list otherwise
               }
               edgeIndex++;
            }
         }

        gpuErrchk(cudaDeviceSynchronize()); //Wait for all kernels to finish

        //Get the CPU to route the large ones
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

               int ret  = LM(graph, gridx, gridy, rTop, rBottom, rRight, rLeft, srcx, srcy, snkx, snky, ind, edgeIndex);
               success[edgeIndex] = ret;            
         }
         cpuRoute.clear();
         
         for(int i = 0; i < s; i++)
         {
            gpuErrchk(cudaStreamDestroy((streams[i])));
         }

         free(streams);
         
         edgeIndex = 0;

         //Determine which edges were successful (and therefore which net)
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
               //Find the routed wires in i's dependency list, if they are present, they can be removed since they're done
               it = std::find(dependencyList[i].begin(), dependencyList[i].end(), routeList[j]);
               if(it != dependencyList[i].end())
               {
                  int index = std::distance(dependencyList[i].begin(), it);
                  dependencyList[i][index] = 4096;
               }
            }
         }
         
         //Remove the routed wires from the dependency lists
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

         //Create a new routing list with the new dependency list
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

      #ifdef DISPLAY
      drawGrid(gridx, gridy, graph, W); 
      #endif

      }

      //Count the total number of successful routes
      numSuccessful = 0;
      for(int i = 0; i < numWires; i++)
      {
         done[i] = 0;
         if(successful[i] == 1)
         {
            numSuccessful++;
         }
      }

      //If not all were successful
      if(numSuccessful != numWires)
      {
         //Do the rip up and get the new list of to-be-routed nets
         ripUpReroute(numWires, graph, gridx, gridy, unsuccessful, W, edges, reRoute, BB, dependencyList, &numSuccessful);
         int count = 0;

         dependencyList.clear();
         dependencyList = vector<vector<int>>(reRoute.size());
         BoundingBox B;

         //Create dependencies based on this new routing order
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
         //Find the nets that can be routed concurrenlty and put them in the routing list
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
      drawGrid(gridx, gridy, graph, W); 
      #endif
   }

   #ifdef DISPLAY
      drawGrid(gridx, gridy, graph, W); 
   #endif
}

//For timing
void sequentialSchedule(
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
   int *graph = (int *)malloc(gridx * gridy * sizeof(int));
   gridToGraph(points, graph, gridx, gridy);

   int attempts = 0;
   int numSuccessful = 0;
   int edgeIndex = 0;

   vector<int> unsuccessful;
   vector<int> successful(numWires);
   vector<int> reRoute;

   int srcx, srcy, snkx, snky;
   int rTop; int rBottom; int rLeft; int rRight; //Region boundaries

   while(numSuccessful < numWires && attempts < 2)
   {
      attempts++;
      for(unsigned int i = 0; i < routeList.size(); i++)
      {
         int ind = routeList[i];
         successful[ind] = 1;
         for(unsigned int j = 0; j < edges[ind].size(); j++)
         {
            srcx = W[ind].pins[edges[ind][j].first][0];
            snkx = W[ind].pins[edges[ind][j].second][0];
            srcy = W[ind].pins[edges[ind][j].first][1];
            snky = W[ind].pins[edges[ind][j].second][1];

            rRight = BB[ind].maxx + 5;
            rRight = rRight > gridx ? gridx - 1 : rRight;

            rLeft = BB[ind].minx - 5;
            rLeft = rLeft < 0 ? 0 : rLeft;

            rTop = BB[ind].miny - 5;
            rTop = rTop < 0 ? 0 : rTop;

            rBottom = BB[ind].maxy + 5;
            rBottom = rBottom > gridy ? gridy - 1 : rBottom;

            
            int ret  = LM(graph, gridx, gridy, rTop, rBottom, rRight, rLeft, srcx, srcy, snkx, snky, ind, edgeIndex);
            if(ret == 0)
            {
               successful[ind] = 0; //If any edges are unsuccessful then this net is unsuccesful
            }
            edgeIndex++;
         }
         if(successful[ind] == 1)
         {
            numSuccessful++;
         }
         else
         {
            unsuccessful.push_back(ind);
         }
      }
      if(numSuccessful != numWires)
      {
         ripUpReroute(numWires, graph, gridx, gridy, unsuccessful, W, edges, reRoute, BB, dependencyList, &numSuccessful);
         unsuccessful.clear();
      }

      
   }

}

//Perform sequential Lee Moore
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

   int foundSink = 0;

   queue<pair<int, int>> Q;

   Q.push(make_pair(srcx, srcy));

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

//Go through the unsuccessfully routed nets and find the offending wire, rip this up and add them both to the routing list
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

                  //Perform an expansion
                  int blockingWire = expand(rTop, rBottom, rRight, rLeft, numWires, ind, graph, gridx, gridy);

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

//Performs a search of a region to find the wire that is most encountered
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