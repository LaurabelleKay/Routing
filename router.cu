#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include <set>
#include <queue>

__global__ void prims(Wire *W, int *e1, int *e2, int np)
{
    int id = threadIdx.x;
    
    //TODO: This doesn't work! 
    //TODO: just use an array that's the size of num_pins, can be free's later -_-
    int **adj = new int*[W[id].numPins];
    int it = 0;

    //Create an adjaceny matrix
    
    for(int i = 0; i < W[id].numPins; i++)
    {
        adj[i] = new int[W[id].numPins];
        for(int j = 0; j < W[id].numPins; j++)
        {
            // ? Could possibly launch another kernel from here to do this in parallel
           adj[i][j] =  abs(W[id].pins[i][0] - W[id].pins[j][0]) - abs(W[id].pins[i][1] - W[id].pins[j][1]);
        }
    }
    

    int numEdges = 0; // Counts the number of edges
    int *set = new int[W[id].numPins];
    set[0] = true;

    int x, y;
    while(numEdges < W[id].numPins)
    {
        int min = 1E09;
        x = 0; y = 0;

        for(int i = 0; i < W[i].numPins; i++)
        {
            //If this pin is in our set
            if(set)
            {
                for(int j = 0; j < W[id].numPins; j++)
                {
                    //Compare with any pins not in the set, where an edge exists
                    if(!set[j] && adj[i][j])
                    {
                        //Find the smallest edge
                        if(adj[i][j] < min)
                        {
                            min = adj[i][j];
                            x = i;
                            y = j;
                        }
                    }
                }
            }
        }
        printf("%d - %d : %d", x, y, adj[x][y]);
        e1[np * id + it] = x;
        e2[np * id + it] = y;
        it++;
        set[y] = true;
        numEdges++;
    }
}

int **spanningTree(Wire *W, int numWires, int numPins)
{
    int *e1 = (int *)malloc(numWires * numPins * sizeof(int));
    int *e2 = (int *)malloc(numWires * numPins * sizeof(int));

    int *e_k1;
    int *e_k2;
    cudaMalloc(&e_k1, numWires * numPins * sizeof(int));
    cudaMalloc(&e_k2, numWires * numPins * sizeof(int));

    cudaMemcpy(e_k1, e1, numWires * numPins * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e_k2, e2, numWires * numPins * sizeof(int), cudaMemcpyHostToDevice);
   
    printf("Spanning tree...\n");
    prims<<< 1, numWires>>>(W, e_k1, e_k2, numPins);
    cudaDeviceSynchronize();
    printf("Done!\n");

    cudaMemcpy(e1, e_k1, numWires * numPins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(e2, e_k2, numWires * numPins * sizeof(int), cudaMemcpyDeviceToHost);

    int **edges = (int **)malloc(2 * sizeof(int *));

    edges[0] = e1;
    edges[1] = e2;

    cudaFree(e_k1);
    cudaFree(e_k2);
    free(e1);
    free(e2);

    return edges;
}