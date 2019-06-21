
#include "common.h"

#define BLOCK_DIM 256
__global__ void conflictsDetection_child_cdp(int *adjacentListD, int *colors, int *conflictD, int maxDegree, int idx, int i){
    int k = blockIdx.x*blockDim.x + threadIdx.x;

    if(k < maxDegree)
    {
        int j = adjacentListD[i*maxDegree + k];
        if (j>i && (colors[i] == colors[j]))
        {
            //conflictD[idx] = min(i,j)+1;	
            //conflictD[idx] = i+1;
            atomicMax(&conflictD[idx], i+1);	

            colors[i] = 0;				// added!!!!!!!!
        }		
    }

}

__global__ void conflictsDetection_cdp(int *adjacentListD, int *boundaryListD, int *colors, int *conflictD, long size, int boundarySize, int maxDegree){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i;


    if (idx < boundarySize){
        i = boundaryListD[idx];
        conflictD[idx] = 0;
        conflictsDetection_child_cdp<<<(int)ceil((float)maxDegree/BLOCK_DIM), BLOCK_DIM>>>(adjacentListD, colors, conflictD, maxDegree, idx, i);
    }

}

void launch_kernel(unsigned int dimGrid_confl, unsigned int dimBlock_confl, int *adjacentListD, int *boundaryListD, int *colors, int *conflictD, long size, int boundarySize, int maxDegree) {
    conflictsDetection_cdp<<<dimGrid_confl, dimBlock_confl>>>(adjacentListD, boundaryListD, colors, conflictD, size, boundarySize, maxDegree);
}

