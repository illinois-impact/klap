
#include "common.h"

__global__ void BFS_child(unsigned int *levels, unsigned int *edgeArrayAux, int curr, int nbr_off, int num_nbr, int *flag) {
    int gid = blockDim.x*blockIdx.x + threadIdx.x;
    if(gid < num_nbr){
        int v = edgeArrayAux[gid + nbr_off];
        if(levels[v] == UINT_MAX) {
            levels[v] = curr + 1;
            *flag = 1;
        }
    }
}

__global__ void BFS_parent(
        unsigned int *levels,
        unsigned int *edgeArray,
        unsigned int *edgeArrayAux,
        unsigned int numVertices,
        int curr,
        int *flag)
{
    int gid = blockDim.x*blockIdx.x + threadIdx.x;
    if(gid < numVertices){
        if(levels[gid] == curr){
            unsigned int nbr_off = edgeArray[gid];
            unsigned int num_nbr = edgeArray[gid + 1] - nbr_off;
            BFS_child<<< (num_nbr - 1)/CHILD_BLOCK_SIZE + 1, CHILD_BLOCK_SIZE>>>(levels, edgeArrayAux, curr, nbr_off, num_nbr, flag);
        }
    }
}

void launch_kernel(unsigned int *d_costArray, unsigned int *d_edgeArray, unsigned int *d_edgeArrayAux, unsigned int numVerts, int iters, int* d_flag) {
    unsigned int numBlocks = (numVerts - 1)/PARENT_BLOCK_SIZE + 1;
    BFS_parent<<<numBlocks,PARENT_BLOCK_SIZE>>>
        (d_costArray, d_edgeArray, d_edgeArrayAux, numVerts, iters, d_flag);
}

