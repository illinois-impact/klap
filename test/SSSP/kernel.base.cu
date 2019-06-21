
#include "common.h"

__global__ void drelax_child(foru *dist, Graph graph, bool *changed, unsigned start, unsigned end) {
    unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;

    if(ii+start < end)
        if (processnode(dist, graph, ii+start)) {
            *changed = true;
        }
}

__global__ void drelax(foru *dist, Graph graph, bool *changed) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned start = id * (MAXBLOCKSIZE / blockDim.x), end = (id + 1) * (MAXBLOCKSIZE / blockDim.x);

    int blocks = (int)ceil((float)(end-start) / (float)BLOCK_DIM);
    int threads = BLOCK_DIM;
    drelax_child<<<blocks, threads>>>(dist, graph, changed, start, end);
}

void launch_kernel(unsigned int nb, unsigned int nt, foru *dist, Graph graph, bool *changed) {
    drelax <<<nb, nt>>> (dist, graph, changed);
}

