
#ifndef _COMMON_H_
#define _COMMON_H_

#include "lonestargpu/lonestargpu.h"
#include "lonestargpu/cutil_subset.h"

#define SSSP_VARIANT "lonestar"
#define BLOCK_DIM 128

inline __device__
bool processedge(foru *dist, Graph &graph, unsigned src, unsigned ii, unsigned &dst) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	dst = graph.getDestination(src, ii);
	if (dst >= graph.nnodes) return 0;

	foru wt = graph.getWeight(src, ii);
	if (wt >= MYINFINITY) return 0;

	foru altdist = dist[src] + wt;
	if (altdist < dist[dst]) {
	 	foru olddist = atomicMin(&dist[dst], altdist);
		if (altdist < olddist) {
			return true;
		} 
		// someone else updated distance to a lower value.
	}
	return false;
}

inline __device__
bool processnode(foru *dist, Graph &graph, unsigned work) {
	//unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nn = work;
	if (nn >= graph.nnodes) return 0;
	bool changed = false;
	
	unsigned neighborsize = graph.getOutDegree(nn);
	for (unsigned ii = 0; ii < neighborsize; ++ii) {
		unsigned dst = graph.nnodes;
		foru olddist = processedge(dist, graph, nn, ii, dst);
		if (olddist) {
			changed = true;
		}
	}
	return changed;
}

__global__ void drelax(foru *dist, Graph graph, bool *changed);

void launch_kernel(unsigned int nb, unsigned int nt, foru *dist, Graph graph, bool *changed);

#endif

