
#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>
using namespace std;

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

const long GRAPHSIZE = 4096;    // number of nodes
const float DENSITY = 0.01;
const long NUMEDGES = DENSITY*GRAPHSIZE*(GRAPHSIZE-1)/2;

const int GRIDSIZE = 2;          // number of blocks
const int BLOCKSIZE = 256;       // number of threads in a block

const int SUBSIZE = GRAPHSIZE/(GRIDSIZE*BLOCKSIZE);

const int SUBSIZE_BOUNDARY = 256;

#ifdef __cplusplus
#define CHECK_EXT extern "C"
#else
#define CHECK_EXT
#endif

CHECK_EXT void cudaGraphColoring(int *adjacentList, int *boundaryList, int *graphColors, int *degreeList, int *conflict, int boundarySize, int maxDegree, int graphSize, int passes, int warmup, int runs, int outputLevel);

void launch_kernel(unsigned int dimGrid_confl, unsigned int dimBlock_confl, int *adjacentListD, int *boundaryListD, int *colors, int *conflictD, long size, int boundarySize, int maxDegree);

#endif // _GRAPHCOLORING_H_
