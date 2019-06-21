/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Bezier Tessellation
 * Version using dynamic parallelism
 *
 */

#include "common.h"

__global__ void computeBezierLinePositions(int lidx, BezierLine *bLines, int nTessPoints){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < nTessPoints){
        float u = (float)idx/(float)(nTessPoints-1);
        float omu = 1.0f - u;

        float B3u[3];

        B3u[0] = omu*omu;
        B3u[1] = 2.0f*u*omu;
        B3u[2] = u*u;

        float2 position = {0,0};

        for (int i = 0; i < 3; i++){
            position = position + B3u[i] * bLines[lidx].CP[i];
        }

        bLines[lidx].vertexPos[idx] = position;
    }
}

__global__ void computeBezierLines(BezierLine *bLines, int nLines){
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (lidx < nLines){
        float curvature = length(bLines[lidx].CP[1] - 0.5f*(bLines[lidx].CP[0] + bLines[lidx].CP[2]))/length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
        int nTessPoints = min(max((int)(curvature*16.0f),4),MAX_TESSELLATION);

        if (bLines[lidx].vertexPos == NULL){
            bLines[lidx].nVertices = nTessPoints;
            cudaMalloc((void **)&bLines[lidx].vertexPos, nTessPoints*sizeof(float2));
        }

        computeBezierLinePositions<<<ceil((float)bLines[lidx].nVertices/32.0f), 32>>>(lidx, bLines, bLines[lidx].nVertices);
    }
}

__global__ void freeVertexMem(BezierLine *bLines, int nLines){
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (lidx < nLines) {
        cudaFree(bLines[lidx].vertexPos);
    }
}

void launch_kernel(int N_LINES, BezierLine *bLines_d) {
    computeBezierLines<<< (N_LINES - 1)/BLOCK_DIM + 1, BLOCK_DIM >>>(bLines_d, N_LINES);
}

void launch_free(int N_LINES, BezierLine *bLines_d) {
    freeVertexMem<<< (N_LINES - 1)/BLOCK_DIM + 1, BLOCK_DIM >>>(bLines_d, N_LINES);
}

