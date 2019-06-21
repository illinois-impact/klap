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

#include <stdio.h>
#include <string.h>
#include "verify.h"

__global__ void freeVertexMem_verify(BezierLine *bLines, int nLines){
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

    if (lidx < nLines)
        cudaFree(bLines[lidx].vertexPos);
}


__global__ void checkVertex(BezierLine *bLines_1, BezierLine *bLines_2, int nLines, int version){
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

#if 1
    if (lidx < nLines){
        if (bLines_1[lidx].nVertices != bLines_2[lidx].nVertices)
            printf("%d Error vertexPos - %d\n", version, lidx);
        for(int j = 0; j < bLines_1[lidx].nVertices; j++){
            //if(version == 12) printf("%f, %f; %f, %f.\n", bLines_1[lidx].vertexPos[j].x, bLines_1[lidx].vertexPos[j].y, bLines_2[lidx].vertexPos[j].x, bLines_2[lidx].vertexPos[j].y);
            if (dif(bLines_1[lidx].vertexPos[j], bLines_2[lidx].vertexPos[j])){
                printf("%d Error lidx=%d j=%d - %f, %f; %f, %f.\n", version, lidx, j, bLines_1[lidx].vertexPos[j].x, bLines_1[lidx].vertexPos[j].y, bLines_2[lidx].vertexPos[j].x, bLines_2[lidx].vertexPos[j].y);
                break;
            }
        }
    }
#endif
}

__global__ void computeBezierLinePositions_verify(int lidx, BezierLine *bLines, int nTessPoints){
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

__global__ void computeBezierLines_cdp_verify(BezierLine *bLines, int nLines){
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;

    int op = lidx < nLines;
    if (op){
        float curvature = length(bLines[lidx].CP[1] - 0.5f*(bLines[lidx].CP[0] + bLines[lidx].CP[2]))/length(bLines[lidx].CP[2] - bLines[lidx].CP[0]);
        int nTessPoints = min(max((int)(curvature*16.0f),4),MAX_TESSELLATION);

        if (bLines[lidx].vertexPos == NULL){
            bLines[lidx].nVertices = nTessPoints;
            cudaMalloc((void **)&bLines[lidx].vertexPos, nTessPoints*sizeof(float2));
        }

        computeBezierLinePositions_verify<<<ceil((float)bLines[lidx].nVertices/32.0f), 32>>>(lidx, bLines, bLines[lidx].nVertices);
    }
}

void verify(int N_LINES, BezierLine *bLines_check_d, BezierLine *bLines_h){

    BezierLine *bLines_d;
    cudaMalloc((void **)&bLines_d, N_LINES*sizeof(BezierLine));
    cudaMemcpy(bLines_d, bLines_h, N_LINES*sizeof(BezierLine), cudaMemcpyHostToDevice);
    computeBezierLines_cdp_verify<<< (unsigned int)ceil((float)N_LINES/(float)BLOCK_DIM), BLOCK_DIM >>>(bLines_d, N_LINES);
    checkVertex<<< (unsigned int)ceil((float)N_LINES/(float)BLOCK_DIM), BLOCK_DIM >>>(bLines_d, bLines_check_d, N_LINES, 1);
    freeVertexMem_verify<<< (unsigned int)ceil((float)N_LINES/(float)BLOCK_DIM), BLOCK_DIM >>>(bLines_d, N_LINES);
    cudaFree(bLines_d);

}

