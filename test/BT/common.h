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

#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <time.h>

#ifndef BLOCK_DIM
#define BLOCK_DIM 64
#endif
#define MAX_TESSELLATION 32 

struct BezierLine{
    float2 CP[3];
    float2 *vertexPos;
    int nVertices;
};

__forceinline__ __host__ __device__ unsigned int dif(float2 a, float2 b)
{
    if(a.x != b.x || a.y != b.y)
      return 1;
    else
      return 0;
}

__forceinline__ __device__ float2 operator+(float2 a, float2 b){
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__forceinline__ __device__ float2 operator-(float2 a, float2 b){
    float2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

__forceinline__ __device__ float2 operator*(float a, float2 b){
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}

__forceinline__ __device__ float length(float2 a){
    return sqrtf(a.x*a.x + a.y*a.y);
}

void launch_kernel(int N_LINES, BezierLine *bLines_d);
void launch_free(int N_LINES, BezierLine *bLines_d);

#endif

