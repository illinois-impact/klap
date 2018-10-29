/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_SCAN_H_
#define _KLAP_SCAN_H_

#include "basic.h"

#include "cub/cub/cub.cuh"

__device__ static int warp_scan_registers(int val) {
    int x = val;
    #pragma unroll
    for(int offset = 1; offset < 32; offset <<= 1){
        // From GTC: Kepler shuffle tips and tricks
        asm volatile("{"
                " .reg .s32 r0;"
                " .reg .pred p;"
                " shfl.up.b32 r0|p, %0, %1, 0x0;"
                " @p add.s32 r0, r0, %0;"
                " mov.s32 %0, r0;"
                "}" : "+r"(x) : "r"(offset));
    }
    return x;
}

// Warp inclusive scan
__device__ static unsigned int warp_inclusive_scan(unsigned int* array) {
    int x = (int) array[lane_id()];
    x = warp_scan_registers(x);
    array[lane_id()] = x;
    return x;
}

__device__ static unsigned int block_scan_registers(int val) {
    int x = val;
    __shared__ int sdata[WARP_SIZE];
    // A. Exclusive scan within each warp
    int warpPrefix = warp_scan_registers(x);
    // B. Store in shared memory
    if(lane_id() == WARP_SIZE - 1)
        sdata[warp_id()] = warpPrefix;
    __syncthreads();
    // C. One warp scans in shared memory
    if(threadIdx.x < WARP_SIZE)
        sdata[threadIdx.x] = warp_scan_registers(sdata[threadIdx.x]);
    __syncthreads();
    // D. Each thread calculates it final value
    return warpPrefix + (warp_id() > 0 ? sdata[warp_id() - 1] : 0);
}

// Block inclusive scan
__device__ static unsigned int block_inclusive_scan(unsigned int* array) {
    int x = (int)array[threadIdx.x];
    int thread_out_element;
    thread_out_element = block_scan_registers(x);
    array[threadIdx.x] = (unsigned int)thread_out_element;
    return thread_out_element;
}

// Kernel inclusive scan
static unsigned int kernel_inclusive_scan(unsigned int* array, unsigned int size){
    void *d_temp_storage=NULL;
    size_t temp_storage_bytes=0;
    // Determine temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, array, array, size);
    // Allocate temporary storage for inclusive prefix sum
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, array, array, size);
    // Free temporary storage
    cudaFree(d_temp_storage);
    // Copy total sum from device and return it
    unsigned int out;
    cudaMemcpy(&out, &array[size - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return out;
}

union scan_counter {
    unsigned long long int fused;
    struct {
        unsigned int idx;
        unsigned int nb;
    };
};

#endif

