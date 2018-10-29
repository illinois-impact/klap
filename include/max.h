/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_MAX_H_
#define _KLAP_MAX_H_

#include "basic.h"

#include "cub/cub/cub.cuh"

// Warp max: Assumes all threads are active. Final result returned by lane 0.
__device__ static unsigned int warp_max(unsigned int* array) {
    int myMax = (int) array[lane_id()];
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        myMax = max(myMax, __shfl_down_sync(0xffffffff, myMax, offset));
    }
    myMax = __shfl_sync(0xffffffff, myMax, 0);
    return (unsigned int) myMax;
}

// Block max
__device__ static unsigned int block_max(unsigned int* array) {

    // Intra-warp reduction
    unsigned int partialMax = warp_max(array + warp_id()*WARP_SIZE);

    // Broadcast partial maxes
    __shared__ unsigned int partialMaxes[MAX_N_WARPS];
    for(int i = threadIdx.x; i < MAX_N_WARPS; i += blockDim.x) {
        partialMaxes[i] = 0;
    }
    __syncthreads();
    if(lane_id() == 0) {
        partialMaxes[warp_id()] = partialMax;
    }
    __syncthreads();

    // Reduce partial maxes
    if(threadIdx.x < MAX_N_WARPS) {

        #if MAX_N_WARPS > WARP_SIZE
        #error Code assumes MAX_N_WARPS <= WARP_SIZE
        #endif
        unsigned int totalMax = warp_max(partialMaxes);

        // Broadcast total max
        if(threadIdx.x == 0) {
            partialMaxes[0] = totalMax;
        }

    }
    __syncthreads();
    unsigned int totalMax = partialMaxes[0];

    return totalMax;

}

// Kernel max
static unsigned int kernel_max(unsigned int* array, unsigned int size) {
    void *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, array, &array[size-1], size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, array, &array[size-1], size);
    unsigned int out;
    cudaMemcpy(&out, &array[size - 1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return out;
}

#endif

