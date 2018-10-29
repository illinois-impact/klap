/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_MALLOC_H_
#define _KLAP_MALLOC_H_

#include "scan.h"

union high_low {
    char* ptr;
    struct {
        int lo;
        int hi;
    };
};

struct poolMetadata {
    unsigned int usedSize;
    unsigned int padding;
};

struct subpoolMetadata {
    unsigned int offset;
    unsigned int size;
};

__device__ __inline__ union high_low __warp_shfl64(union high_low x, int lane) {
    x.lo = __shfl_sync(0xffffffff, x.lo, lane);
    x.hi = __shfl_sync(0xffffffff, x.hi, lane);
    return x;
}

__device__ __inline__ union high_low __block_shfl64(union high_low x, int thread) {
    __shared__ union high_low smem;
    if(threadIdx.x == thread) {
        smem = x;
    }
    __syncthreads();
    return smem;
}

__device__ cudaError_t __cudaMalloc_warp(void** devPtr, size_t size) {
    int x = (int)((size > 0)?(size + sizeof(struct subpoolMetadata)):(0));
    x = warp_scan_registers(x);
    union high_low pool;
    cudaError_t err;
    if(lane_id() == 31) {
        err = cudaMalloc((void**)&(pool.ptr), x + sizeof(struct poolMetadata));
        struct poolMetadata* poolMeta = (struct poolMetadata*) pool.ptr;
        poolMeta->usedSize = x;
    }
    char* myPtr = __warp_shfl64(pool, 31).ptr;
    if(size > 0) {
        unsigned int subPoolSize = size + sizeof(struct subpoolMetadata);
        unsigned int offset = sizeof(struct poolMetadata) + x - subPoolSize;
        struct subpoolMetadata* subpoolMeta = (struct subpoolMetadata*) (myPtr + offset);
        subpoolMeta->offset = offset;
        subpoolMeta->size = subPoolSize;
        char* subpool = myPtr + offset + sizeof(struct subpoolMetadata);
        *devPtr = (void*) subpool;
    }
    return err;
}

__device__ cudaError_t __cudaMalloc_block(void** devPtr, size_t size) {
    int x = (int)((size > 0)?(size + sizeof(struct subpoolMetadata)):(0));
    x = block_scan_registers(x);
    union high_low pool;
    cudaError_t err;
    if(threadIdx.x == blockDim.x - 1) {
        err = cudaMalloc((void**)&(pool.ptr), x + sizeof(struct poolMetadata));
        struct poolMetadata* poolMeta = (struct poolMetadata*) pool.ptr;
        poolMeta->usedSize = x;
    }
    char* myPtr = __block_shfl64(pool, blockDim.x - 1).ptr;
    if(size > 0) {
        unsigned int subPoolSize = size + sizeof(struct subpoolMetadata);
        unsigned int offset = sizeof(struct poolMetadata) + x - subPoolSize;
        struct subpoolMetadata* subpoolMeta = (struct subpoolMetadata*) (myPtr + offset);
        subpoolMeta->offset = offset;
        subpoolMeta->size = subPoolSize;
        char* subpool = myPtr + offset + sizeof(struct subpoolMetadata);
        *devPtr = (void*) subpool;
    }
    return err;
}

__device__ cudaError_t __cudaMalloc_kernel(void** devPtr, size_t size) {
    return __cudaMalloc_block(devPtr, size);
}

__device__ cudaError_t __cudaFree_warp(void* devPtr) {
    char* subpool = (char*) devPtr;
    struct subpoolMetadata* subpoolMeta = (struct subpoolMetadata*) (subpool - sizeof(struct subpoolMetadata));
    unsigned int offset = subpoolMeta->offset;
    unsigned int subPoolSize = subpoolMeta->size;
    char* pool = subpool - sizeof(struct subpoolMetadata) - offset;
    struct poolMetadata* poolMeta = (struct poolMetadata*) pool;
    unsigned int oldUsedSize = atomicSub(&(poolMeta->usedSize), subPoolSize);
    if(oldUsedSize - subPoolSize == 0) {
        return cudaFree(pool);
    } else {
        return cudaSuccess;
    }
}

__device__ cudaError_t __cudaFree_block(void* devPtr) {
    return __cudaFree_warp(devPtr);
}

__device__ cudaError_t __cudaFree_kernel(void* devPtr) {
    return __cudaFree_block(devPtr);
}

#endif

