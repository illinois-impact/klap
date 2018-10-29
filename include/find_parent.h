/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_FIND_H_
#define _KLAP_FIND_H_

#include "basic.h"

#include <thrust/binary_search.h>

__device__ inline bool isParentIdx(unsigned int* array, int childBlockIdx, int parentThreadIdxGuess) {
    int numBlocksBeforeThisParent = (parentThreadIdxGuess == 0)?0:array[parentThreadIdxGuess - 1];
    int numBlocksIncludingThisParent = array[parentThreadIdxGuess];
    return numBlocksBeforeThisParent <= childBlockIdx
        && childBlockIdx < numBlocksIncludingThisParent;
}

// Warp find
__device__ static int __find_parent_idx_warp(unsigned int* array, int childBlockIdx) {

    // Try an initial guess that assumes child blocks are distributed evenly across parent threads
    int parentSize = WARP_SIZE;
    int avgChildBlocksPerParent = (gridDim.x - 1)/parentSize + 1;
    int parentThreadIdxInitialGuess = childBlockIdx/avgChildBlocksPerParent;
    if(isParentIdx(array, childBlockIdx, parentThreadIdxInitialGuess)) {
        return parentThreadIdxInitialGuess;
    }

    // If initial guess fails, search the array using all threads
    __shared__ int parentThreadIdx_s;
    int parentThreadIdxGuess = threadIdx.x;
    if(parentThreadIdxGuess < parentSize && isParentIdx(array, childBlockIdx, parentThreadIdxGuess)) {
        parentThreadIdx_s = parentThreadIdxGuess;
    }

    // Broadcast successful guess and return it
    __syncthreads();
    return parentThreadIdx_s;

}

// Block find
__device__ static int __find_parent_idx_block(unsigned int* array, int childBlockIdx, int parentSize) {

    // Try an initial guess that assumes child blocks are distributed evenly across parent threads
    int avgChildBlocksPerParent = (gridDim.x - 1)/parentSize + 1;
    int parentThreadIdxInitialGuess = childBlockIdx/avgChildBlocksPerParent;
    if(isParentIdx(array, childBlockIdx, parentThreadIdxInitialGuess)) {
        return parentThreadIdxInitialGuess;
    }

    // If initial guess fails, search the array using all threads

    // If we have more child threads than the parent size, search takes one step
    __shared__ int parentThreadIdx_s;
    if(parentSize <= blockDim.x) {
        int parentThreadIdxGuess = threadIdx.x;
        if(parentThreadIdxGuess < parentSize && isParentIdx(array, childBlockIdx, parentThreadIdxGuess)) {
            parentThreadIdx_s = parentThreadIdxGuess;
        }
    }

    // If we have fewer child threads than the parent size, use one warp to search
    // Note: 32-ary search takes 2 steps since we assume parent block size is <= 1024 (32^2)
    else { // parentSize > blockDim.x
        if(warp_id()==0) {
            // Step 1: Find the partition containing the match
            int partitionSize = parentSize/WARP_SIZE;
            int partitionStart = threadIdx.x*partitionSize;
            bool matchingPartitionIssToMyRight = childBlockIdx >= array[partitionStart];
            int matchingPartitionIdx = __popc(__ballot_sync(0xffffffff, matchingPartitionIssToMyRight));
            // Step 2: Find the match in the partiton
            int matchingPartitionStart = (matchingPartitionIdx > 0)?((matchingPartitionIdx - 1)*partitionSize):0;
            int parentThreadIdxGuess = matchingPartitionStart + threadIdx.x;
            bool matchIsToMyRight = (threadIdx.x < partitionSize) && (childBlockIdx >= array[parentThreadIdxGuess]);
            int matchIdx = __popc(__ballot_sync(0xffffffff, matchIsToMyRight));
            parentThreadIdx_s = matchingPartitionStart + matchIdx;
        }
    }      

    // Broadcast successful guess and return it
    __syncthreads();
    return parentThreadIdx_s;

}

// Kernel find
__device__ static int __find_parent_idx_kernel(unsigned int* array, int childBlockIdx, int parentSize) {

    // Try an initial guess that assumes child blocks are distributed evenly across parent threads
    int avgChildBlocksPerParent = (gridDim.x - 1)/parentSize + 1;
    int parentThreadIdxInitialGuess = childBlockIdx/avgChildBlocksPerParent;
    if(isParentIdx(array, childBlockIdx, parentThreadIdxInitialGuess)) {
        return parentThreadIdxInitialGuess;
    }

    // If initial guess fails, search the array using thrust
    thrust::pair<int*, int*> parent = thrust::equal_range(thrust::seq, (int*)array, (int*)(array + parentSize), childBlockIdx);
    return parent.second - (int*)array;

}

#endif

