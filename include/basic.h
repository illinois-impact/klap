/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_BASIC_H_
#define _KLAP_BASIC_H_

#define WARP_SIZE 32
#define MAX_BLOCK_DIM 1024
#define MAX_N_WARPS ((MAX_BLOCK_DIM)/(WARP_SIZE)) // 32

#define _Bool bool

// Lane and warp id
__device__ inline int lane_id(void) { return threadIdx.x % WARP_SIZE; }
__device__ inline int warp_id(void) { return threadIdx.x / WARP_SIZE; }

#endif

