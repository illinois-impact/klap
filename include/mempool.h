/*
 *  Copyright (c) 2018 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _KLAP_MEMPOOL_H_
#define _KLAP_MEMPOOL_H_

class __MemPool {
    public:
        char* pool;
        unsigned int* free_idx;
        __MemPool(unsigned int POOL_SIZE) {
            cudaMalloc(&pool, POOL_SIZE);
            cudaMalloc(&free_idx, sizeof(unsigned int));
        }
        ~__MemPool(){
            cudaDeviceSynchronize();
            cudaFree(pool);
            cudaFree(free_idx);
        }
        void reset() {
            cudaMemset(free_idx, 0, sizeof(unsigned int));
        }
};

class __LocalMemPool {
    public:
        char* pool;
        template<class T>
        __device__ T* allocate(unsigned int n) {
            T* hold = (T*) pool;
            pool += n*sizeof(T);
            return hold;
        }
};

class __GridMemPool {
    public:
        char* pool;
        __GridMemPool(__MemPool& mp) {
            pool = mp.pool;
        }
        template<class T>
        __host__ __device__ T* grid_allocate(unsigned int n) {
            T* hold = (T*) pool;
            pool += n*sizeof(T);
            return hold;
        }
};

class __GlobalMemPool {
    public:
        char* pool;
        unsigned int* free_idx;
        __GlobalMemPool(__MemPool& mp) {
            pool = mp.pool;
            free_idx = mp.free_idx;
        }
        __device__ __LocalMemPool warp_allocate(unsigned int size) {
            unsigned int offset;
            if(threadIdx.x%WARP_SIZE == 0) {
                offset = atomicAdd(free_idx, size);
            }
            __LocalMemPool wp;
            wp.pool = pool + __shfl_sync(0xffffffff, offset, 0);
            return wp;
        }
        __device__ __LocalMemPool block_allocate(unsigned int size) {
            __shared__ unsigned int offset;
            if(threadIdx.x == 0) {
                offset = atomicAdd(free_idx, size);
            }
            __syncthreads();
            __LocalMemPool bp;
            bp.pool = pool + offset;
            return bp;
        }
};

#endif

