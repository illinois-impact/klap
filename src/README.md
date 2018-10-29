
# Instructions for Building the KLAP Compiler

## Overview

KLAP is implemented as a series of Clang passes. To build KLAP, Clang/LLVM must first be built then KLAP can be added as a clang project.

## Instructions for building Clang/LLVM

Refer to the following link for instructions on building Clang/LLVM: https://llvm.org/docs/GettingStarted.html

## Modifying Clang to accept dynamic parallelism code

Clang's semantic checker does not accept dynamic parallelism kernel calls in CUDA kernels. Therefore, the semantic checker needs to be modified to accept such calls. This can be done by editing the file `llvm/tools/clang/lib/Sema/SemaCUDA.cpp` and making the following change:

```
// (a) Can't call global from some contexts until we support CUDA's
// dynamic parallelism.
if (CalleeTarget == CFT_Global &&
    (CallerTarget == CFT_Global || CallerTarget == CFT_Device ||
     (CallerTarget == CFT_HostDevice && getLangOpts().CUDAIsDevice)))
-   return CFP_Never;
+   return CFP_Native;
```

Clang's CUDA runtime wrapper does not support CUDA runtime functions being called from the device. To address this issue, declarations of these runtime functions need to be added to the file `build/lib/clang/7.0.0/include/__clang_cuda_runtime_wrapper.h`. Examples of commonly used runtime functions include:

```
__device__ cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
__device__ cudaError_t cudaMalloc(void **devPtr, size_t size);
__device__ cudaError_t cudaFree(void *devPtr);
__device__ cudaError_t cudaDeviceSynchronize();
__device__ cudaError_t cudaGetLastError();
```

## Adding KLAP to Clang

KLAP can be added as a Clang project as follows:

```
cd llvm/tools/clang/tools/extra
ln -s <path-to-klap-src-directory> klap
echo 'add_subdirectory(klap)' >> CMakeLists.txt
```

After adding KLAP as a Clang project, rebuild Clang/LLVM.

## Instructions for using KLAP

Run KLAP as follows for a list of command line options:

```
klap --help
```

