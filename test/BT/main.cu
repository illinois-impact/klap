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

#include <iostream>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "verify.h"

/**
 * Host main routine
 */
int main(int argc, char **argv){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Parameters
    int warmup      = 1;
    int runs        = 3;
    int outputLevel = 1;
    int N_LINES     = 25600;
    int opt;
    while((opt = getopt(argc, argv, "w:r:o:n:h")) >= 0) {
        switch(opt) {
            case 'w': warmup        = atoi(optarg); break;
            case 'r': runs          = atoi(optarg); break;
            case 'o': outputLevel   = atoi(optarg); break;
            case 'n': N_LINES       = atoi(optarg); break;
            default : std::cerr <<
                      "\nUsage:  ./bt [options]"
                          "\n"
                          "\n    -w <W>    # of warmup runs (default=1)"
                          "\n    -r <R>    # of timed runs (default=3)"
                          "\n    -o <O>    level of output verbosity (0: one CSV row, 1: moderate, 2: verbose)"
                          "\n    -n <N>    # of lines (default=25600)"
                          "\n    -h        help\n\n";
                      exit(0);
        }
    }

    // Set device
    int device = 0;
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties,device);
    cudaSetDevice(device);
    if(outputLevel >= 2) printf("%s\n", device_properties.name);

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, N_LINES);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 24);

    BezierLine *bLines_h = new BezierLine[N_LINES];

    float2 last = {0,0};

    float totalKernelTime = 0;
    for(int run = -warmup; run < runs; run++) {

        if(outputLevel >= 1) {
            if(run < 0) {
                std::cout << "Warmup:\t";
            } else {
                std::cout << "Run " << run << ":\t";
            }
        }

        srand(time(NULL));
        for (int i = 0; i < N_LINES; i++){
            bLines_h[i].CP[0] = last;

            for (int j = 1; j < 3; j++){
                bLines_h[i].CP[j].x = (float)rand()/(float)RAND_MAX;
                bLines_h[i].CP[j].y = (float)rand()/(float)RAND_MAX;
            }
            last = bLines_h[i].CP[2];
            bLines_h[i].vertexPos = NULL;
            bLines_h[i].nVertices = 0;
        }

        // CDP version
        BezierLine *bLines_d;
        cudaMalloc((void **)&bLines_d, N_LINES*sizeof(BezierLine));
        cudaMemcpy(bLines_d, bLines_h, N_LINES*sizeof(BezierLine), cudaMemcpyHostToDevice);
        if(outputLevel >= 2) printf("Computing Bezier Lines (CUDA Dynamic Parallelism Version) ... ");
        cudaEventRecord(start, NULL);

        launch_kernel(N_LINES, bLines_d);

        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);
        if(outputLevel >= 1) std::cout << "run kernel time = " << msecTotal << std::endl;
        if (run >= 0) totalKernelTime += msecTotal;

        //Do something to draw the lines here
        verify(N_LINES, bLines_d, bLines_h);

        // Free memory
        launch_free(N_LINES, bLines_d);
        cudaFree(bLines_d);

    }
    // Timing
    if(outputLevel >= 1) {
        std::cout<< "Average kernel time = " << totalKernelTime/runs << " ms\n";
    } else {
        std::cout<< totalKernelTime/runs;
    }

    delete[] bLines_h;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exitsits
    cudaDeviceReset();

    exit(EXIT_SUCCESS);

}

