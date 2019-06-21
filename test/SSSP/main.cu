/** Single source shortest paths -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Single source shortest paths.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

#include "common.h"

/*************************************************************************************************************/
/*************************************************************************************************************/
__global__
void initialize(foru *dist, unsigned int nv) {
    unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii < nv) {
        dist[ii] = MYINFINITY;
    }
}

__global__
void dverifysolution(foru *dist, Graph graph, unsigned *nerr) {
    unsigned int nn = (blockIdx.x * blockDim.x + threadIdx.x);
    if (nn < graph.nnodes) {
        unsigned int nsrcedges = graph.getOutDegree(nn);
        for (unsigned ii = 0; ii < nsrcedges; ++ii) {
            unsigned int u = nn;
            unsigned int v = graph.getDestination(u, ii);
            foru wt = graph.getWeight(u, ii);
            if (wt > 0 && dist[u] + wt < dist[v]) {
                ++*nerr;
            }
        }
    }	
}

void print_output(const char *filename, foru *hdist, foru *dist, Graph graph)
{
    CUDA_SAFE_CALL(cudaMemcpy(hdist, dist, graph.nnodes * sizeof(foru), cudaMemcpyDeviceToHost));

    //printf("Writing output to %s\n", filename);
    FILE *o = fopen(filename, "w");

    for(int i = 0; i < graph.nnodes; i++) {
        fprintf(o, "%d: %d\n", i, hdist[i]);
    }

    fclose(o);
}

int main(int argc, char *argv[]) {

    // Parameters
    int warmup              = 1;
    int runs                = 3;
    int outputLevel         = 1;
    const char* fileName    = "inputs/rmat12.sym.gr";
    int opt;
    while((opt = getopt(argc, argv, "w:r:o:f:h")) >= 0) {
        switch(opt) {
            case 'w': warmup        = atoi(optarg); break;
            case 'r': runs          = atoi(optarg); break;
            case 'o': outputLevel   = atoi(optarg); break;
            case 'f': fileName      = optarg;       break;
            default : std::cerr <<
                      "\nUsage:  ./bt [options]"
                          "\n"
                          "\n    -w <W>    # of warmup runs (default=1)"
                          "\n    -r <R>    # of timed runs (default=3)"
                          "\n    -o <O>    level of output verbosity (0: one CSV row, 1: moderate, 2: verbose)"
                          "\n    -f <F>    file name (default=inputs/rmat12.sym.gr)"
                          "\n    -h        help\n\n";
                      exit(0);
        }
    }

    foru *dist, *hdist;
    unsigned intzero = 0;
    Graph hgraph, graph;
    unsigned *nerr, hnerr;
    KernelConfig kconf;

    cudaFuncSetCacheConfig(drelax, cudaFuncCachePreferShared);
    cudaGetLastError();

    hgraph.read((char*)fileName);
    //hgraph.optimize();
    long unsigned totalcommu = hgraph.cudaCopy(graph);
    if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");
    if (cudaMalloc((void **)&dist, graph.nnodes * sizeof(foru)) != cudaSuccess) CudaTest("allocating dist failed");
    hdist = (foru *)malloc(graph.nnodes * sizeof(foru));

    float totalIter1Time = 0;
    float totalKernelTime = 0;
    for(int run = -warmup; run < runs; run++) {

        if(outputLevel >= 1) {
            if(run < 0) {
                std::cout << "Warmup:\t";
            } else {
                std::cout << "Run " << run << ":\t";
            }
        }

        kconf.setNumberOfBlockThreads(128);
        kconf.setProblemSize(graph.nnodes);
        kconf.setMaxThreadsPerBlock();	
        kconf.setNumberOfBlockThreads(128);
        initialize <<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph.nnodes);
        CudaTest("initializing failed");

        foru foruzero = 0.0;
        bool *changed, hchanged;
        int iteration = 0;
        double starttime, endtime;
        double starttime2, endtime2, time2;

        kconf.setNumberOfBlockThreads(128);
        kconf.setProblemSize(graph.nnodes);
        cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

        if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

        time2 = 0;
        //printf("nnodes=%d, blocks=%d, threads=%d\n", graph.nnodes, kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads());
        cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, graph.nnodes /*32768*/); // Fixed-size pool
        starttime = rtclock();
        do {
            ++iteration;
            hchanged = false;

            cudaMemcpy(changed, &hchanged, sizeof(hchanged), cudaMemcpyHostToDevice);

            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            starttime2 = rtclock();
            launch_kernel(kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads(), dist, graph, changed);
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            endtime2 = rtclock(); // changed from lsg (for now) which included memcopies of graph too.
            time2 += (endtime2-starttime2);
            if(outputLevel >= 1) printf("iter=%d, time=%f\t", iteration, (endtime2-starttime2)*1000);
            if (iteration == 1 && run >= 0) totalIter1Time += 1000*(endtime2-starttime2);
            CudaTest("solving failed");

            CUDA_SAFE_CALL(cudaMemcpy(&hchanged, changed, sizeof(hchanged), cudaMemcpyDeviceToHost));
        } while (hchanged);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        endtime = rtclock(); // changed from lsg (for now) which included memcopies of graph too.

        CUDA_SAFE_CALL(cudaMemcpy(hdist, dist, graph.nnodes * sizeof(foru), cudaMemcpyDeviceToHost));
        totalcommu += graph.nnodes * sizeof(foru);

        if(outputLevel >= 1) std::cout << "run kernel time = " << 1000*time2 << "\t";
        if (run >= 0) totalKernelTime += 1000*time2;
        //printf("\titerations = %d communication = %.3lf MB.\t", iteration, totalcommu * 1.0 / 1000000);
        if(outputLevel >= 1) printf("runtime [%s] = %f ms\t", SSSP_VARIANT, 1000 * (endtime - starttime));
        //printf("\tdrelax [%s] = %f ms\n", SSSP_VARIANT, 1000 * time2);

        cudaMemcpy(nerr, &intzero, sizeof(intzero), cudaMemcpyHostToDevice);
        kconf.setMaxThreadsPerBlock();
        kconf.setNumberOfBlockThreads(128);
        //printf("verifying.\n");
        dverifysolution<<<kconf.getNumberOfBlocks(), kconf.getNumberOfBlockThreads()>>> (dist, graph, nerr);
        CudaTest("dverifysolution failed");
        cudaMemcpy(&hnerr, nerr, sizeof(hnerr), cudaMemcpyDeviceToHost);
        if(outputLevel >= 1) printf("#errors = %d.\n", hnerr);

    }

    if(outputLevel >= 1) {
        std::cout<< "Average kernel time (iter=1) = " << totalIter1Time/runs << " ms\n";
        std::cout<< "Average kernel time (total) = " << totalKernelTime/runs << " ms\n";
    } else {
        std::cout<< totalIter1Time/runs;
    }

    print_output("outputs/sssp-output.txt", hdist, dist, graph);
    // cleanup left to the OS.

    return 0;
}
