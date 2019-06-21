/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <helper_cuda.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <unistd.h>

#include "common.h"

////////////////////////////////////////////////////////////////////////////////
// Parallel random number generator.
////////////////////////////////////////////////////////////////////////////////
struct Random_generator
{
    __host__ __device__ unsigned int hash(unsigned int a)
    {
        a = (a+0x7ed55d16) + (a<<12);
        a = (a^0xc761c23c) ^ (a>>19);
        a = (a+0x165667b1) + (a<<5);
        a = (a+0xd3a2646c) ^ (a<<9);
        a = (a+0xfd7046c5) + (a<<3);
        a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }

    __host__ __device__ __forceinline__ thrust::tuple<float, float> operator()()
    {
#ifdef __CUDA_ARCH__
        unsigned seed = hash(blockIdx.x*blockDim.x + threadIdx.x);
#else
        unsigned seed = hash(0);
#endif
        thrust::default_random_engine rng(seed);
        thrust::random::uniform_real_distribution<float> distrib;
        return thrust::make_tuple(distrib(rng), distrib(rng));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Make sure a Quadtree is properly defined.
////////////////////////////////////////////////////////////////////////////////
bool check_quadtree(const Quadtree_node *nodes, int idx, int num_pts, Points *pts, Parameters params)
{
    const Quadtree_node &node = nodes[idx];
    int num_points = node.num_points();

    if (params.depth == params.max_depth || num_points <= params.min_points_per_node)
    {
        int num_points_in_children = 0;

        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+0].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+1].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+2].num_points();
        num_points_in_children += nodes[params.num_nodes_at_this_level + 4*idx+3].num_points();

        if (num_points_in_children != node.num_points())
            return false;

        return check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+0, num_pts, pts, Parameters(params, true)) &&
            check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+1, num_pts, pts, Parameters(params, true)) &&
            check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+2, num_pts, pts, Parameters(params, true)) &&
            check_quadtree(&nodes[params.num_nodes_at_this_level], 4*idx+3, num_pts, pts, Parameters(params, true));
    }

    const Bounding_box &bbox = node.bounding_box();

    for (int it = node.points_begin() ; it < node.points_end() ; ++it)
    {
        if (it >= num_pts)
            return false;

        float2 p = pts->get_point(it);

        if (!bbox.contains(p))
            return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Allocate GPU structs, launch kernel and clean up
////////////////////////////////////////////////////////////////////////////////
bool cdpQuadtree(int warp_size, const int num_points, const int max_depth, const int min_points_per_node, int warmup, int runs, int outputLevel)
{
    // Constants to control the algorithm.
    //const int num_points = 8192;//2048;//1024;
    //const int max_depth  = 16;//8;
    //const int min_points_per_node = 8;//16;

    // Allocate memory for points.
    thrust::device_vector<float> x_d0(num_points);
    thrust::device_vector<float> x_d1(num_points);
    thrust::device_vector<float> y_d0(num_points);
    thrust::device_vector<float> y_d1(num_points);

    // Generate random points.
    Random_generator rnd;
    thrust::generate(
            thrust::make_zip_iterator(thrust::make_tuple(x_d0.begin(), y_d0.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(x_d0.end(), y_d0.end())),
            rnd);

    // Host structures to analyze the device ones.
    Points points_init[2];
    points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]), thrust::raw_pointer_cast(&y_d0[0]));
    points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]), thrust::raw_pointer_cast(&y_d1[0]));

    // Allocate memory to store points.
    Points *points;
    cudaMalloc((void **) &points, 2*sizeof(Points));
    //cudaMemcpy(points, points_init, 2*sizeof(Points), cudaMemcpyHostToDevice);

    // We could use a close form...
    int max_nodes = 0;

    for (int i = 0, num_nodes_at_level = 1 ; i < max_depth ; ++i, num_nodes_at_level *= 4)
        max_nodes += num_nodes_at_level;

    // Allocate memory to store the tree.
    Quadtree_node root;
    root.set_range(0, num_points);
    Quadtree_node *nodes;
    cudaMalloc((void **) &nodes, max_nodes*sizeof(Quadtree_node));
    //cudaMemcpy(nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice);

    // We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);
    //cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 65536); // Fixed-size pool

    // Build the quadtree.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    bool ok;
    Quadtree_node *host_nodes = new Quadtree_node[max_nodes];
    float totalKernelTime = 0;
    for(int run = -warmup; run < runs; run++){

        if(outputLevel >= 1) {
            if(run < 0) {
                std::cout << "Warmup:\t";
            } else {
                std::cout << "Run " << run << ":\t";
            }
        }

        Parameters params(max_depth, min_points_per_node);

        cudaMemcpy(points, points_init, 2*sizeof(Points), cudaMemcpyHostToDevice);
        cudaMemcpy(nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice);

        cudaEventRecord(start, NULL);

        launch_kernel(warp_size, nodes, points, params);
        cudaGetLastError();

        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);
        if(outputLevel >= 1) std::cout << "run kernel time = " << msecTotal << "\t";
        if (run >= 0) totalKernelTime += msecTotal;

        // Copy points to CPU.
        thrust::host_vector<float> x_h(x_d0);
        thrust::host_vector<float> y_h(y_d0);
        Points host_points;
        host_points.set(thrust::raw_pointer_cast(&x_h[0]), thrust::raw_pointer_cast(&y_h[0]));

        // Copy nodes to CPU.
        cudaMemcpy(host_nodes, nodes, max_nodes *sizeof(Quadtree_node), cudaMemcpyDeviceToHost);

        // Validate the results.
        ok = check_quadtree(host_nodes, 0, num_points, &host_points, params);
        if(outputLevel >= 1) std::cerr << (ok ? "passed" : "failed") << std::endl;

    }
    // Timing
    if(outputLevel >= 1) {
        std::cout<< "Average kernel time = " << totalKernelTime/runs << " ms\n";
    } else {
        std::cout<< totalKernelTime/runs;
    }

    // Free CPU memory.
    delete[] host_nodes;

    // Free memory.
    cudaFree(nodes);
    cudaFree(points);

    return ok;
}

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

    // Parameters
    int warmup      = 1;
    int runs        = 3;
    int outputLevel = 1;
    int points      = 40000;
    int depth       = 12;
    int min_points  = 1;
    int opt;
    while((opt = getopt(argc, argv, "w:r:o:p:d:m:h")) >= 0) {
        switch(opt) {
            case 'w': warmup        = atoi(optarg); break;
            case 'r': runs          = atoi(optarg); break;
            case 'o': outputLevel   = atoi(optarg); break;
            case 'p': points        = atoi(optarg); break;
            case 'd': depth         = atoi(optarg); break;
            case 'm': min_points    = atoi(optarg); break;
            default : std::cerr <<
                      "\nUsage:  ./qt [options]"
                          "\n"
                          "\n    -w <W>    # of warmup runs (default=1)"
                          "\n    -r <R>    # of timed runs (default=3)"
                          "\n    -o <O>    level of output verbosity (0: one CSV row, 1: moderate, 2: verbose)"
                          "\n    -p <P>    # of points (default=40000)"
                          "\n    -d <D>    depth (default=12)"
                          "\n    -m <M>    minimum # of points (default=1)"
                          "\n    -h        help\n\n";
                      exit(0);
        }
    }

    // Find/set the device.
    // The test requires an architecture SM35 or greater (CDP capable).
    int cuda_device = 0;//findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, cuda_device);
    int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) || deviceProps.major >=4;

    // printf("GPU device %s has compute capabilities (SM %d.%d)\n", deviceProps.name, deviceProps.major, deviceProps.minor);

    if (!cdpCapable)
    {
        std::cerr << "cdpQuadTree requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...\n" << std::endl;
        exit(EXIT_WAIVED);
    }

    bool ok = cdpQuadtree(deviceProps.warpSize, points, depth, min_points, warmup, runs, outputLevel);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return (ok ? EXIT_SUCCESS : EXIT_FAILURE);
}

