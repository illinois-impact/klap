
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "graph.h"

// ****************************************************************************
// Function: verify_results
//
// Purpose:
//  Verify BFS results by comparing the output path lengths from cpu and gpu
//  traversals
//
// Arguments:
//   cpu_cost: path lengths calculated on cpu
//   gpu_cost: path lengths calculated on gpu
//   numVerts: number of vertices in the given graph
//   out_path_lengths: specify if path lengths should be dumped to files
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
inline unsigned int verify_results(unsigned int *cpu_cost, unsigned int *gpu_cost,
        unsigned int numVerts,  bool out_path_lengths)
{
    unsigned int unmatched_nodes=0;
    for(int i=0;i<numVerts;i++)
    {
        if(gpu_cost[i]!=cpu_cost[i])
        {
            unmatched_nodes++;
        }
    }

    // If user wants to write path lengths to file
    if(out_path_lengths)
    {
        std::ofstream bfs_out_cpu("bfs_out_cpu.txt");
        std::ofstream bfs_out_gpu("bfs_out_cuda.txt");
        for(int i=0;i<numVerts;i++)
        {
            if(cpu_cost[i]!=UINT_MAX)
                bfs_out_cpu<<cpu_cost[i]<<"\n";
            else
                bfs_out_cpu<<"0\n";

            if(gpu_cost[i]!=UINT_MAX)
                bfs_out_gpu<<gpu_cost[i]<<"\n";
            else
                bfs_out_gpu<<"0\n";
        }
        bfs_out_cpu.close();
        bfs_out_gpu.close();
    }
    return unmatched_nodes;
}

// ****************************************************************************
// Function: RunTest
//
// Purpose:
//   Runs the BFS benchmark using method 1 (IIIT-BFS method)
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//   G: input graph
//
// Returns:  nothing
//
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunTest(Graph *G, int degree, int runs, int warmup, int outputLevel) {

    typedef char frontier_type;
    typedef unsigned int cost_type;

    // Get graph info
    unsigned int *edgeArray=G->GetEdgeOffsets();
    unsigned int *edgeArrayAux=G->GetEdgeList();
    unsigned int adj_list_length=G->GetAdjacencyListLength();
    unsigned int numVerts = G->GetNumVertices();
    unsigned int numEdges = G->GetNumEdges();

    int *flag;

    // Allocate pinned memory for frontier and cost arrays on CPU
    cost_type  *costArray;
    CUDA_SAFE_CALL(cudaMallocHost((void **)&costArray,sizeof(cost_type)*(numVerts)));
    CUDA_SAFE_CALL(cudaMallocHost((void **)&flag,sizeof(int)));

    // Variables for GPU memory
    // Adjacency lists
    unsigned int *d_edgeArray=NULL,*d_edgeArrayAux=NULL;
    // Cost array
    cost_type *d_costArray;
    // Flag to check when traversal is complete
    int *d_flag;

    // Allocate memory on GPU
    CUDA_SAFE_CALL(cudaMalloc(&d_flag,sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_costArray,sizeof(cost_type)*numVerts));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArray,sizeof(unsigned int)*(numVerts+1)));
    CUDA_SAFE_CALL(cudaMalloc(&d_edgeArrayAux,adj_list_length*sizeof(unsigned int)));

    // Initialize frontier and cost arrays
    for (int index=0;index<numVerts;index++){
        costArray[index]=UINT_MAX;
    }

    // Set vertex to start traversal from
    const unsigned int source_vertex=0; //op.getOptionInt("source_vertex");
    costArray[source_vertex]=0;

    // Initialize timers
    cudaEvent_t start_cuda_event, stop_cuda_event;
    CUDA_SAFE_CALL(cudaEventCreate(&start_cuda_event));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_cuda_event));

    // Transfer frontier, cost array and adjacency lists on GPU
    cudaEventRecord(start_cuda_event, 0);
    CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArray, edgeArray,sizeof(unsigned int)*(numVerts+1),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_edgeArrayAux,edgeArrayAux,sizeof(unsigned int)*adj_list_length,cudaMemcpyHostToDevice));
    cudaEventRecord(stop_cuda_event,0);
    cudaEventSynchronize(stop_cuda_event);
    float inputTransferTime=0;
    cudaEventElapsedTime(&inputTransferTime,start_cuda_event,stop_cuda_event);

    // Get the device properties for kernel configuration
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,device);
    if (devProp.major > 2 || (devProp.major == 1 && devProp.minor >= 2)){
        if(outputLevel >= 1) printf("Running on GPU  %d (%s)\n", device , devProp.name);
    }

    // Get the kernel configuration
    int numBlocks=0;
    numBlocks=(int)ceil((double)numVerts/(double)PARENT_BLOCK_SIZE);
    if (numBlocks > devProp.maxGridSize[0]){
        std::cerr << "Max number of blocks exceeded";
        return;
    }

    unsigned int *cpu_cost = new unsigned int[numVerts];
    // Perform cpu bfs traversal for verifying results
    G->GetVertexLengths(cpu_cost,source_vertex);

    // Start the benchmark
    if(outputLevel >= 1) std::cout<<"Running Benchmark" << std::endl;
    double totalKernelTime = 0;
    for(int run = -warmup; run < runs; run++) {

        if(outputLevel >= 1) {
            if(run < 0) {
                std::cout << "Warmup:\t";
            } else {
                std::cout << "Run " << run << ":\t";
            }
        }

        double runKernelTime = 0;
        *flag = 1; // Flag set when there are nodes to be traversed in frontier
        int iter = 0;
        while (*flag) { // While there are nodes to traverse

            // Set flag to 0
            *flag=0;
            CUDA_SAFE_CALL(cudaMemcpy(d_flag,flag,sizeof(int),cudaMemcpyHostToDevice));

            // Set pending launch count
            cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, numBlocks*PARENT_BLOCK_SIZE);

            // Launch kernel
            cudaEventRecord(start_cuda_event,0);
            launch_kernel(d_costArray,d_edgeArray,d_edgeArrayAux, numVerts, iter, d_flag);
            CHECK_CUDA_ERROR();
            cudaEventRecord(stop_cuda_event,0);
            cudaEventSynchronize(stop_cuda_event);
            float iterKernelTime = 0;
            cudaEventElapsedTime(&iterKernelTime, start_cuda_event, stop_cuda_event);
            runKernelTime += iterKernelTime;

            // Read flag
            CUDA_SAFE_CALL(cudaMemcpy(flag,d_flag,sizeof(int),cudaMemcpyDeviceToHost));

            if(outputLevel >= 1) printf("iter = %d, kernel time = %f\t", iter, iterKernelTime);

            iter++;

        }
        if(outputLevel >= 1) printf("run kernel time = %f\t", runKernelTime);
        if(run >= 0) totalKernelTime += runKernelTime;

        // Copy result back
        CUDA_SAFE_CALL(cudaMemcpy(costArray,d_costArray,sizeof(cost_type)*numVerts,cudaMemcpyDeviceToHost));

        // Count number of vertices visited
        unsigned int numVisited=0;
        for (int i=0;i<numVerts;i++){
            if (costArray[i]!=UINT_MAX)
                numVisited++;
        }

        // Verify Results against serial BFS
        bool dump_paths = 0;
        unsigned int unmatched_verts = verify_results(cpu_cost,costArray,numVerts,dump_paths);
        if(outputLevel >= 1) {
            if (unmatched_verts == 0){
                std::cout << "success\n";
            }
            else{
                std::cout << "failed\n";
            }
        }

        // Reset for next run
        if (run < runs - 1) {
            // Reset the arrays to perform BFS again
            for (int index=0;index<numVerts;index++){
                costArray[index]=UINT_MAX;
            }
            costArray[source_vertex]=0;
            CUDA_SAFE_CALL(cudaMemcpy(d_costArray, costArray,sizeof(cost_type)*numVerts, cudaMemcpyHostToDevice));
        }

    }

    if(outputLevel >= 1) {
        std::cout<<"Average kernel time = " << totalKernelTime/runs << " ms\n";
    } else {
        std::cout<< totalKernelTime/runs;
    }

    // Clean up
    delete[] cpu_cost;
    CUDA_SAFE_CALL(cudaEventDestroy(start_cuda_event));
    CUDA_SAFE_CALL(cudaEventDestroy(stop_cuda_event));
    CUDA_SAFE_CALL(cudaFreeHost(costArray));
    CUDA_SAFE_CALL(cudaFree(d_costArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArray));
    CUDA_SAFE_CALL(cudaFree(d_edgeArrayAux));

}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the BFS benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
// Programmer: Aditya Sarwade
// Creation: June 16, 2011
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(std::string graph_file, int runs, int warmup, int outputLevel, int num_vertices, int degree)
{

    // First, check if the device supports atomics, which are required
    // for this benchmark.  If not, return the "NoResult" sentinel.int device;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    //adjacency list variables
    //number of vertices and edges in graph
    unsigned int numVerts,numEdges;
    //Get the graph filename
    //std::string inFileName = "random"; //op.getOptionString("graph_file");
    std::string inFileName = graph_file;
    //max degree in graph
    Graph *G=new Graph();

    unsigned int **edge_ptr1 = G->GetEdgeOffsetsPtr();
    unsigned int **edge_ptr2 = G->GetEdgeListPtr();
    //Load simple k-way tree or from a file
    if (inFileName == "random")
    {
        //Load simple k-way tree
        //unsigned int prob_sizes[4] = {1000,10000,100000,1000000};
        numVerts = num_vertices; //prob_sizes[3 /*op.getOptionInt("size")-1*/];
        int avg_degree = degree; //op.getOptionInt("degree");
        if(avg_degree<1)
            avg_degree=1;

        //allocate memory for adjacency lists
        //edgeArray =new unsigned int[numVerts+1];
        //edgeArrayAux=new unsigned int[numVerts*(avg_degree+1)];

        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                    sizeof(unsigned int)*(numVerts+1)));
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                    sizeof(unsigned int)*(numVerts*(avg_degree+1))));

        //Generate simple tree
        G->GenerateSimpleKWayGraph(numVerts,avg_degree);
    }
    else
    {
        //open the graph file
        FILE *fp=fopen(inFileName.c_str(),"r");
        if(fp==NULL)
        {
            std::cerr <<"Error: Graph Input File Not Found." << std::endl;
            return;
        }

        //get the number of vertices and edges from the first line
        const char delimiters[]=" \n";
        char charBuf[MAX_LINE_LENGTH];
        fgets(charBuf,MAX_LINE_LENGTH,fp);
        char *temp_token = strtok (charBuf, delimiters);
        while(temp_token[0]=='%')
        {
            fgets(charBuf,MAX_LINE_LENGTH,fp);
            temp_token = strtok (charBuf, delimiters);
        }
        numVerts=atoi(temp_token);
        temp_token = strtok (NULL, delimiters);
        numEdges=atoi(temp_token);

        //allocate pinned memory
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr1,
                    sizeof(unsigned int)*(numVerts+1)));
        CUDA_SAFE_CALL(cudaMallocHost(edge_ptr2,
                    sizeof(unsigned int)*(numEdges*2)));

        fclose(fp);
        //Load the specified graph
        G->LoadMetisGraph(inFileName.c_str());
    }
    if(outputLevel >= 1) {
        std::cout<<"Vertices: "<<G->GetNumVertices() << std::endl;
        std::cout<<"Edges: "<<G->GetNumEdges() << std::endl;
        std::cout<<"Degree: "<< degree << std::endl;
    }

    // Run test
    RunTest(G, degree, runs, warmup, outputLevel);

    //Clean up
    delete G;
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr1));
    CUDA_SAFE_CALL(cudaFreeHost(*edge_ptr2));
}

int main(int argc, char *argv[]) {

    // Parameters
    int warmup          = 1;
    int runs            = 3;
    int outputLevel     = 1;
    std::string graph   = "random";
    int num_vertices    = 10000;
    int degree          = 1000;
    int opt;
    while((opt = getopt(argc, argv, "w:r:o:g:v:d:h")) >= 0) {
        switch(opt) {
            case 'w': warmup          = atoi(optarg); break;
            case 'r': runs            = atoi(optarg); break;
            case 'o': outputLevel     = atoi(optarg); break;
            case 'g': graph           = optarg;       break;
            case 'v': num_vertices    = atoi(optarg); break;
            case 'd': degree          = atoi(optarg); break;
            default : std::cerr <<
                      "\nUsage:  ./bfs [options]"
                          "\n"
                          "\n    -w <W>    # of warmup runs (default=1)"
                          "\n    -r <R>    # of timed runs (default=3)"
                          "\n    -o <O>    level of output verbosity (0: one CSV row, 1: moderate, 2: verbose)"
                          "\n    -g <G>    graph (default=random)"
                          "\n    -v <V>    # of vertices (default=10000)"
                          "\n    -d <D>    degree (default=1000)"
                          "\n    -h        help\n\n";
                      exit(0);
        }
    }

    // Run the benchmark
    RunBenchmark(graph, runs, warmup, outputLevel, num_vertices, degree);

    return 0;
}

