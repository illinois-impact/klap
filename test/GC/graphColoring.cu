
#include "common.h"

__device__ unsigned int numOfConflicts = 0;

//----------------------- SDO improved -----------------------//
//
// Author: Shusen & Pascal
// returns the degree of that node
int __device__ degree_d(int vertex, int *degreeList){
    return degreeList[vertex];
}



// Author: Shusen & Pascal
// saturation of a vertex
int __device__ saturation(int vertex, int *adjacencyList, int *graphColors, int maxDegree, int start, int end){
    int saturation = 0;	
    int colors[256];
    for (int j=0; j<256; j++)
        colors[j] = 0;


    for (int i=0; i<maxDegree; i++){
        if (adjacencyList[vertex*maxDegree + i] < start)
            continue;

        if (adjacencyList[vertex*maxDegree + i] > end)
            break;

        if (adjacencyList[vertex*maxDegree + i] != -1)
            //colors[ graphColors[vertex] ] = 1;			// at each colored set the array to 1
            colors[ graphColors[adjacencyList[vertex*maxDegree + i]] ] = 1;			// at each colored set the array to 1
        else
            break;
    }


    for (int i=1; i<maxDegree+1; i++)					// count the number of 1's but skip uncolored
        if (colors[i] == 1)
            saturation++;

    return saturation;
}




// Author: Shusen & Pascal
// colors the vertex with the min possible color
int __device__ color(int vertex, int *adjacencyList, int *graphColors, int maxDegree, int numColored, int start, int end){
    int colors[256];
    for (int j=0; j<246; j++)
        colors[j] = 0;


    if (graphColors[vertex] == 0)
        numColored++;

    for (int i=0; i<maxDegree; i++){						// set the index of the color to 1

        // Limits color checking to subgraph
        /*
           if (adjacencyList[vertex*maxDegree + i] < start)
           continue;

           if (adjacencyList[vertex*maxDegree + i] > end)
           break;
         */

        if (adjacencyList[vertex*maxDegree + i] != -1)
            colors[  graphColors[  adjacencyList[vertex*maxDegree + i]  ]  ] = 1;
        else 
            break;
    }


    for (int i=1; i<maxDegree+1; i++)					// nodes still equal to 0 are unassigned
        if (colors[i] != 1){
            graphColors[vertex] = i;
            break;
        }

    return numColored;
}




// Author: Shusen & Pascal
// does the coloring
__global__ void colorGraph_SDO(int *adjacencyList, int *graphColors, int *degreeList, int sizeGraph, int maxDegree)
{
    int start, end;
    int subGraphSize, numColored = 0;
    int satDegree, max, index;

    subGraphSize = sizeGraph/(gridDim.x * blockDim.x);
    start = (sizeGraph/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
    end = start + subGraphSize;

    while (numColored < subGraphSize){
        max = -1;

        for (int i=start; i<end; i++){
            if (graphColors[i] == 0)			// not colored
            {
                satDegree = saturation(i,adjacencyList,graphColors, maxDegree, start, end);

                if (satDegree > max){
                    max = satDegree;
                    index = i;				
                }

                if (satDegree == max){
                    if (degree_d(i,degreeList) > degree_d(index,degreeList))
                        index = i;
                }


            }
            numColored = color(index,adjacencyList,graphColors, maxDegree, numColored, start, end);
        }
    }
}



__global__ void colorGraphBis_SDO(int *adjacencyList, int *graphColors, int *degreeList, int sizeGraph, int maxDegree)
{
    int start, end;
    int subGraphSize, numColored = 0;
    int satDegree, max, index;

    subGraphSize = sizeGraph/(gridDim.x * blockDim.x);
    start = (sizeGraph/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
    end = start + subGraphSize;

    while (numColored < subGraphSize){
        max = -1;

        for (int i=start; i<end; i++){
            if (graphColors[i] == 0)			// not colored
            {
                satDegree = saturation(i,adjacencyList,graphColors, maxDegree, start, end);

                if (satDegree > max){
                    max = satDegree;
                    index = i;				
                }

                if (satDegree == max){
                    if (degree_d(i,degreeList) > degree_d(index,degreeList))
                        index = i;
                }
            }

            numColored = color(index, adjacencyList, graphColors, maxDegree, numColored, start, end);
        }
    }
}



__global__ void conflictSolveSDO(int *adjacencyList, int *conflict, int *graphColors, int *degreeList, int sizeGraph, int maxDegree){
    int start, end, index;
    int subGraphSize, numColored = 0;
    int satDegree, max;

    subGraphSize = sizeGraph/(gridDim.x * blockDim.x);
    start = (sizeGraph/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
    end = start + subGraphSize;

    // Set their color to 0
    for (int i=0; i<numOfConflicts; i++)
        graphColors[conflict[i]-1] = 0;


    while (numColored < numOfConflicts){
        max = -1;

        for (int i=0; i<numOfConflicts; i++){
            int vertex = conflict[i]-1;
            if (graphColors[vertex] == 0)                        // not colored
            {
                satDegree = saturation(i,adjacencyList,graphColors, maxDegree, start, end);

                if (satDegree > max){
                    max = satDegree;
                    index = vertex;                              
                }

                if (satDegree == max){
                    if (degree_d(vertex,degreeList) > degree_d(index,degreeList))
                        index = vertex;
                }

                numColored = color(index,adjacencyList,graphColors, maxDegree, numColored, start, end);
            }else{
                numColored++;
            }


        }
    }
}




//----------------------- First Fit Adjacency List -----------------------//
//
// Author: Pascal
// First Fit
#if 0
__global__ void colorGraph_FF(int *adjacencyListD, int *colors, int size, int maxDegree){
    int i, j, start, end;
    int subGraphSize, numColors = 0;

    subGraphSize = size/(gridDim.x * blockDim.x);
    start = (size/gridDim.x * blockIdx.x) + (subGraphSize * threadIdx.x);
    end = start + subGraphSize;


    int degreeArray[100];
    for(i=start; i<end; i++)
    {
        for(j=0; j<maxDegree; j++)
            degreeArray[j] = j+1;


        for (j=0; j<maxDegree; j++){
            int vertexNeigh = i*maxDegree + j;

            if (adjacencyListD[vertexNeigh] != -1){
                if (colors[ adjacencyListD[vertexNeigh] ] != 0)
                    degreeArray[ colors[adjacencyListD[vertexNeigh]] -1 ] = 0;
            }
            else
                break;
        }


        for(j=0; j<maxDegree; j++)
            if(degreeArray[j] != 0){
                colors[i] = degreeArray[j];
                break;
            }

        if(colors[i] > numColors)
            numColors = colors[i];		
    }
}
#endif

//----------------------- Detects conflicts -----------------------//
//
// Author: Peihong
__global__ void conflictsDetection(int *adjacentListD, int *boundaryListD, int *colors, int *conflictD, long size, int boundarySize, int maxDegree){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i, j;


    if (idx < boundarySize){
        i = boundaryListD[idx];
        conflictD[idx] = 0;
        for (int k=0; k < maxDegree; k++)
        {
            j = adjacentListD[i*maxDegree + k];
            if (j>i && (colors[i] == colors[j]))
            {
                //conflictD[idx] = min(i,j)+1;	
                conflictD[idx] = i+1;	

                colors[i] = 0;				// added!!!!!!!!
            }		
        }
    }
}

// Temporary only: will have to be merged in conflictsDetection as a counting reduction operation!!!
void conflictCount(int *conflictD, int *conflictListD, int boundaryCount){
    int conflictCount = 0;

    for (int i=0; i< boundaryCount; i++)
    {
        int node = conflictD[i];

        if (node >= 1)
        {
            conflictListD[conflictCount] = node;
            conflictCount++;
        }
    }

}

//----------------------- Main -----------------------//

extern "C"
void cudaGraphColoring(int *adjacentList, int *boundaryList, int *graphColors, int *degreeList, int *conflict, int boundarySize, int maxDegree, int graphSize, int passes, int warmup, int runs, int outputLevel) {

    int *adjacentListD, *colorsD, *conflictD, *boundaryListD, *degreeListD, *conflictListD;     
    int gridsize = ceil((float)boundarySize/(float)SUBSIZE_BOUNDARY);
    int blocksize = SUBSIZE_BOUNDARY;
    int *numConflicts;

    int *graphColors_cdp = new int[graphSize*sizeof(int)];          
    int *conflict_cdp = new int[boundarySize*sizeof(int)];
    int *colorsD_cdp, *conflictD_cdp;

    cudaEvent_t start_col, start_confl, stop_col, stop_confl, start_mem, stop_mem;         
    float elapsedTime_memory, elapsedTime_col, elapsedTime_confl; 

    //-------------- memory allocation -----------------!
    cudaEventCreate(&start_mem); 
    cudaEventCreate(&stop_mem); 
    cudaEventRecord(start_mem, 0); 

    cudaMalloc((void**)&adjacentListD, graphSize*maxDegree*sizeof(int));
    cudaMalloc((void**)&colorsD, graphSize*sizeof(int));
    cudaMalloc((void**)&conflictD, boundarySize*sizeof(int));
    cudaMalloc((void**)&boundaryListD, boundarySize*sizeof(int));
    cudaMalloc((void**)&degreeListD, graphSize*sizeof(int));
    cudaMalloc((void**)&numConflicts, 1*sizeof(int));
    cudaMalloc((void**)&conflictListD, boundarySize*sizeof(int));

    cudaMalloc((void**)&colorsD_cdp, graphSize*sizeof(int));
    cudaMalloc((void**)&conflictD_cdp, boundarySize*sizeof(int));

    cudaEventRecord(stop_mem, 0); 
    cudaEventSynchronize(stop_mem); 

    float totalKernelTime = 0;
    for(int run = -warmup; run < runs; run++) {

        if(outputLevel >= 1) {
            if(run < 0) {
                std::cout << "Warmup:\t";
            } else {
                std::cout << "Run " << run << ":\t";
            }
        }

        //-------------- memory transfer -----------------!

        memset(conflict, 0, boundarySize*sizeof(int));                        // conflict array initialized to 0
        memset(graphColors, 0, graphSize*sizeof(int));                         // reset colors to 0

        cudaMemcpy(adjacentListD, adjacentList, graphSize*maxDegree*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(colorsD, graphColors, graphSize*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(boundaryListD, boundaryList, boundarySize*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(degreeListD, degreeList, graphSize*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(conflictListD, 0, boundarySize*sizeof(int));

        memset(graphColors_cdp, 0, graphSize*sizeof(int));                         // reset colors to 0 
        memset(conflict_cdp, 0, boundarySize*sizeof(int));  

        cudaMemcpy(colorsD_cdp, graphColors, graphSize*sizeof(int), cudaMemcpyHostToDevice);

        dim3 dimGrid_col(GRIDSIZE);
        dim3 dimBlock_col(BLOCKSIZE);

        unsigned int dimGrid_confl = gridsize;
        unsigned int dimBlock_confl = blocksize;

        //-------------- Sequential Graph coloring -----------------!
        cudaEventCreate(&start_col); 
        cudaEventCreate(&stop_col); 
        cudaEventRecord(start_col, 0); 
        //colorGraph_FF<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, graphSize, maxDegree);				// First Fit
        colorGraph_SDO<<<dimGrid_col, dimBlock_col>>>(adjacentListD, colorsD, degreeListD,graphSize, maxDegree);		// SDO improved
        cudaEventRecord(stop_col, 0); 
        cudaEventSynchronize(stop_col); 

        for (int times = 1; times < passes; times++) {
            if(outputLevel >= 1) printf("times = %d\n", times);
            //-------------- Conflict resolution -----------------!
            cudaEventCreate(&start_confl); 
            cudaEventCreate(&stop_confl); 
            cudaEventRecord(start_confl, 0); 

            conflictsDetection<<<dimGrid_confl, dimBlock_confl>>>(adjacentListD, boundaryListD, colorsD, conflictD, graphSize, boundarySize, maxDegree);

            cudaEventRecord(stop_confl, 0); 
            cudaEventSynchronize(stop_confl); 


            cudaEventCreate(&start_col); 
            cudaEventCreate(&stop_col); 
            cudaEventRecord(start_col, 0); 

            conflictSolveSDO<<<dimGrid_col, dimBlock_col>>>(adjacentListD, conflictListD, colorsD, degreeListD, graphSize, maxDegree);

            cudaEventRecord(stop_col, 0); 
            cudaEventSynchronize(stop_col); 
        }

        cudaMemcpy(colorsD_cdp, colorsD, graphSize*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(conflictD_cdp, conflictD, boundarySize*sizeof(int), cudaMemcpyDeviceToDevice);

        cudaEventCreate(&start_confl); 
        cudaEventCreate(&stop_confl); 
        cudaEventRecord(start_confl, 0); 

        conflictsDetection<<<dimGrid_confl, dimBlock_confl>>>(adjacentListD, boundaryListD, colorsD, conflictD, graphSize, boundarySize, maxDegree);

        cudaEventRecord(stop_confl, 0); 
        cudaEventSynchronize(stop_confl);

        cudaEvent_t start_confl_cdp, stop_confl_cdp;
        cudaEventCreate(&start_confl_cdp);
        cudaEventCreate(&stop_confl_cdp);
        cudaEventRecord(start_confl_cdp, 0); 

        launch_kernel(dimGrid_confl, dimBlock_confl, adjacentListD, boundaryListD, colorsD_cdp, conflictD_cdp, graphSize, boundarySize, maxDegree);

        cudaEventRecord(stop_confl_cdp, 0); 
        cudaEventSynchronize(stop_confl_cdp); 

        //-------------- Cleanup -----------------!
        cudaMemcpy(graphColors, colorsD, graphSize*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(conflict, conflictD, boundarySize*sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemcpy(graphColors_cdp, colorsD_cdp, graphSize*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(conflict_cdp, conflictD_cdp, boundarySize*sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventElapsedTime(&elapsedTime_memory, start_mem, stop_mem); 
        cudaEventElapsedTime(&elapsedTime_col, start_col, stop_col); 
        cudaEventElapsedTime(&elapsedTime_confl, start_confl, stop_confl); 
        float elapsedTime_confl_cdp;
        cudaEventElapsedTime(&elapsedTime_confl_cdp, start_confl_cdp, stop_confl_cdp); 
        if(outputLevel >= 2) {
            cout << "GPU timings ~ Memory transfer: " << elapsedTime_memory  << " ms     Coloring: " 
                << elapsedTime_col << " ms " << endl;
            cout <<  "Conflict: " << elapsedTime_confl << " ms" << endl; 
            cout <<  "Conflict_cdp: " << elapsedTime_confl_cdp << " ms" << endl; 
        }
        if(outputLevel >= 1) std::cout << "run kernel time = " << elapsedTime_confl_cdp << "\t";
        if (run >= 0) totalKernelTime += elapsedTime_confl_cdp;

        // Verify
        for(int i = 0; i < graphSize; i++){
            //printf("graphColors %d - %d ? %d\n", i, graphColors_cdp[i], graphColors[i]);
            if(graphColors_cdp[i] != graphColors[i] && graphColors_cdp[i] != 0 && graphColors[i] != 0){
                fprintf(stderr, "Error graphColors %d - %d ? %d\t", i, graphColors_cdp[i], graphColors[i]);
                fprintf(stderr, "Test failed\n");
                break;
            }
        }
        for(int i = 0; i < boundarySize; i++){
            //printf("conflict %d - %d ? %d\n", i, conflict_cdp[i], conflict[i]);
            if(conflict_cdp[i] != conflict[i]){
                fprintf(stderr, "Error conflict %d - %d != %d\t", i, conflict_cdp[i], conflict[i]);
                fprintf(stderr, "Test failed\n");
                break;
            }
        }
        //printf("Test passed\n");

    }

    if(outputLevel >= 1) {
        std::cout<< "Average kernel time = " << totalKernelTime/runs << " ms\n";
    } else {
        std::cout<< totalKernelTime/runs;
    }

    cudaFree(adjacentListD);
    cudaFree(colorsD);
    cudaFree(conflictD);
    cudaFree(boundaryListD);
    cudaFree(degreeListD);
    cudaFree(numConflicts);
    cudaFree(conflictListD);
    cudaFree(colorsD_cdp);
    cudaFree(conflictD_cdp);

}

