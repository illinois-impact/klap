// Graph coloring

#include <assert.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "common.h"

using namespace std;

//----------------------- Graph initializations -----------------------//
// Author: Shusen

void generateMetisFormatFromAdjacencyList(unsigned int*AdjacencyList, int graphsize, int edgesize, int maxDegree, char* filename)
{
    ofstream metisFile;
    metisFile.open(filename);

    metisFile << graphsize <<" "<< edgesize << endl;
    for(int i=0; i<graphsize; i++)
    {
        for(int j=0; j<maxDegree; j++)
        {
            if (AdjacencyList[i*maxDegree+j] != (unsigned int)-1)
                metisFile << AdjacencyList[i*maxDegree+j]<<" ";
            else
            {
                metisFile << endl;
                continue;
            }
        }
    }

    metisFile.close();
}

void generateMetisFormatFromCompactAdjacencyList(unsigned int* compactAdjacencyList, unsigned int* vertexStartList, int graphsize, int edgesize, int maxDegree, char* filename)
{
    ofstream metisFile;
    metisFile.open(filename);

    metisFile << graphsize <<" "<< edgesize << endl;
    for(int i=0; i<graphsize; i++)
    {
        int idegree = vertexStartList[i+1] - vertexStartList[i];
        for(int j=0; j<idegree; j++)
        {
            metisFile << compactAdjacencyList[ vertexStartList[i] + j]<<" ";
        }
        metisFile << endl;
    }

    metisFile.close();
    //cout << "Write to Metis Format in:" << filename << endl;

}

void getAdjacentCompactListFromSparseMartix_mtx(const char* filename, unsigned int *&compactAdjacencyList, unsigned int *&vertexStartList, int &graphsize, long &edgesize, int &maxDegree)
{
    unsigned int row=0, col=0;
    long entries=0;
    //calculate maxDegree in the following loop
    float donotcare = 0;
    float nodecol = 0;
    float noderow = 0;

    ///////////////////////////////////Read file for the first time ////////////////////////////////////
    ifstream mtxf;
    mtxf.open(filename);
    //cout << string(filename) << endl;
    while(mtxf.peek()=='%')
        mtxf.ignore(512, '\n');//

    mtxf >> row >> col >> entries ;
    //cout<< row <<" " << col <<" " << entries << endl;
    graphsize = col>row? col:row;

    int *graphsizeArray = new int[graphsize];
    memset(graphsizeArray, 0 , sizeof(int)*graphsize);
    edgesize = 0;
    for (long i=0; i<entries; i++)
    {
        mtxf >> noderow >> nodecol >> donotcare;
        //cout << noderow << " " << nodecol << " " << donotcare << endl;
        //assert(noderow!=nodecol);
        if(noderow == nodecol)
            continue;
        else
            edgesize++;
        graphsizeArray[(int)noderow-1]++;
        graphsizeArray[(int)nodecol-1]++;
    }
    //cout << "edgesize: "<< edgesize << endl;
    //	for(int i=0; i<graphsize; i++)
    //		cout << graphsizeArray[i] <<endl;
    //exit(0);
    mtxf.close();

    /////////////////////////////////////close the file/////////////////////////////////////////////
    long listSize = 0;
    //calculate the size of the adjacency list
    maxDegree = 0;
    for(unsigned int i=0; i<graphsize; i++)
    {
        listSize += graphsizeArray[i];
        if(graphsizeArray[i] > maxDegree)
            maxDegree = graphsizeArray[i];
    }
    //cout <<"edge*2: "<<listSize<<endl;
    //cout <<"maxDegree: "<< maxDegree << endl;

    ///////////////////////////////////Read file for the second time ////////////////////////////////////
    mtxf.open(filename);
    //int nodeindex=0, connection=0;
    while(mtxf.peek()=='%')
        mtxf.ignore(512, '\n');//
    mtxf >> donotcare >> donotcare >> donotcare;
    //cout<<donotcare<<endl;
    set<unsigned int>** setArray = new set<unsigned int>* [graphsize];
    assert(setArray);
    memset(setArray, 0 , sizeof(set<unsigned int>*)*graphsize);
    unsigned int x, y;
    //cout<< "finished allocate memory" << endl;
    for(unsigned int i=0; i<entries; i++)
    {
        mtxf >> x >> y >> donotcare;
        x--; y--; //node index start from 0
        //cout << x << " " << y << endl;
        if(x==y)
        {
            continue;
        }
        if (setArray[x] == NULL)
            setArray[x] = new set<unsigned int>();
        if (setArray[y] == NULL)
            setArray[y] = new set<unsigned int>();
        setArray[x]->insert(y);
        setArray[y]->insert(x);
    }
    //cout<< "finished assignment of all the entries" << endl;
    mtxf.close();

    /////////////////////////////////////close the file/////////////////////////////////////////////
    compactAdjacencyList = new unsigned int[listSize];
    memset(compactAdjacencyList, 0, sizeof(unsigned int)*listSize);
    vertexStartList = new unsigned int[graphsize];
    memset(vertexStartList, 0, sizeof(unsigned int)*graphsize);
    unsigned int currentPos = 0;
    for(unsigned int i=0; i<graphsize; i++)
    {
        //cout << "currentPos: " << currentPos << endl;
        if(setArray[i] != NULL)
        {
            vertexStartList[i] = currentPos;
            set<unsigned int>::iterator it = setArray[i]->begin();
            if (i == 1137){
                //cout << "testingggggggggggggggggggggggggggg " << endl;
            }
            for(; it != setArray[i]->end(); it++)
            {
                if (i == 1137){
                    //cout << *it <<  " ";
                }
                compactAdjacencyList[currentPos] = *it;
                currentPos++;
            }
        }
        else
            vertexStartList[i] = currentPos;
    }
    //	for(unsigned int i=0; i<graphsize; i++)
    //		cout<< vertexStartList[i] << " ";
    //cout << "inside function"<< endl;
}

//----------------------- Graph initializations -----------------------//

// Author: Pascal
// genetates a graph
void generateMatrix(int *matrix, int graphSize, int num){
    int x, y;
    int count = 0;

    while (count < num){
        x = rand()%graphSize;
        y = rand()%graphSize;

        if (matrix[x*graphSize + y] != 1)            // if not already assigned an edge
            if (x != y){
                matrix[x*graphSize + y] = 1;       // non directional graph
                matrix[y*graphSize + x] = 1;
                count++;
            }
    }
}

// Author:Peihong
// node index start from 1
// gets an adjacency list from an adjacencyMatrix
void getAdjacentList(int *adjacencyMatrix, int *adjacentList, int size, int maxDegree)
{
    for (int i=0; i<size; i++){
        int nbCount = 0;
        for (int j=0; j<size; j++){
            if ( adjacencyMatrix[i*size + j] == 1)
            {
                adjacentList[i*maxDegree + nbCount] = j;
                nbCount++;
            }
        }
    }

    /*
    // Adj list display
    for (int i=0; i<10; i++){
    for (int j=0; j<maxDegree; j++){
    cout << adjacentList[i*maxDegree + j] << " ";
    }
    cout << endl;
    }
     */
}

// Author: Pascal
// get the degree information for a graph
int getMaxDegree(int *adjacencyMatrix, int size){
    int maxDegree = 0;
    int degree;

    for (int i=0; i<size; i++){
        degree = 0;

        for (int j=0; j<size; j++)
            if (    adjacencyMatrix[i*size + j] == 1)
                degree++;

        if (degree > maxDegree)
            maxDegree = degree;
    }

    return maxDegree;
}

// Author: Pascal
// get the degree of each element in the graph and returns the maximum degree
void getDegreeList(int *adjacencyList, int *degreeList, int sizeGraph, int maxDegree){
    for (int i=0; i<sizeGraph; i++){
        int count = 0;

        for (int j=0; j<maxDegree; j++){
            if (adjacencyList[i*maxDegree + j] != -1)
                count++;
            else
                break;
        }

        degreeList[i] = count;
    }
}

//int inline min(int n1, int n2)
//{
//    if (n1>=n2)
//        return n2;
//    else
//        return n1;
//}

// Author: Peihong
int getBoundaryList(int *adjacencyMatrix, int *boundaryList, int size, int &boundaryCount, int graphSize){
    int maxDegree = 0;
    int degree;

    set<int> boundarySet;
    boundarySet.clear();

    int subSize = graphSize/(GRIDSIZE*BLOCKSIZE);

    for (int i=0; i<size; i++){
        degree = 0;

        int subIdx = i/(float)subSize;
        int start = subIdx * subSize;
        int end = min( (subIdx + 1)*subSize, size );

        for (int j=0; j<size; j++){
            if ( adjacencyMatrix[i*size + j] == 1)
                degree++;
            if ( adjacencyMatrix[i*size + j] == 1 && (j < start || j >= end))
            {
                boundarySet.insert(i);
            }

        }

        if (degree > maxDegree)
            maxDegree = degree;
    }

    boundaryCount = boundarySet.size();

    set<int>::iterator it = boundarySet.begin();
    for (int i=0; it != boundarySet.end(); it++)
    {
        boundaryList[i] = *it;
        i++;
    }

    return maxDegree;
}

//----------------------- Fast Fit Graph Coloring -----------------------//

// Author: Pascal & Shusen
// GraphColor Adjacency list
int colorGraph_FF(int *list, int *colors, int size, int maxDegree){
    int numColors = 0;
    int i, j;

    int * degreeArray;
    degreeArray = new int[maxDegree+1];


    for (i=0; i<size; i++)
    {
        // initialize degree array
        for (j=0; j<=maxDegree; j++)
            degreeArray[j] = j+1;


        // check the colors
        for (j=0; j<maxDegree; j++){
            if (i == j)
                continue;

            // check connected
            if (    list[i*maxDegree + j] != -1)
                if (colors[list[i*maxDegree + j]] != 0)
                    degreeArray[colors[list[i*maxDegree + j]]-1] = 0;   // set connected spots to 0
        }

        for (j=0; j<=maxDegree; j++)
            if (degreeArray[j] != 0){
                colors[i] = degreeArray[j];
                break;
            }

        if (colors[i] > numColors)
            numColors=colors[i];
    }

    delete[] degreeArray;

    return numColors;
}

//----------------------- SDO Improved Graph Coloring -----------------------//

// Author: Pascal
// returns the degree of that node
int degree(int vertex, int *degreeList){
    return degreeList[vertex];
}

// Author: Pascal
// return the saturation of that node
int saturation(int vertex, int *adjacencyList, int *graphColors, int maxDegree){
    int saturation = 0;
    int *colors = new int[maxDegree+1];

    memset(colors, 0, (maxDegree+1)*sizeof(int));           // initialize array


    for (int i=0; i<maxDegree; i++){
        if (adjacencyList[vertex*maxDegree + i] != -1)
            //  colors[ graphColors[vertex] ] = 1;                      // at each colored set the array to 1
            colors[ graphColors[adjacencyList[vertex*maxDegree + i]] ] = 1;                      // at each colored set the array to 1
        else
            break;
    }


    for (int i=1; i<maxDegree+1; i++)                                       // count the number of 1's but skip uncolored
        if (colors[i] == 1)
            saturation++;

    delete[] colors;

    return saturation;
}

// Author: Pascal
// colors the vertex with the min possible color
int color(int vertex, int *adjacencyList, int *graphColors, int maxDegree, int numColored){
    int *colors = new int[maxDegree + 1];
    memset(colors, 0, (maxDegree+1)*sizeof(int));

    if (graphColors[vertex] == 0)
        numColored++;
    //	else
    //		cout << "Old color: " << graphColors[vertex] << "   ";

    for (int i=0; i<maxDegree; i++)                                         // set the index of the color to 1
        if (adjacencyList[vertex*maxDegree + i] != -1)
            colors[  graphColors[  adjacencyList[vertex*maxDegree + i]  ]  ] = 1;
        else {
            break;
        }



    for (int i=1; i<maxDegree+1; i++)                                       // nodes still equal to 0 are unassigned
        if (colors[i] != 1){
            graphColors[vertex] = i;
            //			cout << " New color:" << i << endl;
            break;
        }

    delete[] colors;

    return numColored;
}

// Author: Pascal
int sdoIm(int *adjacencyList, int *graphColors, int *degreeList, int sizeGraph, int maxDegree){
    int satDegree, numColored, max, index;
    numColored = 0;
    int iterations = 0;


    while (numColored < sizeGraph){
        max = -1;

        for (int i=0; i<sizeGraph; i++){
            if (graphColors[i] == 0)                        // not colored
            {
                satDegree = saturation(i,adjacencyList,graphColors, maxDegree);

                if (satDegree > max){
                    max = satDegree;
                    index = i;
                }

                if (satDegree == max){
                    if (degree(i,degreeList) > degree(index,degreeList))
                        index = i;
                }

                numColored = color(index,adjacencyList,graphColors, maxDegree, numColored);
                iterations++;
            }
        }
    }

    return iterations;
}

//----------------------- Conflict Solve -----------------------//

void conflictSolveSDO(int *adjacencyList, int *conflict, int conflictSize, int *graphColors, int *degreeList, int sizeGraph, int maxDegree){
    int satDegree, numColored, max, index;
    numColored = 0;

    // Set their color to 0
    for (int i=0; i<conflictSize; i++)
        graphColors[conflict[i]-1] = 0;


    while (numColored < conflictSize){
        max = -1;

        for (int i=0; i<conflictSize; i++){
            int vertex = conflict[i]-1;
            if (graphColors[vertex] == 0)                        // not colored
            {
                satDegree = saturation(vertex, adjacencyList, graphColors, maxDegree);

                if (satDegree > max){
                    max = satDegree;
                    index = vertex;
                }

                if (satDegree == max){
                    if (degree(vertex,degreeList) > degree(index,degreeList))
                        index = vertex;
                }
            }

            numColored = color(index,adjacencyList,graphColors, maxDegree, numColored);
        }
    }
}

// Author: Pascal & Shusen
// Solves conflicts using Fast Fit
void conflictSolveFF(int *Adjlist, int size, int *conflict, int conflictSize, int *graphColors, int maxDegree){
    int i, j, vertex, *colorList, *setColors;
    colorList = new int[maxDegree];
    setColors = new int[maxDegree];


    // assign colors up to maxDegree in setColors
    for (i=0; i<maxDegree; i++){
        setColors[i] = i+1;
    }


    for (i=0; i<conflictSize; i++){
        memcpy(colorList, setColors, maxDegree*sizeof(int));                    // set the colors in colorList to be same as setColors

        vertex = conflict[i]-1;


        for (j=0; j<maxDegree; j++){                                            // cycle through the graph
            if ( Adjlist[vertex*maxDegree + j] != -1 )                      		//      check if node is connected
                colorList[ graphColors[Adjlist[vertex*maxDegree + j]]-1 ] = 0;
            else
                break;
        }


        for (j=0; j<maxDegree; j++){                                       	// check the colorList array
            if (colorList[j] != 0){                                         //    at the first spot where we have a color not assigned
                graphColors[vertex] = colorList[j];                         //       we assign that color to the node and
                break;                                                      //   	 exit to the next
            }
        }

    }
}

//----------------------- Checking for error -----------------------//

// Author: Pascal
// Checks if a graph has conflicts or not
void checkCorrectColoring(int *adjacencyMatrix, int *graphColors, int graphSize){
    int numErrors = 0;

    //cout << endl << "==================" << endl << "Error checking for Graph" << endl;

    for (int i=0; i<graphSize; i++)                 // we check each row
    {
        int nodeColor = graphColors[i];
        int numErrorsOnRow = 0;

        for (int j=0; j<graphSize;j++){ // check each column in the matrix

            // skip itself
            if (i == j)
                continue;

            if (adjacencyMatrix[i*graphSize + j] == 1)      // there is a connection to that node
                if (graphColors[j] == nodeColor)
                {
                    //cout << "Color collision from: " << i << " @ " << nodeColor << "  to: " << j << " @ " << graphColors[j] << endl;
                    numErrors++;
                    numErrorsOnRow++;
                    exit(1);
                }
        }

        if (numErrorsOnRow != 0) {
            //cout << "Errors for node " << i << " : " << numErrorsOnRow << endl;
        }
    }

    //cout << "Color errors for graph : " << numErrors << endl << "==================== " << endl ;
}

void convert(int *adjacencyMatrix, unsigned int *compactAdjacencyList, unsigned int *vertexStartList, int size, int graphSizeRead, int maxDegree){
    int count;
    for (int i=0; i<graphSizeRead; i++)

    {

        count = 0;
        for (int j=vertexStartList[i]; j<vertexStartList[i+1]; j++){

            adjacencyMatrix[i*size + compactAdjacencyList[j]] = 1;
            count++;
        }
        //cout<< i << endl;
    }

}

int findPower(int x){
    int num = 2;
    int powerIndex = 1;
    while (num <= x){
        powerIndex++;
        num = pow(2,powerIndex);

    }
    //cout << "Closest power: " << num << endl;
    return num;
}
//----------------------- The meat -----------------------//

int main(int argc, char *argv[]){

    // Parameters
    int warmup          = 1;
    int runs            = 3;
    int outputLevel     = 1;
    const char* graph   = "inputs/bcsstk13.mtx";
    int graphSize       = 4096;
    float density       = 0.01;
    int passes          = 1;
    int opt;
    while((opt = getopt(argc, argv, "w:r:o:g:s:d:p:h")) >= 0) {
        switch(opt) {
            case 'w': warmup        = atoi(optarg); break;
            case 'r': runs          = atoi(optarg); break;
            case 'o': outputLevel   = atoi(optarg); break;
            case 'g': graph         = optarg      ; break;
            case 's': graphSize     = atoi(optarg); break;
            case 'd': density       = atof(optarg); break;
            case 'p': passes        = atof(optarg); break;
            default : std::cerr <<
                      "\nUsage:  ./gc [options]"
                          "\n"
                          "\n    -w <W>    # of warmup runs (default=1)"
                          "\n    -r <R>    # of timed runs (default=3)"
                          "\n    -o <O>    level of output verbosity (0: one CSV row, 1: moderate, 2: verbose)"
                          "\n    -g <G>    graph file name (default=inputs/bcsstk13.mtx)"
                          "\n    -s <S>    graph size (default=4096)"
                          "\n    -d <D>    density (default=0.01)"
                          "\n    -p <P>    passes (default=1)"
                          "\n    -h        help\n\n";
                      exit(0);
        }
    }

    int maxDegree, numColorsSeq, boundaryCount;
    unsigned int *compactAdjacencyList;
    unsigned int *vertexStartList;

    bool artificial = false;
    bool sdo = true;

    //const long GRAPHSIZE = 2048;    // number of nodes
    //const float DENSITY = 0.01;
    //const long numEdges = 150000;    // number of edges
    //const long NUMEDGES = DENSITY*GRAPHSIZE*(GRAPHSIZE-1)/2;
    long numEdges = density*graphSize*(graphSize-1)/2;

    int graphSizeRead;
    if (artificial == false)
    {
        getAdjacentCompactListFromSparseMartix_mtx(graph, compactAdjacencyList,  vertexStartList, graphSizeRead, numEdges, maxDegree);
        vertexStartList[graphSizeRead] = numEdges*2;
        graphSize = findPower(graphSizeRead);
    }
    generateMetisFormatFromCompactAdjacencyList(compactAdjacencyList, vertexStartList, graphSizeRead, numEdges, maxDegree, (char*)"outputs/metisTest.txt");

    int *adjacencyMatrix = new int[graphSize*graphSize*sizeof(int)];
    int *graphColors = new int[graphSize*sizeof(int)];
    int *boundaryList = new int[graphSize*sizeof(int)];

    memset(adjacencyMatrix, 0, graphSize*graphSize*sizeof(int));
    memset(graphColors, 0, graphSize*sizeof(int));
    memset(boundaryList, 0, graphSize*sizeof(int));

    boundaryCount = numColorsSeq = 0;

    //long randSeed = time(NULL);
    //srand ( randSeed );  // initialize random numbers
    srand ( 1272167817 );  // initialize random numbers

    cudaEvent_t start, stop, stop_1, stop_4;

    //--------------------- Graph Creation ---------------------!
    // initialize graph
    if (artificial == false)
        convert(adjacencyMatrix, compactAdjacencyList, vertexStartList, graphSize, graphSizeRead, maxDegree);
    else
        generateMatrix(adjacencyMatrix, graphSize, numEdges);

    //cout << "Graph Created!" << endl;

    // Display graph: Adjacency Matrix
    /*
       cout << "Adjacency Matrix:" << endl;
       for (int i=0; i<graphSize; i++){
       for (int j=0; j<graphSize; j++)
       cout << adjacencyMatrix[i*graphSize + j] << "  ";
       cout << endl;
       }
     */

    // determining the maximum degree
    //maxDegree = getMaxDegree(adjacencyMatrix, graphSize);
    cudaEvent_t start_b, stop_b;
    float elapsedTimeBoundary;
    cudaEventCreate(&start_b);
    cudaEventCreate(&stop_b);
    cudaEventRecord(start_b, 0);

    maxDegree = getBoundaryList(adjacencyMatrix, boundaryList, graphSize, boundaryCount, graphSize);	// return maxDegree + boundaryCount (as ref param)

    cudaEventRecord(stop_b, 0);
    cudaEventSynchronize(stop_b);
    cudaEventElapsedTime(&elapsedTimeBoundary, start_b, stop_b);
    //cout<<"Get boundaryList :"<<elapsedTimeBoundary<<" ms"<<endl;
    //cout<<"maxDegree="<<maxDegree<<endl;

    // Get adjacency list
    int *adjacentList = new int[graphSize*maxDegree*sizeof(int)];
    memset(adjacentList, -1, graphSize*maxDegree*sizeof(int));
    getAdjacentList(adjacencyMatrix, adjacentList, graphSize, maxDegree);

    // Get degree List
    int *degreeList = new int[graphSize*sizeof(int)];
    memset(degreeList, 0, graphSize*sizeof(int));

    getDegreeList(adjacentList, degreeList, graphSize, maxDegree);


    //--------------------- Sequential Graph Coloring ---------------------!
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    if (sdo == true)
        sdoIm(adjacentList, graphColors, degreeList, graphSize, maxDegree);
    else
        colorGraph_FF(adjacentList, graphColors, graphSize, maxDegree);



    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTimeCPU;
    cudaEventElapsedTime(&elapsedTimeCPU, start, stop);


    // Get colors
    numColorsSeq = 0;
    for (int i=0; i<graphSize; i++){
        if ( numColorsSeq < graphColors[i] )
            numColorsSeq = graphColors[i];
    }




    //--------------------- Checking for color conflict ---------------------!

    //cout << "Sequential Conflict check:";
    checkCorrectColoring(adjacencyMatrix, graphColors, graphSize);
    //cout << endl;




    //--------------------- Parallel Graph Coloring ---------------------!

    int *conflict = new int[boundaryCount*sizeof(int)];                    // conflict array
    memset(conflict, 0, boundaryCount*sizeof(int));                        // conflict array initialized to 0
    memset(graphColors, 0, graphSize*sizeof(int));                         // reset colors to 0

    //--------------- Steps 1, 2 & 3: Parallel Partitioning + Graph coloring + Conflict Detection

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&stop_1);
    cudaEventCreate(&stop_4);
    cudaEventRecord(start, 0);


    int *conflictTmp = new int[boundaryCount*sizeof(int)];
    memset(conflictTmp, 0, boundaryCount*sizeof(int));

    //cudaGraphColoring(adjacentList, boundaryList, graphColors, conflictTmp, boundaryCount, maxDegree);
    cudaGraphColoring(adjacentList, boundaryList, graphColors, degreeList, conflictTmp, boundaryCount, maxDegree, graphSize, passes, warmup, runs, outputLevel);

    cudaEventRecord(stop_1, 0);
    cudaEventSynchronize(stop_1);

#if 0 // Rest not important for this benchmark
    int interColorsParallel = 0;
    for (int i=0; i<graphSize; i++){
        if ( interColorsParallel < graphColors[i] )
            interColorsParallel = graphColors[i];
    }




    //----- Conflict Count
    int conflictCount = 0;
    for (int i=0; i< boundaryCount; i++)
    {
        int node = conflictTmp[i];

        if(node >= 1)
        {
            conflict[conflictCount] = node;
            conflictCount++;
        }

        //	cout << "i: " << i << "   Node: " << node << endl;
    }
    delete[] conflictTmp;


    cudaEventRecord(stop_4, 0);
    cudaEventSynchronize(stop_4);




    //--------------- Step 4: solve conflicts
    //cout <<"Checkpoint " << endl;

    bool sdoConflictSolver = true;
    if (sdoConflictSolver == true)
        conflictSolveSDO(adjacentList, conflict, conflictCount, graphColors,degreeList, graphSize, maxDegree);
    else
        conflictSolveFF(adjacentList,  graphSize, conflict, conflictCount, graphColors, maxDegree);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTimeGPU, elapsedTimeGPU_1, elapsedTimeGPU_4;
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
    cudaEventElapsedTime(&elapsedTimeGPU_1, start, stop_1);
    cudaEventElapsedTime(&elapsedTimeGPU_4, start, stop_4);


    int numColorsParallel = 0;
    for (int i=0; i<graphSize; i++){
        if ( numColorsParallel < graphColors[i] )
            numColorsParallel = graphColors[i];
    }


    //conflicts
    /*
       cout << "Conclicts: ";
       for (int i=0; i<conflictCount; i++)
       cout << conflict[i] << " colored " <<  graphColors[conflict[i]] << "    ";
       cout << endl;
     */



    // Display information
    /*cout << "List of conflicting nodes:"<<endl;
      for (int k=0; k<conflictCount; k++)
      cout << conflict[k] << "  ";
      cout << endl << endl;  */



    //--------------------- Checking for color conflict ---------------------!

    //cout << endl <<  "Parallel Conflict check:";
    checkCorrectColoring(adjacencyMatrix, graphColors, graphSize);





    //--------------------- Parallel Graph Coloring ---------------------!


    //cout << endl << endl << "Graph Summary" << endl;
    //cout << "Vertices: " << graphSize << "   Edges: " << numEdges << "   Density: " << (2*numEdges)/((float)graphSize*(graphSize-1)) << "   Degree: " << maxDegree << endl;
    //cout << "Random sed used: " << randSeed << endl;

    //cout << endl;
    //cout << "Grid Size: " << GRIDSIZE << "    Block Size: " << BLOCKSIZE << "     Total number of threads: " << GRIDSIZE*BLOCKSIZE << endl;
    //cout << "Graph Subsize: " << graphSize/(GRIDSIZE*BLOCKSIZE) << endl;

    //cout << endl;

    //cout << "Passes: " << passes << endl;

    //if (sdo == true)
    //    if (sdoConflictSolver == true)
    //        cout << "CPU time (SDO): " << elapsedTimeCPU << " ms    -  GPU Time (SDO Solver): " << elapsedTimeGPU << " ms" << endl;
    //    else
    //        cout << "CPU time (SDO): " << elapsedTimeCPU << " ms    -  GPU Time (FF Solver): " << elapsedTimeGPU << " ms" << endl;
    //else
    //    if (sdoConflictSolver == true)
    //        cout << "CPU time (First Fit): " << elapsedTimeCPU << " ms    -  GPU Time (SDO Solver): " << elapsedTimeGPU << " ms" << endl;
    //    else
    //        cout << "CPU time (First Fit): " << elapsedTimeCPU << " ms    -  GPU Time (FF Solver): " << elapsedTimeGPU << " ms" << endl;

    //cout << "ALGO step 1, 2 & 3: " 	<< elapsedTimeGPU_1 << " ms" << endl;
    //cout << "Boundary count: " 		<< elapsedTimeGPU_4 - elapsedTimeGPU_1 << " ms" << endl;
    //cout << "ALGO step 4: " 			<< elapsedTimeGPU - elapsedTimeGPU_4 << " ms" << endl;

    //cout << endl;
    //cout << "Boundary Count: " << boundaryCount << endl;
    //cout << "Conflict count: " << conflictCount << endl;

    //cout << endl;
    //cout << "Colors before solving conflict: " << interColorsParallel << endl;
    //cout << "Sequential Colors: " << numColorsSeq << "      -       Parallel Colors: " << numColorsParallel << endl;

    //cout<<"GPU speed up : "<< elapsedTimeCPU/elapsedTimeGPU<<" x"<<endl;

#endif


    //--------------------- Cleanup ---------------------!

    delete[] adjacencyMatrix;
    delete[] graphColors;
    delete[] conflict;
    delete[] boundaryList;
    delete[] adjacentList;

    return 0;
}

