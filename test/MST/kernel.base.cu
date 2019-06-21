
#include "common.h"

__global__ void dfindelemin2_child_cdp(Graph graph, ComponentSpace cs, foru *eleminwts, unsigned *partners, unsigned *goaheadnodeofcomponent, unsigned src, unsigned id, unsigned srcboss, unsigned degree) {
    unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii < degree){
        foru wt = graph.getWeight(src, ii); //
        if (wt == eleminwts[id]) {
            unsigned dst = graph.getDestination(src, ii); //
            unsigned tempdstboss = cs.find(dst);
            if (tempdstboss == partners[id]) {	// cross-component edge.
                //atomicMin(&goaheadnodeofcomponent[srcboss], id);
                if(atomicCAS(&goaheadnodeofcomponent[srcboss], graph.nnodes, id) == graph.nnodes)
                {
                    //printf("%d: adding %d\n", id, eleminwts[id]);
                    //atomicAdd(wt2, eleminwts[id]);
                }
            }
        }
    }
}

__global__ void dfindelemin2_cdp(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < graph.nnodes) {
        unsigned src = id;
        unsigned srcboss = cs.find(src);
        if(eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != graph.nnodes)
        {
            unsigned degree = graph.getOutDegree(src);
            dfindelemin2_child_cdp<<<(int)ceil((float)degree/BLOCK_DIM), BLOCK_DIM>>>(graph, cs, eleminwts, partners, goaheadnodeofcomponent, src, id, srcboss, degree);
        }
    }
}

__global__ void verify_min_elem_child_cdp(Graph graph, ComponentSpace cs, unsigned minwt_node, foru minwt, /*foru *eleminwts,*/ unsigned *partners, bool *processinnextiteration, /*unsigned *goaheadnodeofcomponent, unsigned src,*/ unsigned id, /*unsigned srcboss,*/ unsigned degree) {
    //bool minwt_found = false;
    unsigned ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii < degree){
        foru wt = graph.getWeight(minwt_node, ii);
        //printf("%d: looking at %d edge %d wt %d (%d)\n", id, minwt_node, ii, wt, minwt);

        if (wt == minwt) {
            //minwt_found = true;
            unsigned dst = graph.getDestination(minwt_node, ii);
            unsigned tempdstboss = cs.find(dst);
            if(tempdstboss == partners[minwt_node] && tempdstboss != id)
            {
                processinnextiteration[minwt_node] = true;
                //printf("%d okay!\n", id);
                return;
            }
        }
    }
    else return;
    //printf("component %d is wrong %d - %d - %d, %d\n", id, minwt_found, minwt, ii, degree); // Thread that would printf "okay" should set a verify[id] element. After child kernel finish, if verify[id]!=1, then printf this wrong message
}

__global__ void verify_min_elem_cdp(unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (inpid < graph.nnodes) id = inpid;


    if (id < graph.nnodes) {

        if(cs.isBoss(id)) {

            if(goaheadnodeofcomponent[id] != graph.nnodes) {
                unsigned minwt_node = goaheadnodeofcomponent[id];
                unsigned degree = graph.getOutDegree(minwt_node);
                foru minwt = minwtcomponent[id];

                if(minwt != MYINFINITY) {
                    verify_min_elem_child_cdp<<<(int)ceil((float)degree/BLOCK_DIM), BLOCK_DIM>>>(graph, cs, minwt_node, minwt, partners, processinnextiteration, /*goaheadnodeofcomponent, src,*/ id, /*srcboss,*/ degree);
                }
            }
        }
    }

}

void launch_find_kernel(unsigned int nb, unsigned int nt, unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
    dfindelemin2_cdp<<<nb, nt>>>(mstwt, graph, cs, eleminwts, minwtcomponent, partners, phore, processinnextiteration, goaheadnodeofcomponent, inpid);
}

void launch_verify_kernel(unsigned int nb, unsigned int nt, unsigned *mstwt, Graph graph, ComponentSpace cs, foru *eleminwts, foru *minwtcomponent, unsigned *partners, unsigned *phore, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid) {
    verify_min_elem_cdp<<<nb, nt>>>(mstwt, graph, cs, eleminwts, minwtcomponent, partners, phore, processinnextiteration, goaheadnodeofcomponent, inpid);
}

