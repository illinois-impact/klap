
#include "common.h"

__global__ void decimate_child(GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, int v, int edoff, int cllen){

    int edndx = threadIdx.x + blockIdx.x * blockDim.x;

    if(edndx < cllen){
        int edge = vars.columns[edoff + edndx];
        int cl = ed.src[edge];

        if(!clauses.sat[cl])
            if(ed.bar[edge] != vars.value[v])
                clauses.sat[cl] = true;
    }
}


__global__ void decimate (GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, int *g_bias_list_vars,
        const int * bias_list_len, int fixperstep)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = blockDim.x * gridDim.x;

    // NOTE: this is slower because the lower-work computation does not use all SMs.

    //for(int l = *bias_list_len - fixperstep + id; l < *bias_list_len; l+=threads)
    for(int l_base = *bias_list_len - fixperstep; l_base < *bias_list_len; l_base+=threads)
    {
        int l = l_base + id;
        if(l < *bias_list_len) {
            int v = g_bias_list_vars[l];
            vars.sat[v] = true;
            int edoff = vars.row_offsets[v];
            int cllen = vars.degree(v);
            decimate_child<<<(int)ceil((float)cllen/(2*32)), 2*32>>>(clauses, vars, ed, v, edoff, cllen);
        }

    }
}

void launch_kernel(unsigned int nb, unsigned int nt, GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, int *g_bias_list_vars,const int * bias_list_len, int fixperstep) {
    decimate<<<nb, nt>>>(clauses, vars, ed, g_bias_list_vars, bias_list_len, fixperstep);
}

