/** Survey propagation -*- C++ -*-
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
 * Implementation of the Survey Propagation Algorithm
 *
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

/* -*- mode: C++ -*- */
#include "common.h"

#include "lonestargpu/lonestargpu.h"
#include "lonestargpu/cuda_launch_config.hpp"
#ifndef _KLAP_
#include "lonestargpu/cutil_subset.h"
#include <cub/cub/cub.cuh>
#else
#define CUDA_SAFE_CALL(x)

namespace cub {
    template<typename T, int i>
    class BlockReduce {
        public:
        typedef int TempStorage;
        BlockReduce(TempStorage t);
        float Sum(float);
    };
    template<typename T>
    class DoubleBuffer {
        public:
        DoubleBuffer(T*,T*);
        T* Current();
    };
    class DeviceRadixSort {
        public:
        static void SortPairs(void*, size_t, cub::DoubleBuffer<float>&, cub::DoubleBuffer<int>, int);
    };
}
#endif

#define WARP_SIZE 32
//__device__ inline int lane_id(void) { return threadIdx.x % WARP_SIZE; }

#define EPSILON 0.01 /* in the source, in the paper 10^-3 */
#define MAXITERATION 1000
#define PARTIAL "outputs/partial.cnf"
#define PARAMAGNET 0.01

KernelConfig kc;
const int nSM = kc.getNumberOfSMs();

void init_from_file(const char *F,
        int max_lit_per_clause,
        struct CSRGraph &clauses,
        struct CSRGraph &vars,
        struct Edge &ed)
{
    FILE *f = fopen(F, "r");

    int nclauses, nvars, ret;
    char line[255];

    if(!f)
    {
        fprintf(stderr, "unable to read file %s.\n", F);
        exit(1);
    }

    while(true)
    {
        if(fgets(line, 255, f))
        {
            if(line[0] != 'c')
                break;
            printf("%s", line);
        }
        else
        {
            fprintf(stderr, "unable to read %s\n", F);
            exit(1);
        }
    }

    ret = sscanf(line, "p cnf %d %d", &nvars, &nclauses);
    assert(ret == 2);

    clauses.nnodes = nclauses;
    vars.nnodes = nvars;
    ed.nedges = clauses.nedges = vars.nedges = nclauses * max_lit_per_clause; // over-estimate

    assert(clauses.alloc());
    assert(vars.alloc());
    assert(ed.alloc());

    int newlit, lit;
    int clndx = 0, litndx = 0, edndx = 0;

    /* read lines of literals terminated by 0 */
    /* assumes literals numbered from 1 */

    do {
        ret = fscanf(f, "%d", &newlit);
        if(ret == EOF) break;

        if(newlit == 0)
        {
            assert(clndx < nclauses);

            clndx++;
            litndx = 0;

            clauses.row_offsets[clndx] = edndx;
            continue;
        }

        assert(litndx < max_lit_per_clause);

        // convert to zero-based
        lit = ((newlit < 0) ? -newlit : newlit) - 1;

        assert(lit >= 0);

        ed.src[edndx] = clndx;
        ed.dst[edndx] = lit;
        ed.bar[edndx] = newlit < 0;
        ed.eta[edndx] = (float)(rand()) / (float)RAND_MAX;

        // essentially clause -> edge
        clauses.columns[clauses.row_offsets[clndx] + litndx] = edndx;

        // record size of every var node
        vars.row_offsets[lit]++;

        litndx++;
        edndx++;
    } while(true);
    fclose(f);

    clauses.nedges = vars.nedges = ed.nedges = edndx;

    clauses.set_last_offset();
    vars.set_last_offset();

    /* populate vars */
    // exclusive-sum
    for(int i = 0, sum = 0; i < vars.nnodes; i++)
    {
        int size = vars.row_offsets[i];
        vars.row_offsets[i] = sum;
        sum += size;
    }

    int *varndx = (int *) calloc(vars.nedges, sizeof(int));

    for(int i = 0; i < ed.nedges; i++)
    {
        unsigned var = ed.dst[i];
        vars.columns[vars.row_offsets[var] + varndx[var]++] = i;
    }

    //printf("read %d clauses, %d variables, %d literals\n", clauses.nnodes, vars.nnodes, ed.nedges);
}

void print_solution(const char *sol, const CSRGraph &vars)
{
    FILE *f = fopen(sol, "w");
    int i;
    for(i = 0; i < vars.nnodes; i++)
    {
        if(vars.sat[i])
            fprintf(f, "%d\n", vars.value[i] ? (i + 1) : -(i + 1));
    }
    fclose(f);
}

void dump_formula(const char *output, const CSRGraph &clauses, const CSRGraph &vars, const Edge &ed)
{
    FILE *of = fopen(output, "w");

    fprintf(of, "p cnf %d %d\n", vars.nnodes, clauses.nnodes);

    for(int cl = 0; cl < clauses.nnodes; cl++) {
        unsigned offset = clauses.row_offsets[cl];

        for(int i = 0; i < clauses.degree(cl); i++) {
            unsigned edndx = clauses.columns[offset + i];
            fprintf(of, "%d ", ed.bar[edndx] ? -(ed.dst[edndx]+1) : (ed.dst[edndx]+1));
        }
        fprintf(of, "0\n");
    }
}

void dump_partial(const char *output, const CSRGraph &clauses, const CSRGraph &vars, const Edge &ed)
{
    FILE *of = fopen(output, "w");

    int sat = 0;
    for(int cl = 0; cl < clauses.nnodes; cl++)
        if(clauses.sat[cl]) sat++;

    fprintf(of, "p cnf %d %d\n", vars.nnodes, clauses.nnodes - sat);

    for(int cl = 0; cl < clauses.nnodes; cl++) {
        if(clauses.sat[cl])
            continue;

        unsigned offset = clauses.row_offsets[cl];

        for(int i = 0; i < clauses.degree(cl); i++) {
            unsigned edndx = clauses.columns[offset + i];

            if(vars.sat[ed.dst[edndx]])
                continue;

            fprintf(of, "%d ", ed.bar[edndx] ? -(ed.dst[edndx]+1) : (ed.dst[edndx]+1));
        }
        fprintf(of, "0\n");
    }
}

__global__ void calc_pi_values(GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = blockDim.x * gridDim.x;

    // over all a -> j
    for(int edndx = id; edndx < ed.nedges; edndx += threads)
    {
        int j = ed.dst[edndx];
        int a = ed.src[edndx];

        if(clauses.sat[a] || vars.sat[j])
            continue;

        int V_j = vars.row_offsets[j];
        int V_j_len = vars.degree(j);

        float pi_0 = 1.0;
        float V_s_a = 1.0;
        float V_u_a = 1.0;

        // over all b E V(j)
        for(int i = 0; i < V_j_len; i++)
        {
            int ed_btoj = vars.columns[V_j + i];

            int b = ed.src[ed_btoj];

            if(clauses.sat[b])
                continue;

            if(b != a)
            {
                pi_0 *= (1 - ed.eta[ed_btoj]);

                if(ed.bar[ed_btoj] == ed.bar[edndx])
                    V_s_a *= (1 - ed.eta[ed_btoj]);
                else
                    V_u_a *= (1 - ed.eta[ed_btoj]);
            }
        }

        ed.pi_0[edndx] = pi_0;
        ed.pi_U[edndx] = (1 - V_u_a) * (V_s_a);
        ed.pi_S[edndx] = (1 - V_s_a) * (V_u_a);

        //printf("%f %f %f\n", ed.pi_0[edndx,
    }
}

__global__ void update_eta(GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, float *max_eps)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = blockDim.x * gridDim.x;
    float eps, lmaxeps = 0;

    for(int edndx = id; edndx < ed.nedges; edndx+=threads)
    {
        int a = ed.src[edndx];
        int i = ed.dst[edndx];

        // as these are "removed"
        if(clauses.sat[a] || vars.sat[i])
            continue;

        int clndx = clauses.row_offsets[a];
        int nlit = clauses.degree(a);

        float new_eta = 1.0;

        for(int aedndx = 0; aedndx < nlit; aedndx++)
        {
            int jedndx = clauses.columns[clndx + aedndx];

            int j = ed.dst[jedndx];

            if(j == i)
                continue;

            if(vars.sat[j])
                continue;

            float sum = ed.pi_0[jedndx] + ed.pi_S[jedndx] + ed.pi_U[jedndx];

            if(sum == 0.0) { // TODO: non-standard ...
                new_eta = 0;
                break;
            }

            new_eta *= ed.pi_U[jedndx] / sum;
        }

        eps = fabs(new_eta - ed.eta[edndx]);
        if(eps > lmaxeps)
            lmaxeps = eps;

        ed.eta[edndx] = new_eta;
    }

    if(lmaxeps)
        atomicMax((int *) max_eps, __float_as_int(lmaxeps));
}

__global__ void update_bias(GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, float *bias_list, int *bias_list_vars, int *bias_len, float *g_summag)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int threads = blockDim.x * gridDim.x;
    float maxmag;
    float summag = 0;

    for(int v = id; v < vars.nnodes; v+=threads)
    {
        if(vars.sat[v])
            continue;

        float pi_0_hat = 1.0, V_minus = 1.0, V_plus = 1.0;
        float pi_P_hat, pi_M_hat;

        int edoff = vars.row_offsets[v];
        int ncl = vars.degree(v);

        // a E v(i)
        for(int edndx = 0; edndx < ncl; edndx++)
        {
            int edge = vars.columns[edoff + edndx];
            int cl = ed.src[edge];

            if(clauses.sat[cl])
                continue;

            pi_0_hat *= (1 - ed.eta[edge]);

            if(ed.bar[edge])
                V_minus *= (1 - ed.eta[edge]);
            else
                V_plus *= (1 - ed.eta[edge]);
        }

        pi_P_hat = (1 - V_plus) * V_minus;
        pi_M_hat = (1 - V_minus) * V_plus;

        float W_plus, W_minus; //, W_zero;

        if (((pi_0_hat + pi_P_hat + pi_M_hat)) == 0.0)
        {
            W_plus = 0.0;
            W_minus = 0.0;
        }
        else
        {
            W_plus = pi_P_hat / (pi_0_hat + pi_P_hat + pi_M_hat);
            W_minus = pi_M_hat / (pi_0_hat + pi_P_hat + pi_M_hat);
        }

        //W_zero = 1 - W_plus - W_minus;

        vars.bias[v] = fabs(W_plus - W_minus);
        vars.value[v] = (W_plus > W_minus);

        maxmag = W_plus > W_minus ? W_plus : W_minus;
        summag += maxmag;

        int ndx = atomicAdd(bias_len, 1);
        bias_list[ndx] = vars.bias[v];
        bias_list_vars[ndx] = v;
    }

    typedef cub::BlockReduce<float, 384> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    summag = BlockReduce(temp_storage).Sum(summag);

    if(threadIdx.x == 0)
        atomicAdd(g_summag, summag);
}

int compare_float(const void *x, const void *y)
{
    float xx = *(float *)x, yy = *(float *)y;

    // reverse order
    if(xx > yy)
        return -1;
    else if (xx < yy)
        return 1;

    return 0;
}

float sort_bias_list(cub::DoubleBuffer<float> &db_bias_list,
        cub::DoubleBuffer<int> &db_bias_list_vars,
        int *g_bias_list_len, float summag, int& fixperstep)
{
    int bias_list_len;
    static void *d_temp_storage = NULL;
    static size_t temp_storage_bytes = 0;

    CUDA_SAFE_CALL(cudaMemcpy(&bias_list_len, g_bias_list_len, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    float r = 0;

    if(bias_list_len)
    {
        r = (summag / bias_list_len);

#if PRINT
        printf("<bias>:%f\n", r);
#endif

        if(r < PARAMAGNET)
            return r;

        if(d_temp_storage == NULL)
        {
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_bias_list,
                    db_bias_list_vars, bias_list_len);

            // Allocate temporary storage for sorting operation
            CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        }

        // Run sorting operation
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, db_bias_list,
                db_bias_list_vars, bias_list_len);

        if(fixperstep > bias_list_len)
            fixperstep = 1;
    }

    return r;
}

int converge(GPUCSRGraph &g_cl, GPUCSRGraph &g_vars, Edge &g_ed, float *g_max_eps)
{
    float max_eps;
    int i = 0;
    const size_t cpv_res = maximum_residency(calc_pi_values, 384, 0);//Comment = 10
    const size_t ue_res = maximum_residency(update_eta, 384, 0);//Comment = 10...

    do {
        calc_pi_values<<<nSM * cpv_res, 384>>>(g_cl, g_vars, g_ed);

        max_eps = 0;
        CUDA_SAFE_CALL(cudaMemcpy(g_max_eps, &max_eps,
                    sizeof(float), cudaMemcpyHostToDevice));

        update_eta<<<nSM * ue_res, 384>>>(g_cl, g_vars, g_ed, g_max_eps);

        CUDA_SAFE_CALL(cudaMemcpy(&max_eps, g_max_eps,
                    sizeof(float), cudaMemcpyDeviceToHost));

    } while(max_eps > EPSILON && i++ < MAXITERATION);

    if(max_eps <= EPSILON) {
#if PRINT
        printf("converged in %d iterations max eps %f\n", i, max_eps);
#endif
        return 1;
    } else {
        printf("SP UN-CONVERGED, max eps %f\n", max_eps);
        //TODO write out formula?
        exit(1);
    }

    return 0;
}

int build_list(GPUCSRGraph &g_cl, GPUCSRGraph &g_vars, Edge &g_ed, float *g_summag,
        cub::DoubleBuffer<float> &db_bias_list, cub::DoubleBuffer<int> &db_bias_list_vars,
        int *g_bias_list_len, int &fixperstep)
{
    float summag;
    int bias_list_len;
    static size_t updb_res = maximum_residency(update_bias, 384, 0);//Comment

    summag = 0;
    CUDA_SAFE_CALL(cudaMemcpy(g_summag, &summag, sizeof(summag), cudaMemcpyHostToDevice));

    bias_list_len = 0;
    CUDA_SAFE_CALL(cudaMemcpy(g_bias_list_len, &bias_list_len, sizeof(int) * 1, cudaMemcpyHostToDevice));
    update_bias<<<nSM * updb_res, 384>>>(g_cl, g_vars, g_ed, db_bias_list.Current(),
            db_bias_list_vars.Current(), g_bias_list_len, g_summag);
    CUDA_SAFE_CALL(cudaMemcpy(&summag, g_summag, sizeof(summag), cudaMemcpyDeviceToHost));

    float limitbias = sort_bias_list(db_bias_list, db_bias_list_vars, g_bias_list_len, summag, fixperstep);
    if(limitbias < PARAMAGNET)
    {
        //printf("paramagnetic state\n");
        return 1;
    }

    return 0;
}

int compare_solutions(const CSRGraph &vars, const CSRGraph &vars_, char* version)
{
    int error = 0;
    int i;
    for(i = 0; i < vars.nnodes; i++)
    {
        if(vars.sat[i]){
            //printf("Version %s vars %d -> %d - %d\n", version, i, vars.value[i], vars_.value[i]);
            if(vars.value[i] != vars_.value[i]){
                error = 1;
                printf("Error in version %s vars %d -> %d != %d\n", version, i, vars.value[i], vars_.value[i]);
            }
        }
    }
    return error;
}

//********************************************************************************************************************


int main(int argc, char *argv[])
{
    // Parameters
    int warmup              = 1;
    int runs                = 3;
    int outputLevel         = 1;
    const char* fileName    = "inputs/random-42000-10000-3.cnf";
    int max_literals        = 10000;
    int opt;
    while((opt = getopt(argc, argv, "w:r:o:f:m:h")) >= 0) {
        switch(opt) {
            case 'w': warmup        = atoi(optarg); break;
            case 'r': runs          = atoi(optarg); break;
            case 'o': outputLevel   = atoi(optarg); break;
            case 'f': fileName      = optarg;       break;
            case 'm': max_literals  = atoi(optarg); break;
            default : std::cerr <<
                      "\nUsage:  ./bt [options]"
                          "\n"
                          "\n    -w <W>    # of warmup runs (default=1)"
                          "\n    -r <R>    # of timed runs (default=3)"
                          "\n    -o <O>    level of output verbosity (0: one CSV row, 1: moderate, 2: verbose)"
                          "\n    -f <F>    file name (default=inputs/random-42000-10000-3.cnf)"
                          "\n    -m <M>    maximum literals (default=10000)"
                          "\n    -h        help\n\n";
                      exit(0);
        }
    }

    //srand(7);
    double starttime, endtime, runtime;
    CSRGraph cl, vars;
    Edge ed;
    GPUCSRGraph g_cl, g_vars;
    GPUEdge g_ed;
    float *g_max_eps;
    float *g_bias_list, *g_bias_list_2;
    int *g_bias_list_vars, *g_bias_list_vars_2;
    int *g_bias_list_len;
    float *g_summag;
    const size_t d2_res = maximum_residency(decimate, 384, 0);
    init_from_file(fileName, max_literals, cl, vars, ed);
    g_vars.from_cpu(vars);
    CUDA_SAFE_CALL(cudaMalloc(&g_max_eps, sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&g_summag, sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&g_bias_list, g_vars.nnodes * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&g_bias_list_2, g_vars.nnodes * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&g_bias_list_vars, g_vars.nnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&g_bias_list_vars_2, g_vars.nnodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&g_bias_list_len, sizeof(int)));
    cub::DoubleBuffer<float> db_bias_list(g_bias_list, g_bias_list_2);
    cub::DoubleBuffer<int> db_bias_list_vars(g_bias_list_vars, g_bias_list_vars_2);
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, d2_res*nSM*384); // Fixed-size pool

    float totalKernelTime = 0;
    for(int run = -warmup; run < runs; run++) {

        if(outputLevel >= 1) {
            if(run < 0) {
                std::cout << "Warmup:\t";
            } else {
                std::cout << "Run " << run << ":\t";
            }
        }

        g_cl.from_cpu(cl);
        g_ed.from_cpu(ed);
        int canfix = 0.01 * vars.nnodes;
        if(canfix < 1) canfix = 1;
        int round;
        int breaker = 54;
        //int error = 0;
        runtime = 0;
        round = 0;
        // Start execution
        while(converge(g_cl, g_vars, g_ed, g_max_eps)) {
            //printf("round = %d\n", round++);
            if(round == breaker) break;

            if(build_list(g_cl, g_vars, g_ed, g_summag,
                        db_bias_list, db_bias_list_vars,
                        g_bias_list_len,
                        canfix))
                break;

            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            starttime = rtclock();

            launch_kernel(d2_res * nSM, 384, g_cl, g_vars, g_ed,
                    db_bias_list_vars.Current(), g_bias_list_len, canfix);

            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            endtime = rtclock();
            runtime += (1000.0 * (endtime - starttime));
        };

        if(outputLevel >= 1) std::cout << "run kernel time = " << runtime << std::endl;
        if (run >= 0) totalKernelTime += runtime;
        g_cl.to_cpu(cl);
        g_vars.to_cpu(vars);
        g_ed.to_cpu(ed);
        print_solution("outputs/sp_sol.dat", vars);
        //char version[10];
        //strcpy(version, "CDP");
        //error = compare_solutions(vars_gold, vars, version);
        //if(error) printf("Test failed %s\n", version);
        //else printf("Test passed %s\n", version);
        dump_partial(PARTIAL, cl, vars, ed);

        if(run < runs) {
            init_from_file(fileName, max_literals, cl, vars, ed);
            g_vars.from_cpu(vars);
        }

    }

    if(outputLevel >= 1) {
        std::cout<< "Average kernel time = " << totalKernelTime/runs << " ms\n";
    } else {
        std::cout<< totalKernelTime/runs;
    }

}

