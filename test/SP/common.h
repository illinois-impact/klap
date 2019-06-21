
#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <assert.h>

struct CSRGraph {
  int nnodes;
  int nedges;

  int *row_offsets;
  int *columns;

  bool *sat;

  float *bias;
  bool *value;
  
  bool alloc()
  {
    assert(nnodes > 0);
    assert(nedges > 0);
    
    row_offsets = (int *) calloc(nnodes + 1, sizeof(*row_offsets));
    columns = (int *) calloc(nedges, sizeof(*columns));

    sat = (bool *) calloc(nnodes, sizeof(bool));
    bias = (float *) calloc(nnodes, sizeof(float));
    value = (bool *) calloc(nnodes, sizeof(bool));

    return (row_offsets != NULL) && (columns != NULL) && sat && bias && value;
  }
  
  void set_last_offset()
  {
    row_offsets[nnodes] = nedges;
  }

  int degree(const int node) const
  {
    return row_offsets[node + 1] - row_offsets[node];
  }

  void dump_edges() const
  {
    int i;
    for(i = 0; i < nedges; i++)
      printf("%d ", columns[i]);
    printf("\n");
  }

  void dump_offsets() const
  {
    int i;
    for(i = 0; i < nnodes; i++)
      printf("%d ", row_offsets[i]);
    printf("\n");
  }
};

struct GPUCSRGraph : CSRGraph {
  bool alloc()
  {
    assert(nnodes > 0);
    assert(nedges > 0);
    
    cudaMalloc(&row_offsets, (nnodes + 1) * sizeof(*row_offsets));
    cudaMalloc(&columns, nedges * sizeof(*columns));

    cudaMalloc(&sat, nnodes * sizeof(*sat));

    cudaMalloc(&bias, nnodes * sizeof(*bias));

    cudaMalloc(&value, nnodes * sizeof(*value));

    return (row_offsets != NULL) && (columns != NULL) && sat && bias && value;
  }

  bool from_cpu(CSRGraph &cpu)
  {
    nnodes = cpu.nnodes;
    nedges = cpu.nedges;
    
    assert(alloc());
    
    cudaMemcpy(row_offsets, cpu.row_offsets, (nnodes + 1) * sizeof(*row_offsets), cudaMemcpyHostToDevice);

    cudaMemcpy(columns, cpu.columns, (nedges * sizeof(*columns)), cudaMemcpyHostToDevice);

    cudaMemcpy(sat, cpu.sat, (nnodes * sizeof(*sat)), cudaMemcpyHostToDevice);

    cudaMemcpy(bias, cpu.bias, (nnodes * sizeof(*bias)), cudaMemcpyHostToDevice);

    cudaMemcpy(value, cpu.value, (nnodes * sizeof(*value)), cudaMemcpyHostToDevice);

    return true;
  }

  bool to_cpu(CSRGraph &cpu, bool alloc = false)
  {
    if(alloc)
      {
	cpu.nnodes = nnodes;
	cpu.nedges = nedges;

	assert(cpu.alloc());

      }
    assert(nnodes == cpu.nnodes);
    assert(nedges == cpu.nedges);
        
    cudaMemcpy(cpu.row_offsets, row_offsets, (nnodes + 1) * sizeof(*row_offsets), cudaMemcpyDeviceToHost);

    cudaMemcpy(cpu.columns, columns, nedges * sizeof(*columns), cudaMemcpyDeviceToHost);

    cudaMemcpy(cpu.sat, sat, (nnodes * sizeof(*sat)), cudaMemcpyDeviceToHost);

    cudaMemcpy(cpu.bias, bias, (nnodes * sizeof(*bias)), cudaMemcpyDeviceToHost);

    cudaMemcpy(cpu.value, value, (nnodes * sizeof(*value)), cudaMemcpyDeviceToHost);


    return true;
  }

  __device__
    int degree(const int node) const
  {
    return row_offsets[node + 1] - row_offsets[node];
  } 
};

struct Edge {
  int nedges;
  int *src;
  int *dst;  
  bool *bar;
  float *eta;
  float *pi_0;
  float *pi_S;
  float *pi_U;

  bool alloc()
  {
    assert(nedges > 0);
    
    src = (int*) calloc(nedges, sizeof(*src));
    dst = (int*) calloc(nedges, sizeof(*dst));
    bar = (bool*) calloc(nedges, sizeof(*bar));
    eta = (float*) calloc(nedges, sizeof(*eta));
    pi_0 = (float*) calloc(nedges, sizeof(*pi_0));
    pi_S = (float*) calloc(nedges, sizeof(*pi_S));
    pi_U = (float*) calloc(nedges, sizeof(*pi_U));

    return (src && dst && bar && eta && pi_0 && pi_S && pi_U);
  }
};

struct GPUEdge : Edge {
  bool alloc()
  {
    assert(nedges > 0);
    
    cudaMalloc(&src, (nedges) * (sizeof(*src)));
    cudaMalloc(&dst, (nedges) * (sizeof(*dst)));
    cudaMalloc(&bar, (nedges) * (sizeof(*bar)));
    cudaMalloc(&eta, (nedges) * (sizeof(*eta)));
    cudaMalloc(&pi_0, (nedges) * (sizeof(*pi_0)));
    cudaMalloc(&pi_S, (nedges) * (sizeof(*pi_S)));
    cudaMalloc(&pi_U, (nedges) * (sizeof(*pi_U)));
 
    return (src && dst && bar && eta && pi_0 && pi_S && pi_U);
  }

  bool from_cpu(Edge &cpu)
  {
    nedges = cpu.nedges;
    
    assert(alloc());
    
    cudaMemcpy(src, cpu.src, (nedges) * sizeof(*src), cudaMemcpyHostToDevice);
    cudaMemcpy(dst, cpu.dst, (nedges) * sizeof(*dst), cudaMemcpyHostToDevice);
    cudaMemcpy(bar, cpu.bar, (nedges) * sizeof(*bar), cudaMemcpyHostToDevice);
    cudaMemcpy(eta, cpu.eta, (nedges) * sizeof(*eta), cudaMemcpyHostToDevice);
    cudaMemcpy(pi_0, cpu.pi_0, (nedges) * sizeof(*pi_0), cudaMemcpyHostToDevice);
    cudaMemcpy(pi_S, cpu.pi_S, (nedges) * sizeof(*pi_S), cudaMemcpyHostToDevice);
    cudaMemcpy(pi_U, cpu.pi_U, (nedges) * sizeof(*pi_U), cudaMemcpyHostToDevice);

    return true;
  }

  bool to_cpu(Edge &cpu, bool alloc = false)
  {
    if(alloc)
      {
	cpu.nedges = nedges;

	assert(cpu.alloc());
      }

    assert(nedges == cpu.nedges);

    cudaMemcpy(cpu.src, src, (nedges) * sizeof(*src), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.dst, dst, (nedges) * sizeof(*dst), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.bar, bar, (nedges) * sizeof(*bar), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.eta, eta, (nedges) * sizeof(*eta), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.pi_0, pi_0, (nedges) * sizeof(*pi_0), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.pi_S, pi_S, (nedges) * sizeof(*pi_S), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu.pi_U, pi_U, (nedges) * sizeof(*pi_U), cudaMemcpyDeviceToHost);
        
    return true;
  }
};

__global__ void decimate (GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, int *g_bias_list_vars, const int * bias_list_len, int fixperstep);

void launch_kernel(unsigned int nb, unsigned int nt, GPUCSRGraph clauses, GPUCSRGraph vars, Edge ed, int *g_bias_list_vars,const int * bias_list_len, int fixperstep);

#endif

