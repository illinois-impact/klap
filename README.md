
# KLAP

## Overview

KLAP is a source-to-source compiler that optimizes CUDA code which uses dynamic parallelism to implement applications with nested parallelism. KLAP aggregates dynamic launches across warps, blocks, and grids to reduce the total number of grid launches and increase their granularity.

## Instructions

Refer to `src` for instructions on how to build the compiler.

Refer to `include` for instructions on how to setup the runtime.

## Citation

Please cite the following paper if you find this work useful:

* I. El Hajj, J. Gómez-Luna, C. Li, L.-W. Chang, D. Milojicic, W.-M. Hwu.
  **KLAP: Kernel Launch Aggregation and Promotion for Optimizing Dynamic Parallelism**.
  In *Proceedings of the 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)*, 2016.

