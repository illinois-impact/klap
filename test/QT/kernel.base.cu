
#include "common.h"

////////////////////////////////////////////////////////////////////////////////
// Build a quadtree on the GPU. Use CUDA Dynamic Parallelism.
//
// The algorithm works as follows. The host (CPU) launches one block of
// NUM_THREADS_PER_BLOCK threads. That block will do the following steps:
//
// 1- Check the number of points and its depth.
//
// We impose a maximum depth to the tree and a minimum number of points per
// node. If the maximum depth is exceeded or the minimum number of points is
// reached. The threads in the block exit.
//
// Before exiting, they perform a buffer swap if it is needed. Indeed, the
// algorithm uses two buffers to permute the points and make sure they are
// properly distributed in the quadtree. By design we want all points to be
// in the first buffer of points at the end of the algorithm. It is the reason
// why we may have to swap the buffer before leavin (if the points are in the
// 2nd buffer).
//
// 2- Count the number of points in each child.
//
// If the depth is not too high and the number of points is sufficient, the
// block has to dispatch the points into four geometrical buckets: Its
// children. For that purpose, we compute the center of the bounding box and
// count the number of points in each quadrant.
//
// The set of points is divided into sections. Each section is given to a
// warp of threads (32 threads). Warps use __ballot and __popc intrinsics
// to count the points. See the Programming Guide for more information about
// those functions.
//
// 3- Scan the warps' results to know the "global" numbers.
//
// Warps work independently from each other. At the end, each warp knows the
// number of points in its section. To know the numbers for the block, the
// block has to run a scan/reduce at the block level. It's a traditional
// approach. The implementation in that sample is not as optimized as what
// could be found in fast radix sorts, for example, but it relies on the same
// idea.
//
// 4- Move points.
//
// Now that the block knows how many points go in each of its 4 children, it
// remains to dispatch the points. It is straightforward.
//
// 5- Launch new blocks.
//
// The block launches four new blocks: One per children. Each of the four blocks
// will apply the same algorithm.
////////////////////////////////////////////////////////////////////////////////
#ifndef NUM_THREADS_PER_BLOCK
#define NUM_THREADS_PER_BLOCK 128 // Do not use less than 128 threads.
#endif
//template< int NUM_THREADS_PER_BLOCK >
__global__
void build_quadtree_kernel(Quadtree_node *nodes, Points *points, Parameters params)
{
    // The number of warps in a block.
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

    // Shared memory to store the number of points.
    extern __shared__ int smem[];

    // s_num_pts[4][NUM_WARPS_PER_BLOCK];
    // Addresses of shared memory.
    volatile int *s_num_pts[4];

    for (int i = 0 ; i < 4 ; ++i)
        s_num_pts[i] = (volatile int *) &smem[i*NUM_WARPS_PER_BLOCK];

    // Compute the coordinates of the threads in the block.
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    // Mask for compaction.
    int lane_mask_lt = (1 << lane_id) - 1; // Same as: asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt) );

    // The current node.
    Quadtree_node &node = nodes[blockIdx.x];
    node.set_id(node.id() + blockIdx.x);

    // The number of points in the node.
    int num_points = node.num_points();

    //
    // 1- Check the number of points and its depth.
    //

    // Stop the recursion here. Make sure points[0] contains all the points.
    if (params.depth >= params.max_depth || num_points <= params.min_points_per_node)
    {
        if (params.point_selector == 1)
        {
            int it = node.points_begin(), end = node.points_end();

            for (it += threadIdx.x ; it < end ; it += NUM_THREADS_PER_BLOCK)
                if (it < end)
                    points[0].set_point(it, points[1].get_point(it));
        }

        return;
    }

    // Compute the center of the bounding box of the points.
    const Bounding_box &bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(center);

    // Find how many points to give to each warp.
    int num_points_per_warp = max(warpSize, (num_points + NUM_WARPS_PER_BLOCK-1) / NUM_WARPS_PER_BLOCK);

    // Each warp of threads will compute the number of points to move to each quadrant.
    int range_begin = node.points_begin() + warp_id * num_points_per_warp;
    int range_end   = min(range_begin + num_points_per_warp, node.points_end());

    //
    // 2- Count the number of points in each child.
    //

    // Reset the counts of points per child.
    if (lane_id == 0)
    {
        s_num_pts[0][warp_id] = 0;
        s_num_pts[1][warp_id] = 0;
        s_num_pts[2][warp_id] = 0;
        s_num_pts[3][warp_id] = 0;
    }

    // Input points.
    const Points &in_points = points[params.point_selector];

    // Compute the number of points.
    for (int range_it = range_begin + lane_id ; __any(range_it < range_end) ; range_it += warpSize)
    {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Load the coordinates of the point.
        float2 p = is_active ? in_points.get_point(range_it) : make_float2(0.0f, 0.0f);

        // Count top-left points.
        int num_pts = __popc(__ballot(is_active && p.x < center.x && p.y >= center.y));

        if (num_pts > 0 && lane_id == 0)
            s_num_pts[0][warp_id] += num_pts;

        // Count top-right points.
        num_pts = __popc(__ballot(is_active && p.x >= center.x && p.y >= center.y));

        if (num_pts > 0 && lane_id == 0)
            s_num_pts[1][warp_id] += num_pts;

        // Count bottom-left points.
        num_pts = __popc(__ballot(is_active && p.x < center.x && p.y < center.y));

        if (num_pts > 0 && lane_id == 0)
            s_num_pts[2][warp_id] += num_pts;

        // Count bottom-right points.
        num_pts = __popc(__ballot(is_active && p.x >= center.x && p.y < center.y));

        if (num_pts > 0 && lane_id == 0)
            s_num_pts[3][warp_id] += num_pts;
    }

    // Make sure warps have finished counting.
    __syncthreads();

    //
    // 3- Scan the warps' results to know the "global" numbers.
    //

    // First 4 warps scan the numbers of points per child (inclusive scan).
    if (warp_id < 4)
    {
        int num_pts = lane_id < NUM_WARPS_PER_BLOCK ? s_num_pts[warp_id][lane_id] : 0;
#pragma unroll

        for (int offset = 1 ; offset < NUM_WARPS_PER_BLOCK ; offset *= 2)
        {
            int n = __shfl_up(num_pts, offset, NUM_WARPS_PER_BLOCK);

            if (lane_id >= offset)
                num_pts += n;
        }

        if (lane_id < NUM_WARPS_PER_BLOCK)
            s_num_pts[warp_id][lane_id] = num_pts;
    }

    __syncthreads();

    // Compute global offsets.
    if (warp_id == 0)
    {
        int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK-1];

        for (int row = 1 ; row < 4 ; ++row)
        {
            int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK-1];

            if (lane_id < NUM_WARPS_PER_BLOCK)
                s_num_pts[row][lane_id] += sum;

            sum += tmp;
        }
    }

    __syncthreads();

    // Make the scan exclusive.
    if (threadIdx.x < 4*NUM_WARPS_PER_BLOCK)
    {
        int val = threadIdx.x == 0 ? 0 : smem[threadIdx.x-1];
        val += node.points_begin();
        smem[threadIdx.x] = val;
    }

    __syncthreads();

    //
    // 4- Move points.
    //

    // Output points.
    Points &out_points = points[(params.point_selector+1) % 2];

    // Reorder points.
    for (int range_it = range_begin + lane_id ; __any(range_it < range_end) ; range_it += warpSize)
    {
        // Is it still an active thread?
        bool is_active = range_it < range_end;

        // Load the coordinates of the point.
        float2 p = is_active ? in_points.get_point(range_it) : make_float2(0.0f, 0.0f);

        // Count top-left points.
        bool pred = is_active && p.x < center.x && p.y >= center.y;
        int vote = __ballot(pred);
        int dest = s_num_pts[0][warp_id] + __popc(vote & lane_mask_lt);

        if (pred)
            out_points.set_point(dest, p);

        if (lane_id == 0)
            s_num_pts[0][warp_id] += __popc(vote);

        // Count top-right points.
        pred = is_active && p.x >= center.x && p.y >= center.y;
        vote = __ballot(pred);
        dest = s_num_pts[1][warp_id] + __popc(vote & lane_mask_lt);

        if (pred)
            out_points.set_point(dest, p);

        if (lane_id == 0)
            s_num_pts[1][warp_id] += __popc(vote);

        // Count bottom-left points.
        pred = is_active && p.x < center.x && p.y < center.y;
        vote = __ballot(pred);
        dest = s_num_pts[2][warp_id] + __popc(vote & lane_mask_lt);

        if (pred)
            out_points.set_point(dest, p);

        if (lane_id == 0)
            s_num_pts[2][warp_id] += __popc(vote);

        // Count bottom-right points.
        pred = is_active && p.x >= center.x && p.y < center.y;
        vote = __ballot(pred);
        dest = s_num_pts[3][warp_id] + __popc(vote & lane_mask_lt);

        if (pred)
            out_points.set_point(dest, p);

        if (lane_id == 0)
            s_num_pts[3][warp_id] += __popc(vote);
    }

    __syncthreads();

    //
    // 5- Launch new blocks.
    //

    // The last thread launches new blocks.
    if (threadIdx.x == NUM_THREADS_PER_BLOCK-1)
    {
        // The children.
        Quadtree_node *children = &nodes[params.num_nodes_at_this_level];

        // The offsets of the children at their level.
        int child_offset = 4*node.id();

        // Set IDs.
        children[child_offset+0].set_id(4*node.id()+ 0);
        children[child_offset+1].set_id(4*node.id()+ 4);
        children[child_offset+2].set_id(4*node.id()+ 8);
        children[child_offset+3].set_id(4*node.id()+12);

        // Points of the bounding-box.
        const float2 &p_min = bbox.get_min();
        const float2 &p_max = bbox.get_max();

        // Set the bounding boxes of the children.
        children[child_offset+0].set_bounding_box(p_min.x , center.y, center.x, p_max.y);    // Top-left.
        children[child_offset+1].set_bounding_box(center.x, center.y, p_max.x , p_max.y);    // Top-right.
        children[child_offset+2].set_bounding_box(p_min.x , p_min.y , center.x, center.y);   // Bottom-left.
        children[child_offset+3].set_bounding_box(center.x, p_min.y , p_max.x , center.y);   // Bottom-right.

        // Set the ranges of the children.
        children[child_offset+0].set_range(node.points_begin(),   s_num_pts[0][warp_id]);
        children[child_offset+1].set_range(s_num_pts[0][warp_id], s_num_pts[1][warp_id]);
        children[child_offset+2].set_range(s_num_pts[1][warp_id], s_num_pts[2][warp_id]);
        children[child_offset+3].set_range(s_num_pts[2][warp_id], s_num_pts[3][warp_id]);

        // Launch 4 children.
        build_quadtree_kernel<<<4, NUM_THREADS_PER_BLOCK, 4 *NUM_WARPS_PER_BLOCK *sizeof(int)>>>(children, points, Parameters(params, true));
    }
}

void launch_kernel(int warp_size, Quadtree_node *nodes, Points *points, Parameters params) {
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
    const size_t smem_size = 4*NUM_WARPS_PER_BLOCK*sizeof(int);
    build_quadtree_kernel<<<1, NUM_THREADS_PER_BLOCK, smem_size>>>(nodes, points, params);
}

