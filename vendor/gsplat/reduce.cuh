#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#define MAX_INIT 0.0
#define WARP_SIZE 64

namespace cg = cooperative_groups;

__inline__ __device__ float warp_reduce_sum(float val, const int tile) {
    for ( int offset = tile / 2; offset > 0; offset /= 2 )
        val += __shfl_down(val, offset);

    return val;
}

__inline__ __device__ float block_reduce_sum(float val, const int tile) {
    static __shared__ int shared[32]; // Shared memory for 32 partial sums
    const int warpSize = tile / 2;
    const int wid  = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    warp_reduce_sum(val, tile);

    if (blockDim.x < warpSize) return val;
    if (lane == 0) shared[wid] = val;

    __syncthreads( ); // Wait for all partial reductions

    // Read from shared memory only if the warp exists
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if ( wid == 0 ) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.;
        val = warp_reduce_sum(val, tile);
    }

    return val;    
}

__inline__ __device__ float warp_reduce_max(float val, const int tile) {
    for (int offset = tile / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_xor(val, offset));
    return val;
}

__inline__ __device__ float block_reduce_max(float val, const int tile) {
    __shared__ float shared[32];
    const int warpSize = tile / 2;
    const int wid  = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;

    val = warp_reduce_max(val, tile);

    if (blockDim.x < warpSize) return val;
    if (lane == 0) shared[wid] = val;

    __syncthreads( );

    if ( wid == 0 )
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : MAX_INIT;
        val = warp_reduce_max(val, tile);

    return val;
}
