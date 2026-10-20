// Strata K12: raw-pointer dispatch for upstream trellis-only reconstruction.
#include "reconstruct.cuh"
#include <cassert>
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"

template <int K, int cb, bool HALF = false>
__device__ __forceinline__
void reconstruct_tile
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 16 * K + (HALF ? 8 : 0);  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb, HALF>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
//    __shared__ half tile[16 * 8 * 16];
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    reconstruct_tile<K, cb, HALF>(g_unpacked, g_packed, packed_blocks_n, packed_n_offset);
}

// Batched variant: blockIdx.z selects the matrix from a pointer table, outputs are consecutive
// [k, n] slabs out_stride halfs apart. Whole matrices only.
template <int K, int cb, bool HALF = false>
__global__ __launch_bounds__(256)
void reconstruct_batch_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* const* __restrict__ packed_ptrs,
    int packed_blocks_n,
    size_t out_stride
)
{
    int b = blockIdx.z;
    reconstruct_tile<K, cb, HALF>(g_unpacked + (size_t) b * out_stride, packed_ptrs[b], packed_blocks_n, 0);
}

namespace strata_exl3 {

// K is uniform for the block. Each branch calls the upstream tile decoder.
__global__ __launch_bounds__(256)
void reconstruct_mul1_job(half* unpacked, const ReconstructJob* job, int blocks_n) {
    const uint16_t* packed = job->packed;
    switch (job->bits) {
        case 1: reconstruct_tile<1, 2, false>(unpacked, packed, blocks_n, 0); break;
        case 2: reconstruct_tile<2, 2, false>(unpacked, packed, blocks_n, 0); break;
        case 3: reconstruct_tile<3, 2, false>(unpacked, packed, blocks_n, 0); break;
        case 4: reconstruct_tile<4, 2, false>(unpacked, packed, blocks_n, 0); break;
        case 5: reconstruct_tile<5, 2, false>(unpacked, packed, blocks_n, 0); break;
        case 6: reconstruct_tile<6, 2, false>(unpacked, packed, blocks_n, 0); break;
        default: assert(false && "K12: expected integer K1..K6");
    }
}

void reconstruct_mul1(half* unpacked, const ReconstructJob* job,
                       int k, int n, cudaStream_t stream) {
    reconstruct_mul1_job<<<dim3(n / 128, k / 16), 256, 0, stream>>>(unpacked, job, n / 16);
}

}  // namespace strata_exl3
