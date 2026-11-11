// SPDX-License-Identifier: MIT
//
// GGUF MoE vector projection for several routes of one expert per CTA (IQ2_S). The dot product of
// a row is ggml's vec_dot_iq2_s_q8_1 (copied and adapted from llama.cpp b2899 ggml-cuda/vecdotq.cuh,
// Copyright (C) 2023-2024 The ggml authors), split in two: the weight sub-block is unpacked once,
// then multiplied with the activations of each route. A route's result is the value moe_vec_q
// writes for it: the same integer sums, the same float expression, the same warp reduction.
#pragma once

#include "vecdotq.cuh"
#include <cstdint>

namespace sglang {
namespace gguf {

// The routes route[0..kN) of one expert through row `row`: the weight sub-block is unpacked once and
// multiplied with each route's activations. kN is a compile-time count (no test inside the loops).
template <typename scalar_t, int kN>
static __device__ __forceinline__ void moe_vec_multi_iq2_s_routes(
    const block_iq2_s* __restrict__ x,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int32_t* __restrict__ sids,
    const int blocks_per_row,
    const int nrows,
    const int token_stride,
    const int top_k,
    const int row) {
  const int ib32 = threadIdx.x % QI2_S;
  const block_q8_1* y[kN];
  float tmp[kN];
#pragma unroll
  for (int r = 0; r < kN; ++r) {
    y[r] = (const block_q8_1*)(((const int*)vy) + (int64_t)(sids[r] / top_k) * token_stride);
    tmp[r] = 0.0f;
  }

  for (int i = threadIdx.x / QI2_S; i < blocks_per_row; i += WARP_SIZE / QI2_S) {
    const block_iq2_s* bq2 = &x[i];
    const uint8_t* signs = bq2->qs + QK_K / 8 + 4 * ib32;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >> 4;
    int grid[8];
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const uint32_t* grid_pos =
          (const uint32_t*)(iq2s_grid + (bq2->qs[4 * ib32 + l] | ((bq2->qh[ib32] << (8 - 2 * l)) & 0x300)));
      const uint32_t signs0 = __vcmpeq4(((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
      const uint32_t signs1 = __vcmpeq4(((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201);
      grid[2 * l + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
      grid[2 * l + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
    }
    const float dw = __half2float(bq2->d);
#pragma unroll
    for (int r = 0; r < kN; ++r) {
      const block_q8_1* bq8_1 = y[r] + i * (QK_K / QK8_1);
      const int* q8 = (const int*)bq8_1[ib32].qs;
      int sumi1 = 0;
      sumi1 = __dp4a(grid[0], q8[0], sumi1);
      sumi1 = __dp4a(grid[1], q8[1], sumi1);
      sumi1 = __dp4a(grid[2], q8[2], sumi1);
      sumi1 = __dp4a(grid[3], q8[3], sumi1);
      int sumi2 = 0;
      sumi2 = __dp4a(grid[4], q8[4], sumi2);
      sumi2 = __dp4a(grid[5], q8[5], sumi2);
      sumi2 = __dp4a(grid[6], q8[6], sumi2);
      sumi2 = __dp4a(grid[7], q8[7], sumi2);
      const float d = dw * __low2float(bq8_1[ib32].ds) * 0.25f;
      tmp[r] += d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
    }
  }

#pragma unroll
  for (int r = 0; r < kN; ++r) {
    float t = tmp[r];
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      t += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), t, mask);
    }
    if (threadIdx.x == 0) {
      dst[(int64_t)sids[r] * nrows + row] = t;
    }
  }
}

// grid (ceil(nrows / blockDim.y), lanes), block (32, rows per CTA).
// vx: packed experts [E][nrows][ncols / 256] block_iq2_s; vy: q8_1 rows of token_stride ints;
// dst: [numel, nrows]. sorted_ids / expert_ids / num_post_padded come from moe_align_block_size
// with block size kRoutes: a sorted id s < numel is route s, its activation row is s / top_k
// (top_k = 1 for the down projection); s >= numel pads. The CTA of lane y walks the blocks y,
// y + lanes, ... below num_post_padded and skips those of a negative expert: the grid does not
// grow with the routes (graph-capture safe, and no CTA is launched for a block that has no work;
// the blocks in use are consecutive, so the lanes share them evenly).
// A block's routes come first and its padding last: with kPrefix the block runs the code built
// for its route count; without it (or for a block with a hole) each route is tested in the loops.
template <typename scalar_t, int kRoutes, bool kPrefix>
static __global__ void moe_vec_multi_iq2_s(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_post_padded,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int top_k,
    const int numel,
    const int num_experts,
    const int max_blocks) {
  static_assert(kRoutes == 2 || kRoutes == 4 || kRoutes == 8, "2, 4 or 8 routes a block");
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= nrows) {
    return;
  }
  const int num_blocks = min(num_post_padded[0] / kRoutes, max_blocks);
  const int blocks_per_row = ncols / QK_K;

  for (int jb = blockIdx.y; jb < num_blocks; jb += gridDim.y) {
    const int expert = expert_ids[jb];
    if (expert < 0 || expert >= num_experts) {
      continue;
    }
    const int32_t* sids = sorted_ids + jb * kRoutes;
    const block_iq2_s* x =
        ((const block_iq2_s*)vx) + (int64_t)expert * nrows * blocks_per_row + (int64_t)row * blocks_per_row;

    int valid = 0;   // routes of the block
    int prefix = 0;  // leading routes of the block
#pragma unroll
    for (int r = 0; r < kRoutes; ++r) {
      const bool ok = sids[r] >= 0 && sids[r] < numel;
      valid += ok;
      prefix += ok && prefix == r;
    }

    if (kPrefix && prefix == valid) {
// One straight-line body per route count; the count is the same for every thread of the warp.
#define SGLANG_VEC_MULTI_CASE(N)                                                                                  \
  case N:                                                                                                         \
    if constexpr (N <= kRoutes) {                                                                                 \
      moe_vec_multi_iq2_s_routes<scalar_t, N>(x, vy, dst, sids, blocks_per_row, nrows, token_stride, top_k, row); \
    }                                                                                                             \
    break;
      switch (valid) {
        SGLANG_VEC_MULTI_CASE(1)
        SGLANG_VEC_MULTI_CASE(2)
        SGLANG_VEC_MULTI_CASE(3)
        SGLANG_VEC_MULTI_CASE(4)
        SGLANG_VEC_MULTI_CASE(5)
        SGLANG_VEC_MULTI_CASE(6)
        SGLANG_VEC_MULTI_CASE(7)
        SGLANG_VEC_MULTI_CASE(8)
        default:
          break;  // no route in the block
      }
#undef SGLANG_VEC_MULTI_CASE
      continue;
    }

    int route[kRoutes];
    const block_q8_1* y[kRoutes];
    float tmp[kRoutes];
#pragma unroll
    for (int r = 0; r < kRoutes; ++r) {
      const int sid = sids[r];
      const bool ok = sid >= 0 && sid < numel;
      route[r] = ok ? sid : -1;
      y[r] = (const block_q8_1*)(((const int*)vy) + (int64_t)(ok ? sid / top_k : 0) * token_stride);
      tmp[r] = 0.0f;
    }

    const int ib32 = threadIdx.x % QI2_S;
    for (int i = threadIdx.x / QI2_S; i < blocks_per_row; i += WARP_SIZE / QI2_S) {
      const block_iq2_s* bq2 = &x[i];
      const uint8_t* signs = bq2->qs + QK_K / 8 + 4 * ib32;
      const uint8_t ls1 = bq2->scales[ib32] & 0xf;
      const uint8_t ls2 = bq2->scales[ib32] >> 4;
      int grid[8];
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint32_t* grid_pos =
            (const uint32_t*)(iq2s_grid + (bq2->qs[4 * ib32 + l] | ((bq2->qh[ib32] << (8 - 2 * l)) & 0x300)));
        const uint32_t signs0 = __vcmpeq4(((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t signs1 = __vcmpeq4(((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201);
        grid[2 * l + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
        grid[2 * l + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
      }
      const float dw = __half2float(bq2->d);
#pragma unroll
      for (int r = 0; r < kRoutes; ++r) {
        if (route[r] < 0) {
          continue;
        }
        const block_q8_1* bq8_1 = y[r] + i * (QK_K / QK8_1);
        const int* q8 = (const int*)bq8_1[ib32].qs;
        int sumi1 = 0;
        sumi1 = __dp4a(grid[0], q8[0], sumi1);
        sumi1 = __dp4a(grid[1], q8[1], sumi1);
        sumi1 = __dp4a(grid[2], q8[2], sumi1);
        sumi1 = __dp4a(grid[3], q8[3], sumi1);
        int sumi2 = 0;
        sumi2 = __dp4a(grid[4], q8[4], sumi2);
        sumi2 = __dp4a(grid[5], q8[5], sumi2);
        sumi2 = __dp4a(grid[6], q8[6], sumi2);
        sumi2 = __dp4a(grid[7], q8[7], sumi2);
        const float d = dw * __low2float(bq8_1[ib32].ds) * 0.25f;
        tmp[r] += d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
      }
    }

#pragma unroll
    for (int r = 0; r < kRoutes; ++r) {
      float t = tmp[r];
#pragma unroll
      for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        t += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), t, mask);
      }
      if (threadIdx.x == 0 && route[r] >= 0) {
        dst[(int64_t)route[r] * nrows + row] = t;
      }
    }
  }
}

}  // namespace gguf
}  // namespace sglang
