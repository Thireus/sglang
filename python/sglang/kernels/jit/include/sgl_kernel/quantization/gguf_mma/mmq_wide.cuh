// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// A wide-block variant of the GGUF MoE MMQ kernel for IQ2_S (mmq_id.cuh, ported from ik_llama.cpp):
// a CTA owns up to kWide sorted routes of one expert instead of 16 and multiplies only the groups
// of 8 columns that hold a route, 32 columns of activations at a time. The weight tile of an expert
// (128 rows x 256 values, unpacked into shared memory) is then loaded once per kWide routes
// instead of once per 16, and a block that is partly filled does not pay for its empty columns.
//
// The arithmetic of a (route, row) is that of gguf_moe_mmq_kernel with mmq_x = 16: the same tile
// loader, the same m16n8k16 products, the same float expression accumulated in the same order
// (QK_K block, half, 32-value group); only the column a route sits in differs. The output bits
// are the same.
#pragma once

#include <sgl_kernel/quantization/gguf_mma/mmq_id.cuh>

#include <cstdint>

namespace sglang {
namespace gguf_mma {

constexpr int kWideGroup = 8;  // columns per group: tile_C::J
constexpr int kWidePass = 32;  // columns of the y tile gathered per pass

template <int kWide>
constexpr __host__ __device__ int mmq_wide_pass_cols() {
  return kWide < kWidePass ? kWide : kWidePass;
}

template <int kWide>
constexpr __host__ __device__ int mmq_wide_shared_words() {
  // ids_dst + ids_src, the y tile of one pass padded to the thread count, the x tile
  return 2 * kWide + pad_to(mmq_wide_pass_cols<kWide>() * kTileYK, kNWarps * kWarpSize) + kMmqY * kTileXK_Q3_K;
}

// vec_dot_q8_0_16_q8_1_mma for kGroups groups of 8 columns (granularity 8: one warp owns 16 rows
// of N and every column group).
template <int kGroups>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma_groups(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {
  typedef tile<16, 4, int> tile_A;
  typedef tile<16, 8, int> tile_A_8;
  typedef tile<8, 4, int> tile_B;
  typedef tile<16, 8, int> tile_C;
  static_assert(tile_C::J == kWideGroup, "one group is one C tile");

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + kTileNeK * 2;
  const int* y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  const int i0 = threadIdx.y * tile_A::I;

  tile_A A[8];
  float dA[tile_C::ne / 2][8];

#pragma unroll
  for (int k01 = 0; k01 < kTileNeK; k01 += 8) {
    const int k0 = k00 + k01;
    load_ldmatrix(((tile_A_8*)A)[k01 / 8], x_qs + i0 * kTileXK_Q3_K + k0, kTileXK_Q3_K);
  }

#pragma unroll
  for (int l = 0; l < tile_C::ne / 2; ++l) {
    const int i = i0 + tile_C::get_i(2 * l);
#pragma unroll
    for (int k01 = 0; k01 < kTileNeK; k01 += 4) {
      const int k0 = k00 + k01;
      dA[l][k01 / 4] = x_df[i * kTileXK_Q3_K + k0 / 4];
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < kGroups * kWideGroup; j0 += kWideGroup) {
#pragma unroll
    for (int k01 = 0; k01 < kTileNeK; k01 += 8) {
      tile_B B[2];
      float dB[tile_C::ne / 2];

      load_generic(B[0], y_qs + j0 * kTileYK + (k01 + 0), kTileYK);
      load_generic(B[1], y_qs + j0 * kTileYK + (k01 + tile_B::J), kTileYK);

#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * kTileYK + k01 / kQI8_1];
      }

      tile_C C[2];
      mma(C[0], A[k01 / 4 + 0], B[0]);
      mma(C[1], A[k01 / 4 + 1], B[1]);

#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        sum[j0 / kWideGroup * tile_C::ne + l] +=
            dB[l % 2] * (C[0].x[l] * dA[l / 2][k01 / 4 + 0] + C[1].x[l] * dA[l / 2][k01 / 4 + 1]);
      }
    }
  }
}

// One pass over kNow groups (up to 32 columns) of a block: gather their activations for the
// 128-value half k00 of the loaded x tile through the route ids, then multiply.
template <int kNow>
static __device__ __forceinline__ void mmq_wide_pass(
    const int* __restrict__ y,
    const int64_t b128,
    const int* __restrict__ ids_pass,
    int* __restrict__ tile_y,
    const int* __restrict__ tile_x,
    float* __restrict__ sum,
    const int k00) {
  constexpr int kWords = kNow * kWideGroup * kTileYK;
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
#pragma unroll
  for (int l0 = 0; l0 < kWords; l0 += kNWarps * kWarpSize) {
    const int l = l0 + tid;
    if (l < kWords) {
      const int col = l / kTileYK;
      const int w = l - col * kTileYK;
      tile_y[l] = y[(b128 + ids_pass[col]) * kTileYK + w];
    }
  }

  __syncthreads();

  vec_dot_q8_0_16_q8_1_mma_groups<kNow>(tile_x, tile_y, sum, k00);

  __syncthreads();
}

// One block with its first kGroups column groups in use, kGroups a compile-time count: the passes
// of 32 columns, the groups and the write-back have constant bounds, as in the 16-route kernel
// (a run-time group count inside the loops cost 40 % on SM120).
template <typename scalar_t, int kGroups, bool need_check>
static __device__ __forceinline__ void mmq_wide_block(
    const char* __restrict__ xt,
    const int* __restrict__ y,
    const int* __restrict__ ids_dst,
    const int* __restrict__ ids_src,
    int* __restrict__ tile_y,
    int* __restrict__ tile_x,
    scalar_t* __restrict__ out,
    const int blocks_per_row,
    const int nrows_x,
    const int stride_row_x,
    const int nrows_y,
    const int tile_x_max_i) {
  typedef tile<16, 8, int> tile_C;
  constexpr int kPassGroups = kWidePass / kWideGroup;
  static_assert(kGroups >= 1 && kGroups <= 4 * kPassGroups, "at most four passes");

  float sum[kGroups * tile_C::ne] = {0.0f};

// Pass P covers the groups [P * kPassGroups, min(kGroups, (P + 1) * kPassGroups)).
#define SGLANG_MMQ_WIDE_PASS(P)                                                                                  \
  if constexpr (kGroups > (P) * kPassGroups) {                                                                   \
    constexpr int kNow = kGroups - (P) * kPassGroups < kPassGroups ? kGroups - (P) * kPassGroups : kPassGroups;  \
    mmq_wide_pass<kNow>(                                                                                         \
        y, b128, ids_src + (P) * kWidePass, tile_y, tile_x, sum + (P) * kPassGroups * tile_C::ne, k * kTileNeK); \
  }
  for (int kb0 = 0; kb0 < blocks_per_row; ++kb0) {
    load_tiles_iq2_s<kMmqY, need_check>(xt, tile_x, kb0, tile_x_max_i, stride_row_x);

#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int64_t b128 = (int64_t)(2 * kb0 + k) * nrows_y;
      SGLANG_MMQ_WIDE_PASS(0)
      SGLANG_MMQ_WIDE_PASS(1)
      SGLANG_MMQ_WIDE_PASS(2)
      SGLANG_MMQ_WIDE_PASS(3)
    }
  }
#undef SGLANG_MMQ_WIDE_PASS

  const int i0 = threadIdx.y * tile_C::I;
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
      const int route = ids_dst[g * kWideGroup + tile_C::get_j(l)];
      if (route < 0) {
        continue;
      }
      const int i = i0 + tile_C::get_i(l);
      if (need_check && i > tile_x_max_i) {
        continue;
      }
      out[(int64_t)route * nrows_x + i] = static_cast<scalar_t>(sum[g * tile_C::ne + l]);
    }
  }
}

// grid (num_blocks of kWide sorted rows, ceil(N / 128)), block (32, 8).
// x: IQ2_S weights [E, N, K / 256 * 82]; y: [K / 128][nrows_y] block_q8_1_mmq; dst: [numel, N].
// sorted_ids / expert_ids / num_tokens_post_padded come from moe_align_block_size with block size
// kWide (pad_expert_ids): a sorted id s < numel is route s, its activation row is s / top_k
// (top_k = 1 for the down projection); s >= numel pads. A CTA past num_tokens_post_padded, of a
// negative expert or with no route leaves at once, so the grid is sized from the static bound.
// The y tile holds one pass of 32 columns: with the x tile the CTA stays under 50 KiB of shared
// memory whatever kWide (two CTAs a multiprocessor on a card with 100 KiB each, as the 16-route
// kernel). kSkip = false multiplies every group of every block (a reference for measurements).
template <typename scalar_t, int kWide, bool need_check, bool kSkip>
__global__ void __launch_bounds__(kWarpSize* kNWarps, 1) gguf_moe_mmq_wide_iq2_s_kernel(
    const char* __restrict__ x,
    const int* __restrict__ y,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    scalar_t* __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int stride_row_x,
    const int64_t stride_expert_x,
    const int nrows_y,
    const int top_k,
    const int numel,
    const int num_experts) {
  typedef tile<16, 8, int> tile_C;
  constexpr int kMaxGroups = kWide / kWideGroup;
  static_assert(kWide == 16 || kWide == 32 || kWide == 64 || kWide == 128, "kWide: 16, 32, 64 or 128");
  static_assert(kNWarps * tile_C::I == kMmqY, "nwarps*tile_C::I != mmq_y");

  extern __shared__ int data_mul_mat_q[];
  int* ids_dst = data_mul_mat_q;
  int* ids_src = ids_dst + kWide;
  int* tile_y = ids_src + kWide;
  int* tile_x = tile_y + pad_to(mmq_wide_pass_cols<kWide>() * kTileYK, kNWarps * kWarpSize);

  const int jt = blockIdx.x;
  const int it = blockIdx.y;
  const int col0 = jt * kWide;
  if (col0 >= num_tokens_post_padded[0]) {
    return;
  }
  const int expert = expert_ids[jt];
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
  for (int j = tid; j < kWide; j += kNWarps * kWarpSize) {
    const int sid = sorted_ids[col0 + j];
    const bool valid = sid >= 0 && sid < numel;
    ids_dst[j] = valid ? sid : -1;
    ids_src[j] = valid ? sid / top_k : 0;
  }
  __syncthreads();

  // Groups in use: up to the last column that holds a route (the align kernel puts an expert's
  // routes first and its padding last; a hole would only cost its group's products).
  int groups = kMaxGroups;
  if (kSkip) {
    groups = 0;
#pragma unroll
    for (int g = 0; g < kMaxGroups; ++g) {
      bool any = false;
#pragma unroll
      for (int j = 0; j < kWideGroup; ++j) {
        any = any || ids_dst[g * kWideGroup + j] >= 0;
      }
      if (any) {
        groups = g + 1;
      }
    }
  }

  const char* xt = x + (int64_t)expert * stride_expert_x + (int64_t)it * kMmqY * stride_row_x;
  const int tile_x_max_i = nrows_x - it * kMmqY - 1;
  const int blocks_per_row = ncols_x / QK_K;
  scalar_t* out = dst + it * kMmqY;

// One straight-line body per group count; the count is the same for every thread of the CTA.
#define SGLANG_MMQ_WIDE_CASE(G)                                                                                        \
  case G:                                                                                                              \
    if constexpr (G <= kMaxGroups) {                                                                                   \
      mmq_wide_block<scalar_t, G, need_check>(                                                                         \
          xt, y, ids_dst, ids_src, tile_y, tile_x, out, blocks_per_row, nrows_x, stride_row_x, nrows_y, tile_x_max_i); \
    }                                                                                                                  \
    break;
  switch (groups) {
    SGLANG_MMQ_WIDE_CASE(1)
    SGLANG_MMQ_WIDE_CASE(2)
    SGLANG_MMQ_WIDE_CASE(3)
    SGLANG_MMQ_WIDE_CASE(4)
    SGLANG_MMQ_WIDE_CASE(5)
    SGLANG_MMQ_WIDE_CASE(6)
    SGLANG_MMQ_WIDE_CASE(7)
    SGLANG_MMQ_WIDE_CASE(8)
    SGLANG_MMQ_WIDE_CASE(9)
    SGLANG_MMQ_WIDE_CASE(10)
    SGLANG_MMQ_WIDE_CASE(11)
    SGLANG_MMQ_WIDE_CASE(12)
    SGLANG_MMQ_WIDE_CASE(13)
    SGLANG_MMQ_WIDE_CASE(14)
    SGLANG_MMQ_WIDE_CASE(15)
    SGLANG_MMQ_WIDE_CASE(16)
    default:
      break;  // no route in the block
  }
#undef SGLANG_MMQ_WIDE_CASE
}

}  // namespace gguf_mma
}  // namespace sglang
