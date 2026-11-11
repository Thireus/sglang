// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// int8 tensor-core MoE matmul (MMQ) for the GGUF types IQ2_S (22), IQ4_KS (144) and IQ2_KS (145).
// Ported from ik_llama.cpp 7c1deaf3 (unchanged at 39fc3128):
//   ggml/src/ggml-cuda/mmq_id_common.cuh: tile sizes, vec_dot_q8_0_q8_1_mma (m16n8k32),
//     vec_dot_q8_0_16_q8_1_mma (m16n8k16, Turing+ branch), load_tiles_iq2_s, mmq_write_back_mma_id,
//     mul_mat_q_process_tile_id;
//   ggml/src/ggml-cuda/template-instances/mmq-instance-iq4_ks_id.cu: load_tiles_iq4_ks;
//   ggml/src/ggml-cuda/template-instances/mmq-instance-iq2_ks_id.cu: load_tiles_iq2_ks.
// Only the NVIDIA int8-MMA branches are kept (sm_80+, built for SM120 here).
//
// Changes from ik:
//   - routing comes from SGLang's moe_align_block_size (sorted_token_ids, expert_ids per block of
//     mmq_x sorted rows, num_tokens_post_padded) instead of ik's mmq_ids_helper / expert_bounds;
//     each CTA owns one block of mmq_x sorted rows of one expert and one 128-row tile of N, with
//     plain tiling (no stream-k), and leaves early past num_tokens_post_padded or for expert < 0,
//     so the grid can be sized from the static bound and the launch is graph-capture safe;
//   - the activations are quantised once per source row ([K / 128][rows] block_q8_1_mmq) and the
//     y tile is gathered through the sorted ids, instead of ik's per-route gathered copy;
//   - the output is written as the input dtype at [route, N]; padded and invalid routes are skipped;
//   - IQ2_S always takes the dequant-exact block-16 path (load_tiles_iq2_s + m16n8k16): ik's requant
//     path (mmq_x >= 40, load_tiles_iq2_s_q8) re-rounds the weights and is not ported.
#pragma once

#include <sgl_kernel/quantization/gguf/ggml-common.h>

#include <sgl_kernel/quantization/gguf/iqk_types.cuh>
#include <sgl_kernel/quantization/gguf_mma/mma.cuh>
#include <sgl_kernel/quantization/gguf_mma/quantize_mmq.cuh>

#include <cstdint>

namespace sglang {
namespace gguf_mma {

constexpr int kMmqY = 128;    // rows of N per CTA (get_mmq_y_device, Volta+)
constexpr int kNWarps = 8;    // mmq_get_nwarps_device
constexpr int kTileNeK = 32;  // MMQ_TILE_NE_K
constexpr int kQI8_0 = 8;     // QI8_0
constexpr int kQI8_1 = 8;     // QI8_1
constexpr int kQR2_S = 4;     // ik QR2_S
constexpr int kQI2_S = QK_K / (4 * kQR2_S);
constexpr int kTileXK_Q8_0 = 2 * kTileNeK + 2 * kTileNeK / kQI8_0 + 4;  // MMQ_MMA_TILE_X_K_Q8_0 = 76
constexpr int kTileXK_Q3_K = 2 * kTileNeK + kTileNeK / 2 + 4;           // MMQ_MMA_TILE_X_K_Q3_K = 84
static_assert(kTileXK_Q8_0 % 8 == 4, "Wrong padding.");
static_assert(kTileXK_Q3_K % 8 == 4, "Wrong padding.");

constexpr __host__ __device__ int mmq_granularity(const int mmq_x) {
  return mmq_x >= 48 ? 16 : 8;  // mmq_get_granularity_device, Turing+ MMA
}

constexpr __host__ __device__ int pad_to(const int x, const int n) {
  return (x + n - 1) / n * n;
}

static __device__ __forceinline__ int mmq_get_int_b2(const void* x, const int i32) {
  const uint16_t* x16 = (const uint16_t*)x;  // 2-byte alignment
  return x16[2 * i32 + 0] | (x16[2 * i32 + 1] << 16);
}

static __device__ __forceinline__ int mmq_get_int_b4(const void* x, const int i32) {
  return ((const int*)x)[i32];  // 4-byte alignment
}

// ----------------------------------------------------------------------------------------------
// Tile loaders: one call fills x_tile with mmq_y rows x 256 K values (one QK_K block) of int8
// quants plus fp32 scales. x points at the first row of the tile; stride is the row size in bytes
// (row header included); kbx0 is the QK_K block index along K.

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_iq4_ks(
    const char* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2 * kTileNeK);

  const int kqsx = threadIdx.x / 4;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += 4 * kNWarps) {
    int i = i0 + 4 * threadIdx.y + threadIdx.x % 4;

    if (need_check) {
      i = min(i, i_max);
    }

    const float* dptr = (const float*)(x + (int64_t)i * stride);
    const block_iq4_ks* bxi = (const block_iq4_ks*)(dptr + 1) + kbx0;
    const int ls = (bxi->scales[kqsx] & 254) - 127;

    auto values = iqk_iq4k_values + ((bxi->scales[kqsx] & 1) << 4);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q4 = mmq_get_int_b4(bxi->qs, 4 * kqsx + j);
      const int2 v = iqk_get_int_from_table_16(q4, values);
      x_qs[i * kTileXK_Q8_0 + 8 * kqsx + j + 0] = v.x;
      x_qs[i * kTileXK_Q8_0 + 8 * kqsx + j + 4] = v.y;
    }
    x_df[i * kTileXK_Q8_0 + kqsx] = dptr[0] * ls;
  }
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_iq2_ks(
    const char* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2 * kTileNeK);

  const int kqsx = threadIdx.x % 16;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += 2 * kNWarps) {
    int i = i0 + 2 * threadIdx.y + threadIdx.x / 16;

    if (need_check) {
      i = min(i, i_max);
    }

    const block_iq2_ks* bxi = (const block_iq2_ks*)(x + (int64_t)i * stride + sizeof(half)) + kbx0;

    uint16_t extra = bxi->extra >> 4 * (kqsx / 8);
    int q2 = mmq_get_int_b2(bxi->qs, kqsx);

    uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;
    uint32_t val1 = ((q2 >> 0) & 0x33333333) | ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
    uint32_t val2 = ((q2 >> 2) & 0x33333333) | ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
    int2 v1 = iqk_get_int_from_table_8(val1, iqk_iq2nl_values);
    int2 v2 = iqk_get_int_from_table_8(val2, iqk_iq2nl_values);

    x_qs[i * kTileXK_Q8_0 + kqsx % 8 + 32 * (kqsx / 8) + 0] = v1.x;
    x_qs[i * kTileXK_Q8_0 + kqsx % 8 + 32 * (kqsx / 8) + 8] = v2.x;
    x_qs[i * kTileXK_Q8_0 + kqsx % 8 + 32 * (kqsx / 8) + 16] = v1.y;
    x_qs[i * kTileXK_Q8_0 + kqsx % 8 + 32 * (kqsx / 8) + 24] = v2.y;
  }

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += kNWarps * 8) {
    int i = i0 + threadIdx.y * 8 + threadIdx.x / 4;

    if (need_check) {
      i = min(i, i_max);
    }

    const half* dptr = (const half*)(x + (int64_t)i * stride);
    const float d = __half2float(dptr[0]);
    const block_iq2_ks* bxi = (const block_iq2_ks*)(dptr + 1) + kbx0;
    const int ls1 = ((bxi->scales[threadIdx.x % 4] >> 0) & 0xf) | ((bxi->extra >> (4 + 2 * (threadIdx.x % 4))) & 0x10);
    const int ls2 = ((bxi->scales[threadIdx.x % 4] >> 4) & 0xf) | ((bxi->extra >> (5 + 2 * (threadIdx.x % 4))) & 0x10);

    x_df[i * kTileXK_Q8_0 + 2 * (threadIdx.x % 4) + 0] = d * (ls1 - 16);
    x_df[i * kTileXK_Q8_0 + 2 * (threadIdx.x % 4) + 1] = d * (ls2 - 16);
  }
}

template <int mmq_y, bool need_check>
static __device__ __forceinline__ void load_tiles_iq2_s(
    const char* __restrict__ x, int* __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + kTileNeK * 2);

  constexpr int threads_per_row = (256 / (4 * kQR2_S)) / 2;
  constexpr int nrows = kWarpSize / threads_per_row;
  const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += kNWarps * nrows) {
    int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;

    if (need_check) {
      i = min(i, i_max);
    }

    const block_iq2_s* bxi = (const block_iq2_s*)(x + (int64_t)i * stride) + kbx0;

    const int qs_packed = mmq_get_int_b2(bxi->qs, kqsx);
    const uint8_t* qs = (const uint8_t*)&qs_packed;

    const int qh = bxi->qh[kqsx];

    const int signs_packed_32 = mmq_get_int_b2(bxi->qs, QK_K / 32 + kqsx);
    const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;

#pragma unroll
    for (int l = 0; l < kQR2_S; ++l) {
      const int* grid_pos = (const int*)(iq2s_grid + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));

      const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
      const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

      const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
      const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

      x_qs[i * kTileXK_Q3_K + 8 * kqsx + (2 * l + 0)] = grid_l;
      x_qs[i * kTileXK_Q3_K + 8 * kqsx + (2 * l + 1)] = grid_h;
    }

    const int ls = bxi->scales[kqsx];
    const float d = __half2float(bxi->d);
    x_df[i * kTileXK_Q3_K + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
    x_df[i * kTileXK_Q3_K + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
  }
}

// ----------------------------------------------------------------------------------------------
// Tile products. x: the loaded x tile; y: mmq_x columns of one block_q8_1_mmq (36 words each);
// k00: 0 or 32, the half of the 64-word x tile that this y block covers.

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void
vec_dot_q8_0_q8_1_mma(const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {
  typedef tile<16, 8, int> tile_A;
  typedef tile<8, 8, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int granularity = mmq_granularity(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * kTileYK);

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + 2 * kTileNeK;
  const int* y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  tile_A A[ntx][kTileNeK / kQI8_0];
  float dA[ntx][tile_C::ne / 2][kTileNeK / kQI8_0];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < kTileNeK; k01 += kQI8_0) {
      const int k0 = k00 + k01;
      load_ldmatrix(A[n][k01 / kQI8_0], x_qs + (i0 + n * tile_A::I) * kTileXK_Q8_0 + k0, kTileXK_Q8_0);
    }

#pragma unroll
    for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);
#pragma unroll
      for (int k01 = 0; k01 < kTileNeK; k01 += kQI8_0) {
        const int k0 = k00 + k01;
        dA[n][l][k01 / kQI8_0] = x_df[i * kTileXK_Q8_0 + k0 / kQI8_0];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < kTileNeK; k01 += kQI8_0) {
      tile_B B;
      float dB[tile_C::ne / 2];

      load_generic(B, y_qs + j0 * kTileYK + k01, kTileYK);  // faster than load_ldmatrix

#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * kTileYK + k01 / kQI8_1];
      }

#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01 / kQI8_0], B);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l] * dA[n][l / 2][k01 / kQI8_0] * dB[l % 2];
        }
      }
    }
  }
}

template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void
vec_dot_q8_0_16_q8_1_mma(const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {
  typedef tile<16, 4, int> tile_A;
  typedef tile<16, 8, int> tile_A_8;
  typedef tile<8, 4, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int granularity = mmq_granularity(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * kTileYK);

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + kTileNeK * 2;
  const int* y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_A::I);

  tile_A A[ntx][8];
  float dA[ntx][tile_C::ne / 2][8];

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < kTileNeK; k01 += 8) {
      const int k0 = k00 + k01;
      load_ldmatrix(((tile_A_8*)A[n])[k01 / 8], x_qs + (i0 + n * tile_A::I) * kTileXK_Q3_K + k0, kTileXK_Q3_K);
    }

#pragma unroll
    for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_C::I + tile_C::get_i(2 * l);
#pragma unroll
      for (int k01 = 0; k01 < kTileNeK; k01 += 4) {
        const int k0 = k00 + k01;
        dA[n][l][k01 / 4] = x_df[i * kTileXK_Q3_K + k0 / 4];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < kTileNeK; k01 += 8) {  // QR3_K * VDR_Q3_K_Q8_1_MMQ
      tile_B B[2];
      float dB[tile_C::ne / 2];

      load_generic(B[0], y_qs + j0 * kTileYK + (k01 + 0), kTileYK);
      load_generic(B[1], y_qs + j0 * kTileYK + (k01 + tile_B::J), kTileYK);

#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * kTileYK + k01 / kQI8_1];
      }

#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C[2];
        mma(C[0], A[n][k01 / 4 + 0], B[0]);
        mma(C[1], A[n][k01 / 4 + 1], B[1]);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
              dB[l % 2] * (C[0].x[l] * dA[n][l / 2][k01 / 4 + 0] + C[1].x[l] * dA[n][l / 2][k01 / 4 + 1]);
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------------------------

template <int kType>
struct MmqTraits;

template <>
struct MmqTraits<GGML_TYPE_IQ2_S_ID> {
  static constexpr int kTileXK = kTileXK_Q3_K;
  static constexpr int kRowMeta = 0;
  static constexpr int kBlockBytes = sizeof(block_iq2_s);
  template <int mmq_y, bool need_check>
  static __device__ __forceinline__ void load(const char* x, int* t, int kb, int i_max, int stride) {
    load_tiles_iq2_s<mmq_y, need_check>(x, t, kb, i_max, stride);
  }
  template <int mmq_x, int mmq_y>
  static __device__ __forceinline__ void dot(const int* x, const int* y, float* sum, int k00) {
    vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>(x, y, sum, k00);
  }
};

template <>
struct MmqTraits<GGML_TYPE_IQ4_KS_ID> {
  static constexpr int kTileXK = kTileXK_Q8_0;
  static constexpr int kRowMeta = IQ4_KS_ROW_META;
  static constexpr int kBlockBytes = sizeof(block_iq4_ks);
  template <int mmq_y, bool need_check>
  static __device__ __forceinline__ void load(const char* x, int* t, int kb, int i_max, int stride) {
    load_tiles_iq4_ks<mmq_y, need_check>(x, t, kb, i_max, stride);
  }
  template <int mmq_x, int mmq_y>
  static __device__ __forceinline__ void dot(const int* x, const int* y, float* sum, int k00) {
    vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y>(x, y, sum, k00);
  }
};

template <>
struct MmqTraits<GGML_TYPE_IQ2_KS_ID> {
  static constexpr int kTileXK = kTileXK_Q8_0;
  static constexpr int kRowMeta = IQ2_KS_ROW_META;
  static constexpr int kBlockBytes = sizeof(block_iq2_ks);
  template <int mmq_y, bool need_check>
  static __device__ __forceinline__ void load(const char* x, int* t, int kb, int i_max, int stride) {
    load_tiles_iq2_ks<mmq_y, need_check>(x, t, kb, i_max, stride);
  }
  template <int mmq_x, int mmq_y>
  static __device__ __forceinline__ void dot(const int* x, const int* y, float* sum, int k00) {
    vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y>(x, y, sum, k00);
  }
};

template <int kType, int mmq_x>
constexpr __host__ __device__ int mmq_shared_words() {
  // ids_dst + ids_src, the y tile padded to the thread count, the x tile
  return 2 * mmq_x + pad_to(mmq_x * kTileYK, kNWarps * kWarpSize) + kMmqY * MmqTraits<kType>::kTileXK;
}

template <typename scalar_t, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void mmq_write_back_mma_id(
    const float* __restrict__ sum,
    const int* __restrict__ ids_dst,
    scalar_t* __restrict__ dst,
    const int stride,
    const int i_max) {
  typedef tile<16, 8, int> tile_C;
  constexpr int granularity = mmq_granularity(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx = rows_per_warp / tile_C::I;
  static_assert(kNWarps * tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        const int route = ids_dst[j];
        if (route < 0) {
          continue;
        }
        const int i = i0 + n * tile_C::I + tile_C::get_i(l);
        if (need_check && i > i_max) {
          continue;
        }
        dst[(int64_t)route * stride + i] = static_cast<scalar_t>(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

// grid (num_blocks of mmq_x sorted rows, ceil(N / 128)), block (32, 8).
// x: weights [E, N, row_bytes]; y: [K / 128][nrows_y] block_q8_1_mmq; dst: [numel, N].
// A sorted id s < numel is route s: its activation row is s / top_k (top_k = 1 for the down
// projection, whose input is already per route); s >= numel is padding.
template <typename scalar_t, int kType, int mmq_x, bool need_check>
__global__ void __launch_bounds__(kWarpSize* kNWarps, 1) gguf_moe_mmq_kernel(
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
  using Traits = MmqTraits<kType>;
  static_assert(mmq_x % mmq_granularity(mmq_x) == 0, "mmq_x must be a multiple of the granularity");

  extern __shared__ int data_mul_mat_q[];
  int* ids_dst = data_mul_mat_q;
  int* ids_src = ids_dst + mmq_x;
  int* tile_y = ids_src + mmq_x;
  int* tile_x = tile_y + pad_to(mmq_x * kTileYK, kNWarps * kWarpSize);

  const int jt = blockIdx.x;
  const int it = blockIdx.y;
  const int col0 = jt * mmq_x;
  if (col0 >= num_tokens_post_padded[0]) {
    return;
  }
  const int expert = expert_ids[jt];
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
  for (int j = tid; j < mmq_x; j += kNWarps * kWarpSize) {
    const int sid = sorted_ids[col0 + j];
    const bool valid = sid >= 0 && sid < numel;
    ids_dst[j] = valid ? sid : -1;
    ids_src[j] = valid ? sid / top_k : 0;
  }
  __syncthreads();

  const char* xt = x + (int64_t)expert * stride_expert_x + (int64_t)it * kMmqY * stride_row_x;
  const int tile_x_max_i = nrows_x - it * kMmqY - 1;
  const int blocks_per_row = ncols_x / QK_K;

  float sum[mmq_x * kMmqY / (kNWarps * kWarpSize)] = {0.0f};

  for (int kb0 = 0; kb0 < blocks_per_row; ++kb0) {
    Traits::template load<kMmqY, need_check>(xt, tile_x, kb0, tile_x_max_i, stride_row_x);

#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int64_t b128 = (int64_t)(2 * kb0 + k) * nrows_y;
#pragma unroll
      for (int l0 = 0; l0 < mmq_x * kTileYK; l0 += kNWarps * kWarpSize) {
        const int l = l0 + tid;
        if (l < mmq_x * kTileYK) {
          const int col = l / kTileYK;
          const int w = l - col * kTileYK;
          tile_y[l] = y[(b128 + ids_src[col]) * kTileYK + w];
        }
      }

      __syncthreads();

      Traits::template dot<mmq_x, kMmqY>(tile_x, tile_y, sum, k * kTileNeK);

      __syncthreads();
    }
  }

  mmq_write_back_mma_id<scalar_t, mmq_x, kMmqY, need_check>(sum, ids_dst, dst + it * kMmqY, nrows_x, tile_x_max_i);
}

}  // namespace gguf_mma
}  // namespace sglang
