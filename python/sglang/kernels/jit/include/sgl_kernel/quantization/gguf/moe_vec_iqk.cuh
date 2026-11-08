// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// GGUF vector MoE projection for ik_llama.cpp's row-header types IQ4_KS (144) and IQ2_KS (145).
// vec_dot_iq4_ks_q8_1 / vec_dot_iq2_ks_q8_1 are copied from ik_llama.cpp 7c1deaf3
// ggml/src/ggml-cuda/template-instances/mmvq-instance-iq{4,2}_ks.cu (CUDA branch). The kernel
// follows moe_vec_q (moe_vec.cuh) but addresses rows in bytes: every row starts with a
// 4-byte (IQ4_KS) or 2-byte (IQ2_KS) scale before its first block, so the block-count expert
// offset of moe_vec_q would mis-stride them. JIT-only: the AOT build does not include this file.
#pragma once

#include "iqk_types.cuh"
#include "moe_vec.cuh"

typedef void (*vec_dot_iqk_cuda_t)(
    const void* __restrict__ vrow, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs, float* result);

static __device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs, float* result) {
  float scale = *(const float*)vbq;
  const block_iq4_ks* bq4 = (const block_iq4_ks*)((const char*)vbq + sizeof(float)) + kbx;

  // iqs is 0...28
  const int ib32 = iqs / 4;
  const int32_t* q8 = (const int*)bq8_1[ib32].qs;
  const uint32_t* q4 = (const uint32_t*)bq4->qs + 4 * ib32;
  const float dl = scale * ((bq4->scales[ib32] & 254) - 127);
  auto values = iqk_iq4k_values + ((bq4->scales[ib32] & 1) << 4);
  int sumi = 0;
  for (int j = 0; j < 4; ++j) {
    auto v = iqk_get_int_from_table_16(q4[j], values);
    sumi = __dp4a(v.x, q8[j + 0], sumi);
    sumi = __dp4a(v.y, q8[j + 4], sumi);
  }
  *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

static __device__ __forceinline__ void vec_dot_iq2_ks_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs, float* result) {
  float scale = __half2float(*(const half*)vbq);
  const block_iq2_ks* bq2 = (const block_iq2_ks*)((const char*)vbq + sizeof(half)) + kbx;

  int i4 = iqs / 4;  // 0...7: q8 blocks 4*(i4/4) ... 4*(i4/4)+3
  const int32_t* q8_1 = (const int*)bq8_1[4 * (i4 / 4) + 0].qs + 2 * (i4 % 4);
  const int32_t* q8_2 = (const int*)bq8_1[4 * (i4 / 4) + 1].qs + 2 * (i4 % 4);
  const int32_t* q8_3 = (const int*)bq8_1[4 * (i4 / 4) + 2].qs + 2 * (i4 % 4);
  const int32_t* q8_4 = (const int*)bq8_1[4 * (i4 / 4) + 3].qs + 2 * (i4 % 4);

  const uint16_t* q2 = (const uint16_t*)bq2->qs + 16 * (i4 / 4) + 4 * (i4 % 4);
  const uint16_t extra = bq2->extra >> 4 * (i4 / 4);

  uint32_t val1 = q2[0] | (q2[1] << 16), val2 = q2[2] | (q2[3] << 16);

  int32_t scales32;
  const uint16_t* scales16 = (const uint16_t*)bq2->scales;
  scales32 = __vsub4((scales16[i4 / 4] | (scales16[i4 / 4] << 12)) & 0x0f0f0f0f, 0x10101010);
  int8_t* s8 = (int8_t*)&scales32;
  s8[0] += ((extra >> 4) & 0x10);
  s8[1] += ((extra >> 6) & 0x10);
  s8[2] += ((extra >> 5) & 0x10);
  s8[3] += ((extra >> 7) & 0x10);

  uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;

  uint32_t this_extra = ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
  uint32_t idx1 = ((val1 >> 0) & 0x33333333) | this_extra;
  uint32_t idx2 = ((val2 >> 0) & 0x33333333) | this_extra;
  int2 v1 = iqk_get_int_from_table_8(idx1, iqk_iq2nl_values);
  int2 v2 = iqk_get_int_from_table_8(idx2, iqk_iq2nl_values);

  int sumi1 = __dp4a(v2.x, q8_1[1], __dp4a(v1.x, q8_1[0], 0)) * s8[0];
  int sumi3 = __dp4a(v2.y, q8_3[1], __dp4a(v1.y, q8_3[0], 0)) * s8[1];

  this_extra = ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
  idx1 = ((val1 >> 2) & 0x33333333) | this_extra;
  idx2 = ((val2 >> 2) & 0x33333333) | this_extra;
  v1 = iqk_get_int_from_table_8(idx1, iqk_iq2nl_values);
  v2 = iqk_get_int_from_table_8(idx2, iqk_iq2nl_values);

  int sumi2 = __dp4a(v2.x, q8_2[1], __dp4a(v1.x, q8_2[0], 0)) * s8[2];
  int sumi4 = __dp4a(v2.y, q8_4[1], __dp4a(v1.y, q8_4[0], 0)) * s8[3];

  *result +=
      scale * (__low2float(bq8_1[4 * (i4 / 4) + 0].ds) * sumi1 + __low2float(bq8_1[4 * (i4 / 4) + 1].ds) * sumi2 +
               __low2float(bq8_1[4 * (i4 / 4) + 2].ds) * sumi3 + __low2float(bq8_1[4 * (i4 / 4) + 3].ds) * sumi4);
}

template <typename scalar_t, typename block_q_t, int row_meta, int vdr, vec_dot_iqk_cuda_t vec_dot>
static __global__ void moe_vec_iqk(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* topk_ids,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride) {
  constexpr int qi = QK_K / 8;  // QI4_XS: 32 four-value words per block
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;
  const auto token = blockIdx.z / topk;
  const auto expert = topk_ids[blockIdx.z];

  if (row >= nrows) {
    return;
  }
  if (expert < 0) {
    if (threadIdx.x == 0) {
      dst[blockIdx.z * nrows + row] = scalar_t(0.0f);
    }
    return;
  }

  const int blocks_per_row = ncols / QK_K;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const int64_t row_size = row_meta + (int64_t)blocks_per_row * sizeof(block_q_t);
  const void* x = (const char*)vx + ((int64_t)expert * nrows + row) * row_size;
  const block_q8_1* y = (const block_q8_1*)(((const int*)vy) + token * token_stride);

  float tmp = 0.0f;
  for (int i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int iqs = vdr * (threadIdx.x % (qi / vdr));
    vec_dot(x, &y[i * (QK_K / QK8_1)], i, iqs, &tmp);
  }

#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp, mask);
  }

  if (threadIdx.x == 0) {
    dst[blockIdx.z * nrows + row] = tmp;
  }
}

#define SGLANG_MOE_VEC_IQK_LAUNCHER(name, block, meta, dot)                                                \
  template <typename scalar_t>                                                                             \
  static void name(                                                                                        \
      const void* vx,                                                                                      \
      const void* vy,                                                                                      \
      scalar_t* dst,                                                                                       \
      const int* topk_ids,                                                                                 \
      const int top_k,                                                                                     \
      const int tokens,                                                                                    \
      const int ncols,                                                                                     \
      const int nrows,                                                                                     \
      const int token_stride,                                                                              \
      cudaStream_t stream) {                                                                               \
    const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;                               \
    const dim3 block_nums(block_num_y, 1, tokens * top_k);                                                 \
    const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);                                                  \
    moe_vec_iqk<scalar_t, block, meta, 4, dot>                                                             \
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride); \
  }

SGLANG_MOE_VEC_IQK_LAUNCHER(moe_vec_iq4_ks_q8_1_cuda, block_iq4_ks, IQ4_KS_ROW_META, vec_dot_iq4_ks_q8_1)
SGLANG_MOE_VEC_IQK_LAUNCHER(moe_vec_iq2_ks_q8_1_cuda, block_iq2_ks, IQ2_KS_ROW_META, vec_dot_iq2_ks_q8_1)

#undef SGLANG_MOE_VEC_IQK_LAUNCHER
