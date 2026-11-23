// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// The scalar types of ik_llama.cpp (one scale times an integer code), after
// dequantize_row_q6_0 (ggml-quants.c), dequantize_row_q8_KV and dequantize_row_q8_k_r8
// (iqk/iqk_quantize.cpp) at 7c1deaf3:
//   Q6_0     133  blocks of 32: half scale, 8 bytes of high bit pairs, 16 bytes of low nibbles
//   Q8_KV    151  a row: float scale, 4 bytes ik uses for its dot products, then one int8 a value
//   Q8_K_R8  399  blocks of 256: half scale, 256 int8. ik stores eight rows interleaved
//                 (eight scales, then the eight rows' values four at a time); the loader
//                 regroups them into one row after the other, which is the layout read here.
#pragma once

#include "common.cuh"

namespace sglang {
namespace gguf {
namespace dequant {

template <>
struct Format<133> {
  static constexpr int kBlock = 32;
  static constexpr int kBlockBytes = 26;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(blk);
    const uint8_t* qh = blk + 2;
    const uint8_t* qs = blk + 10;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int p = kUnit * u + i;
      const int j = p & 15;
      const int h = qh[j & 7] >> (4 * (j >> 3));
      const int q = p < 16 ? ((qs[j] & 0x0F) | ((h << 4) & 0x30)) : ((qs[j] >> 4) | ((h << 2) & 0x30));
      v[i] = static_cast<float>(q - 32) * d;
    }
  }
};

template <>
struct Format<151> {
  static constexpr int kBlock = 32;
  static constexpr int kBlockBytes = 32;
  static constexpr int kRowMeta = 8;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = d * static_cast<float>(load_i8(blk + kUnit * u + i));
    }
  }
};

template <>
struct Format<399> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 258;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(blk);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = d * static_cast<float>(load_i8(blk + 2 + kUnit * u + i));
    }
  }
};

}  // namespace dequant
}  // namespace gguf
}  // namespace sglang
