// SPDX-License-Identifier: MIT
//
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// The ternary BitNet types of ik_llama.cpp, after dequantize_row_iq1_bn and
// dequantize_row_iq2_bn (iqk/iqk_quantize.cpp at 7c1deaf3). A row starts with its scale
// (IQ1_BN a half, IQ2_BN a float) and every weight is that scale times -1, 0 or 1:
//   IQ1_BN  134  blocks of 64 in 13 bytes: 12 bytes of five base-3 digits each, and one byte
//                whose first four digits are the 16th weight of each run of 16
//   IQ2_BN  135  blocks of 64 in 16 bytes: two bits a weight, the four bit pairs of byte j
//                being weights j, j + 16, j + 32 and j + 48
// ik's CPU functions return the codes and leave the row scale to the dot product; its CUDA
// dequantiser and this one return scale * code.
#pragma once

#include "common.cuh"

namespace sglang {
namespace gguf {
namespace dequant {

template <>
struct Format<134> {
  static constexpr int kBlock = 64;
  static constexpr int kBlockBytes = 13;
  static constexpr int kRowMeta = 2;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    // IQ1BNQuantizer::k_mult: digit m of a byte q is ((q * 3^(4 - m)) mod 256) * 3 / 256
    const int mult[5] = {81, 27, 9, 3, 1};
    const float d = load_f16(row);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int p = kUnit * u + i;
      const int g = p >> 4;
      const int r = p & 15;
      const int q = r < 15 ? blk[3 * g + r / 5] : blk[12];
      const int m = r < 15 ? mult[r % 5] : mult[g];
      const int t = (q * m) & 0xFF;
      const int code = ((t + (t >> 1)) >> 7) - 1;
      v[i] = d * static_cast<float>(code);
    }
  }
};

template <>
struct Format<135> {
  static constexpr int kBlock = 64;
  static constexpr int kBlockBytes = 16;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int p = kUnit * u + i;
      const int code = ((blk[p & 15] >> (2 * (p >> 4))) & 3) - 1;
      v[i] = d * static_cast<float>(code);
    }
  }
};

}  // namespace dequant
}  // namespace gguf
}  // namespace sglang
