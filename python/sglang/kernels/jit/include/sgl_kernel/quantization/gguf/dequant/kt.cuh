// SPDX-License-Identifier: MIT
//
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// ik_llama.cpp's trellis types, after dequantize_row_iq{1,2,3,4}_kt and
// QuantizerIQKT<..., is_int = true>::set_values (iqk/iqk_quantize.cpp at 7c1deaf3).
//
// There is no code table: an index seeds a 32-bit generator, x <- 0xCBAC1FED * x, and each step
// gives one value, the sum of the four 6-bit fields of x minus 126. An index stands for a group
// of 8 values (4 for IQ4_KT). Every row starts with a float scale; a block of 256 adds an
// integer scale for each run of 32 (IQ1_KT, IQ4_KT) or two for each run of 32 (IQ2_KT, IQ3_KT):
//   IQ1_KT  158  13-bit index a group of 8, scale = iq4k[4 bits]
//   IQ2_KT  153  16-bit index a group of 8, scale = iq4k[4 bits]
//   IQ3_KT  154  16-bit index a group of 8 for the magnitudes, 4-bit scale, one sign bit a weight
//   IQ4_KT  155  15-bit index a group of 4, 7-bit scale, one bit a run of 32 picks the seed offset
//
// A row whose length is not a multiple of 256 ends in ik's 32-value tail blocks, which have a
// layout of their own; the host side refuses such rows.
#pragma once

#include "../iqk_types.cuh"
#include "common.cuh"

namespace sglang {
namespace gguf {
namespace dequant {

// One step of the generator: the next value of the group seeded into x.
__device__ __forceinline__ float trellis_next(uint32_t& x) {
  x *= 0xCBAC1FEDu;
  const uint32_t s = x & 0x3F3F3F3Fu;
  const int sum = static_cast<int>((s & 0xFF) + ((s >> 8) & 0xFF) + ((s >> 16) & 0xFF) + (s >> 24));
  return static_cast<float>(sum) - 126.f;
}

// row: float d; block_iq1_kt: sh[8], ql[32], qh[16]
template <>
struct Format<158> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 56;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
    const int ib = u >> 2;
    const int ig = u & 3;
    const uint32_t sh = blk[ib];
    const float sl = d * static_cast<float>(iqk_iq4k_values[sh & 0x0F]);
    const uint32_t qh = blk[40 + 4 * (ib & 3) + ig];
    const uint32_t index = blk[8 + u] | ((qh << (8 - 4 * (ib >> 2))) & 0xF00u) | ((sh << (8 - ig)) & 0x1000u);
    uint32_t x = index + 4096u;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = sl * trellis_next(x);
    }
  }
};

// row: float d; block_iq2_kt: scales[4], u16 ql[32]
template <>
struct Format<153> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 68;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
    const int s = blk[(u >> 2) & 3];
    const float sl = d * static_cast<float>(iqk_iq4k_values[(u >> 4) ? s >> 4 : s & 0x0F]);
    uint32_t x = load_u16(blk + 4 + 2 * u) + 4096u;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = sl * trellis_next(x);
    }
  }
};

// row: float d; block_iq3_kt: scales[4], u16 ql[32], qh[32]. The sign of weight w of the block is
// bit w / 32 of qh[w % 32].
template <>
struct Format<154> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 100;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row) * 1.01f;
    const int s = blk[(u >> 2) & 3];
    const float sl = d * static_cast<float>((u >> 4) ? s >> 4 : s & 0x0F);
    uint32_t x = load_u16(blk + 4 + 2 * u) + 4096u;
    const uint8_t* qh = blk + 68 + 8 * (u & 3);
    const int bit = u >> 2;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const float magnitude = sl * fabsf(trellis_next(x));
      v[i] = ((qh[i] >> bit) & 1) ? -magnitude : magnitude;
    }
  }
};

// row: float d; block_iq4_kt: u32 qs[8] (bit 0 the offset, bits 1-7 the scale, bits 8-31 the top
// three index bits of the 8 groups), ql[64], qh[32]
template <>
struct Format<155> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 128;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
    const int ib = u >> 2;
    const uint32_t sh = load_u32(blk + 4 * ib);
    const uint32_t offset = (sh & 1) ? 32768u + 4096u : 4096u;
    const float sl = d * static_cast<float>(static_cast<int>((sh & 0xFF) >> 1) - 64);
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const int ig = 2 * (u & 3) + g;
      const int jj = 8 * ib + ig;
      const uint32_t qh = blk[96 + (jj & 31)];
      const uint32_t index =
          blk[32 + jj] | ((qh << (8 - 4 * (jj >> 5))) & 0xF00u) | (((sh >> (8 + 3 * ig)) & 7u) << 12);
      uint32_t x = index + offset;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[4 * g + i] = sl * trellis_next(x);
      }
    }
  }
};

}  // namespace dequant
}  // namespace gguf
}  // namespace sglang
