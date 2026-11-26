// SPDX-License-Identifier: MIT
//
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// ik_llama.cpp's non-linear types, after the dequantize_row_<type> functions of
// iqk/iqk_quantize.cpp at 7c1deaf3. All of them split a block of 256 into sub-blocks with an
// integer scale each, and look every code up in a small table of int8 values that has two
// halves (an "extra" bit a sub-block picks the shifted half).
//
//   K family, the block carries its own half scale d:
//     IQ2_K  137   2 bits, iq2nl,  16-value sub-blocks, 4-bit scales
//     IQ3_K  138   3 bits, iq3nl,  16-value sub-blocks, 4-bit magnitudes and a sign bit
//     IQ4_K  139   4 bits, iq4k,   16-value sub-blocks, 6-bit scales
//     IQ5_K  140   5 bits, iq5nl,  16-value sub-blocks, 6-bit scales
//     IQ6_K  141   6 bits, a cubic in the code, 16-value sub-blocks, int8 scales
//   KS family, the ROW carries the scale d (a float, or a half for the 2-byte headers):
//     IQ4_KS   144   4 bits, iq4k,  32-value sub-blocks, 7-bit scales
//     IQ2_KS   145   2 bits, iq2nl, 32-value sub-blocks, 5-bit scales
//     IQ4_KSS  146   4 bits, iq4k,  32-value sub-blocks; 16-bit words whose low bits spell the scale
//     IQ5_KS   152   5 bits, iq5nl, 32-value sub-blocks, 7-bit scales
//     IQ3_KS   156   3 bits, iq3nl, 32-value sub-blocks, 5-bit scales
//     IQ2_KL   157   5 bits for a PAIR of values (iq2kl), 32-value sub-blocks, 6-bit scales
#pragma once

#include "../iqk_types.cuh"
#include "common.cuh"

namespace sglang {
namespace gguf {
namespace dequant {

// ggml-common.h of ik_llama.cpp 7c1deaf3 (iq2nl_values and iq4k_values are in iqk_types.cuh).
static const __device__ int8_t iq3nl_values[16] = {-63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51};

static const __device__ int8_t iq5nl_values[64] = {
    -126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36,  -30,  -24,  -18, -12, -6,  -1,  5,   11,  17,  23,  29,
    36,   43,   51,   59,  68,  77,  87,  97,  109, 121, -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28,
    -22,  -16,  -10,  -4,  1,   7,   13,  19,  25,  31,  38,   45,   53,   61,  70,  79,  89,  99,  111, 123};

// A pair of int8 values an entry: the low byte is the first value.
static const __device__ uint16_t iq2kl_values[32] = {0xe9c1, 0x0dc1, 0xc1d8, 0xf6d8, 0x0dd8, 0x2fd8, 0xd8e9, 0xe9e9,
                                                     0x01e9, 0x0de9, 0x1ce9, 0xc1f6, 0x01f6, 0x0df6, 0x2ff6, 0xe901,
                                                     0xf601, 0x0101, 0x0d01, 0x1c01, 0xd80d, 0xe90d, 0xf60d, 0x010d,
                                                     0x0d0d, 0xc11c, 0xe91c, 0x011c, 0x1c1c, 0x2f1c, 0xe92f, 0x0d2f};

// ---------------------------------------------------------------------------------------- K

// block_iq2_k: half d, u16 extra, scales[8], qs[64]
template <>
struct Format<137> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 76;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(blk);
    const uint32_t extra = load_u16(blk + 2);
    const int ib = u >> 2;       // 32 values
    const int h = (u >> 1) & 1;  // their first or second 16
    const int s = blk[4 + ib];
    const float dl = d * static_cast<float>((h ? s >> 4 : s & 0x0F) - 8);
    const int8_t* values = iqk_iq2nl_values + (((extra >> (2 * ib + h)) & 1) << 2);
    const uint8_t* qs = blk + 12 + 32 * (ib >> 2) + 16 * h + 8 * (u & 1);
    const int shift = 2 * (ib & 3);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = dl * static_cast<float>(values[(qs[i] >> shift) & 3]);
    }
  }
};

// block_iq3_k: half d, u16 extra, u16 scales_h, scales_l[8], qs[64], qh[32]
template <>
struct Format<138> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 110;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(blk);
    const uint32_t extra = load_u16(blk + 2);
    const uint32_t sh = load_u16(blk + 4);
    const int ib = u >> 2;
    const int h = (u >> 1) & 1;
    const int bit = 2 * ib + h;
    const int s = blk[6 + ib];
    const int mag = 2 * (h ? s >> 4 : s & 0x0F) + 1;
    const float dl = d * static_cast<float>(((sh >> bit) & 1) ? -mag : mag);
    const int8_t* values = iq3nl_values + (((extra >> bit) & 1) << 3);
    const int j0 = 16 * h + 8 * (u & 1);
    const uint8_t* qs = blk + 14 + 32 * (ib >> 2) + j0;
    const uint8_t* qh = blk + 78 + j0;
    const int shift = 2 * (ib & 3);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = dl * static_cast<float>(values[((qs[i] >> shift) & 3) | (((qh[i] >> ib) & 1) << 2)]);
    }
  }
};

// block_iq4_k: half d, u16 extra, scales_h[4], scales_l[8], qs[128]
template <>
struct Format<139> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 144;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(blk);
    const uint32_t extra = load_u16(blk + 2);
    const int ib = u >> 2;
    const int h = (u >> 1) & 1;
    const int sh = blk[4 + (ib >> 1)] >> (4 * (ib & 1));
    const int sl = blk[8 + ib];
    const int ls = h ? ((sl >> 4) | ((sh << 2) & 0x30)) : ((sl & 0x0F) | ((sh << 4) & 0x30));
    const float dl = d * static_cast<float>(ls - 32);
    const int8_t* values = iqk_iq4k_values + (((extra >> (2 * ib + h)) & 1) << 4);
    const uint8_t* qs = blk + 16 + 16 * ib + 8 * (u & 1);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = dl * static_cast<float>(values[h ? qs[i] >> 4 : qs[i] & 0x0F]);
    }
  }
};

// block_iq5_k: half d, u16 extra, scales_h[4], scales_l[8], qs[128], qh[32]
template <>
struct Format<140> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 176;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(blk);
    const uint32_t extra = load_u16(blk + 2);
    const int i64 = u >> 3;      // 64 values
    const int c = (u >> 1) & 3;  // their 16-value quarter
    const int t = c & 1;         // which 16 bytes of qs and qh
    const int hi = c >> 1;       // low or high nibbles
    const int sh = blk[4 + i64];
    const int sl = blk[8 + 2 * i64 + hi];
    const int ls = (t ? sl >> 4 : sl & 0x0F) | (((sh << 4) >> (2 * c)) & 0x30);
    const float dl = d * static_cast<float>(ls - 32);
    const int8_t* values = iq5nl_values + (((extra >> (4 * i64 + c)) & 1) << 5);
    const int j0 = 16 * t + 8 * (u & 1);
    const uint8_t* qs = blk + 16 + 32 * i64 + j0;
    const uint8_t* qh = blk + 144 + j0;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int hb = qh[i] >> (2 * i64);
      const int q = hi ? ((qs[i] >> 4) | ((hb & 2) << 3)) : ((qs[i] & 0x0F) | ((hb & 1) << 4));
      v[i] = dl * static_cast<float>(values[q]);
    }
  }
};

// block_iq6_k: half d, u16 extra, int8 scales[16], qs[128], qh[64]
template <>
struct Format<141> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 212;
  static constexpr int kRowMeta = 0;

  __device__ static __forceinline__ void unit(const uint8_t*, const uint8_t* blk, int u, float* v) {
    // A_IQ6K, B_IQ6K, C_IQ6K, D_IQ6K and S_IQ6K of ggml-common.h
    constexpr float kA = -127.f;
    constexpr float kB = 6.2568f;
    constexpr float kC = 0.11218f;
    constexpr float kD = 0.0011972f;
    const float d = load_f16(blk);
    const uint32_t extra = load_u16(blk + 2);
    const int i64 = u >> 3;
    const int c = (u >> 1) & 3;
    const int t = c & 1;
    const int hi = c >> 1;
    const float dl = d * static_cast<float>(load_i8(blk + 4 + 4 * i64 + c));
    const float m = ((extra >> (4 * i64 + c)) & 1) ? 1.f : 0.f;
    const int j0 = 16 * t + 8 * (u & 1);
    const uint8_t* qs = blk + 20 + 32 * i64 + j0;
    const uint8_t* qh = blk + 148 + 32 * (i64 >> 1) + j0;
    const int shift = 4 * (i64 & 1);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int hb = qh[i] >> shift;
      const float q = static_cast<float>(hi ? ((qs[i] >> 4) | ((hb & 0x0C) << 2)) : ((qs[i] & 0x0F) | ((hb & 3) << 4)));
      // A + q * (B + q * (-C + q * D)) + m, each multiply-add rounded once
      const float poly = fmaf(q, fmaf(q, fmaf(q, kD, -kC), kB), kA);
      const float value = poly + m;
      v[i] = dl * value;
    }
  }
};

// --------------------------------------------------------------------------------------- KS

// row: float d; block_iq4_ks: scales[8], qs[128]
template <>
struct Format<144> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 136;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
    const int ib = u >> 2;
    const int hi = (u >> 1) & 1;
    const int s = blk[ib];
    const float dl = d * static_cast<float>((s & 254) - 127);
    const int8_t* values = iqk_iq4k_values + ((s & 1) << 4);
    const uint8_t* qs = blk + 8 + 16 * ib + 8 * (u & 1);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = dl * static_cast<float>(values[hi ? qs[i] >> 4 : qs[i] & 0x0F]);
    }
  }
};

// row: half d; block_iq2_ks: u16 extra, scales[4], qs[64]
template <>
struct Format<145> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 70;
  static constexpr int kRowMeta = 2;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(row);
    const int i64 = u >> 3;
    const int h = (u >> 2) & 1;  // first or second 32 of the 64
    const uint32_t extra = load_u16(blk) >> (2 * i64);
    const int s = blk[2 + i64];
    const int ls = h ? ((s >> 4) | ((extra >> 5) & 0x10)) : ((s & 0x0F) | ((extra >> 4) & 0x10));
    const float dl = d * static_cast<float>(ls - 16);
    const int8_t* values = iqk_iq2nl_values + (((extra >> h) & 1) << 2);
    const uint8_t* qs = blk + 6 + 32 * (i64 >> 1) + 8 * (u & 3);
    const int shift = 4 * (i64 & 1) + 2 * h;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = dl * static_cast<float>(values[(qs[i] >> shift) & 3]);
    }
  }
};

// row: float d; block_iq4_kss: u16 qs[64]. Bit 0 of the eight words of a sub-block is its scale;
// the other 15 bits, xored with themselves shifted down by one, are the 16 nibbles.
template <>
struct Format<146> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 128;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
    const int ib = u >> 2;
    const int hi = (u >> 1) & 1;
    const uint8_t* words = blk + 16 * ib;
    int ls = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      ls |= (words[2 * w] & 1) << w;
    }
    const float dl = d * static_cast<float>((ls & 254) - 127);
    const int8_t* values = iqk_iq4k_values + ((ls & 1) << 4);
    const int w0 = 4 * (u & 1);  // the first of the four words these eight nibbles sit in
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      uint32_t a = load_u16(words + 2 * (w0 + (i >> 1))) & 0xFFFEu;
      a ^= a >> 1;
      const int byte = (a >> (8 * (i & 1))) & 0xFF;
      v[i] = dl * static_cast<float>(values[hi ? byte >> 4 : byte & 0x0F]);
    }
  }
};

// row: float d; block_iq5_ks: scales[8], qs[128], qh[32]
template <>
struct Format<152> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 168;
  static constexpr int kRowMeta = 4;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f32(row);
    const int i64 = u >> 3;
    const int h = (u >> 2) & 1;
    const int s = blk[2 * i64 + h];
    const float dl = d * static_cast<float>((s & 254) - 127);
    const int8_t* values = iq5nl_values + ((s & 1) << 5);
    const int l0 = 8 * (u & 3);
    const uint8_t* qs = blk + 8 + 32 * i64 + l0;
    const uint8_t* qh = blk + 136 + l0;
    const int bit = 2 * i64 + h;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int q = (h ? qs[i] >> 4 : qs[i] & 0x0F) | (((qh[i] >> bit) & 1) << 4);
      v[i] = dl * static_cast<float>(values[q]);
    }
  }
};

// row: half d; block_iq3_ks: u16 extra, scales[4], qs[64], qh[32]
template <>
struct Format<156> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 102;
  static constexpr int kRowMeta = 2;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(row);
    const uint32_t extra = load_u16(blk);
    const int ib = u >> 2;  // 32 values: 0..7
    const int s = blk[2 + (ib & 3)];
    const int ls = (ib < 4 ? s & 0x0F : s >> 4) | (((extra >> ib) & 1) << 4);
    const float dl = d * static_cast<float>(ls - 16);
    const int8_t* values = iq3nl_values + (((extra >> (8 + ib)) & 1) << 3);
    const int l0 = 8 * (u & 3);
    const uint8_t* qs = blk + 6 + 32 * (ib >> 2) + l0;
    const uint8_t* qh = blk + 70 + l0;
    const int shift = 2 * (ib & 3);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      v[i] = dl * static_cast<float>(values[((qs[i] >> shift) & 3) | (((qh[i] >> ib) & 1) << 2)]);
    }
  }
};

// row: half d; block_iq2_kl: u16 scales_h, scales_l[4], qs[64], qh[16]
template <>
struct Format<157> {
  static constexpr int kBlock = 256;
  static constexpr int kBlockBytes = 86;
  static constexpr int kRowMeta = 2;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(row);
    const uint32_t sh = load_u16(blk);
    const int i64 = u >> 3;
    const int h = (u >> 2) & 1;  // first or second 32 of the 64
    const int sl = blk[2 + ((2 * i64 + h) & 3)] >> (4 * (i64 >> 1));
    const int ls = (sl & 0x0F) | (((sh >> (4 * i64 + 2 * h)) & 3) << 4);
    const float dl = d * static_cast<float>(ls - 32);
    const uint8_t* qs = blk + 6 + 16 * i64 + 4 * (u & 3);  // four pairs
    const uint8_t* qh = blk + 70 + 4 * (u & 3);
    const int bit = 2 * i64 + h;
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int j = i >> 1;
      const int index = (h ? qs[j] >> 4 : qs[j] & 0x0F) | (((qh[j] >> bit) & 1) << 4);
      const int value = static_cast<int8_t>((iq2kl_values[index] >> (8 * (i & 1))) & 0xFF);
      v[i] = dl * static_cast<float>(value);
    }
  }
};

}  // namespace dequant
}  // namespace gguf
}  // namespace sglang
