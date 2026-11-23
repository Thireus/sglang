// SPDX-License-Identifier: MIT
//
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// IQ1_S_R4 (219) and IQ1_M_R4 (229) of ik_llama.cpp, after dequantize_row_iq1_s_r4 and
// dequantize_row_iq1_m_r4 (iqk/iqk_quantize.cpp at 7c1deaf3). They are quantisers of their own
// (blocks of 32 and one half scale a ROW), not IQ1_S / IQ1_M regrouped, but they share the IQ1_S
// codebook: 11 bits pick 8 values of -1, 0 or 1, shifted by +-1/8.
//
// ik stores four rows interleaved (four scales, then blocks holding the four rows' bytes). The
// loader regroups them into one row after the other, each with the bytes of its own blocks in
// the order ik reads them, and that is the layout read here:
//   IQ1_S_R4  row: half d; block of 6: qs[4] (the low index bytes of its four groups of 8),
//             u16 qh (3 high index bits a group, 3 bits of scale, the sign of the shift)
//   IQ1_M_R4  row: half d; block of 7: qs[4] (two groups of 8 for each run of 16), qh[2] (a run
//             of 16: two times 3 high index bits and a shift sign), scales (a nibble a run)
#pragma once

#include "../ggml-common.h"
#include "common.cuh"

namespace sglang {
namespace gguf {
namespace dequant {

// The 8 values of codebook entry `index`, each plus `delta`, times `dl`. iq1s_grid_gpu packs a
// value + 1 into each nibble: values 0..3 are the low nibbles of bytes 0..3, 4..7 the high ones.
__device__ __forceinline__ void iq1s_group(int index, float delta, float dl, float* v) {
  const uint32_t grid = static_cast<uint32_t>(iq1s_grid_gpu[index]);
#pragma unroll
  for (int i = 0; i < kUnit; ++i) {
    const int value = static_cast<int>((grid >> (8 * (i & 3) + 4 * (i >> 2))) & 0x0F) - 1;
    v[i] = dl * (static_cast<float>(value) + delta);
  }
}

template <>
struct Format<219> {
  static constexpr int kBlock = 32;
  static constexpr int kBlockBytes = 6;
  static constexpr int kRowMeta = 2;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(row);
    const uint32_t qh = load_u16(blk + 4);
    const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000u) ? -IQ1S_DELTA : IQ1S_DELTA;
    iq1s_group(blk[u] | (((qh >> (3 * u)) & 7) << 8), delta, dl, v);
  }
};

template <>
struct Format<229> {
  static constexpr int kBlock = 32;
  static constexpr int kBlockBytes = 7;
  static constexpr int kRowMeta = 2;

  __device__ static __forceinline__ void unit(const uint8_t* row, const uint8_t* blk, int u, float* v) {
    const float d = load_f16(row);
    const int run = u >> 1;  // a run of 16 values
    const int half = u & 1;  // its first or second group of 8
    const int scales = blk[6];
    const float dl = d * static_cast<float>(run ? scales >> 4 : scales & 0x0F);
    const int qh = blk[4 + run];
    const int index = blk[u] | (half ? (qh & 0x70) << 4 : (qh & 0x07) << 8);
    const float delta = (qh & (half ? 0x80 : 0x08)) ? -IQ1M_DELTA : IQ1M_DELTA;
    iq1s_group(index, delta, dl, v);
  }
};

}  // namespace dequant
}  // namespace gguf
}  // namespace sglang
