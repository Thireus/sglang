// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// Block layouts, value tables and table lookups of ik_llama.cpp's IQ4_KS (ggml type 144)
// and IQ2_KS (ggml type 145), copied from ik_llama.cpp 7c1deaf3:
// ggml/src/ggml-common.h (block_iq4_ks, block_iq2_ks, iq2nl_values, iq4k_values),
// ggml/src/ggml-cuda/iqk_cuda_common.h (get_int_from_table_8) and
// ggml/src/ggml-cuda/vecdotq.cuh (get_int_from_table_16).
//
// Both types carry a per-row header before the first block: IQ4_KS one float (4 bytes),
// IQ2_KS one half (2 bytes). A row of K values is therefore
// meta + K / 256 * sizeof(block) bytes (ggml_row_size), not a whole number of blocks.
#pragma once

#include "ggml-common.h"

#define GGML_TYPE_IQ2_S_ID 22
#define GGML_TYPE_IQ4_KS_ID 144
#define GGML_TYPE_IQ2_KS_ID 145

typedef struct {
  uint8_t scales[QK_K / 32];
  uint8_t qs[QK_K / 2];
} block_iq4_ks;
static_assert(sizeof(block_iq4_ks) == QK_K / 32 + QK_K / 2, "wrong iq4_ks block size/padding");
#define IQ4_KS_ROW_META 4

typedef struct {
  uint16_t extra;
  uint8_t scales[QK_K / 64];
  uint8_t qs[QK_K / 4];
} block_iq2_ks;
static_assert(sizeof(block_iq2_ks) == sizeof(uint16_t) + QK_K / 64 + QK_K / 4, "wrong iq2_ks block size/padding");
#define IQ2_KS_ROW_META 2

// The lookups below read the tables as 32-bit words.
alignas(16) static const __device__ int8_t iqk_iq2nl_values[8] = {-31, -13, 1, 17, -26, -8, 6, 22};

alignas(16) static const __device__ int8_t iqk_iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18, -6,  5, 17, 29, 42, 57, 73, 93, 117};

// 8 3-bit indices per byte pair -> 8 int8 values (iq2nl table of IQ2_KS).
static __device__ __forceinline__ int2 iqk_get_int_from_table_8(const int& q4, const int8_t* values) {
  const uint32_t* values32 = (const uint32_t*)values;
  uint32_t v1 = __byte_perm(values32[0], values32[1], q4);
  uint32_t v2 = __byte_perm(values32[0], values32[1], q4 >> 16);
  return make_int2(__byte_perm(v1, v2, 0x6420), __byte_perm(v1, v2, 0x7531));
}

// 8 4-bit indices -> 8 int8 values (iq4k table of IQ4_KS).
static __device__ __forceinline__ int2 iqk_get_int_from_table_16(const int& q4, const int8_t* values) {
  uint32_t v1, v2, v3, v4, mask;
  const uint32_t* values32 = (const uint32_t*)values;

  mask = (0x32103210 | ((q4 & 0x88888888) >> 1));
  v1 = __byte_perm(values32[0], values32[1], q4);
  v2 = __byte_perm(values32[2], values32[3], q4);
  v3 = __byte_perm(v1, v2, mask);
  v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
  v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
  v4 = __byte_perm(v1, v2, mask >> 16);

  return make_int2(__byte_perm(v3, v4, 0x6420), __byte_perm(v3, v4, 0x7531));
}
