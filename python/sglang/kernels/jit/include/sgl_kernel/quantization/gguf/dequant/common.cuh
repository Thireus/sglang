// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// Dequantising ggml / ik_llama.cpp rows on the GPU: what every format shares.
//
// A format is a specialisation of Format<ggml type id> with
//   kBlock       values a block
//   kBlockBytes  bytes a block
//   kRowMeta     bytes before the first block of a row (ggml's row_meta_size: a row scale)
//   unit(row, blk, u, v)  decodes values [kUnit * u, kUnit * (u + 1)) of the block at `blk`
//                         of the row at `row` into v[0 .. kUnit), as fp32
// so a row of K values is kRowMeta + K / kBlock * kBlockBytes bytes (ggml_row_size).
//
// Every unit() follows ik_llama.cpp's CPU dequantize_row_<type> (ggml-quants.c and
// iqk/iqk_quantize.cpp at 7c1deaf3) operation for operation: integers are combined as integers,
// converted to fp32 once, and each fp32 product is one multiply, so the fp32 value of a weight
// is bit-identical to what ik_llama.cpp computes on the CPU. The one multiply-add chain there is
// (IQ6_K) is written with explicit fused multiply-adds, the way the CPU build contracts it.
//
// Rows are read byte by byte: a row header of 2 bytes or a block of 13 or 102 bytes leaves the
// 16- and 32-bit fields of a block on odd addresses.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sglang {
namespace gguf {
namespace dequant {

// Values one kernel thread decodes and writes.
inline constexpr int kUnit = 8;

template <int kQuantType>
struct Format;

__device__ __forceinline__ uint32_t load_u16(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

// A half scale as fp32 (exact, as GGML_FP16_TO_FP32).
__device__ __forceinline__ float load_f16(const uint8_t* p) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(load_u16(p))));
}

__device__ __forceinline__ float load_f32(const uint8_t* p) {
  return __uint_as_float(load_u32(p));
}

__device__ __forceinline__ int load_i8(const uint8_t* p) {
  return static_cast<int8_t>(*p);
}

}  // namespace dequant
}  // namespace gguf
}  // namespace sglang
