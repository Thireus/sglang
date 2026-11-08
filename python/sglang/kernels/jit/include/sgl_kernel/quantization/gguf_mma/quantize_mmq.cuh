// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// The MMQ activation layout block_q8_1_mmq and its producer, adapted from ik_llama.cpp 7c1deaf3
// ggml/src/ggml-cuda/mmq_id_common.cuh (block_q8_1_mmq) and quantize.cu (quantize_mmq_q8_1, D4
// layout: IQ2_S, IQ4_KS and IQ2_KS all take one fp32 scale per 32 values and no partial sums).
//
// Changes from ik: the input is the bf16/fp16/fp32 activation itself, and the scale is rounded
// exactly as SGLang's vector path does (quantize_q8_1.cuh: d = amax / 127, q = round(x / d), the
// dot product uses half(d)), so the MMQ and vector kernels see the same int8 values and scales
// and switching between them at the token threshold does not move the numerics.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace sglang {
namespace gguf_mma {

// 128 values: 4 fp32 scales (one per 32 values) followed by 128 int8 quants, 144 bytes.
struct block_q8_1_mmq {
  float d4[4];
  int8_t qs[4 * 32];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "Unexpected block_q8_1_mmq size");

// block_q8_1_mmq in 32-bit words: the y tile row stride.
constexpr int kTileYK = 36;

constexpr int kQuantizeMmqBlock = 128;

// x [nrows, k] row-major -> y [k / 128][nrows] block_q8_1_mmq. k % 128 == 0.
template <typename scalar_t>
__global__ void
quantize_mmq_q8_1_d4(const scalar_t* __restrict__ x, block_q8_1_mmq* __restrict__ y, const int k, const int nrows) {
  const int ix0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
  if (ix0 >= k) {
    return;  // k % 128 == 0: whole warps leave together
  }
  const int64_t row = blockIdx.x;  // rows on x: tokens * top_k may exceed 65535
  const scalar_t* xr = x + row * k + ix0;
  const float x0 = static_cast<float>(xr[0]);
  const float x1 = static_cast<float>(xr[1]);
  const float x2 = static_cast<float>(xr[2]);
  const float x3 = static_cast<float>(xr[3]);

  float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
  }

  const float d = amax / 127;
  char4 q;
  q.x = amax == 0.0f ? 0 : roundf(x0 / d);
  q.y = amax == 0.0f ? 0 : roundf(x1 / d);
  q.z = amax == 0.0f ? 0 : roundf(x2 / d);
  q.w = amax == 0.0f ? 0 : roundf(x3 / d);

  const int64_t ib = (int64_t)(ix0 / 128) * nrows + row;
  const int iqs = ix0 % 128;
  reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
  if (iqs % 32 == 0) {
    y[ib].d4[iqs / 32] = __half2float(__float2half(d));
  }
}

template <typename scalar_t>
static void quantize_mmq_q8_1_d4_cuda(const scalar_t* x, void* y, const int k, const int nrows, cudaStream_t stream) {
  const dim3 grid(nrows, (k / 4 + kQuantizeMmqBlock - 1) / kQuantizeMmqBlock, 1);
  quantize_mmq_q8_1_d4<scalar_t><<<grid, kQuantizeMmqBlock, 0, stream>>>(x, (block_q8_1_mmq*)y, k, nrows);
}

}  // namespace gguf_mma
}  // namespace sglang
