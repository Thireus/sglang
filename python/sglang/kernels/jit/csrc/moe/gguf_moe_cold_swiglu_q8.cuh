// SPDX-License-Identifier: Apache-2.0
//
// The clamped SwiGLU of a DECODE cold-tier call with the down projection's activation
// quantisation as a second output: one launch where silu_mul_clamp_kernel and quantize_q8_1 took
// two, and only for the rows of the cold routes (ids[row] >= 0; the two kernels walk every row).
//
//   output[row]    silu_mul_clamp_kernel's value (the same silu_and_mul expression, built with the
//                  same flags: bit-equal per row);
//   quantized[row] ggml's Q8_1 blocks of that row, as quantize_q8_1 writes them from the STORED
//                  values (rounded to the dtype): per 32 values d = amax / 127, q = round(v / d),
//                  and the block's sum in ggml's order (value i with i ^ 16, then ^ 8, ^ 4, ^ 2, ^ 1).
//
// This module is built with silu_and_mul_clamp's fast-math flags (the SwiGLU must round alike),
// quantize_q8_1 is built without them. The quantisation therefore asks for its two divisions and
// its additions by intrinsic (__fdiv_rn, __fadd_rn: IEEE whatever the flags say). What the flags
// still change is the treatment of subnormal numbers, and those never reach the result: a block
// whose d is subnormal stores d = 0 as a half in both builds, and the projection multiplies the
// block's integer sums by that d.
// The padding blocks of a quantised row (a width that is not a multiple of 512) are left alone:
// the projection reads the blocks of its columns only.
#pragma once

#include <sgl_kernel/quantization/gguf/ggml-common.h>

#include "../deepseek_v4/silu_and_mul_masked_post_quant.cuh"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {

struct ColdSwigluQ8Params {
  const void* __restrict__ input;
  void* __restrict__ output;
  int32_t* __restrict__ quantized;
  const int32_t* __restrict__ ids;
  float swiglu_limit;
  uint32_t out_vecs;  // output values / 8
  uint32_t rows;
  uint32_t q_stride;  // int32 words a row of `quantized`
};

// The largest grid: with at most this many rows every row has a CTA of its own.
constexpr uint32_t kColdSwigluQ8MaxGrid = 4096;

// grid (min(rows, kColdSwigluQ8MaxGrid)), block (a multiple of 32 threads): a thread holds 8
// consecutive values, four neighbouring threads one Q8_1 block. Every thread of a warp runs every
// pass, so the shuffles below always see their partners (a thread past the row's width holds zeros
// and stores nothing).
template <typename DType>
__global__
__launch_bounds__(1024, 2) void gguf_moe_cold_swiglu_q8_kernel(const ColdSwigluQ8Params __grid_constant__ params) {
  using namespace device;
  static_assert(sizeof(DType) == 2, "only fp16/bf16 supported");
  using DType2 = packed_t<DType>;
  constexpr uint32_t kVecSize = 16 / sizeof(DType);
  static_assert(kVecSize == 8 && QK8_1 == 4 * kVecSize, "four threads of 8 values a Q8_1 block");
  static_assert(sizeof(block_q8_1) == 9 * sizeof(int32_t), "a Q8_1 block is 9 words");
  using Vec = AlignedVector<DType2, kVecSize / 2>;

  const auto input = static_cast<const Vec*>(params.input);
  auto output = static_cast<Vec*>(params.output);
  const float limit = params.swiglu_limit;
  const uint32_t passes = (params.out_vecs + blockDim.x - 1) / blockDim.x;
  const uint32_t lane = threadIdx.x & 3u;  // the thread's place among the four of its block

  for (uint32_t row = blockIdx.x; row < params.rows; row += gridDim.x) {
    if (params.ids[row] < 0) {
      continue;  // not a cold route: nothing reads this row
    }
    const uint64_t input_row = static_cast<uint64_t>(row) * 2 * params.out_vecs;
    int32_t* q_row = params.quantized + static_cast<uint64_t>(row) * params.q_stride;
    for (uint32_t pass = 0; pass < passes; ++pass) {
      const uint32_t vec_id = pass * blockDim.x + threadIdx.x;
      const bool live = vec_id < params.out_vecs;
      float v[kVecSize];
      if (live) {
        const auto gate = input[input_row + vec_id];
        const auto up = input[input_row + params.out_vecs + vec_id];
        Vec out;
#pragma unroll
        for (uint32_t i = 0; i < kVecSize / 2; ++i) {
          out[i] = cast<DType2>(silu_and_mul<true>(to_bf16x2(gate[i]), to_bf16x2(up[i]), limit));
        }
        output[static_cast<uint64_t>(row) * params.out_vecs + vec_id] = out;
#pragma unroll
        for (uint32_t i = 0; i < kVecSize / 2; ++i) {
          // the quantisation takes the value as it is stored
          const auto [v0, v1] = cast<fp32x2_t>(out[i]);
          v[2 * i] = v0;
          v[2 * i + 1] = v1;
        }
      } else {
#pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++i) {
          v[i] = 0.0f;
        }
      }

      float amax = 0.0f;
      float s[kVecSize];
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
        s[i] = v[i];
      }
      // quantize_q8_1 reduces a block over 32 lanes with xor masks 16, 8, 4, 2, 1; value j of the
      // block is value j % 8 of thread j / 8 here, so masks 16 and 8 are the threads 2 and 1 away
      // and masks 4, 2, 1 pair values inside a thread.
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        s[i] = __fadd_rn(s[i], __shfl_xor_sync(0xFFFFFFFFu, s[i], 2));
      }
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        s[i] = __fadd_rn(s[i], __shfl_xor_sync(0xFFFFFFFFu, s[i], 1));
      }
      const float s4_0 = __fadd_rn(s[0], s[4]);
      const float s4_1 = __fadd_rn(s[1], s[5]);
      const float s4_2 = __fadd_rn(s[2], s[6]);
      const float s4_3 = __fadd_rn(s[3], s[7]);
      const float sum = __fadd_rn(__fadd_rn(s4_0, s4_2), __fadd_rn(s4_1, s4_3));

      const float d = __fdiv_rn(amax, 127.0f);
      uint32_t words[2] = {0u, 0u};
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        const int8_t q = amax == 0.0f ? 0 : roundf(__fdiv_rn(v[i], d));
        words[i / 4] |= static_cast<uint32_t>(static_cast<uint8_t>(q)) << (8 * (i % 4));
      }
      if (live) {
        int32_t* block = q_row + static_cast<uint64_t>(vec_id / 4) * 9;
        block[1 + 2 * lane] = static_cast<int32_t>(words[0]);
        block[2 + 2 * lane] = static_cast<int32_t>(words[1]);
        if (lane == 0) {
          const uint32_t ds = static_cast<uint32_t>(__half_as_ushort(__float2half(d))) |
                              (static_cast<uint32_t>(__half_as_ushort(__float2half(sum))) << 16);
          block[0] = static_cast<int32_t>(ds);
        }
      }
    }
  }
}

/// \brief The clamped SwiGLU of the rows whose id is >= 0, and its Q8_1 quantisation.
/// \param input [rows, 2 * out_dim], bf16 / fp16 (gate then up).
/// \param output [rows, out_dim]; the rows of a negative id are not touched.
/// \param quantized int32 [rows, padded out_dim / 32 * 9] (the vector kernels' Q8_1 scratch); the
///        blocks of the columns of a row with a non-negative id are written.
/// \param ids int32 [rows]: the cold id of each route (negative = not a cold route).
template <typename DType>
struct ColdSwigluQ8Kernel {
  static constexpr auto kernel = gguf_moe_cold_swiglu_q8_kernel<DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView quantized,
      const tvm::ffi::TensorView ids,
      const double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"rows"};
    auto D = SymbolicSize{"gate_up_dim"};  // 2 * out_dim
    auto H = SymbolicSize{"out_dim"};
    auto Q = SymbolicSize{"quantized_words"};
    device.set_options<kDLGPU>();

    TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({M, H}).with_dtype<DType>().with_device(device).verify(output);
    TensorMatcher({M, Q}).with_dtype<int32_t>().with_device(device).verify(quantized);
    TensorMatcher({M}).with_dtype<int32_t>().with_device(device).verify(ids);
    RuntimeCheck(D.unwrap() == 2 * H.unwrap(), "input last dim must be 2 * output last dim");

    const auto out_dim = static_cast<uint32_t>(H.unwrap());
    RuntimeCheck(out_dim > 0 && out_dim % QK8_1 == 0, "out_dim must be a multiple of 32");
    const int64_t padded = (static_cast<int64_t>(out_dim) + 511) / 512 * 512;
    RuntimeCheck(Q.unwrap() == padded / 32 * 9, "quantized must hold padded out_dim / 32 * 9 words a row");
    RuntimeCheck(M.unwrap() <= std::numeric_limits<int32_t>::max(), "too many rows");
    const auto rows = static_cast<uint32_t>(M.unwrap());
    if (rows == 0) return;
    const auto out_vecs = out_dim / 8u;
    const auto num_threads = std::min((out_vecs + 31u) / 32u * 32u, 1024u);

    const auto params = ColdSwigluQ8Params{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .quantized = static_cast<int32_t*>(quantized.data_ptr()),
        .ids = static_cast<const int32_t*>(ids.data_ptr()),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .out_vecs = out_vecs,
        .rows = rows,
        .q_stride = static_cast<uint32_t>(Q.unwrap()),
    };
    LaunchKernel(std::min(rows, kColdSwigluQ8MaxGrid), num_threads, device.unwrap())(kernel, params);
  }
};

}  // namespace sglang
