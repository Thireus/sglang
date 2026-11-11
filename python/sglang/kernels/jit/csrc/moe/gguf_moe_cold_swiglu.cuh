// SPDX-License-Identifier: Apache-2.0
//
// The clamped SwiGLU of a COMPACT cold-tier call: silu_mul_clamp_kernel's result (the same
// silu_and_mul expression, built with the same flags: bit-equal per row) over the first
// row_count[0] rows of buffers that are sized for every route. The row count is read ON THE
// DEVICE, so the host launches this kernel without waiting for the gate/up projection that
// decides it. The grid does not depend on the count either: CTA b takes rows b, b + grid,
// b + 2 * grid, ... below the count, and a CTA with no row leaves at once.
#pragma once

#include "../deepseek_v4/silu_and_mul_masked_post_quant.cuh"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {

struct ColdSwigluParams {
  const void* __restrict__ input;
  void* __restrict__ output;
  const int32_t* __restrict__ row_count;
  float swiglu_limit;
  uint32_t out_vecs;
  uint32_t rows;  // the buffers' rows: row_count[0] is clamped to it
};

// The largest grid: with at most this many rows every row has a CTA of its own.
constexpr uint32_t kColdSwigluMaxGrid = 4096;

template <typename DType>
__global__
__launch_bounds__(1024, 2) void gguf_moe_cold_swiglu_kernel(const ColdSwigluParams __grid_constant__ params) {
  using namespace device;
  static_assert(sizeof(DType) == 2, "only fp16/bf16 supported");
  using DType2 = packed_t<DType>;
  constexpr auto kVecSize = 16 / sizeof(DType);
  static_assert(kVecSize % 2 == 0 && kVecSize > 0);
  using Vec = AlignedVector<DType2, kVecSize / 2>;

  const int32_t counted = params.row_count[0];
  const uint32_t count =
      counted <= 0 ? 0u : (static_cast<uint32_t>(counted) < params.rows ? static_cast<uint32_t>(counted) : params.rows);
  const auto input = static_cast<const Vec*>(params.input);
  auto output = static_cast<Vec*>(params.output);
  const float limit = params.swiglu_limit;

  for (uint32_t row = blockIdx.x; row < count; row += gridDim.x) {
    const uint64_t input_row = static_cast<uint64_t>(row) * 2 * params.out_vecs;
    for (uint32_t vec_id = threadIdx.x; vec_id < params.out_vecs; vec_id += blockDim.x) {
      const auto gate = input[input_row + vec_id];
      const auto up = input[input_row + params.out_vecs + vec_id];
      Vec out;

#pragma unroll
      for (uint32_t i = 0; i < kVecSize / 2; ++i) {
        out[i] = cast<DType2>(silu_and_mul<true>(to_bf16x2(gate[i]), to_bf16x2(up[i]), limit));
      }

      output[static_cast<uint64_t>(row) * params.out_vecs + vec_id] = out;
    }
  }
}

/// \brief output[r] = clamped SwiGLU of input[r] (gate || up) for r < row_count[0]; the other rows are not touched.
/// \param input [rows, 2 * out_dim], bf16 / fp16.
/// \param output [rows, out_dim].
/// \param row_count int32 [1] on the device: the rows to compute (clamped to `rows`).
template <typename DType>
struct ColdSwigluKernel {
  static constexpr auto kernel = gguf_moe_cold_swiglu_kernel<DType>;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView row_count,
      const double swiglu_limit) {
    using namespace host;

    auto device = SymbolicDevice{};
    auto M = SymbolicSize{"rows"};
    auto D = SymbolicSize{"gate_up_dim"};  // 2 * out_dim
    auto H = SymbolicSize{"out_dim"};
    device.set_options<kDLGPU>();

    TensorMatcher({M, D}).with_dtype<DType>().with_device(device).verify(input);
    TensorMatcher({M, H}).with_dtype<DType>().with_device(device).verify(output);
    TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(row_count);
    RuntimeCheck(D.unwrap() == 2 * H.unwrap(), "input last dim must be 2 * output last dim");

    constexpr uint32_t kVecSize = 16 / sizeof(DType);
    const auto out_dim = static_cast<uint32_t>(H.unwrap());
    RuntimeCheck(out_dim > 0, "out_dim must be positive");
    RuntimeCheck(out_dim % kVecSize == 0, "out_dim must be divisible by vector size");
    RuntimeCheck(M.unwrap() <= std::numeric_limits<int32_t>::max(), "too many rows");
    const auto rows = static_cast<uint32_t>(M.unwrap());
    if (rows == 0) return;
    const auto out_vecs = out_dim / kVecSize;
    const auto num_threads = std::min(out_vecs, 1024u);

    const auto params = ColdSwigluParams{
        .input = input.data_ptr(),
        .output = output.data_ptr(),
        .row_count = static_cast<const int32_t*>(row_count.data_ptr()),
        .swiglu_limit = static_cast<float>(swiglu_limit),
        .out_vecs = out_vecs,
        .rows = rows,
    };
    LaunchKernel(std::min(rows, kColdSwigluMaxGrid), num_threads, device.unwrap())(kernel, params);
  }
};

}  // namespace sglang
