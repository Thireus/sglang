// SPDX-License-Identifier: MIT
//
// Host entry of the GGUF MoE vector projection that serves several routes of one expert per CTA
// (IQ2_S). Device code: sgl_kernel/quantization/gguf/moe_vec_multi.cuh.
#pragma once

#include <sgl_kernel/quantization/gguf/ggml-common.h>
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/quantization/gguf/moe_vec_multi.cuh>
#include <sgl_kernel/quantization/gguf/quantize_q8_1.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

/// \brief Quantize input rows to Q8_1 and project the routes of `moe_align_block_size`, one expert block per CTA.
/// \param input Activations [tokens, K]; route s reads row s / top_k (top_k = 1 for the down projection).
/// \param weight Packed IQ2_S experts [E, row, K / 256 * 82].
/// \param sorted_token_ids Route ids grouped per expert in blocks of kRoutes; ids >= tokens * top_k pad.
/// \param expert_ids Expert of each block; negative blocks are skipped. Its size sets the grid.
/// \param num_tokens_post_padded int32[>=1]; blocks at or past it are skipped on the device.
/// \param output Zero-initialised [tokens * top_k, row]; only valid routes are written.
/// \param quantized_input Private Q8_1 scratch, int32 [tokens, padded K / 32 * 9] (the vector op's format).
template <typename scalar_t, int kQuantType, int kRoutes, int kRows>
void gguf_moe_vec_multi(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView quantized_input,
    int64_t top_k,
    int64_t row,
    int64_t tokens) {
  using namespace host;
  static_assert(kQuantType == 22, "the multi-route vector kernel serves IQ2_S (22) only");
  static_assert(kRoutes >= 1 && kRoutes <= 16 && kRows >= 1 && kRows <= 32);
  constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();
  CHECK_HOST(tokens >= 0 && tokens <= kMaxIndex && top_k > 0 && top_k <= kMaxIndex);
  CHECK_HOST(row > 0 && row <= kMaxIndex);
  CHECK_HOST(tokens <= kMaxIndex / top_k);

  auto device = SymbolicDevice{};
  auto columns = SymbolicSize{"columns"};
  auto experts = SymbolicSize{"experts"};
  auto padded = SymbolicSize{"sorted_capacity"};
  auto blocks = SymbolicSize{"blocks"};
  TensorMatcher({tokens, columns}).with_dtype<scalar_t>().template with_device<kDLGPU>(device).verify(input);
  const int64_t k = columns.unwrap();
  CHECK_HOST(k > 0 && k % QK_K == 0 && k <= kMaxIndex - 511);
  const int64_t packed_columns = k / QK_K * static_cast<int64_t>(sizeof(block_iq2_s));
  TensorMatcher({experts, row, packed_columns})
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(2)
      .verify(weight);
  CHECK_HOST(experts.unwrap() > 0 && experts.unwrap() <= kMaxIndex / row / (k / QK_K));
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(sorted_token_ids);
  TensorMatcher({blocks}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
  CHECK_HOST(blocks.unwrap() <= (padded.unwrap() + kRoutes - 1) / kRoutes);
  CHECK_HOST(blocks.unwrap() <= 65535) << "GGUF multi-route vector blocks exceed the CUDA y-grid limit";
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
  CHECK_HOST(num_tokens_post_padded.numel() >= 1);
  TensorMatcher({tokens * top_k, row}).with_dtype<scalar_t>().with_device(device).verify(output);
  const int64_t padded_k = (k + 511) / 512 * 512;
  TensorMatcher({tokens, padded_k / 32 * 9}).with_dtype<int32_t>().with_device(device).verify(quantized_input);
  CHECK_HOST(tokens <= kMaxIndex / padded_k && tokens * top_k <= kMaxIndex / row);
  if (tokens == 0 || blocks.unwrap() == 0) return;

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  quantize_row_q8_1_cuda<scalar_t>(
      static_cast<const scalar_t*>(input.data_ptr()),
      quantized_input.data_ptr(),
      static_cast<int>(k),
      static_cast<int>(tokens),
      stream);
  const dim3 grid(static_cast<unsigned>((row + kRows - 1) / kRows), static_cast<unsigned>(blocks.unwrap()), 1);
  const dim3 block(WARP_SIZE, kRows, 1);
  gguf::moe_vec_multi_iq2_s<scalar_t, kRoutes><<<grid, block, 0, stream>>>(
      weight.data_ptr(),
      quantized_input.data_ptr(),
      static_cast<scalar_t*>(output.data_ptr()),
      static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),
      static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
      static_cast<int>(k),
      static_cast<int>(row),
      static_cast<int>(padded_k / 32 * 9),
      static_cast<int>(top_k),
      static_cast<int>(tokens * top_k),
      static_cast<int>(experts.unwrap()));
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace sglang
