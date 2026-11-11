// SPDX-License-Identifier: MIT
//
// Host entry of the wide-block GGUF MoE int8 tensor-core matmul (IQ2_S): up to kWide routes of one
// expert per CTA, only the column groups in use multiplied. Device code:
// sgl_kernel/quantization/gguf_mma/mmq_wide.cuh (on top of the ik_llama.cpp port in mmq_id.cuh).
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/quantization/gguf_mma/mmq_wide.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

namespace gguf_mmq_wide_detail {

// sorted_rows == nullptr: route s is written to output row s and the output is [tokens * top_k, N].
// Otherwise sorted_rows[c] is the output row of sorted column c (a compact output of any row count).
template <typename scalar_t, int kQuantType, int kWide, bool kSkip>
void launch(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView sorted_token_ids,
    const tvm::ffi::TensorView* sorted_rows,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView quantized_input,
    int64_t top_k,
    int64_t row,
    int64_t tokens) {
  using namespace host;
  static_assert(kQuantType == GGML_TYPE_IQ2_S_ID, "the wide-block MMQ kernel serves IQ2_S (22) only");
  constexpr int64_t kMaxIndex = std::numeric_limits<int32_t>::max();
  CHECK_HOST(tokens >= 0 && tokens <= kMaxIndex && top_k > 0 && top_k <= kMaxIndex);
  CHECK_HOST(row > 0 && row <= kMaxIndex);
  CHECK_HOST(tokens <= kMaxIndex / top_k);

  auto device = SymbolicDevice{};
  auto columns = SymbolicSize{"columns"};
  auto experts = SymbolicSize{"experts"};
  auto row_bytes = SymbolicSize{"row_bytes"};
  auto padded = SymbolicSize{"sorted_capacity"};
  auto blocks = SymbolicSize{"blocks"};
  auto out_rows = SymbolicSize{"output_rows"};
  TensorMatcher({tokens, columns}).with_dtype<scalar_t>().template with_device<kDLGPU>(device).verify(input);
  const int64_t k = columns.unwrap();
  CHECK_HOST(k > 0 && k % QK_K == 0 && k <= kMaxIndex);
  TensorMatcher({experts, row, row_bytes}).with_dtype<uint8_t>().with_device(device).ensure_alignment(2).verify(weight);
  CHECK_HOST(row_bytes.unwrap() == k / QK_K * static_cast<int64_t>(sizeof(block_iq2_s)))
      << "GGUF MMQ row size differs from ggml_row_size for IQ2_S";
  CHECK_HOST(experts.unwrap() > 0 && experts.unwrap() <= kMaxIndex);
  CHECK_HOST(row * row_bytes.unwrap() <= kMaxIndex) << "one expert row block must be addressable with int32 offsets";
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(sorted_token_ids);
  TensorMatcher({blocks}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
  CHECK_HOST(blocks.unwrap() <= (padded.unwrap() + kWide - 1) / kWide);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
  CHECK_HOST(num_tokens_post_padded.numel() >= 1);
  if (sorted_rows == nullptr) {
    TensorMatcher({tokens * top_k, row}).with_dtype<scalar_t>().with_device(device).verify(output);
  } else {
    TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(*sorted_rows);
    TensorMatcher({out_rows, row}).with_dtype<scalar_t>().with_device(device).verify(output);
    CHECK_HOST(out_rows.unwrap() <= kMaxIndex / row);
  }
  TensorMatcher({k / 128, tokens, gguf_mma::kTileYK}).with_dtype<int32_t>().with_device(device).verify(quantized_input);
  CHECK_HOST(tokens * top_k <= kMaxIndex / row);
  if (tokens == 0 || blocks.unwrap() == 0) return;

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  gguf_mma::quantize_mmq_q8_1_d4_cuda<scalar_t>(
      static_cast<const scalar_t*>(input.data_ptr()),
      quantized_input.data_ptr(),
      static_cast<int>(k),
      static_cast<int>(tokens),
      stream);

  constexpr int kSharedBytes = gguf_mma::mmq_wide_shared_words<kWide>() * sizeof(int);
  const bool need_check = row % gguf_mma::kMmqY != 0;
  auto kernel = need_check ? gguf_mma::gguf_moe_mmq_wide_iq2_s_kernel<scalar_t, kWide, true, kSkip>
                           : gguf_mma::gguf_moe_mmq_wide_iq2_s_kernel<scalar_t, kWide, false, kSkip>;
  if constexpr (kSharedBytes > 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
  }
  const dim3 grid(
      static_cast<unsigned>(blocks.unwrap()), static_cast<unsigned>((row + gguf_mma::kMmqY - 1) / gguf_mma::kMmqY), 1);
  const dim3 block(gguf_mma::kWarpSize, gguf_mma::kNWarps, 1);
  kernel<<<grid, block, kSharedBytes, stream>>>(
      static_cast<const char*>(weight.data_ptr()),
      static_cast<const int*>(quantized_input.data_ptr()),
      static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
      sorted_rows == nullptr ? nullptr : static_cast<const int32_t*>(sorted_rows->data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),
      static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
      static_cast<scalar_t*>(output.data_ptr()),
      static_cast<int>(k),
      static_cast<int>(row),
      static_cast<int>(row_bytes.unwrap()),
      static_cast<int64_t>(row) * row_bytes.unwrap(),
      static_cast<int>(tokens),
      static_cast<int>(top_k),
      static_cast<int>(tokens * top_k),
      static_cast<int>(experts.unwrap()));
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace gguf_mmq_wide_detail

/// \brief Project the routes of `moe_align_block_size` through packed IQ2_S experts with int8 MMA.
/// \param input Activations [tokens, K]; route s reads row s / top_k (top_k = 1 for the down projection).
/// \param weight Packed experts [E, N, K / 256 * 82].
/// \param sorted_token_ids Route ids grouped per expert in blocks of kWide; ids >= tokens * top_k pad.
/// \param expert_ids Expert of each block; negative blocks are skipped. Its size bounds the grid.
/// \param num_tokens_post_padded int32[>=1]; blocks at or past it are skipped on the device.
/// \param output [tokens * top_k, N]; only the rows of valid routes are written (a caller that reads the
///        others zero-fills it first).
/// \param quantized_input Private scratch, int32 [K / 128, tokens, 36].
template <typename scalar_t, int kQuantType, int kWide, bool kSkip>
void gguf_moe_mmq_wide(
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
  gguf_mmq_wide_detail::launch<scalar_t, kQuantType, kWide, kSkip>(
      input,
      weight,
      sorted_token_ids,
      nullptr,
      expert_ids,
      num_tokens_post_padded,
      output,
      quantized_input,
      top_k,
      row,
      tokens);
}

/// \brief The same projection into a compact output: sorted column c is written to row sorted_rows[c].
/// \param sorted_rows int32, aligned with sorted_token_ids; read for the valid routes only.
/// \param output [rows, N] for any row count; the caller sizes it for the rows it mapped.
template <typename scalar_t, int kQuantType, int kWide, bool kSkip>
void gguf_moe_mmq_wide_rows(
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView sorted_rows,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView quantized_input,
    int64_t top_k,
    int64_t row,
    int64_t tokens) {
  gguf_mmq_wide_detail::launch<scalar_t, kQuantType, kWide, kSkip>(
      input,
      weight,
      sorted_token_ids,
      &sorted_rows,
      expert_ids,
      num_tokens_post_padded,
      output,
      quantized_input,
      top_k,
      row,
      tokens);
}

}  // namespace sglang
