// SPDX-License-Identifier: MIT
//
// Host entry of the GGUF MoE int8 tensor-core matmul (IQ2_S / IQ4_KS / IQ2_KS). The device code is
// ported from ik_llama.cpp (Copyright (C) 2024-2025 Iwan Kawrakow and the ik_llama.cpp authors,
// MIT), see sgl_kernel/quantization/gguf_mma/mmq_id.cuh.
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/quantization/gguf_mma/mmq_id.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

/// \brief Project the routes of `moe_align_block_size` through packed GGUF experts with int8 MMA.
/// \param input Activations [tokens, K]; route s reads row s / top_k (top_k = 1 for the down projection).
/// \param weight Packed experts [E, N, row_bytes], row_bytes = row header + K / 256 * block bytes.
/// \param sorted_token_ids Route ids grouped per expert in blocks of kBlock; ids >= tokens * top_k pad.
/// \param expert_ids Expert of each block; negative blocks are skipped. Its size sets the grid.
/// \param num_tokens_post_padded int32[>=1]; blocks at or past it are skipped on the device.
/// \param output Zero-initialised [tokens * top_k, N]; only valid routes are written.
/// \param quantized_input Private scratch, int32 [K / 128, tokens, 36].
template <typename scalar_t, int kQuantType, int kBlock>
void gguf_moe_mmq(
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
  using Traits = gguf_mma::MmqTraits<kQuantType>;
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
  TensorMatcher({tokens, columns}).with_dtype<scalar_t>().template with_device<kDLGPU>(device).verify(input);
  const int64_t k = columns.unwrap();
  CHECK_HOST(k > 0 && k % QK_K == 0 && k <= kMaxIndex);
  TensorMatcher({experts, row, row_bytes})
      .with_dtype<uint8_t>()
      .with_device(device)
      .ensure_alignment(Traits::kRowMeta == 4 ? 4 : 2)
      .verify(weight);
  CHECK_HOST(row_bytes.unwrap() == Traits::kRowMeta + k / QK_K * Traits::kBlockBytes)
      << "GGUF MMQ row size differs from ggml_row_size for type " << kQuantType;
  CHECK_HOST(row_bytes.unwrap() % (Traits::kRowMeta == 4 ? 4 : 2) == 0);
  CHECK_HOST(experts.unwrap() > 0 && experts.unwrap() <= kMaxIndex);
  CHECK_HOST(row * row_bytes.unwrap() <= kMaxIndex) << "one expert row block must be addressable with int32 offsets";
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(sorted_token_ids);
  TensorMatcher({blocks}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
  CHECK_HOST(blocks.unwrap() <= (padded.unwrap() + kBlock - 1) / kBlock);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
  CHECK_HOST(num_tokens_post_padded.numel() >= 1);
  TensorMatcher({tokens * top_k, row}).with_dtype<scalar_t>().with_device(device).verify(output);
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

  constexpr int kSharedBytes = gguf_mma::mmq_shared_words<kQuantType, kBlock>() * sizeof(int);
  const bool need_check = row % gguf_mma::kMmqY != 0;
  auto kernel = need_check ? gguf_mma::gguf_moe_mmq_kernel<scalar_t, kQuantType, kBlock, true>
                           : gguf_mma::gguf_moe_mmq_kernel<scalar_t, kQuantType, kBlock, false>;
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

}  // namespace sglang
