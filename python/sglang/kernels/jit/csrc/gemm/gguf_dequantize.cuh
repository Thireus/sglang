// SPDX-License-Identifier: Apache-2.0
//
// Packed rows of a ggml / ik_llama.cpp quant type -> dense weights, for the types that have no
// matmul kernel of their own: the caller multiplies with the dense result.
//
// One thread decodes one unit of 8 consecutive values (gguf::dequant::Format<type>::unit) as
// fp32 and rounds each to the output dtype (to nearest even for bf16 / fp16; fp32 is stored as
// it is), so the fp32 output is bit-identical to ik_llama.cpp's CPU dequantisation and a 16-bit
// output is that value rounded once. Thread g of the launch writes values [8 g, 8 g + 8) of the
// row-major output: the grid is one flat run of units, cut into CTAs of kThreads, and the last
// CTA stops at the end of the last row. Every output value is written exactly once.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/quantization/gguf/dequant/bn.cuh>
#include <sgl_kernel/quantization/gguf/dequant/common.cuh>
#include <sgl_kernel/quantization/gguf/dequant/iq1_r4.cuh>
#include <sgl_kernel/quantization/gguf/dequant/iqk.cuh>
#include <sgl_kernel/quantization/gguf/dequant/kt.cuh>
#include <sgl_kernel/quantization/gguf/dequant/scalar.cuh>

#include <cstdint>
#include <limits>

namespace sglang {

inline constexpr int kGgufDequantizeThreads = 256;

template <typename scalar_t, int kQuantType>
__global__ void __launch_bounds__(kGgufDequantizeThreads) gguf_dequantize_kernel(
    const uint8_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int64_t units,
    const int64_t units_per_row,
    const int64_t row_bytes) {
  using namespace device;
  using Format = gguf::dequant::Format<kQuantType>;
  constexpr int kUnit = gguf::dequant::kUnit;
  constexpr int kUnitsPerBlock = Format::kBlock / kUnit;

  const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= units) return;
  const int64_t r = g / units_per_row;
  const int c = static_cast<int>(g - r * units_per_row);
  const uint8_t* row = weight + r * row_bytes;
  const uint8_t* blk = row + Format::kRowMeta + static_cast<int64_t>(c / kUnitsPerBlock) * Format::kBlockBytes;

  float v[kUnit];
  Format::unit(row, blk, c % kUnitsPerBlock, v);

  scalar_t* out = output + g * kUnit;
#pragma unroll
  for (int i = 0; i < kUnit; ++i) {
    out[i] = cast<scalar_t>(v[i]);
  }
}

/// \brief Dequantise packed rows of ggml type kQuantType into a dense matrix.
/// \param weight uint8 [rows, row_bytes], contiguous; row_bytes is ggml_row_size of the output width
///        (the row header of the type included). For the types ik stores as interleaved row groups
///        (IQ1_S_R4, IQ1_M_R4, Q8_K_R8) the rows are the regrouped, one-after-the-other layout.
/// \param output [rows, columns] in scalar_t (fp32, fp16 or bf16), contiguous; every value is written.
template <typename scalar_t, int kQuantType>
void gguf_dequantize(tvm::ffi::TensorView weight, tvm::ffi::TensorView output) {
  using namespace host;
  using Format = gguf::dequant::Format<kQuantType>;
  constexpr int kUnit = gguf::dequant::kUnit;
  static_assert(Format::kBlock % kUnit == 0, "a block is a whole number of units");

  auto device = SymbolicDevice{};
  auto rows = SymbolicSize{"rows"};
  auto row_bytes = SymbolicSize{"row_bytes"};
  auto columns = SymbolicSize{"columns"};
  TensorMatcher({rows, row_bytes}).with_dtype<uint8_t>().template with_device<kDLGPU>(device).verify(weight);
  TensorMatcher({rows, columns}).with_dtype<scalar_t>().with_device(device).verify(output);

  const int64_t n = rows.unwrap();
  const int64_t k = columns.unwrap();
  CHECK_HOST(k > 0 && k % Format::kBlock == 0) << "GGUF type " << kQuantType << ": a row of " << k
                                               << " values is not a whole number of blocks of " << Format::kBlock;
  const int64_t expected = Format::kRowMeta + k / Format::kBlock * Format::kBlockBytes;
  CHECK_HOST(row_bytes.unwrap() == expected) << "GGUF type " << kQuantType << ": " << row_bytes.unwrap()
                                             << " bytes a row, expected " << expected << " for " << k << " values";
  if (n == 0) return;

  const int64_t units_per_row = k / kUnit;
  CHECK_HOST(n <= std::numeric_limits<int64_t>::max() / units_per_row);
  const int64_t units = n * units_per_row;
  const int64_t blocks = (units + kGgufDequantizeThreads - 1) / kGgufDequantizeThreads;
  CHECK_HOST(blocks <= std::numeric_limits<int32_t>::max())
      << "GGUF dequantise: the matrix is too large for one launch";

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  LaunchKernel(dim3(static_cast<unsigned>(blocks), 1, 1), dim3(kGgufDequantizeThreads, 1, 1), stream)(
      gguf_dequantize_kernel<scalar_t, kQuantType>,
      static_cast<const uint8_t*>(weight.data_ptr()),
      static_cast<scalar_t*>(output.data_ptr()),
      units,
      units_per_row,
      row_bytes.unwrap());
}

}  // namespace sglang
