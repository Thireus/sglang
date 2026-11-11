// SPDX-License-Identifier: Apache-2.0
//
// Route weights and the sum over top-k for a COMPACT expert output: y holds one row per active
// route only ([active, hidden]); rows[token, slot] is the row of that route or -1. Per token and
// column the result is what apply_gguf_moe_weights followed by moe_sum (top_k outside 2..4, i.e.
// torch.sum over the top-k axis) gives on the dense [tokens * top_k, hidden] layout:
//   term(slot) = active ? round_to_dtype(float(y) * float(weight)) : +0
//   out        = round_to_dtype(((acc0 + acc1) + acc2) + acc3),  acc[slot % 4] += term(slot)
// with fp32 accumulators that start at +0: the four interleaved accumulators and the order of
// their combination are those of torch's CUDA reduction for a short reduced axis, so the bits
// match the dense path (an inactive slot adds +0 there, which changes nothing here either).
//
// Two more layouts for a DECODE batch, whose y keeps the dense layout (one row per route, the
// rows of the inactive routes undefined):
//   kDense  rows[token, slot] is only the route's activity (>= 0 active, e.g. the cold ids): the
//           route's row is token * top_k + slot;
//   kFold   out already holds the other tier's output and the sum is added to it in place, as
//           torch's add_ does after the dense path: out = round_to_dtype(float(out) + float(sum)),
//           with sum rounded to the dtype first (the tensor add_ reads). A token with no active
//           route still adds +0 (a -0 in out becomes +0, as add_ makes it).
// Three launches (route weights, moe_sum, add_) become one; the rounding points and the order
// of the additions are the same.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {

constexpr int kColdSumMaxTopK = 16;

// grid (tokens), one CTA per token; kVec two-byte values per thread (16: one 32-byte load a row where
// the architecture has them, else 8). The token's rows and weights are read once by the first threads
// and shared, and a thread then makes one load per ACTIVE route. kTopK is a template argument: the
// slot loop unrolls and the four accumulators stay in registers (indexed by a run-time slot they
// would live in local memory).
template <typename scalar_t, int kVec, int kTopK, bool kDense, bool kFold>
__global__ void gguf_moe_cold_sum_kernel(
    const scalar_t* __restrict__ y,
    const int32_t* __restrict__ rows,
    const float* __restrict__ weights,
    scalar_t* __restrict__ out,
    const int hidden) {
  constexpr int top_k = kTopK;
  using namespace device;
  using vec_t = AlignedVector<scalar_t, kVec>;
  __shared__ int32_t s_rows[kColdSumMaxTopK];
  __shared__ float s_weights[kColdSumMaxTopK];

  const int64_t token = blockIdx.x;
  if (threadIdx.x < static_cast<unsigned>(top_k)) {
    const int32_t r = rows[token * top_k + threadIdx.x];
    s_rows[threadIdx.x] = r;
    s_weights[threadIdx.x] = r >= 0 ? weights[token * top_k + threadIdx.x] : 0.0f;
  }
  __syncthreads();

  for (int v = threadIdx.x; v < hidden / kVec; v += blockDim.x) {
    float acc[4][kVec];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        acc[a][i] = 0.0f;
      }
    }

#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      const int32_t r = s_rows[slot];
      if (r < 0) {
        continue;  // the dense path adds +0 here: no change to an accumulator that starts at +0
      }
      const float w = s_weights[slot];
      const int64_t row = kDense ? token * top_k + slot : static_cast<int64_t>(r);
      vec_t x;
      x.load(y + row * hidden, v);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        // the weighted value is rounded to the output dtype before it is summed, as the dense path stores it
        const scalar_t term = cast<scalar_t>(cast<fp32_t>(x[i]) * w);
        acc[slot & 3][i] += cast<fp32_t>(term);
      }
    }

    vec_t o;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      o[i] = cast<scalar_t>(((acc[0][i] + acc[1][i]) + acc[2][i]) + acc[3][i]);
    }
    if constexpr (kFold) {
      vec_t other;
      other.load(out + token * hidden, v);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        // torch's add_ on the two rounded tensors: one fp32 addition, rounded to the dtype
        o[i] = cast<scalar_t>(cast<fp32_t>(other[i]) + cast<fp32_t>(o[i]));
      }
    }
    o.store(out + token * hidden, v);
  }
}

/// \brief out[token] = sum over the token's active routes of weight * y[row], in the dense path's rounding and order.
/// \param y Compact expert outputs [active, hidden] (rows past the active ones may exist and are not read); with
///        kDense one row per route, [tokens * top_k, hidden] (the rows of the inactive routes are not read).
/// \param rows int32 [tokens, top_k]: the row of each route in y, negative for an inactive route (with kDense only
///        the sign is used).
/// \param weights fp32 [tokens, top_k], contiguous.
/// \param out [tokens, hidden]; every row is written (zeros for a token with no active route). With kFold it holds
///        the other tier's output on entry and that output plus the sum on return.
template <typename scalar_t, int kTopK, bool kDense, bool kFold>
void gguf_moe_cold_sum(
    tvm::ffi::TensorView y, tvm::ffi::TensorView rows, tvm::ffi::TensorView weights, tvm::ffi::TensorView out) {
  using namespace host;
  static_assert(sizeof(scalar_t) == 2, "bf16 / fp16 outputs");
  static_assert(kTopK > 0 && kTopK <= kColdSumMaxTopK, "top-k of the compact sum");
  auto device = SymbolicDevice{};
  auto active = SymbolicSize{"active"};
  auto hidden = SymbolicSize{"hidden"};
  auto tokens = SymbolicSize{"tokens"};
  TensorMatcher({active, hidden}).with_dtype<scalar_t>().template with_device<kDLGPU>(device).verify(y);
  TensorMatcher({tokens, kTopK}).with_dtype<int32_t>().with_device(device).verify(rows);
  TensorMatcher({tokens, kTopK}).with_dtype<float>().with_device(device).verify(weights);
  TensorMatcher({tokens, hidden}).with_dtype<scalar_t>().with_device(device).verify(out);
  const int64_t h = hidden.unwrap();
  CHECK_HOST(h > 0 && h % 8 == 0 && h <= std::numeric_limits<int32_t>::max());
  CHECK_HOST(tokens.unwrap() <= std::numeric_limits<int32_t>::max());
  if constexpr (kDense) {
    CHECK_HOST(active.unwrap() == tokens.unwrap() * kTopK) << "a dense y holds one row per route";
  }
  if (tokens.unwrap() == 0) return;

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  constexpr int kWide = device::kMaxVecBytes >= 32 ? 16 : 8;
  // 32-byte vectors need rows on 32-byte boundaries: a width that is a multiple of 16 values and aligned tensors
  const bool wide = kWide == 16 && h % 16 == 0 && reinterpret_cast<std::uintptr_t>(y.data_ptr()) % 32 == 0 &&
                    reinterpret_cast<std::uintptr_t>(out.data_ptr()) % 32 == 0;
  const int64_t vecs = h / (wide ? 16 : 8);
  const unsigned threads = static_cast<unsigned>(std::max<int64_t>(std::min<int64_t>((vecs + 31) / 32 * 32, 512), 32));
  const dim3 grid(static_cast<unsigned>(tokens.unwrap()), 1, 1);
  const auto y_ptr = static_cast<const scalar_t*>(y.data_ptr());
  const auto rows_ptr = static_cast<const int32_t*>(rows.data_ptr());
  const auto weights_ptr = static_cast<const float*>(weights.data_ptr());
  const auto out_ptr = static_cast<scalar_t*>(out.data_ptr());
  if constexpr (kWide == 16) {
    if (wide) {
      gguf_moe_cold_sum_kernel<scalar_t, 16, kTopK, kDense, kFold>
          <<<grid, threads, 0, stream>>>(y_ptr, rows_ptr, weights_ptr, out_ptr, static_cast<int>(h));
      CHECK_CUDA(cudaGetLastError());
      return;
    }
  }
  gguf_moe_cold_sum_kernel<scalar_t, 8, kTopK, kDense, kFold>
      <<<grid, threads, 0, stream>>>(y_ptr, rows_ptr, weights_ptr, out_ptr, static_cast<int>(h));
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace sglang
