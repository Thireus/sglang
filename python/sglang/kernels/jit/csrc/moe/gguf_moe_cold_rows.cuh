// SPDX-License-Identifier: Apache-2.0
//
// The row map of a COMPACT cold-tier call, in one launch and without a host read: the cold tier
// keeps one row per cold route (a route whose cold id is >= 0) in buffers sized for every route,
// and the number of cold routes stays on the device.
//
//   rows[route]      the route's row in route order (0, 1, 2, ... over the cold routes), -1 for a
//                    route that is not cold;
//   sorted_rows[c]   the row of sorted column c of moe_align_block_size's output, for the columns
//                    below num_tokens_post_padded: rows[sorted_ids[c]], or `numel` for a padding
//                    column (an id outside [0, numel)), which every consumer skips as an id past
//                    its rows;
//   count[0]         the number of cold routes.
//
// One CTA: each thread counts the cold routes of its strip of the ids, a block-wide prefix sum
// gives every strip its first row, and the strips are walked a second time. The sorted columns
// are mapped by the same CTA after a barrier.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {

constexpr int kColdRowsThreads = 1024;

__global__ void __launch_bounds__(kColdRowsThreads, 1) gguf_moe_cold_rows_kernel(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t* rows,
    int32_t* __restrict__ sorted_rows,
    int32_t* __restrict__ count,
    const int numel,
    const int padded) {
  __shared__ int32_t s_warp_total[kColdRowsThreads / 32];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int strip = (numel + kColdRowsThreads - 1) / kColdRowsThreads;
  const int begin = min(tid * strip, numel);
  const int end = min(begin + strip, numel);

  int mine = 0;
  for (int i = begin; i < end; ++i) {
    mine += ids[i] >= 0 ? 1 : 0;
  }

  // inclusive prefix sum inside the warp, then over the warps
  int inclusive = mine;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int below = __shfl_up_sync(0xFFFFFFFF, inclusive, offset);
    if (lane >= offset) {
      inclusive += below;
    }
  }
  if (lane == 31) {
    s_warp_total[warp] = inclusive;
  }
  __syncthreads();
  int before = lane < warp ? s_warp_total[lane] : 0;  // 32 warps: one lane per earlier warp
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    before += __shfl_xor_sync(0xFFFFFFFF, before, offset);
  }

  int next = before + inclusive - mine;
  for (int i = begin; i < end; ++i) {
    rows[i] = ids[i] >= 0 ? next++ : -1;
  }
  if (tid == kColdRowsThreads - 1) {
    count[0] = before + inclusive;
  }
  __syncthreads();  // every row of the map is written

  const int used = min(max(num_tokens_post_padded[0], 0), padded);
  for (int c = tid; c < used; c += kColdRowsThreads) {
    const int sid = sorted_ids[c];
    sorted_rows[c] = (sid >= 0 && sid < numel) ? rows[sid] : numel;
  }
}

/// \brief The compact row of every cold route, of every sorted column, and the cold route count.
/// \param ids int32 [tokens, top_k], contiguous: the cold id of each route, negative = not cold.
/// \param sorted_ids int32 [padded]: moe_align_block_size's sorted route ids.
/// \param num_tokens_post_padded int32 [>= 1]: the sorted columns in use.
/// \param rows int32 [tokens, top_k], written whole.
/// \param sorted_rows int32 [padded]; the columns below num_tokens_post_padded are written.
/// \param count int32 [1].
inline void gguf_moe_cold_rows(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView sorted_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView rows,
    tvm::ffi::TensorView sorted_rows,
    tvm::ffi::TensorView count) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto tokens = SymbolicSize{"tokens"};
  auto top_k = SymbolicSize{"top_k"};
  auto padded = SymbolicSize{"sorted_capacity"};
  TensorMatcher({tokens, top_k}).with_dtype<int32_t>().template with_device<kDLGPU>(device).verify(ids);
  TensorMatcher({tokens, top_k}).with_dtype<int32_t>().with_device(device).verify(rows);
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(sorted_rows);
  TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).verify(num_tokens_post_padded);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(count);
  CHECK_HOST(num_tokens_post_padded.numel() >= 1);
  const int64_t numel = tokens.unwrap() * top_k.unwrap();
  CHECK_HOST(numel <= std::numeric_limits<int32_t>::max() && padded.unwrap() <= std::numeric_limits<int32_t>::max());

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  gguf_moe_cold_rows_kernel<<<1, kColdRowsThreads, 0, stream>>>(
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(sorted_ids.data_ptr()),
      static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
      static_cast<int32_t*>(rows.data_ptr()),
      static_cast<int32_t*>(sorted_rows.data_ptr()),
      static_cast<int32_t*>(count.data_ptr()),
      static_cast<int>(numel),
      static_cast<int>(padded.unwrap()));
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace sglang
