// SPDX-License-Identifier: Apache-2.0
//
// The route split of a hot/cold MoE layer in one launch. Each route's expert id is looked up in
// the layer's two remap tables, both indexed by id + 1 so that the padded route (-1) has entry 0:
//   hot_ids[r]  = remap_hot[ids[r] + 1]    the hot slot, or the hot count for a cold or padded route
//   cold_ids[r] = remap_cold[ids[r] + 1]   the cold slot, or -1 for a hot or padded route
// The values torch gives with one add and two index_select calls (three launches). An id outside
// the tables takes entry 0, the padded route, instead of a read past them.
//
// gguf_moe_split_align does the same for a DECODE batch and, in the same launch, groups the cold
// routes per cold expert in blocks for the multi-route vector kernel (what moe_align_block_size
// gives the cold tier in two more launches): sorted_ids holds each cold expert's routes padded to
// whole blocks with `routes` (an id past the routes), the groups in expert order; expert_ids the
// expert of each block and -1 past the blocks in use; num_post_padded[0] the columns in use. The
// routes that are not cold get no block (moe_align_block_size gives them blocks of expert -1,
// which the kernels skip). The order of the routes inside a group is not fixed (as there): each
// route's projection is computed by itself, so no output bit depends on it.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace sglang {

constexpr int kSplitThreads = 256;
constexpr int64_t kSplitMaxGrid = 1024;

// grid (CTAs of kSplitThreads threads), a route per thread; a CTA strides over the routes when the
// grid is capped.
__global__ void gguf_moe_split_kernel(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ remap_hot,
    const int32_t* __restrict__ remap_cold,
    int32_t* __restrict__ hot_ids,
    int32_t* __restrict__ cold_ids,
    const int numel,
    const int entries) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
    const uint32_t shifted = static_cast<uint32_t>(ids[i]) + 1u;
    const uint32_t entry = shifted < static_cast<uint32_t>(entries) ? shifted : 0u;
    hot_ids[i] = remap_hot[entry];
    cold_ids[i] = remap_cold[entry];
  }
}

constexpr int kSplitAlignThreads = 1024;
constexpr int kSplitAlignMaxExperts = kSplitAlignThreads;  // a thread per cold expert in the scan

// One CTA of 1024 threads: the split with a count per cold expert, a block-wide prefix sum of the
// padded counts (a thread per expert: inside the warp, then over the 32 warps), and the routes
// placed in their expert's group through a cursor per expert.
__global__ void __launch_bounds__(kSplitAlignThreads, 1) gguf_moe_split_align_kernel(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ remap_hot,
    const int32_t* __restrict__ remap_cold,
    int32_t* __restrict__ hot_ids,
    int32_t* cold_ids,
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_post_padded,
    const int numel,
    const int entries,
    const int num_cold,
    const int block,
    const int sorted_capacity,
    const int block_capacity) {
  __shared__ int32_t s_count[kSplitAlignMaxExperts];  // routes of each cold expert, then its group's cursor
  __shared__ int32_t s_start[kSplitAlignMaxExperts];  // first column of each cold expert's group
  __shared__ int32_t s_warp_total[kSplitAlignThreads / 32];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  s_count[tid] = 0;
  for (int c = tid; c < sorted_capacity; c += kSplitAlignThreads) {
    sorted_ids[c] = numel;  // padding: an id past the routes
  }
  for (int b = tid; b < block_capacity; b += kSplitAlignThreads) {
    expert_ids[b] = -1;
  }
  __syncthreads();

  for (int i = tid; i < numel; i += kSplitAlignThreads) {
    const uint32_t shifted = static_cast<uint32_t>(ids[i]) + 1u;
    const uint32_t entry = shifted < static_cast<uint32_t>(entries) ? shifted : 0u;
    const int32_t cold = remap_cold[entry];
    hot_ids[i] = remap_hot[entry];
    cold_ids[i] = cold;
    if (cold >= 0 && cold < num_cold) {
      atomicAdd(&s_count[cold], 1);
    }
  }
  __syncthreads();

  // thread e: cold expert e's routes padded to whole blocks (0 for a thread past the cold experts)
  const int mine = (s_count[tid] + block - 1) / block * block;
  s_count[tid] = 0;  // from here the cursor of the expert's group
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
  const int start = before + inclusive - mine;
  s_start[tid] = start;
  if (tid == kSplitAlignThreads - 1) {
    num_post_padded[0] = before + inclusive;
  }
  for (int b = start / block; b < (start + mine) / block && b < block_capacity; ++b) {
    expert_ids[b] = tid;
  }
  __syncthreads();

  for (int i = tid; i < numel; i += kSplitAlignThreads) {
    const int32_t cold = cold_ids[i];  // this thread wrote it above
    if (cold >= 0 && cold < num_cold) {
      const int column = s_start[cold] + atomicAdd(&s_count[cold], 1);
      if (column < sorted_capacity) {
        sorted_ids[column] = i;
      }
    }
  }
}

/// \brief hot_ids = remap_hot[ids + 1] and cold_ids = remap_cold[ids + 1], one launch.
/// \param ids int32 [routes], contiguous: the routed expert of each route, -1 for a padded route.
/// \param remap_hot int32 [entries]: entry 0 is the padded route's.
/// \param remap_cold int32 [entries].
/// \param hot_ids int32 [routes], written whole.
/// \param cold_ids int32 [routes], written whole.
inline void gguf_moe_split(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView remap_hot,
    tvm::ffi::TensorView remap_cold,
    tvm::ffi::TensorView hot_ids,
    tvm::ffi::TensorView cold_ids) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto routes = SymbolicSize{"routes"};
  auto entries = SymbolicSize{"entries"};
  TensorMatcher({routes}).with_dtype<int32_t>().template with_device<kDLGPU>(device).verify(ids);
  TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(hot_ids);
  TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(cold_ids);
  TensorMatcher({entries}).with_dtype<int32_t>().with_device(device).verify(remap_hot);
  TensorMatcher({entries}).with_dtype<int32_t>().with_device(device).verify(remap_cold);
  const int64_t numel = routes.unwrap();
  CHECK_HOST(numel <= std::numeric_limits<int32_t>::max());
  CHECK_HOST(entries.unwrap() >= 1 && entries.unwrap() <= std::numeric_limits<int32_t>::max());
  if (numel == 0) return;

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  const int64_t grid = std::min<int64_t>((numel + kSplitThreads - 1) / kSplitThreads, kSplitMaxGrid);
  gguf_moe_split_kernel<<<static_cast<unsigned>(grid), kSplitThreads, 0, stream>>>(
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(remap_hot.data_ptr()),
      static_cast<const int32_t*>(remap_cold.data_ptr()),
      static_cast<int32_t*>(hot_ids.data_ptr()),
      static_cast<int32_t*>(cold_ids.data_ptr()),
      static_cast<int>(numel),
      static_cast<int>(entries.unwrap()));
  CHECK_CUDA(cudaGetLastError());
}

/// \brief The split, and the cold routes grouped per cold expert in blocks, one launch.
/// \param ids int32 [routes], contiguous: the routed expert of each route, -1 for a padded route.
/// \param remap_hot, remap_cold int32 [entries]: entry 0 is the padded route's; cold slots are below num_cold.
/// \param hot_ids, cold_ids int32 [routes], written whole.
/// \param sorted_ids int32 [capacity], written whole: per cold expert its routes then padding (`routes`).
/// \param expert_ids int32 [block capacity], written whole: the cold expert of each block, -1 past the blocks in use.
/// \param num_post_padded int32 [1]: the columns of sorted_ids in use.
/// \param num_cold The layer's cold experts (1..1024).
/// \param block Routes of one expert per block (the multi-route kernel's).
inline void gguf_moe_split_align(
    tvm::ffi::TensorView ids,
    tvm::ffi::TensorView remap_hot,
    tvm::ffi::TensorView remap_cold,
    tvm::ffi::TensorView hot_ids,
    tvm::ffi::TensorView cold_ids,
    tvm::ffi::TensorView sorted_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_post_padded,
    int64_t num_cold,
    int64_t block) {
  using namespace host;
  auto device = SymbolicDevice{};
  auto routes = SymbolicSize{"routes"};
  auto entries = SymbolicSize{"entries"};
  auto capacity = SymbolicSize{"sorted_capacity"};
  auto blocks = SymbolicSize{"block_capacity"};
  TensorMatcher({routes}).with_dtype<int32_t>().template with_device<kDLGPU>(device).verify(ids);
  TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(hot_ids);
  TensorMatcher({routes}).with_dtype<int32_t>().with_device(device).verify(cold_ids);
  TensorMatcher({entries}).with_dtype<int32_t>().with_device(device).verify(remap_hot);
  TensorMatcher({entries}).with_dtype<int32_t>().with_device(device).verify(remap_cold);
  TensorMatcher({capacity}).with_dtype<int32_t>().with_device(device).verify(sorted_ids);
  TensorMatcher({blocks}).with_dtype<int32_t>().with_device(device).verify(expert_ids);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(num_post_padded);
  const int64_t numel = routes.unwrap();
  CHECK_HOST(entries.unwrap() >= 1 && entries.unwrap() <= std::numeric_limits<int32_t>::max());
  CHECK_HOST(num_cold >= 1 && num_cold <= kSplitAlignMaxExperts) << "one scan thread per cold expert";
  CHECK_HOST(block >= 1 && block <= 1024);
  CHECK_HOST(numel <= std::numeric_limits<int32_t>::max() / block);
  // every group padded to whole blocks: at most a block per route, and at most block - 1 columns of padding per expert
  const int64_t needed = std::min(numel * block, numel + num_cold * (block - 1));
  CHECK_HOST(capacity.unwrap() >= needed && capacity.unwrap() <= std::numeric_limits<int32_t>::max());
  CHECK_HOST(blocks.unwrap() * block >= needed && blocks.unwrap() <= std::numeric_limits<int32_t>::max());

  const auto stream = LaunchKernel::resolve_device(device.unwrap());
  gguf_moe_split_align_kernel<<<1, kSplitAlignThreads, 0, stream>>>(
      static_cast<const int32_t*>(ids.data_ptr()),
      static_cast<const int32_t*>(remap_hot.data_ptr()),
      static_cast<const int32_t*>(remap_cold.data_ptr()),
      static_cast<int32_t*>(hot_ids.data_ptr()),
      static_cast<int32_t*>(cold_ids.data_ptr()),
      static_cast<int32_t*>(sorted_ids.data_ptr()),
      static_cast<int32_t*>(expert_ids.data_ptr()),
      static_cast<int32_t*>(num_post_padded.data_ptr()),
      static_cast<int>(numel),
      static_cast<int>(entries.unwrap()),
      static_cast<int>(num_cold),
      static_cast<int>(block),
      static_cast<int>(capacity.unwrap()),
      static_cast<int>(blocks.unwrap()));
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace sglang
