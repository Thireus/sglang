// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// The half-float variant of the staged GGUF MoE kernel for IQ2_S (mmq_staged.cuh): the same
// asynchronous staging of the packed rows one tile ahead, the same grid table and sign handling
// (from the ik_llama.cpp port in mmq_id.cuh), but the products run in the hot tier's arithmetic
// instead of int8:
//
//   - the weights of a tile are unpacked to fp16 with the scale of their 16-value group folded
//     in: sign * grid * (2 * ls + 1), an integer of at most 1,333, exact in fp16. The bytes become
//     half floats with ik's byte-permute trick (requant_int_q8 in ik's vecdotq.cuh: 0x64uu is
//     1024 + uu in fp16) and one fused multiply-add per pair of values;
//   - the activations are NOT quantised to int8: they are taken as fp16 (a bf16 value converts
//     exactly; a row whose largest magnitude would overflow fp16 is scaled by a power of two and
//     scaled back at the end, gguf_moe_f16_prepare below);
//   - m16n8k16 half-float products accumulate in fp32 inside the tensor cores over a 128-value
//     half of a QK_K block; the block's scale d / 8 is then applied once per (row, half) in fp32.
//
// Weights exact, activations exact, products exact (11 x 8 significant bits), sums in fp32: the
// result differs from the fp32 matrix product of the dequantised weights only by the order and
// the rounding of the fp32 additions. It is NOT the int8 kernels' result (those round every
// activation to 8 bits per 32-value group): this path is behind SGLANG_HOTCOLD_PREFILL_FLOAT.
#pragma once

#include <sgl_kernel/quantization/gguf_mma/mmq_staged.cuh>

#include <cstdint>
#include <cuda_fp16.h>

namespace sglang {
namespace gguf_mma {

constexpr int kF16Half = 128;                 // values of K per x tile: half a QK_K block
constexpr int kF16TileXK = kF16Half / 2 + 4;  // 68 words a row: 64 of values, 4 of padding (ldmatrix, as kTileXK_Q3_K)
constexpr int kF16TileYK = kF16Half / 2 + 4;  // the same for a column of activations
static_assert(kF16TileXK % 8 == 4, "Wrong padding.");

template <int kWide>
constexpr __host__ __device__ int f16_staged_shared_words() {
  // ids_dst + ids_src + the columns' row scales, the grid table, two buffers of staged packed rows, the rows'
  // block scales, the y tile of one half, the x tile of one half
  return 3 * kWide + kStGridWords + 2 * kStStageWords + kMmqY + kWide * kF16TileYK + kMmqY * kF16TileXK;
}

// ----------------------------------------------------------------------------------------------
// Activations: any float row -> fp16, with a power-of-two scale per row that keeps the largest
// magnitude below 2^15 (1 for every row that fits, which is every row seen in practice).
// grid (rows), block 256; k values per row.
template <typename scalar_t>
__global__ void gguf_moe_f16_prepare_kernel(
    const scalar_t* __restrict__ x, __half* __restrict__ xh, float* __restrict__ xscale, const int k) {
  constexpr int kThreads = 256;
  __shared__ float warp_max[kThreads / kWarpSize];
  const int64_t row = blockIdx.x;
  const scalar_t* xr = x + row * k;

  float amax = 0.0f;
  for (int i = threadIdx.x; i < k; i += kThreads) {
    amax = fmaxf(amax, fabsf(static_cast<float>(xr[i])));
  }
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, kWarpSize));
  }
  if (threadIdx.x % kWarpSize == 0) {
    warp_max[threadIdx.x / kWarpSize] = amax;
  }
  __syncthreads();
  amax = warp_max[0];
#pragma unroll
  for (int w = 1; w < kThreads / kWarpSize; ++w) {
    amax = fmaxf(amax, warp_max[w]);
  }

  // down = 2^-s with s the smallest shift that brings amax under 2^15; 1 when it already is (or is not finite)
  float down = 1.0f;
  if (amax >= 32768.0f && amax <= 3.0e38f) {
    down = scalbnf(1.0f, 14 - ilogbf(amax));
  }
  for (int i = threadIdx.x; i < k; i += kThreads) {
    xh[row * k + i] = __float2half(static_cast<float>(xr[i]) * down);
  }
  if (threadIdx.x == 0) {
    xscale[row] = 1.0f / down;
  }
}

template <typename scalar_t>
static void
gguf_moe_f16_prepare_cuda(const scalar_t* x, void* xh, void* xscale, const int k, const int rows, cudaStream_t stream) {
  gguf_moe_f16_prepare_kernel<scalar_t><<<rows, 256, 0, stream>>>(x, (__half*)xh, (float*)xscale, k);
}

// ----------------------------------------------------------------------------------------------

static __device__ __forceinline__ void load_ldmatrix_x2(int* __restrict__ regs, const int* __restrict__ xs) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(regs[0]), "=r"(regs[1]) : "l"(xs));
}

// D (16 x 8, fp32) += A (16 x 16, fp16, row major) * B (16 x 8, fp16, column major)
static __device__ __forceinline__ void
mma_f16(float* __restrict__ d, const int* __restrict__ a, const int* __restrict__ b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  int* di = (int*)d;  // the fp32 accumulators as register bit patterns, as ggml's mma.cuh passes them
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, "
      "%3};"
      : "+r"(di[0]), "+r"(di[1]), "+r"(di[2]), "+r"(di[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  __trap();
#endif
}

// 32 bytes as one global load (stored to shared memory as two 16-byte values).
struct alignas(32) F16Vec32 {
  StVec16 h[2];
};
constexpr int kF16ColVecs = kF16Half * 2 / 32;  // 128 fp16 values of a column: 8 vectors of 32 bytes

template <int kGroups>
constexpr __host__ __device__ int f16_staged_y_regs() {
  return (kGroups * kWideGroup * kF16ColVecs + kStThreads - 1) / kStThreads;
}

// Issue the loads of the fp16 activations of the block's first kGroups column groups for 128-value
// half `half128` of K into registers, 32 bytes a load (y: [rows, K] fp16, 32-byte aligned rows).
template <int kGroups>
static __device__ __forceinline__ void f16_staged_fetch_y(
    const int* __restrict__ y,
    const int* __restrict__ ids_src,
    F16Vec32* __restrict__ regs,
    const int half128,
    const int words_per_row) {
  constexpr int kVecs = kGroups * kWideGroup * kF16ColVecs;
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
#pragma unroll
  for (int m = 0; m < f16_staged_y_regs<kGroups>(); ++m) {
    const int l = tid + m * kStThreads;
    if (l < kVecs) {
      const int col = l / kF16ColVecs;
      const int v = l - col * kF16ColVecs;
      regs[m] = ((const F16Vec32*)(y + (int64_t)ids_src[col] * words_per_row + half128 * (kF16Half / 2)))[v];
    }
  }
}

template <int kGroups>
static __device__ __forceinline__ void f16_staged_store_y(int* __restrict__ tile_y, const F16Vec32* __restrict__ regs) {
  constexpr int kVecs = kGroups * kWideGroup * kF16ColVecs;
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
#pragma unroll
  for (int m = 0; m < f16_staged_y_regs<kGroups>(); ++m) {
    const int l = tid + m * kStThreads;
    if (l < kVecs) {
      const int col = l / kF16ColVecs;
      // a column of the y tile is 68 words: 16-byte aligned, not 32
      StVec16* dst = (StVec16*)(tile_y + col * kF16TileYK) + 2 * (l - col * kF16ColVecs);
      dst[0] = regs[m].h[0];
      dst[1] = regs[m].h[1];
    }
  }
}

// Half `half` (128 values of K) of the staged tile -> fp16 with the 16-value scales folded in.
// One (row, 32 values) per thread; the fields and the signed grid values are those of
// load_tiles_iq2_s. tile_d gets the row's block scale d / 8.
template <bool need_check>
static __device__ __forceinline__ void f16_staged_unpack_iq2_s(
    const char* __restrict__ xt,
    const int* __restrict__ stage,
    const int* __restrict__ grid,
    int* __restrict__ x_tile,
    float* __restrict__ tile_d,
    const int kb0,
    const int half,
    const int i_max,
    const int stride) {
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
  const int q = tid % 4;  // the 32-value group inside the half
  int i = tid / 4;        // kStThreads / 4 = kMmqY rows
  if (need_check) {
    i = min(i, i_max);
  }
  const int kqsx = 4 * half + q;

  const int offset = (int)(mmq_staged_block_addr(xt, i, kb0, stride) & (kStVecBytes - 1));
  const block_iq2_s* bxi = (const block_iq2_s*)((const char*)(stage + i * kStRowWords) + offset);

  const int qs_packed = mmq_get_int_b2(bxi->qs, kqsx);
  const uint8_t* qs = (const uint8_t*)&qs_packed;

  const int qh = bxi->qh[kqsx];

  const int signs_packed_32 = mmq_get_int_b2(bxi->qs, QK_K / 32 + kqsx);
  const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;

  const int ls = bxi->scales[kqsx];
  // (m, m) and (-1152 m, -1152 m): a byte u = v + 128 placed under 0x64 reads 1152 + v in fp16, and
  // (1152 + v) * m - 1152 * m = v * m exactly (one rounding of an integer below 2,048)
  const __half2 m_lo = __half2half2(__int2half_rn(2 * (ls & 0x0F) + 1));
  const __half2 m_hi = __half2half2(__int2half_rn(2 * (ls >> 4) + 1));
  const __half2 bias = __float2half2_rn(-1152.0f);
  const __half2 b_lo = __hmul2(m_lo, bias);
  const __half2 b_hi = __hmul2(m_hi, bias);

  StVec16* dst = (StVec16*)(x_tile + i * kF16TileXK + q * 16);
#pragma unroll
  for (int l = 0; l < kQR2_S; ++l) {
    const uint64_t entry = *(const uint64_t*)(grid + 2 * (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));

    const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
    const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

    const unsigned grid_l = (unsigned)__vsub4((int)(uint32_t)entry ^ signs0, signs0) ^ 0x80808080u;
    const unsigned grid_h = (unsigned)__vsub4((int)(uint32_t)(entry >> 32) ^ signs1, signs1) ^ 0x80808080u;

    const __half2 m = l < kQR2_S / 2 ? m_lo : m_hi;
    const __half2 b = l < kQR2_S / 2 ? b_lo : b_hi;
    unsigned v01 = __byte_perm(grid_l, 0x64646464u, 0x4140);
    unsigned v23 = __byte_perm(grid_l, 0x64646464u, 0x4342);
    unsigned v45 = __byte_perm(grid_h, 0x64646464u, 0x4140);
    unsigned v67 = __byte_perm(grid_h, 0x64646464u, 0x4342);
    *(__half2*)&v01 = __hfma2(*(const __half2*)&v01, m, b);
    *(__half2*)&v23 = __hfma2(*(const __half2*)&v23, m, b);
    *(__half2*)&v45 = __hfma2(*(const __half2*)&v45, m, b);
    *(__half2*)&v67 = __hfma2(*(const __half2*)&v67, m, b);
    StVec16 out;
    out.w[0] = v01;
    out.w[1] = v23;
    out.w[2] = v45;
    out.w[3] = v67;
    dst[l] = out;
  }

  if (q == 0) {
    tile_d[i] = __half2float(bxi->d) * 0.125f;
  }
}

// The products of one half for this warp's column groups (even groups: warps 0..7, odd groups:
// warps 8..15, 16 rows of N a warp): eight m16n8k16 steps accumulate in fp32 in the tensor
// cores, then one multiply-add per element applies the rows' block scales.
template <int kGroups>
static __device__ __forceinline__ void f16_staged_dot(
    const int* __restrict__ x, const float* __restrict__ tile_d, const int* __restrict__ y, float* __restrict__ sum) {
  typedef tile<16, 8, int> tile_C;
  constexpr int kMine = (kGroups + 1) / 2;
  constexpr int kSteps = kF16Half / 16;

  const int part = threadIdx.y / kNWarps;
  if (part >= kGroups) {
    return;  // one group: the second half of the warps has no column
  }
  const int i0 = (threadIdx.y % kNWarps) * tile_C::I;

  float c[kMine][tile_C::ne] = {{0.0f}};
#pragma unroll
  for (int s = 0; s < kSteps; ++s) {
    tile<16, 8, int> a;
    load_ldmatrix(a, x + i0 * kF16TileXK + 8 * s, kF16TileXK);
#pragma unroll
    for (int n = 0; n < kMine; ++n) {
      const int group = 2 * n + part;
      if (group < kGroups) {
        int b[2];
        load_ldmatrix_x2(
            b, y + (group * kWideGroup + threadIdx.x % 8) * kF16TileYK + 8 * s + ((threadIdx.x / 8) & 1) * 4);
        mma_f16(c[n], a.x, b);
      }
    }
  }

  const float d0 = tile_d[i0 + tile_C::get_i(0)];
  const float d1 = tile_d[i0 + tile_C::get_i(2)];
#pragma unroll
  for (int n = 0; n < kMine; ++n) {
    if (2 * n + part < kGroups) {
      sum[n * tile_C::ne + 0] += d0 * c[n][0];
      sum[n * tile_C::ne + 1] += d0 * c[n][1];
      sum[n * tile_C::ne + 2] += d1 * c[n][2];
      sum[n * tile_C::ne + 3] += d1 * c[n][3];
    }
  }
}

template <typename scalar_t, int kWide, int kGroups, bool need_check>
static __device__ __forceinline__ void f16_staged_block(
    const char* __restrict__ xt,
    const int* __restrict__ y,
    const int* __restrict__ ids_dst,
    const int* __restrict__ ids_src,
    const float* __restrict__ col_scale,
    const int* __restrict__ grid,
    int* __restrict__ stage,
    float* __restrict__ tile_d,
    int* __restrict__ tile_y,
    int* __restrict__ tile_x,
    scalar_t* __restrict__ out,
    const int blocks_per_row,
    const int nrows_x,
    const int stride_row_x,
    const int words_per_row_y,
    const int tile_x_max_i) {
  typedef tile<16, 8, int> tile_C;
  constexpr int kMine = (kGroups + 1) / 2;

  float sum[kMine * tile_C::ne] = {0.0f};
  F16Vec32 regs_y[f16_staged_y_regs<kGroups>()];

  mmq_staged_copy_x<need_check>(xt, stage, 0, tile_x_max_i, stride_row_x);
  mmq_staged_cp_async_commit();
  f16_staged_fetch_y<kGroups>(y, ids_src, regs_y, 0, words_per_row_y);

  for (int kb0 = 0; kb0 < blocks_per_row; ++kb0) {
    const int* cur = stage + (kb0 & 1) * kStStageWords;
    if (kb0 + 1 < blocks_per_row) {
      // the next tile's packed rows, on their way while this one is unpacked and multiplied
      mmq_staged_copy_x<need_check>(xt, stage + ((kb0 + 1) & 1) * kStStageWords, kb0 + 1, tile_x_max_i, stride_row_x);
      mmq_staged_cp_async_commit();
      mmq_staged_cp_async_wait<1>();
    } else {
      mmq_staged_cp_async_wait<0>();
    }
    f16_staged_store_y<kGroups>(tile_y, regs_y);  // the first half's activations

    __syncthreads();

    f16_staged_fetch_y<kGroups>(y, ids_src, regs_y, 2 * kb0 + 1, words_per_row_y);
    f16_staged_unpack_iq2_s<need_check>(xt, cur, grid, tile_x, tile_d, kb0, 0, tile_x_max_i, stride_row_x);

    __syncthreads();

    f16_staged_dot<kGroups>(tile_x, tile_d, tile_y, sum);

    __syncthreads();

    f16_staged_store_y<kGroups>(tile_y, regs_y);  // the second half's
    if (kb0 + 1 < blocks_per_row) {
      f16_staged_fetch_y<kGroups>(y, ids_src, regs_y, 2 * kb0 + 2, words_per_row_y);
    }
    f16_staged_unpack_iq2_s<need_check>(xt, cur, grid, tile_x, tile_d, kb0, 1, tile_x_max_i, stride_row_x);

    __syncthreads();

    f16_staged_dot<kGroups>(tile_x, tile_d, tile_y, sum);

    __syncthreads();
  }

  const int part = threadIdx.y / kNWarps;
  const int i0 = (threadIdx.y % kNWarps) * tile_C::I;
#pragma unroll
  for (int n = 0; n < kMine; ++n) {
    const int group = 2 * n + part;
    if (group >= kGroups) {
      continue;
    }
#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
      const int col = group * kWideGroup + tile_C::get_j(l);
      const int route = ids_dst[col];
      if (route < 0) {
        continue;
      }
      const int i = i0 + tile_C::get_i(l);
      if (need_check && i > tile_x_max_i) {
        continue;
      }
      out[(int64_t)route * nrows_x + i] = static_cast<scalar_t>(sum[n * tile_C::ne + l] * col_scale[col]);
    }
  }
}

// grid (num_blocks of kWide sorted rows, ceil(N / 128)), block (32, 16).
// x: IQ2_S weights [E, N, K / 256 * 82]; y: fp16 activations [nrows_y, K] with their row scales
// yscale [nrows_y] (gguf_moe_f16_prepare); the routing arguments are those of the staged MMQ kernel.
template <typename scalar_t, int kWide, bool need_check>
__global__ void __launch_bounds__(kStThreads, 1) gguf_moe_f16_staged_iq2_s_kernel(
    const char* __restrict__ x,
    const int* __restrict__ y,
    const float* __restrict__ yscale,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_rows,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    scalar_t* __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int stride_row_x,
    const int64_t stride_expert_x,
    const int top_k,
    const int numel,
    const int num_experts) {
  constexpr int kMaxGroups = kWide / kWideGroup;
  static_assert(kWide == 32 || kWide == 64, "kWide: 32 or 64");

  extern __shared__ int data_mul_mat_q[];
  int* ids_dst = data_mul_mat_q;
  int* ids_src = ids_dst + kWide;
  float* col_scale = (float*)(ids_src + kWide);
  int* grid = ids_src + 2 * kWide;
  int* stage = grid + kStGridWords;
  float* tile_d = (float*)(stage + 2 * kStStageWords);
  int* tile_y = stage + 2 * kStStageWords + kMmqY;
  int* tile_x = tile_y + kWide * kF16TileYK;
  static_assert(
      (3 * kWide + kStGridWords + 2 * kStStageWords + kMmqY) % 4 == 0,
      "the staged rows and the tiles must stay 16-byte aligned");

  const int jt = blockIdx.x;
  const int it = blockIdx.y;
  const int col0 = jt * kWide;
  if (col0 >= num_tokens_post_padded[0]) {
    return;
  }
  const int expert = expert_ids[jt];
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
  for (int j = tid; j < kWide; j += kStThreads) {
    const int sid = sorted_ids[col0 + j];
    const bool valid = sid >= 0 && sid < numel;
    const int src = valid ? sid / top_k : 0;
    ids_dst[j] = valid ? (sorted_rows != nullptr ? sorted_rows[col0 + j] : sid) : -1;
    ids_src[j] = src;
    col_scale[j] = yscale[src];
  }
#pragma unroll
  for (int n = tid; n < kStGridWords / 2; n += kStThreads) {
    ((uint64_t*)grid)[n] = iq2s_grid[n];
  }
  __syncthreads();

  int groups = 0;
#pragma unroll
  for (int g = 0; g < kMaxGroups; ++g) {
    bool any = false;
#pragma unroll
    for (int j = 0; j < kWideGroup; ++j) {
      any = any || ids_dst[g * kWideGroup + j] >= 0;
    }
    if (any) {
      groups = g + 1;
    }
  }

  const char* xt = x + (int64_t)expert * stride_expert_x + (int64_t)it * kMmqY * stride_row_x;
  const int tile_x_max_i = nrows_x - it * kMmqY - 1;
  const int blocks_per_row = ncols_x / QK_K;
  scalar_t* out = dst + it * kMmqY;

#define SGLANG_F16_STAGED_CASE(G)                       \
  case G:                                               \
    if constexpr (G <= kMaxGroups) {                    \
      f16_staged_block<scalar_t, kWide, G, need_check>( \
          xt,                                           \
          y,                                            \
          ids_dst,                                      \
          ids_src,                                      \
          col_scale,                                    \
          grid,                                         \
          stage,                                        \
          tile_d,                                       \
          tile_y,                                       \
          tile_x,                                       \
          out,                                          \
          blocks_per_row,                               \
          nrows_x,                                      \
          stride_row_x,                                 \
          ncols_x / 2,                                  \
          tile_x_max_i);                                \
    }                                                   \
    break;
  switch (groups) {
    SGLANG_F16_STAGED_CASE(1)
    SGLANG_F16_STAGED_CASE(2)
    SGLANG_F16_STAGED_CASE(3)
    SGLANG_F16_STAGED_CASE(4)
    SGLANG_F16_STAGED_CASE(5)
    SGLANG_F16_STAGED_CASE(6)
    SGLANG_F16_STAGED_CASE(7)
    SGLANG_F16_STAGED_CASE(8)
    default:
      break;  // no route in the block
  }
#undef SGLANG_F16_STAGED_CASE
}

}  // namespace gguf_mma
}  // namespace sglang
