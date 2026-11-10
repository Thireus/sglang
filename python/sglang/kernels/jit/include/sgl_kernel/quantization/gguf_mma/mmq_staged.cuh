// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// A staged variant of the wide-block GGUF MoE MMQ kernel for IQ2_S (mmq_wide.cuh, on top of the
// ik_llama.cpp port in mmq_id.cuh). The arithmetic of a (route, row) is unchanged: the same tile
// values, the same m16n8k16 products, the same float expression accumulated in the same order
// (QK_K block, half, 32-value group), so the output bits are those of gguf_moe_mmq_kernel. What
// changes is where the bytes come from and when:
//
//   - one CTA of 16 warps per multiprocessor (93 KiB of shared memory) instead of two of 8 warps
//     that leave the first-level cache almost no room;
//   - the packed rows of a tile (128 rows x 82 bytes, about 10 KiB that no cache holds) are
//     copied into shared memory by ASYNCHRONOUS copies (cp.async, 16 bytes each: a row block sits
//     in at most six aligned 16-byte vectors), one tile ahead and into the buffer the previous
//     tile left: a plain load stalls its warp until the bytes arrive (about a microsecond from
//     memory), a cp.async does not, so the multiprocessor unpacks and multiplies one tile while
//     the next one is on its way. The tile is unpacked from shared memory against the 8 KiB grid
//     table, which sits in shared memory too: the unpack makes no global load at all (the
//     wide-block kernel spends about three quarters of its time waiting for the fifteen
//     scattered loads per thread and iteration of its unpack);
//   - the activations of all the block's columns and both halves are copied with 16-byte loads
//     and sit in shared memory at once (no passes); the two halves of the warps share the column
//     groups (even / odd);
//   - the activations of the next tile travel the same way when the block has at most half its
//     columns in use (the y tile then holds two blocks of two halves); a fuller block loads them
//     into registers before the unpack and stores them after the products, which stalls its
//     warps for the second-level cache once a tile;
//   - inside shared memory the same economy: 16-byte stores of the unpacked values, 8-byte loads
//     of the grid entries, the scales of a row and of a column as 16-byte loads, the activation
//     words of a group through ldmatrix (the y tile's row stride is padded for it, as ggml's).
#pragma once

#include <sgl_kernel/quantization/gguf_mma/mmq_wide.cuh>

#include <cstdint>

namespace sglang {
namespace gguf_mma {

constexpr int kStWarps = 2 * kNWarps;  // 16 warps: 512 threads
constexpr int kStThreads = kStWarps * kWarpSize;
constexpr int kStBlockBytes = 82;                          // sizeof(block_iq2_s)
constexpr int kStVecBytes = 16;                            // one asynchronous copy of the packed rows
constexpr int kStRowVecs = 6;                              // 82 bytes at any even offset below 16 end before byte 96
constexpr int kStRowWords = kStRowVecs * kStVecBytes / 4;  // a staged row: 24 words
constexpr int kStStageVecs = kMmqY * kStRowVecs;           // 768 copies a tile
constexpr int kStStageWords = kMmqY * kStRowWords;         // 3,072 words a buffer; there are two
constexpr int kStGridWords = 2 * 1024;                     // iq2s_grid as 32-bit words
constexpr int kStYVecs = kTileYK / 4;                      // a block_q8_1_mmq as 16-byte vectors: 9
static_assert(sizeof(block_iq2_s) == kStBlockBytes, "IQ2_S block size");
static_assert(kStStageVecs <= 2 * kStThreads, "at most two copies of the staged tile per thread");
static_assert(
    kStStageWords % 4 == 0 && kStGridWords % 4 == 0, "the staged rows and the tiles must stay 16-byte aligned");

// 16 bytes as one load or store.
struct alignas(16) StVec16 {
  uint32_t w[4];
};
struct alignas(16) StFloat4 {
  float f[4];
};
struct alignas(8) StFloat2 {
  float f[2];
};

template <int kWide>
constexpr __host__ __device__ int mmq_staged_shared_words() {
  // ids_dst + ids_src, the grid table, two buffers of staged packed rows, the y tile of both halves, the x tile
  return 2 * kWide + kStGridWords + 2 * kStStageWords + 2 * kWide * kTileYK + kMmqY * kTileXK_Q3_K;
}

// Row block (i, kb0) starts at an even offset below 16 inside an aligned 16-byte vector and ends
// inside the sixth one at the latest. With the weight tensor starting on a 16-byte boundary and its
// size a multiple of 16 (checked on the host), every vector that holds a byte of a block lies inside
// the tensor; a vector the block does not reach is not copied.
static __device__ __forceinline__ uintptr_t
mmq_staged_block_addr(const char* xt, const int i, const int kb0, const int stride) {
  return (uintptr_t)(xt + (int64_t)i * stride + (int64_t)kb0 * kStBlockBytes);
}

// cp.async (sm_80+): 16 bytes from global to shared memory without stalling the warp. The copies a
// thread issues before a commit form a group; wait_group N returns when at most N of the thread's
// most recent groups are still in flight.
static __device__ __forceinline__ void
mmq_staged_cp_async_16(void* __restrict__ shared, const void* __restrict__ global) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" : : "r"(dst), "l"(global) : "memory");
}

static __device__ __forceinline__ void mmq_staged_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" : : : "memory");
}

template <int kPending>
static __device__ __forceinline__ void mmq_staged_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" : : "n"(kPending) : "memory");
}

// Issue the copies of the packed rows of tile kb0 into a staging buffer (vector v of the staged
// tile: row v / 6, vector v % 6 of that row's window). The caller closes the group.
template <bool need_check>
static __device__ __forceinline__ void mmq_staged_copy_x(
    const char* __restrict__ xt, int* __restrict__ stage, const int kb0, const int i_max, const int stride) {
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
#pragma unroll
  for (int m = 0; m < (kStStageVecs + kStThreads - 1) / kStThreads; ++m) {
    const int v = tid + m * kStThreads;
    if (v < kStStageVecs) {
      int i = v / kStRowVecs;
      const int part = v - i * kStRowVecs;
      if (need_check) {
        i = min(i, i_max);
      }
      const uintptr_t p = mmq_staged_block_addr(xt, i, kb0, stride);
      if (kStVecBytes * part < (int)(p & (kStVecBytes - 1)) + kStBlockBytes) {
        mmq_staged_cp_async_16(stage + 4 * v, (const void*)((p & ~(uintptr_t)(kStVecBytes - 1)) + kStVecBytes * part));
      }
    }
  }
}

// Issue the copies of the activations of the block's first kGroups column groups for both halves
// of QK_K block kb0 into a y buffer of two halves of half_words words each. The caller closes the group.
template <int kGroups>
static __device__ __forceinline__ void mmq_staged_copy_y(
    const int* __restrict__ y,
    const int* __restrict__ ids_src,
    int* __restrict__ ybuf,
    const int kb0,
    const int nrows_y,
    const int half_words) {
  constexpr int kVecs = kGroups * kWideGroup * kStYVecs;  // one half
#pragma unroll
  for (int m = 0; m < (2 * kVecs + kStThreads - 1) / kStThreads; ++m) {
    const int l2 = threadIdx.y * kWarpSize + threadIdx.x + m * kStThreads;
    if (l2 < 2 * kVecs) {
      const int h = l2 >= kVecs ? 1 : 0;
      const int l = l2 - h * kVecs;
      const int col = l / kStYVecs;
      const int v = l - col * kStYVecs;
      mmq_staged_cp_async_16(
          ybuf + h * half_words + 4 * l,
          (const StVec16*)(y + ((int64_t)(2 * kb0 + h) * nrows_y + ids_src[col]) * kTileYK) + v);
    }
  }
}

template <int kGroups>
constexpr __host__ __device__ int mmq_staged_y_regs() {
  return (2 * kGroups * kWideGroup * kStYVecs + kStThreads - 1) / kStThreads;
}

// Issue the loads of the activations of the block's first kGroups column groups for both
// 128-value halves of QK_K block kb0 into registers, 16 bytes a load (a block_q8_1_mmq is 9 of them).
template <int kGroups>
static __device__ __forceinline__ void mmq_staged_fetch_y(
    const int* __restrict__ y,
    const int* __restrict__ ids_src,
    StVec16* __restrict__ regs,
    const int kb0,
    const int nrows_y) {
  constexpr int kVecs = kGroups * kWideGroup * kStYVecs;  // one half
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
#pragma unroll
  for (int m = 0; m < mmq_staged_y_regs<kGroups>(); ++m) {
    const int l2 = tid + m * kStThreads;
    if (l2 < 2 * kVecs) {
      const int h = l2 >= kVecs ? 1 : 0;
      const int l = l2 - h * kVecs;
      const int col = l / kStYVecs;
      const int v = l - col * kStYVecs;
      regs[m] = ((const StVec16*)(y + ((int64_t)(2 * kb0 + h) * nrows_y + ids_src[col]) * kTileYK))[v];
    }
  }
}

template <int kGroups, int kWide>
static __device__ __forceinline__ void mmq_staged_store_y(int* __restrict__ tile_y, const StVec16* __restrict__ regs) {
  constexpr int kVecs = kGroups * kWideGroup * kStYVecs;
  const int tid = threadIdx.y * kWarpSize + threadIdx.x;
#pragma unroll
  for (int m = 0; m < mmq_staged_y_regs<kGroups>(); ++m) {
    const int l2 = tid + m * kStThreads;
    if (l2 < 2 * kVecs) {
      const int h = l2 >= kVecs ? 1 : 0;
      ((StVec16*)(tile_y + h * (kWide * kTileYK)))[l2 - h * kVecs] = regs[m];
    }
  }
}

// load_tiles_iq2_s (mmq_id.cuh) reading the staged copy of the packed rows and the grid table in
// shared memory: the same statements per (row, 32 values), so the same tile values and scales.
template <bool need_check>
static __device__ __forceinline__ void mmq_staged_unpack_iq2_s(
    const char* __restrict__ xt,
    const int* __restrict__ stage,
    const int* __restrict__ grid,
    int* __restrict__ x_tile,
    const int kb0,
    const int i_max,
    const int stride) {
  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + kTileNeK * 2);

  constexpr int threads_per_row = (256 / (4 * kQR2_S)) / 2;
  constexpr int nrows = kWarpSize / threads_per_row;
  const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
  for (int i0 = 0; i0 < kMmqY; i0 += kStWarps * nrows) {
    int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;

    if (need_check) {
      i = min(i, i_max);
    }

    // (with need_check the rows past i_max read row i_max's block out of staged row i_max, as load_tiles_iq2_s does)
    const int offset = (int)(mmq_staged_block_addr(xt, i, kb0, stride) & (kStVecBytes - 1));
    const block_iq2_s* bxi = (const block_iq2_s*)((const char*)(stage + i * kStRowWords) + offset);

    const int qs_packed = mmq_get_int_b2(bxi->qs, kqsx);
    const uint8_t* qs = (const uint8_t*)&qs_packed;

    const int qh = bxi->qh[kqsx];

    const int signs_packed_32 = mmq_get_int_b2(bxi->qs, QK_K / 32 + kqsx);
    const uint8_t* signs_packed_8 = (const uint8_t*)&signs_packed_32;

    StVec16 q[2];  // the unit's eight words of values: x_qs[i * kTileXK_Q3_K + 8 * kqsx + 0..7]
#pragma unroll
    for (int l = 0; l < kQR2_S; ++l) {
      // one 8-byte load of the grid entry: its two words are grid_pos[0] and grid_pos[1] of load_tiles_iq2_s
      const uint64_t entry = *(const uint64_t*)(grid + 2 * (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
      const int grid_pos0 = (int)(uint32_t)entry;
      const int grid_pos1 = (int)(uint32_t)(entry >> 32);

      const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
      const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

      const int grid_l = __vsub4(grid_pos0 ^ signs0, signs0);
      const int grid_h = __vsub4(grid_pos1 ^ signs1, signs1);

      q[l / 2].w[2 * (l % 2) + 0] = (uint32_t)grid_l;
      q[l / 2].w[2 * (l % 2) + 1] = (uint32_t)grid_h;
    }
    StVec16* dst = (StVec16*)(x_qs + i * kTileXK_Q3_K + 8 * kqsx);
    dst[0] = q[0];
    dst[1] = q[1];

    const int ls = bxi->scales[kqsx];
    const float d = __half2float(bxi->d);
    StFloat2 df;
    df.f[0] = ((ls & 0x0F) * d + d / 2) / 4;
    df.f[1] = ((ls >> 4) * d + d / 2) / 4;
    *(StFloat2*)(x_df + i * kTileXK_Q3_K + 2 * kqsx) = df;
  }
}

// Four 8 x 8 matrices of 16-bit elements in one load: matrix m is rows 0..7 (stride words apart) at
// words 4 m .. 4 m + 3 of xs0; a thread gets word threadIdx.x % 4 of row threadIdx.x / 4 of each.
static __device__ __forceinline__ void
load_ldmatrix_x4_rows(int* __restrict__ regs, const int* __restrict__ xs0, const int stride) {
  const int* xs = xs0 + (threadIdx.x % 8) * stride + (threadIdx.x / 8) * 4;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
               : "l"(xs));
}

// vec_dot_q8_0_16_q8_1_mma for the column groups of this warp: warps 0..7 own 16 rows of N each
// and the even groups, warps 8..15 the same rows and the odd groups. sum holds the warp's groups
// in order ((kGroups + 1) / 2 groups of tile_C::ne values).
template <int kGroups>
static __device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma_staged(
    const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int k00) {
  typedef tile<16, 4, int> tile_A;
  typedef tile<16, 8, int> tile_A_8;
  typedef tile<8, 4, int> tile_B;
  typedef tile<16, 8, int> tile_C;
  static_assert(tile_C::J == kWideGroup, "one group is one C tile");
  constexpr int kMine = (kGroups + 1) / 2;

  const int part = threadIdx.y / kNWarps;
  if (part >= kGroups) {
    return;  // one group: the second half of the warps has no column
  }

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + kTileNeK * 2;
  const int* y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  const int i0 = (threadIdx.y % kNWarps) * tile_A::I;

  tile_A A[8];
  float dA[tile_C::ne / 2][8];

#pragma unroll
  for (int k01 = 0; k01 < kTileNeK; k01 += 8) {
    const int k0 = k00 + k01;
    load_ldmatrix(((tile_A_8*)A)[k01 / 8], x_qs + i0 * kTileXK_Q3_K + k0, kTileXK_Q3_K);
  }

  // the eight scales of a row for this half (x_df[i * kTileXK_Q3_K + (k00 + k01) / 4], k01 = 0, 4, .., 28): two 16-byte
  // loads
#pragma unroll
  for (int l = 0; l < tile_C::ne / 2; ++l) {
    const int i = i0 + tile_C::get_i(2 * l);
    const StFloat4* row = (const StFloat4*)(x_df + i * kTileXK_Q3_K + k00 / 4);
    const StFloat4 lo = row[0];
    const StFloat4 hi = row[1];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      dA[l][k] = lo.f[k];
      dA[l][4 + k] = hi.f[k];
    }
  }

#pragma unroll
  for (int n = 0; n < kMine; ++n) {
    const int group = 2 * n + part;
    if (group < kGroups) {
      const int j0 = group * kWideGroup;
      // the group's activation words of this half: word w of column j0 + threadIdx.x / 4 is yq[w / 16][w / 4 % 4]
      // at w + threadIdx.x % 4 (what load_generic reads one word at a time), and the four scales of the
      // thread's two columns (y_df[j * kTileYK + k01 / 8])
      int yq[2][4];
      load_ldmatrix_x4_rows(yq[0], y_qs + j0 * kTileYK, kTileYK);
      load_ldmatrix_x4_rows(yq[1], y_qs + j0 * kTileYK + 16, kTileYK);
      StFloat4 dBv[tile_C::ne / 2];
#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        dBv[l] = *(const StFloat4*)(y_df + (j0 + tile_C::get_j(l)) * kTileYK);
      }
#pragma unroll
      for (int k01 = 0; k01 < kTileNeK; k01 += 8) {
        tile_B B[2];
        float dB[tile_C::ne / 2];

        B[0].x[0] = yq[k01 / 16][(k01 % 16) / 4 + 0];
        B[1].x[0] = yq[k01 / 16][(k01 % 16) / 4 + 1];

#pragma unroll
        for (int l = 0; l < tile_C::ne / 2; ++l) {
          dB[l] = dBv[l].f[k01 / kQI8_1];
        }

        tile_C C[2];
        mma(C[0], A[k01 / 4 + 0], B[0]);
        mma(C[1], A[k01 / 4 + 1], B[1]);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[n * tile_C::ne + l] +=
              dB[l % 2] * (C[0].x[l] * dA[l / 2][k01 / 4 + 0] + C[1].x[l] * dA[l / 2][k01 / 4 + 1]);
        }
      }
    }
  }
}

// One block with its first kGroups column groups in use (a compile-time count, as in the wide-block
// kernel). Per QK_K block: issue the copies of the next tile's packed rows into the other staging
// buffer, wait for this tile's (issued one block ago), unpack, multiply both halves.
template <typename scalar_t, int kWide, int kGroups, bool need_check>
static __device__ __forceinline__ void mmq_staged_block(
    const char* __restrict__ xt,
    const int* __restrict__ y,
    const int* __restrict__ ids_dst,
    const int* __restrict__ ids_src,
    const int* __restrict__ grid,
    int* __restrict__ stage,
    int* __restrict__ tile_y,
    int* __restrict__ tile_x,
    scalar_t* __restrict__ out,
    const int blocks_per_row,
    const int nrows_x,
    const int stride_row_x,
    const int nrows_y,
    const int tile_x_max_i) {
  typedef tile<16, 8, int> tile_C;
  constexpr int kMine = (kGroups + 1) / 2;

  float sum[kMine * tile_C::ne] = {0.0f};

  if constexpr (kGroups <= kWide / (2 * kWideGroup)) {
    // At most half the columns in use: the y tile holds two blocks (each two halves of kWide / 2 columns), and
    // the activations are copied one tile ahead with the packed rows. No load stalls a warp inside the loop.
    constexpr int kHalfWords = (kWide / 2) * kTileYK;
    mmq_staged_copy_x<need_check>(xt, stage, 0, tile_x_max_i, stride_row_x);
    mmq_staged_copy_y<kGroups>(y, ids_src, tile_y, 0, nrows_y, kHalfWords);
    mmq_staged_cp_async_commit();

    for (int kb0 = 0; kb0 < blocks_per_row; ++kb0) {
      const int* cur = stage + (kb0 & 1) * kStStageWords;
      const int* y_cur = tile_y + (kb0 & 1) * 2 * kHalfWords;
      if (kb0 + 1 < blocks_per_row) {
        // The next tile, into the buffers the previous tile left (its unpack and its products are done): on its
        // way from memory while this tile is unpacked and multiplied. Then wait for this tile's copies only.
        const int other = (kb0 + 1) & 1;
        mmq_staged_copy_x<need_check>(xt, stage + other * kStStageWords, kb0 + 1, tile_x_max_i, stride_row_x);
        mmq_staged_copy_y<kGroups>(y, ids_src, tile_y + other * 2 * kHalfWords, kb0 + 1, nrows_y, kHalfWords);
        mmq_staged_cp_async_commit();
        mmq_staged_cp_async_wait<1>();
      } else {
        mmq_staged_cp_async_wait<0>();
      }

      __syncthreads();  // every thread's copies of this tile have landed

      mmq_staged_unpack_iq2_s<need_check>(xt, cur, grid, tile_x, kb0, tile_x_max_i, stride_row_x);

      __syncthreads();

      vec_dot_q8_0_16_q8_1_mma_staged<kGroups>(tile_x, y_cur, sum, 0);
      vec_dot_q8_0_16_q8_1_mma_staged<kGroups>(tile_x, y_cur + kHalfWords, sum, kTileNeK);

      __syncthreads();
    }
  } else {
    StVec16 regs_y[mmq_staged_y_regs<kGroups>()];

    mmq_staged_copy_x<need_check>(xt, stage, 0, tile_x_max_i, stride_row_x);
    mmq_staged_cp_async_commit();
    mmq_staged_fetch_y<kGroups>(y, ids_src, regs_y, 0, nrows_y);

    for (int kb0 = 0; kb0 < blocks_per_row; ++kb0) {
      const int* cur = stage + (kb0 & 1) * kStStageWords;
      if (kb0 + 1 < blocks_per_row) {
        // The next tile's packed rows, into the buffer the previous tile left (its unpack is done): on their way
        // from memory while this tile is unpacked and multiplied. Then wait for this tile's copies only.
        mmq_staged_copy_x<need_check>(xt, stage + ((kb0 + 1) & 1) * kStStageWords, kb0 + 1, tile_x_max_i, stride_row_x);
        mmq_staged_cp_async_commit();
        mmq_staged_cp_async_wait<1>();
      } else {
        mmq_staged_cp_async_wait<0>();
      }
      mmq_staged_store_y<kGroups, kWide>(tile_y, regs_y);

      __syncthreads();  // every thread's copies of this tile have landed

      if (kb0 + 1 < blocks_per_row) {
        mmq_staged_fetch_y<kGroups>(y, ids_src, regs_y, kb0 + 1, nrows_y);
      }

      mmq_staged_unpack_iq2_s<need_check>(xt, cur, grid, tile_x, kb0, tile_x_max_i, stride_row_x);

      __syncthreads();

      vec_dot_q8_0_16_q8_1_mma_staged<kGroups>(tile_x, tile_y, sum, 0);
      vec_dot_q8_0_16_q8_1_mma_staged<kGroups>(tile_x, tile_y + kWide * kTileYK, sum, kTileNeK);

      __syncthreads();
    }
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
      const int route = ids_dst[group * kWideGroup + tile_C::get_j(l)];
      if (route < 0) {
        continue;
      }
      const int i = i0 + tile_C::get_i(l);
      if (need_check && i > tile_x_max_i) {
        continue;
      }
      out[(int64_t)route * nrows_x + i] = static_cast<scalar_t>(sum[n * tile_C::ne + l]);
    }
  }
}

// grid (num_blocks of kWide sorted rows, ceil(N / 128)), block (32, 16).
// The arguments are those of gguf_moe_mmq_wide_iq2_s_kernel (sorted_rows may be null); x and y must
// start on 16-byte boundaries (the host entry checks both).
template <typename scalar_t, int kWide, bool need_check>
__global__ void __launch_bounds__(kStThreads, 1) gguf_moe_mmq_staged_iq2_s_kernel(
    const char* __restrict__ x,
    const int* __restrict__ y,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_rows,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    scalar_t* __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int stride_row_x,
    const int64_t stride_expert_x,
    const int nrows_y,
    const int top_k,
    const int numel,
    const int num_experts) {
  constexpr int kMaxGroups = kWide / kWideGroup;
  static_assert(kWide == 32 || kWide == 64, "kWide: 32 or 64");

  extern __shared__ int data_mul_mat_q[];
  int* ids_dst = data_mul_mat_q;
  int* ids_src = ids_dst + kWide;
  int* grid = ids_src + kWide;
  int* stage = grid + kStGridWords;
  int* tile_y = stage + 2 * kStStageWords;
  int* tile_x = tile_y + 2 * kWide * kTileYK;

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
    ids_dst[j] = valid ? (sorted_rows != nullptr ? sorted_rows[col0 + j] : sid) : -1;
    ids_src[j] = valid ? sid / top_k : 0;
  }
#pragma unroll
  for (int n = tid; n < kStGridWords / 2; n += kStThreads) {
    ((uint64_t*)grid)[n] = iq2s_grid[n];
  }
  __syncthreads();

  // Groups in use: up to the last column that holds a route (the align kernel puts an expert's
  // routes first and its padding last; a hole would only cost its group's products).
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

// One straight-line body per group count; the count is the same for every thread of the CTA.
#define SGLANG_MMQ_STAGED_CASE(G)                       \
  case G:                                               \
    if constexpr (G <= kMaxGroups) {                    \
      mmq_staged_block<scalar_t, kWide, G, need_check>( \
          xt,                                           \
          y,                                            \
          ids_dst,                                      \
          ids_src,                                      \
          grid,                                         \
          stage,                                        \
          tile_y,                                       \
          tile_x,                                       \
          out,                                          \
          blocks_per_row,                               \
          nrows_x,                                      \
          stride_row_x,                                 \
          nrows_y,                                      \
          tile_x_max_i);                                \
    }                                                   \
    break;
  switch (groups) {
    SGLANG_MMQ_STAGED_CASE(1)
    SGLANG_MMQ_STAGED_CASE(2)
    SGLANG_MMQ_STAGED_CASE(3)
    SGLANG_MMQ_STAGED_CASE(4)
    SGLANG_MMQ_STAGED_CASE(5)
    SGLANG_MMQ_STAGED_CASE(6)
    SGLANG_MMQ_STAGED_CASE(7)
    SGLANG_MMQ_STAGED_CASE(8)
    default:
      break;  // no route in the block
  }
#undef SGLANG_MMQ_STAGED_CASE
}

// quantize_mmq_q8_1_d4 over the first row_count[0] rows only, the count read ON THE DEVICE (a compact
// cold-tier call: the down projection's input holds one row per cold route in a buffer sized for every
// route, and the host does not know how many there are). The statements of a row are those of
// quantize_mmq_q8_1_d4: the same int8 values and scales. y keeps the layout of `nrows` rows per block.
// grid (min(nrows, kStCountedGrid), ceil(k / 4 / 128)), block (128): CTA (b, c) takes rows b, b + grid.x, ...
constexpr int kStCountedGrid = 4096;

template <typename scalar_t>
__global__ void quantize_mmq_q8_1_d4_counted(
    const scalar_t* __restrict__ x,
    block_q8_1_mmq* __restrict__ y,
    const int32_t* __restrict__ row_count,
    const int k,
    const int nrows) {
  const int ix0 = (blockDim.x * blockIdx.y + threadIdx.x) * 4;
  if (ix0 >= k) {
    return;  // k % 128 == 0: whole warps leave together
  }
  const int counted = row_count[0];
  const int count = counted < nrows ? counted : nrows;
  for (int64_t row = blockIdx.x; row < count; row += gridDim.x) {
    const scalar_t* xr = x + row * k + ix0;
    const float x0 = static_cast<float>(xr[0]);
    const float x1 = static_cast<float>(xr[1]);
    const float x2 = static_cast<float>(xr[2]);
    const float x3 = static_cast<float>(xr[3]);

    float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
    }

    const float d = amax / 127;
    char4 q;
    q.x = amax == 0.0f ? 0 : roundf(x0 / d);
    q.y = amax == 0.0f ? 0 : roundf(x1 / d);
    q.z = amax == 0.0f ? 0 : roundf(x2 / d);
    q.w = amax == 0.0f ? 0 : roundf(x3 / d);

    const int64_t ib = (int64_t)(ix0 / 128) * nrows + row;
    const int iqs = ix0 % 128;
    reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
    if (iqs % 32 == 0) {
      y[ib].d4[iqs / 32] = __half2float(__float2half(d));
    }
  }
}

template <typename scalar_t>
static void quantize_mmq_q8_1_d4_counted_cuda(
    const scalar_t* x, void* y, const int32_t* row_count, const int k, const int nrows, cudaStream_t stream) {
  const dim3 grid(
      nrows < kStCountedGrid ? nrows : kStCountedGrid, (k / 4 + kQuantizeMmqBlock - 1) / kQuantizeMmqBlock, 1);
  quantize_mmq_q8_1_d4_counted<scalar_t>
      <<<grid, kQuantizeMmqBlock, 0, stream>>>(x, (block_q8_1_mmq*)y, row_count, k, nrows);
}

}  // namespace gguf_mma
}  // namespace sglang
