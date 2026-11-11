// SPDX-License-Identifier: MIT
//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow (ikawrakow) and the ik_llama.cpp authors
//
// int8 tensor-core tiles for the GGUF MoE MMQ kernel, copied from ik_llama.cpp 7c1deaf3
// ggml/src/ggml-cuda/mma_new.cuh (namespace ggml_cuda_mma), reduced to the NVIDIA int8 tiles the
// IQ2_S / IQ4_KS / IQ2_KS paths use: tile<16,8>/<8,8> with m16n8k32 and tile<16,4>/<8,4> with
// m16n8k16. Requires sm_80 or newer (SM120: IMMA.16832.S8 / IMMA.16816.S8).
//
// A is a row-major M x K tile, B a column-major K x N tile, C a column-major M x N tile; J counts
// 32-bit elements. get_i / get_j give the physical position of the lth 32-bit element of a thread.
// load_ldmatrix pointers must be shared memory and 16-byte aligned.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace sglang {
namespace gguf_mma {

constexpr int kWarpSize = 32;

template <int I_, int J_, typename T>
struct tile {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I * J / kWarpSize;
  T x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && (J == 4 || J == 8)) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return (l / 2) * 8 + threadIdx.x / 4;
    } else {
      static_assert(I == -1 && J == -1, "template specialization not implemented");
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 4) {
      return threadIdx.x % 4;
    } else if constexpr (I == 8 && J == 8) {
      return 4 * l + threadIdx.x % 4;
    } else if constexpr (I == 16 && J == 8) {
      return 2 * (threadIdx.x % 4) + l % 2;
    } else {
      static_assert(I == -1 && J == -1, "template specialization not implemented");
    }
  }
};

template <int I, int J, typename T>
static __device__ __forceinline__ void load_generic(tile<I, J, T>& t, const T* __restrict__ xs0, const int stride) {
#pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
  }
}

template <typename T>
static __device__ __forceinline__ void load_ldmatrix(tile<16, 8, T>& t, const T* __restrict__ xs0, const int stride) {
  int* xi = (int*)t.x;
  const int* xs = (const int*)xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
               : "l"(xs));
}

static __device__ __forceinline__ void mma(tile<16, 8, int>& D, const tile<16, 4, int>& A, const tile<8, 4, int>& B) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
  __trap();
#endif
}

static __device__ __forceinline__ void mma(tile<16, 8, int>& D, const tile<16, 8, int>& A, const tile<8, 8, int>& B) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, "
      "%3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
  __trap();
#endif
}

}  // namespace gguf_mma
}  // namespace sglang
