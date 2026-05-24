// Workaround: glibc 2.40+ declares rsqrt/rsqrtf with noexcept(true),
// but CUDA 13.1 math_functions.h declares them without noexcept.
#ifndef CUDA_GLIBC_COMPAT_H
#define CUDA_GLIBC_COMPAT_H

#ifdef __CUDACC__
// SIMD macro overrides to prevent syntax errors when renaming rsqrt
#define __DECL_SIMD___glibc_rsqrt
#define __DECL_SIMD___glibc_rsqrtf
#define __DECL_SIMD___glibc_rsqrtl
#define __DECL_SIMD___glibc_rsqrtf32
#define __DECL_SIMD___glibc_rsqrtf64
#define __DECL_SIMD___glibc_rsqrtf128
#define __DECL_SIMD___glibc_rsqrtf32x
#define __DECL_SIMD___glibc_rsqrtf64x

#define rsqrt __glibc_rsqrt
#define rsqrtf __glibc_rsqrtf
#include <math.h>
#include <cmath>
#undef rsqrt
#undef rsqrtf
#endif

#endif
