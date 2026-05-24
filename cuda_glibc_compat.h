// Workaround: glibc 2.40+ declares rsqrt/rsqrtf with noexcept(true),
// but CUDA 13.1 math_functions.h declares them without noexcept.
// Pre-define them to prevent the glibc declarations from conflicting.
#ifndef CUDA_GLIBC_COMPAT_H
#define CUDA_GLIBC_COMPAT_H

// Prevent glibc from declaring rsqrt (it's in __MATHCALL_VEC which expands from bits/mathcalls.h)  
// We do this by including math.h FIRST with rsqrt guard
#ifdef __cplusplus
#include <cstddef>
// Define __FAST_MATH__ guard won't work since rsqrt is unconditional in glibc 2.43
#endif

#endif
