// Workaround for glibc 2.40+ rsqrt/rsqrtf conflict with CUDA 13.1
// glibc added rsqrt() and rsqrtf() which clash with CUDA's __device__ declarations
#include <cmath>
#ifdef __CUDACC__
// Suppress the duplicate declaration by pre-including math.h before CUDA headers
#endif
