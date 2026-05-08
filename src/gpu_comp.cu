/**
 * gpu_comp.cu — LZ4 Compression and Decompression using NVIDIA nvCOMPDx
 */
#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef USE_NVCOMPDX
#include <nvcompdx.hpp>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/* Define the nvCOMPDx operator types for Block-level LZ4 */
using LZ4Compressor = nvcompdx::Compressor<
    nvcompdx::algorithm::LZ4,
    nvcompdx::execution::Block,
    uint8_t
>;

using LZ4Decompressor = nvcompdx::Decompressor<
    nvcompdx::algorithm::LZ4,
    nvcompdx::execution::Block,
    uint8_t
>;

extern "C" {
/* Device-side compression using nvCOMPDx block-cooperative execution */
__device__ void device_compress_lz4(const uint8_t *in_data, size_t in_len,
                                    uint8_t *out_data, size_t out_max,
                                    size_t *actual_out_size)
{
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for nvCOMP workspace
    __shared__ uint8_t workspace[LZ4Compressor::shared_memory_size];
    
    LZ4Compressor comp(workspace);
    
    // Compress
    size_t written = 0;
    comp.execute(block, in_data, in_len, out_data, out_max, &written);
    
    if (block.thread_rank() == 0) {
        *actual_out_size = written;
    }
}

/* Device-side decompression using nvCOMPDx block-cooperative execution */
__device__ void device_decompress_lz4(const uint8_t *in_data, size_t in_len,
                                      uint8_t *out_data, size_t out_max,
                                      size_t *actual_out_size)
{
    cg::thread_block block = cg::this_thread_block();
    
    __shared__ uint8_t workspace[LZ4Decompressor::shared_memory_size];
    
    LZ4Decompressor decomp(workspace);
    
    size_t written = 0;
    decomp.execute(block, in_data, in_len, out_data, out_max, &written);
    
    if (block.thread_rank() == 0) {
        *actual_out_size = written;
    }
}
}

#else /* USE_NVCOMPDX */

extern "C" {
/* Stubs if nvCOMPDx (MathDx) is not installed.
 * WARNING: These stubs perform a memcpy (no actual compression).
 * Compression ratio will be 1:1. Enable USE_NVCOMPDX for real LZ4. */
__device__ void device_compress_lz4(const uint8_t *in_data, size_t in_len,
                                    uint8_t *out_data, size_t out_max,
                                    size_t *actual_out_size)
{
    /* Fallback stub: memory copy (no compression) */
    size_t copy_len = (in_len < out_max) ? in_len : out_max;

    /* Only use vectorized uint4 loads if both pointers are 16-byte aligned (C-4) */
    if (((uintptr_t)in_data & 0xF) == 0 && ((uintptr_t)out_data & 0xF) == 0) {
        size_t copy_len_uint4 = copy_len / 16;
        uint4 *out_4 = (uint4 *)out_data;
        const uint4 *in_4 = (const uint4 *)in_data;
        for (size_t i = threadIdx.x; i < copy_len_uint4; i += blockDim.x) {
            out_4[i] = in_4[i];
        }
        /* Tail bytes */
        for (size_t i = copy_len_uint4 * 16 + threadIdx.x; i < copy_len; i += blockDim.x) {
            out_data[i] = in_data[i];
        }
    } else {
        /* Byte-level fallback for unaligned data */
        for (size_t i = threadIdx.x; i < copy_len; i += blockDim.x) {
            out_data[i] = in_data[i];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        *actual_out_size = copy_len;
    }
}

__device__ void device_decompress_lz4(const uint8_t *in_data, size_t in_len,
                                      uint8_t *out_data, size_t out_max,
                                      size_t *actual_out_size)
{
    /* Fallback stub: memory copy (no decompression) */
    size_t copy_len = (in_len < out_max) ? in_len : out_max;

    /* Only use vectorized uint4 loads if both pointers are 16-byte aligned (C-4) */
    if (((uintptr_t)in_data & 0xF) == 0 && ((uintptr_t)out_data & 0xF) == 0) {
        size_t copy_len_uint4 = copy_len / 16;
        uint4 *out_4 = (uint4 *)out_data;
        const uint4 *in_4 = (const uint4 *)in_data;
        for (size_t i = threadIdx.x; i < copy_len_uint4; i += blockDim.x) {
            out_4[i] = in_4[i];
        }
        for (size_t i = copy_len_uint4 * 16 + threadIdx.x; i < copy_len; i += blockDim.x) {
            out_data[i] = in_data[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < copy_len; i += blockDim.x) {
            out_data[i] = in_data[i];
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        *actual_out_size = copy_len;
    }
}
}

#endif /* USE_NVCOMPDX */
