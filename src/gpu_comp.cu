/**
 * gpu_comp.cu — LZ4 Compression and Decompression
 *
 * When USE_NVCOMPDX is defined (nvCOMPDx / MathDx installed):
 *   Uses NVIDIA's nvCOMPDx library for real device-side LZ4 compression.
 *   The compressor/decompressor operates at warp level within the persistent
 *   kernel's thread block.
 *
 * When USE_NVCOMPDX is NOT defined:
 *   Falls back to vectorized memcpy stubs (1:1 ratio, no actual compression).
 */
#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <stdio.h>

#ifdef USE_NVCOMPDX
#include <nvcompdx.hpp>

using namespace nvcompdx;

/* nvCOMPDx LZ4 compressor/decompressor descriptors for sm_75 (Turing).
 * MaxUncompChunkSize = 1<<20 (1MB) — our largest single-chunk size.
 * Uses Warp-level execution to fit within the persistent kernel's block. */
using LZ4CompDesc = decltype(
    Algorithm<algorithm::lz4>() +
    DataType<datatype::uint8>() +
    Direction<direction::compress>() +
    MaxUncompChunkSize<(1 << 20)>() +  /* 1 MB max chunk */
    Warp() +
    SM<750>()
);

using LZ4DecompDesc = decltype(
    Algorithm<algorithm::lz4>() +
    DataType<datatype::uint8>() +
    Direction<direction::decompress>() +
    Warp() +
    SM<750>()
);

extern "C" {

/**
 * Device-side LZ4 compression using nvCOMPDx.
 * Called from the persistent kernel — a single warp within the block executes this.
 * Requires shared memory and a global scratch buffer.
 *
 * NOTE: The caller (persistent kernel) must allocate:
 *   - Shared memory: LZ4CompDesc::shmem_size_group() bytes (aligned)
 *   - Global scratch: pre-allocated per-block temp buffer
 */
__device__ void device_compress_lz4(const uint8_t *in_data, size_t in_len,
                                    uint8_t *out_data, size_t out_max,
                                    size_t *actual_out_size)
{
    /* Only the first warp in the block executes compression.
     * nvCOMPDx warp-level API requires exactly one warp. */
    if (threadIdx.x >= 32) {
        __syncthreads();
        return;
    }

    /* For data larger than MaxUncompChunkSize, fall back to memcpy */
    if (in_len > (1 << 20)) {
        if (threadIdx.x == 0) {
            size_t copy_len = (in_len < out_max) ? in_len : out_max;
            /* Simple byte copy for oversized data */
            for (size_t i = 0; i < copy_len; i++) {
                out_data[i] = in_data[i];
            }
            *actual_out_size = copy_len;
        }
        __syncwarp();
        return;
    }

    auto compressor = LZ4CompDesc();

    /* Use dynamic shared memory for scratch — the persistent kernel
     * allocates this based on the SM's available shared memory. */
    extern __shared__ uint8_t comp_shared_scratch[];

    /* nvCOMPDx LZ4 in warp mode does not need global temp memory
     * for a single chunk. Pass nullptr. */
    compressor.execute(
        in_data,            /* input (uncompressed) */
        out_data,           /* output (compressed) */
        in_len,             /* input size */
        actual_out_size,    /* output size (written by API) */
        comp_shared_scratch,/* shared memory scratch */
        nullptr             /* global temp (not needed for single-chunk warp) */
    );
    __syncwarp();
}

/**
 * Device-side LZ4 decompression using nvCOMPDx.
 */
__device__ void device_decompress_lz4(const uint8_t *in_data, size_t in_len,
                                      uint8_t *out_data, size_t out_max,
                                      size_t *actual_out_size)
{
    if (threadIdx.x >= 32) {
        __syncthreads();
        return;
    }

    auto decompressor = LZ4DecompDesc();

    extern __shared__ uint8_t comp_shared_scratch[];

    decompressor.execute(
        in_data,            /* input (compressed) */
        out_data,           /* output (decompressed) */
        in_len,             /* compressed input size */
        actual_out_size,    /* decompressed output size */
        comp_shared_scratch,/* shared memory scratch */
        nullptr             /* global temp */
    );
    __syncwarp();
}

} /* extern "C" */

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
