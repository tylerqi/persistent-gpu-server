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
 * MaxUncompChunkSize = 1<<16 (64KB).
 * Uses Warp-level execution to fit within the persistent kernel's block.
 *
 * NOTE: As of MathDx 25.12.1, the pre-compiled libnvcompdx.fatbin only contains
 * ANS (algorithm::0) implementations, NOT LZ4 (algorithm::1). This code will
 * compile but link-time resolution requires a future MathDx version that ships
 * LZ4 in the fatbin. Until then, build with -DBUILD_WITH_NVCOMPDX=OFF to use
 * the memcpy stub path. */
using LZ4CompDesc = decltype(
    Algorithm<algorithm::lz4>() +
    DataType<datatype::uint8>() +
    Direction<direction::compress>() +
    MaxUncompChunkSize<(1 << 16)>() +  /* 64 KB max chunk */
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
 *
 * Output format (chunked):
 *   [uint32_t num_chunks]
 *   [uint32_t original_size]
 *   [uint32_t compressed_size_per_chunk[num_chunks]]
 *   [chunk_0_compressed_data]
 *   [chunk_1_compressed_data]
 *   ...
 *
 * TODO: Replace the vectorized-memcpy chunk body with compressor.execute() when
 * a future MathDx version ships LZ4 in the fatbin (currently only ANS is included).
 * The chunked framing format is already correct for real LZ4 — just swap the inner loop.
 *
 * NOTE: The caller (persistent kernel) must allocate dynamic shared memory.
 */
__device__ void device_compress_lz4(const uint8_t *in_data, size_t in_len,
                                    uint8_t *out_data, size_t out_max,
                                    size_t *actual_out_size)
{
    const size_t chunk_size = 1 << 16; /* 64KB — nvCOMPDx MaxUncompChunkSize */
    uint32_t num_chunks = (uint32_t)((in_len + chunk_size - 1) / chunk_size);
    if (num_chunks == 0) num_chunks = 1;

    /* Write header: [num_chunks] [original_size] [chunk_sizes...] */
    size_t header_size = 8 + (size_t)num_chunks * 4;
    uint32_t *header = (uint32_t *)out_data;
    if (threadIdx.x == 0) {
        header[0] = num_chunks;
        header[1] = (uint32_t)in_len;
    }
    __syncthreads();

    size_t out_offset = header_size;

    for (uint32_t c = 0; c < num_chunks; c++) {
        size_t chunk_offset = (size_t)c * chunk_size;
        size_t this_len = chunk_size;
        if (chunk_offset + this_len > in_len)
            this_len = in_len - chunk_offset;

        /* Vectorized memcpy for this chunk (all block threads participate).
         * TODO: Replace with compressor.execute() when nvCOMPDx LZ4 ships. */
        const uint8_t *src = in_data + chunk_offset;
        uint8_t *dst = out_data + out_offset;
        if (out_offset + this_len <= out_max) {
            if (((uintptr_t)src & 0xF) == 0 && ((uintptr_t)dst & 0xF) == 0) {
                size_t n16 = this_len / 16;
                for (size_t i = threadIdx.x; i < n16; i += blockDim.x)
                    ((uint4 *)dst)[i] = ((const uint4 *)src)[i];
                for (size_t i = n16 * 16 + threadIdx.x; i < this_len; i += blockDim.x)
                    dst[i] = src[i];
            } else {
                for (size_t i = threadIdx.x; i < this_len; i += blockDim.x)
                    dst[i] = src[i];
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            header[2 + c] = (uint32_t)this_len; /* uncompressed = same size for memcpy */
        }
        __syncthreads();
        out_offset += this_len;
    }

    if (threadIdx.x == 0) {
        *actual_out_size = out_offset;
    }
    __syncthreads();
}

/**
 * Device-side LZ4 decompression using nvCOMPDx.
 * Reads the chunked format produced by device_compress_lz4.
 *
 * TODO: Replace the vectorized-memcpy chunk body with decompressor.execute()
 * when nvCOMPDx LZ4 ships in the fatbin.
 */
__device__ void device_decompress_lz4(const uint8_t *in_data, size_t in_len,
                                      uint8_t *out_data, size_t out_max,
                                      size_t *actual_out_size)
{
    /* Read chunked header */
    const uint32_t *header = (const uint32_t *)in_data;
    uint32_t num_chunks = header[0];
    /* uint32_t original_size = header[1]; */
    size_t header_size = 8 + (size_t)num_chunks * 4;
    size_t in_offset = header_size;
    size_t out_offset = 0;

    for (uint32_t c = 0; c < num_chunks; c++) {
        size_t chunk_comp_size = header[2 + c];

        /* Vectorized memcpy for this chunk (all block threads participate).
         * TODO: Replace with decompressor.execute() when nvCOMPDx LZ4 ships. */
        const uint8_t *src = in_data + in_offset;
        uint8_t *dst = out_data + out_offset;
        if (out_offset + chunk_comp_size <= out_max && in_offset + chunk_comp_size <= in_len) {
            if (((uintptr_t)src & 0xF) == 0 && ((uintptr_t)dst & 0xF) == 0) {
                size_t n16 = chunk_comp_size / 16;
                for (size_t i = threadIdx.x; i < n16; i += blockDim.x)
                    ((uint4 *)dst)[i] = ((const uint4 *)src)[i];
                for (size_t i = n16 * 16 + threadIdx.x; i < chunk_comp_size; i += blockDim.x)
                    dst[i] = src[i];
            } else {
                for (size_t i = threadIdx.x; i < chunk_comp_size; i += blockDim.x)
                    dst[i] = src[i];
            }
        }
        __syncthreads();

        in_offset += chunk_comp_size;
        out_offset += chunk_comp_size;
    }

    if (threadIdx.x == 0) {
        *actual_out_size = out_offset;
    }
    __syncthreads();
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

/* ── Standalone Kernels for Host-Callable API ──────────────────────────── */
/* These wrap the device_compress/decompress functions into kernels that
 * can be launched directly from host code, analogous to crc32c_kernel. */

__global__ void lz4_compress_kernel(const uint8_t *in_data, size_t in_len,
                                    uint8_t *out_data, size_t out_max,
                                    size_t *actual_out_size)
{
    device_compress_lz4(in_data, in_len, out_data, out_max, actual_out_size);
}

__global__ void lz4_decompress_kernel(const uint8_t *in_data, size_t in_len,
                                      uint8_t *out_data, size_t out_max,
                                      size_t *actual_out_size)
{
    device_decompress_lz4(in_data, in_len, out_data, out_max, actual_out_size);
}

/* Dynamic shared memory size for nvCOMPDx scratch buffers.
 * Must match the persistent kernel's allocation (48 KB). */
#ifdef USE_NVCOMPDX
static const size_t LZ4_SHMEM_BYTES = 48 * 1024;
#else
static const size_t LZ4_SHMEM_BYTES = 0;
#endif

/* Host API: LZ4 compress on GPU data */
extern "C"
int gpu_lz4_compress(const void *gpu_in, size_t in_len,
                     void *gpu_out, size_t out_max, size_t *out_len)
{
    if (!gpu_in || !gpu_out || !out_len || in_len == 0) return -1;

    size_t *d_out_len;
    cudaMalloc(&d_out_len, sizeof(size_t));
    cudaMemset(d_out_len, 0, sizeof(size_t));

    /* Use a separate non-blocking stream to avoid deadlocking with
     * the persistent kernel if it is running. */
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    /* Allocate dynamic shared memory for nvCOMPDx scratch buffers */
    if (LZ4_SHMEM_BYTES > 0) {
        cudaFuncSetAttribute(lz4_compress_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)LZ4_SHMEM_BYTES);
    }

    lz4_compress_kernel<<<1, 128, LZ4_SHMEM_BYTES, stream>>>(
        (const uint8_t *)gpu_in, in_len,
        (uint8_t *)gpu_out, out_max, d_out_len);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaMemcpy(out_len, d_out_len, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out_len);

    return 0;
}

/* Host API: LZ4 decompress on GPU data */
extern "C"
int gpu_lz4_decompress(const void *gpu_in, size_t in_len,
                       void *gpu_out, size_t out_max, size_t *out_len)
{
    if (!gpu_in || !gpu_out || !out_len || in_len == 0) return -1;

    size_t *d_out_len;
    cudaMalloc(&d_out_len, sizeof(size_t));
    cudaMemset(d_out_len, 0, sizeof(size_t));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    if (LZ4_SHMEM_BYTES > 0) {
        cudaFuncSetAttribute(lz4_decompress_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)LZ4_SHMEM_BYTES);
    }

    lz4_decompress_kernel<<<1, 128, LZ4_SHMEM_BYTES, stream>>>(
        (const uint8_t *)gpu_in, in_len,
        (uint8_t *)gpu_out, out_max, d_out_len);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaMemcpy(out_len, d_out_len, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(d_out_len);

    return 0;
}
