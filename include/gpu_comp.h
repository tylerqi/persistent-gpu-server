/**
 * gpu_comp.h — GPU Compression API (LZ4)
 *
 * Host-callable LZ4 compression/decompression functions for use outside
 * the persistent kernel. The persistent kernel calls internal device
 * functions directly via gpu_comp_internal.h.
 *
 * When USE_NVCOMPDX is defined: uses real nvCOMPDx LZ4 compression.
 * When USE_NVCOMPDX is NOT defined: memcpy stub (1:1 ratio).
 */
#ifndef GPU_COMP_H
#define GPU_COMP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compress data using LZ4 on GPU. Synchronous — blocks until done.
 *
 * @param gpu_in    Pointer to input data in GPU memory
 * @param in_len    Length of input data in bytes
 * @param gpu_out   Pointer to output buffer in GPU memory
 * @param out_max   Maximum size of output buffer in bytes
 * @param out_len   Receives the actual compressed size (host pointer)
 * @return 0 on success, -1 on failure
 */
int gpu_lz4_compress(const void *gpu_in, size_t in_len,
                     void *gpu_out, size_t out_max, size_t *out_len);

/**
 * Decompress LZ4-compressed data on GPU. Synchronous — blocks until done.
 *
 * @param gpu_in    Pointer to compressed input data in GPU memory
 * @param in_len    Length of compressed input in bytes
 * @param gpu_out   Pointer to output buffer in GPU memory
 * @param out_max   Maximum size of output buffer in bytes
 * @param out_len   Receives the actual decompressed size (host pointer)
 * @return 0 on success, -1 on failure
 */
int gpu_lz4_decompress(const void *gpu_in, size_t in_len,
                       void *gpu_out, size_t out_max, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* GPU_COMP_H */
