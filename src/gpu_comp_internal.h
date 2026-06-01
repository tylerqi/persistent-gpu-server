/**
 * gpu_comp.h — Internal device-side operation declarations
 *
 * Declares __device__ functions for LZ4 compression/decompression and
 * EC parity generation. These are called by the persistent kernel in
 * gpu_engine.cu. Not part of the public API.
 */
#ifndef GPU_COMP_INTERNAL_H
#define GPU_COMP_INTERNAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

__device__ int device_compress_lz4(const uint8_t *in_data, size_t in_len,
                                   uint8_t *out_data, size_t out_max,
                                   size_t *actual_out_size);

__device__ int device_decompress_lz4(const uint8_t *in_data, size_t in_len,
                                     uint8_t *out_data, size_t out_max,
                                     size_t *actual_out_size);

__device__ void device_ec_encode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size, uint32_t ec_mode,
                                 const uint8_t *encode_matrix);

__device__ void device_ec_decode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size,
                                 const uint32_t *failed_idx, uint32_t failed_cnt,
                                 uint32_t ec_mode,
                                 const uint8_t *decode_matrix);

#ifdef __cplusplus
}
#endif

#endif
