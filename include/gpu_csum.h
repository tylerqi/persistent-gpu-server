/**
 * gpu_csum.h — GPU Checksum API (CRC32C, SHA256)
 *
 * Standalone host-callable checksum functions for testing/benchmarking
 * outside of the persistent kernel. The persistent kernel calls internal
 * device functions directly.
 */
#ifndef GPU_CSUM_H
#define GPU_CSUM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute CRC32C on GPU memory. Synchronous — blocks until done.
 *
 * @param gpu_data  Pointer to data in GPU memory
 * @param len       Length in bytes
 * @param crc_out   Receives the CRC32C value (host pointer)
 * @return 0 on success
 */
int gpu_crc32c(const void *gpu_data, size_t len, uint32_t *crc_out);

/**
 * Compute SHA256 on GPU memory. Synchronous — blocks until done.
 *
 * @param gpu_data    Pointer to data in GPU memory
 * @param len         Length in bytes
 * @param hash_out    Receives the 32-byte SHA256 hash (host pointer)
 * @return 0 on success
 */
int gpu_sha256(const void *gpu_data, size_t len, uint8_t hash_out[32]);

/**
 * Initialize GPU checksum subsystem.
 * Must be called before gpu_engine_init() to populate __constant__ CRC tables.
 * Safe to call multiple times (idempotent).
 */
void gpu_csum_init(void);

/**
 * CPU reference CRC32C for verification.
 * Uses a simple software implementation (not ISA-L).
 */
uint32_t cpu_crc32c(const void *data, size_t len);

/**
 * CPU reference SHA256 for verification.
 * Uses a simple software implementation.
 */
void cpu_sha256(const void *data, size_t len, uint8_t hash_out[32]);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CSUM_H */
