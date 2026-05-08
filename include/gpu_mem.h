/**
 * gpu_mem.h — GPU Memory Utilities
 *
 * Convenience wrappers around cudaMalloc/cudaFree for host code.
 * These call CUDA APIs directly and will DEADLOCK if used while
 * the persistent kernel is running. Use gpu_mempool for dynamic
 * allocation during engine lifetime.
 */
#ifndef GPU_MEM_H
#define GPU_MEM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate GPU device memory.
 * WARNING: Triggers implicit device sync — do NOT call while engine is running.
 *
 * @param size  Number of bytes to allocate
 * @return Pointer to device memory, or NULL on failure
 */
void *gpu_mem_alloc(size_t size);

/**
 * Free GPU device memory.
 * WARNING: Triggers implicit device sync — do NOT call while engine is running.
 *
 * @param ptr  Pointer previously returned by gpu_mem_alloc (may be NULL)
 */
void gpu_mem_free(void *ptr);

/**
 * Copy data from host to device.
 *
 * @param dst   Device pointer
 * @param src   Host pointer
 * @param size  Number of bytes
 * @return 0 on success, -1 on failure
 */
int gpu_mem_copy_h2d(void *dst, const void *src, size_t size);

/**
 * Copy data from device to host.
 *
 * @param dst   Host pointer
 * @param src   Device pointer
 * @param size  Number of bytes
 * @return 0 on success, -1 on failure
 */
int gpu_mem_copy_d2h(void *dst, const void *src, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* GPU_MEM_H */
