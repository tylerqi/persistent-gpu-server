/**
 * gpu_mempool.h — Lock-Free GPU Memory Pool
 *
 * Provides a CPU-side lock-free allocator for managing fixed-size blocks
 * of GPU memory. Useful for allocating device memory while the persistent
 * kernel is running without triggering implicit CUDA driver synchronization
 * (which would otherwise cause a deadlock).
 */
#ifndef GPU_MEMPOOL_H
#define GPU_MEMPOOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_mempool gpu_mempool_t;

/**
 * Create a new GPU memory pool.
 * Allocates `block_size * block_count` bytes of device memory.
 *
 * @param pool_out    Pointer to receive the memory pool handle
 * @param block_size  Size of each allocation block in bytes
 * @param block_count Total number of blocks to manage
 * @return 0 on success, negative on error
 */
int gpu_mempool_create(gpu_mempool_t **pool_out, size_t block_size, uint32_t block_count);

/**
 * Destroy the GPU memory pool and free the underlying device memory.
 *
 * @param pool The pool handle
 */
void gpu_mempool_destroy(gpu_mempool_t *pool);

/**
 * Allocate a block of device memory from the pool.
 * Thread-safe and lock-free.
 *
 * @param pool The pool handle
 * @return Pointer to device memory, or NULL if pool is empty
 */
void *gpu_mempool_alloc(gpu_mempool_t *pool);

/**
 * Return a block of device memory to the pool.
 * Thread-safe and lock-free.
 *
 * @param pool The pool handle
 * @param ptr  Pointer previously returned by gpu_mempool_alloc
 */
void gpu_mempool_free(gpu_mempool_t *pool, void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* GPU_MEMPOOL_H */
