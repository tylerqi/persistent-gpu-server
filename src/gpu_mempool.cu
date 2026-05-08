/**
 * gpu_mempool.cu — Lock-Free GPU Memory Pool Implementation
 */
#include "gpu_mempool.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

struct gpu_mempool {
    void *gpu_base;
    size_t block_size;
    uint32_t total_blocks;

    /* Lock-free stack of free block indices */
    uint32_t *next_idx_array;
    volatile uint64_t head; /* [32-bit generation counter | 32-bit index] */

    /* Bitmap tracking allocated blocks for double-free detection (M-2) */
    volatile uint32_t *alloc_bitmap; /* 1 bit per block: 1=allocated, 0=free */
};

#define MEMPOOL_NULL_IDX 0xFFFFFFFF

int gpu_mempool_create(gpu_mempool_t **pool_out, size_t block_size, uint32_t block_count)
{
    if (!pool_out || block_size == 0 || block_count == 0) return -1;

    gpu_mempool_t *pool = (gpu_mempool_t *)calloc(1, sizeof(gpu_mempool_t));
    if (!pool) return -1;

    pool->block_size = block_size;
    pool->total_blocks = block_count;

    cudaError_t err = cudaMalloc(&pool->gpu_base, block_size * block_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_mempool_create: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(pool);
        return -1;
    }

    pool->next_idx_array = (uint32_t *)malloc(sizeof(uint32_t) * block_count);
    if (!pool->next_idx_array) {
        cudaFree(pool->gpu_base);
        free(pool);
        return -1;
    }

    /* Bitmap: ceil(block_count / 32) uint32_t words, all bits 0 (free) */
    uint32_t bitmap_words = (block_count + 31) / 32;
    pool->alloc_bitmap = (volatile uint32_t *)calloc(bitmap_words, sizeof(uint32_t));
    if (!pool->alloc_bitmap) {
        free(pool->next_idx_array);
        cudaFree(pool->gpu_base);
        free(pool);
        return -1;
    }

    /* Initialize free list: 0 -> 1 -> 2 -> ... -> NULL */
    for (uint32_t i = 0; i < block_count - 1; i++) {
        pool->next_idx_array[i] = i + 1;
    }
    pool->next_idx_array[block_count - 1] = MEMPOOL_NULL_IDX;

    /* Head starts with generation 0, index 0 */
    pool->head = 0;

    *pool_out = pool;
    return 0;
}

void gpu_mempool_destroy(gpu_mempool_t *pool)
{
    if (!pool) return;
    if (pool->gpu_base) cudaFree(pool->gpu_base);
    if (pool->next_idx_array) free(pool->next_idx_array);
    if (pool->alloc_bitmap) free((void *)pool->alloc_bitmap);
    free(pool);
}

void *gpu_mempool_alloc(gpu_mempool_t *pool)
{
    if (!pool) return NULL;

    uint64_t old_head = pool->head;
    while (1) {
        uint32_t idx = (uint32_t)(old_head & 0xFFFFFFFF);
        if (idx == MEMPOOL_NULL_IDX) {
            return NULL; /* Out of memory */
        }

        uint32_t next_idx = pool->next_idx_array[idx];
        uint32_t next_gen = (uint32_t)(old_head >> 32) + 1;
        uint64_t new_head = ((uint64_t)next_gen << 32) | next_idx;

        if (__sync_bool_compare_and_swap(&pool->head, old_head, new_head)) {
            /* Mark block as allocated in bitmap */
            uint32_t word = idx / 32;
            uint32_t bit = 1u << (idx % 32);
            __sync_fetch_and_or(&pool->alloc_bitmap[word], bit);
            return (char *)pool->gpu_base + ((size_t)idx * pool->block_size);
        }
        old_head = pool->head;
    }
}

void gpu_mempool_free(gpu_mempool_t *pool, void *ptr)
{
    if (!pool || !ptr) return;

    size_t offset = (char *)ptr - (char *)pool->gpu_base;
    uint32_t idx = (uint32_t)(offset / pool->block_size);

    /* Basic sanity check to avoid corrupting the list */
    if (idx >= pool->total_blocks || offset % pool->block_size != 0) {
        fprintf(stderr, "gpu_mempool_free: invalid pointer (out of range)\n");
        return;
    }

    /* Double-free detection (M-2): clear the allocated bit atomically.
     * If the bit was already 0, this is a double-free. */
    uint32_t word = idx / 32;
    uint32_t bit = 1u << (idx % 32);
    uint32_t old_bits = __sync_fetch_and_and(&pool->alloc_bitmap[word], ~bit);
    if (!(old_bits & bit)) {
        fprintf(stderr, "gpu_mempool_free: DOUBLE FREE detected for block %u!\n", idx);
        return; /* Don't corrupt the free list */
    }

    uint64_t old_head = pool->head;
    while (1) {
        pool->next_idx_array[idx] = (uint32_t)(old_head & 0xFFFFFFFF);

        uint32_t next_gen = (uint32_t)(old_head >> 32) + 1;
        uint64_t new_head = ((uint64_t)next_gen << 32) | idx;

        if (__sync_bool_compare_and_swap(&pool->head, old_head, new_head)) {
            break;
        }
        old_head = pool->head;
    }
}
