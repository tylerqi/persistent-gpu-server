/**
 * gpu_engine.cu — Persistent CUDA Kernel Engine Implementation
 *
 * Launches a persistent kernel at init time. The kernel runs forever,
 * polling a pinned-memory work queue for items submitted by CPU threads.
 */
#include "gpu_engine.h"
#include "gpu_error.h"
#include "gpu_comp.h"
#include "gpu_csum.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>

/* ── Forward declarations of device-side compute functions ─────────────── */
__device__ uint32_t device_crc32c(const uint8_t *data, size_t len);
__device__ void     device_sha256(const uint8_t *data, size_t len, uint8_t *out);

/* ── Internal engine structure ─────────────────────────────────────────── */
struct gpu_engine {
    /* Pinned (host+device visible) work queue */
    gpu_work_item_t    *queue;          /* GPU_QUEUE_SIZE items */
    gpu_result_t       *results;        /* GPU_QUEUE_SIZE result slots */

    /* Queue head: next slot to be written by producer (CPU) */
    volatile uint64_t  *head;           /* pinned memory */
    /* Queue tail: next slot to be consumed by consumer (GPU) */
    volatile uint64_t  *tail;           /* pinned memory */
    /* Shutdown flag */
    volatile int       *shutdown;       /* pinned memory */

    /* Metrics */
    volatile uint64_t   queue_full_events;
    volatile uint64_t   items_submitted;
    volatile uint64_t   items_completed;
    volatile uint64_t   total_poll_spins;

    /* CUDA stream for the persistent kernel */
    cudaStream_t        stream;
};

/* ── Persistent kernel ─────────────────────────────────────────────────── */
__global__ void persistent_kernel(
    gpu_work_item_t    *queue,
    gpu_result_t       *results,
    volatile uint64_t  *head,
    volatile uint64_t  *tail,
    volatile int       *shutdown)
{
    /* Each block cooperatively processes work items.
     * Only thread 0 of each block polls for work; all threads in the
     * block participate in the compute via shared memory coordination.
     * For simplicity in v1: only thread 0 in each block does work. */
    const int tid = threadIdx.x;

    __shared__ uint64_t shared_tail;

    while (!(*shutdown)) {
        /* ── Leader thread: try to claim a work item ──────────────── */
        if (tid == 0) {
            uint64_t cur_head = *head;
            uint64_t my_tail = *tail;

            if (my_tail >= cur_head) {
                shared_tail = (uint64_t)-1;
            } else {
                unsigned long long old_tail = atomicCAS((unsigned long long *)tail, 
                                                        (unsigned long long)my_tail, 
                                                        (unsigned long long)(my_tail + 1));
                if (old_tail != my_tail) {
                    shared_tail = (uint64_t)-1;
                } else {
                    shared_tail = my_tail;
                }
            }
        }
        __syncthreads();

        uint64_t my_tail = shared_tail;
        if (my_tail == (uint64_t)-1) {
            /* No work available or failed claim — progressive backoff sleep */
            if (tid == 0) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(1000); /* 1µs — reduces PCIe coherence storm */
#endif
            }
            __syncthreads();
            continue;
        }

        /* ── Got a work item ──────────────────────────────────────── */
        uint32_t slot = (uint32_t)(my_tail & GPU_QUEUE_MASK);
        volatile gpu_work_item_t *item = &queue[slot];
        volatile gpu_result_t *result = &results[slot];

        /* Wait until CPU has fully written the item */
        if (tid == 0) {
            int wait_spins = 0;
            while (item->op_type == GPU_OP_INVALID && !(*shutdown)) {
#if __CUDA_ARCH__ >= 700
                /* Progressive backoff: 100ns → 1µs as item takes longer to arrive */
                __nanosleep(wait_spins < 100 ? 100 : 1000);
#endif
                wait_spins++;
            }
        }
        __syncthreads();
        if (*shutdown) break;

        /* Thread 0 copies the item into shared memory for block-wide consistency */
        __shared__ gpu_work_item_t s_item;
        if (tid == 0) {
            /* Copy the struct */
            s_item = *(gpu_work_item_t*)item;
            result->error_code = GPU_SUCCESS;
        }
        __syncthreads();

        /* ── Dispatch ─────────────────────────────────────────────── */
        switch (s_item.op_type) {
        case GPU_OP_NOP:
            /* Nothing to do */
            break;

        case GPU_OP_CRC32C:
            if (tid == 0) {
                if (s_item.data_ptr == NULL || s_item.data_len == 0) {
                    result->error_code = GPU_ERR_INVAL;
                } else {
                    /* Write CRC result to result slot (unified API) */
                    result->crc32c_result = device_crc32c(
                        (const uint8_t *)s_item.data_ptr, s_item.data_len);
                }
            }
            break;

        case GPU_OP_SHA256:
            if (tid == 0) {
                if (s_item.data_ptr == NULL || s_item.data_len == 0) {
                    result->error_code = GPU_ERR_INVAL;
                } else {
                    /* Write SHA256 result via temp buffer to avoid stripping volatile (H-5) */
                    uint8_t sha_tmp[32];
                    device_sha256((const uint8_t *)s_item.data_ptr,
                                  s_item.data_len, sha_tmp);
                    for (int si = 0; si < 32; si++)
                        result->sha256_result[si] = sha_tmp[si];
                }
            }
            break;

        case GPU_OP_EC_ENCODE:
            if (s_item.stripe_cnt == 0 || s_item.cell_size == 0 ||
                s_item.stripe_cnt > 16 || s_item.parity_cnt > 4) {
                if (tid == 0) result->error_code = GPU_ERR_INVAL;
            } else {
                device_ec_encode((void *const *)s_item.ec_ptrs,
                                 (void *const *)s_item.parity_ptrs,
                                 s_item.stripe_cnt, s_item.parity_cnt,
                                 s_item.cell_size);
            }
            break;

        case GPU_OP_EC_DECODE:
            /* EC decode (data reconstruction) is not yet implemented (C-2) */
            if (tid == 0) result->error_code = GPU_ERR_NOSYS;
            break;

        case GPU_OP_COMPRESS_LZ4:
            if (s_item.data_ptr == NULL || s_item.comp_out_ptr == NULL) {
                if (tid == 0) result->error_code = GPU_ERR_INVAL;
            } else {
                size_t actual_out = 0;
                device_compress_lz4((const uint8_t *)s_item.data_ptr, s_item.data_len,
                                    (uint8_t *)s_item.comp_out_ptr, s_item.comp_max_size,
                                    &actual_out);
                if (tid == 0) result->actual_comp_size = actual_out;
            }
            break;

        case GPU_OP_DECOMPRESS_LZ4:
            if (s_item.data_ptr == NULL || s_item.comp_out_ptr == NULL) {
                if (tid == 0) result->error_code = GPU_ERR_INVAL;
            } else {
                size_t actual_out = 0;
                device_decompress_lz4((const uint8_t *)s_item.data_ptr, s_item.data_len,
                                      (uint8_t *)s_item.comp_out_ptr, s_item.comp_max_size,
                                      &actual_out);
                if (tid == 0) result->actual_comp_size = actual_out;
            }
            break;

        default:
            if (tid == 0) result->error_code = GPU_ERR_INVAL;
            break;
        }
        __syncthreads();

        /* ── Mark result as ready (visible to CPU) ────────────────── */
        if (tid == 0) {
            result->ticket = my_tail; /* Generation counter */
            item->op_type = GPU_OP_INVALID; /* Free slot for next use */
            __threadfence_system(); /* Ensure all result fields visible before READY */
            result->status = GPU_RESULT_READY;
            __threadfence_system(); /* Ensure READY is visible to CPU */
        }
        __syncthreads();
    }
}

/* ── Host API ──────────────────────────────────────────────────────────── */

#define CUDA_CHECK(call) do {                                   \
    cudaError_t _err = (call);                                  \
    if (_err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(_err));  \
        goto init_cleanup;                                      \
    }                                                           \
} while(0)

int gpu_engine_init(gpu_engine_t **engine_out)
{
    if (!engine_out) return -1;

    /* Initialize subsystems (must happen before kernel launch) */
    gpu_csum_init();

    gpu_engine_t *eng = (gpu_engine_t *)calloc(1, sizeof(gpu_engine_t));
    if (!eng) return -1;

    /* Pre-declare ALL variables before first CUDA_CHECK to allow goto cleanup
     * in C++ (goto cannot jump over variable declarations with initializers). */
    int device = 0;
    cudaDeviceProp prop;
    int num_blocks = 0;
    int threads_per_block = 128;
    gpu_work_item_t *d_queue = NULL;
    gpu_result_t *d_results = NULL;
    volatile uint64_t *d_head = NULL, *d_tail = NULL;
    volatile int *d_shutdown = NULL;
    size_t shmem_bytes = 0;

    /* Allocate pinned memory (visible to both CPU and GPU) */
    CUDA_CHECK(cudaHostAlloc(&eng->queue, sizeof(gpu_work_item_t) * GPU_QUEUE_SIZE,
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&eng->results, sizeof(gpu_result_t) * GPU_QUEUE_SIZE,
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&eng->head, sizeof(uint64_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&eng->tail, sizeof(uint64_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void **)&eng->shutdown, sizeof(int),
                             cudaHostAllocMapped));

    /* Initialize */
    memset((void *)eng->queue, 0, sizeof(gpu_work_item_t) * GPU_QUEUE_SIZE);
    memset((void *)eng->results, 0, sizeof(gpu_result_t) * GPU_QUEUE_SIZE);
    *(uint64_t *)eng->head = 0;
    *(uint64_t *)eng->tail = 0;
    *(int *)eng->shutdown = 0;

    /* Create stream and launch persistent kernel */
    CUDA_CHECK(cudaStreamCreateWithFlags(&eng->stream, cudaStreamNonBlocking));

    /* Get device SM count for optimal launch */
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    /* Launch 1 block per SM — balances throughput vs. polling overhead (H-7).
     * Each block has 128 threads for cooperative EC/compress execution. */
    num_blocks = prop.multiProcessorCount;

    /* Get device pointers for mapped memory */
    CUDA_CHECK(cudaHostGetDevicePointer(&d_queue, eng->queue, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_results, eng->results, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((uint64_t **)&d_head, (void *)eng->head, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((uint64_t **)&d_tail, (void *)eng->tail, 0));
    CUDA_CHECK(cudaHostGetDevicePointer((int **)&d_shutdown, (void *)eng->shutdown, 0));

    /* Clear any residual CUDA errors before launch check */
    cudaGetLastError();

    /* Dynamic shared memory for nvCOMPDx compression scratch buffers.
     * Without nvCOMPDx, no extra shared memory is needed. */
#ifdef USE_NVCOMPDX
    /* Query shared memory requirements from nvCOMPDx at runtime.
     * We allocate enough for the larger of compress/decompress. */
    shmem_bytes = 48 * 1024;  /* 48 KB — conservative for LZ4 warp-level */
    /* Ensure the GPU can support this amount of dynamic shared memory */
    cudaFuncSetAttribute(persistent_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shmem_bytes);
#endif

    persistent_kernel<<<num_blocks, threads_per_block, shmem_bytes, eng->stream>>>(
        d_queue, d_results, d_head, d_tail, d_shutdown);

    /* Check for launch errors */
    {
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
            gpu_engine_fini(eng);
            return -1;
        }
    }

    *engine_out = eng;
    return 0;

init_cleanup:
    /* Resource cleanup on CUDA_CHECK failure (C-5) */
    gpu_engine_fini(eng);
    return -1;
}

void gpu_engine_fini(gpu_engine_t *engine)
{
    if (!engine) return;

    /* Signal shutdown */
    if (engine->shutdown) {
        *(int *)engine->shutdown = 1;
        __sync_synchronize();
    }

    /* Wait for kernel to exit */
    if (engine->stream) {
        cudaStreamSynchronize(engine->stream);
        cudaStreamDestroy(engine->stream);
    }

    /* Free pinned memory */
    if (engine->queue)    cudaFreeHost(engine->queue);
    if (engine->results)  cudaFreeHost(engine->results);
    if (engine->head)     cudaFreeHost((void *)engine->head);
    if (engine->tail)     cudaFreeHost((void *)engine->tail);
    if (engine->shutdown) cudaFreeHost((void *)engine->shutdown);

    free(engine);
}

int gpu_engine_submit(gpu_engine_t *engine, const gpu_work_item_t *item,
                      uint64_t *ticket_out)
{
    if (!engine || !item || !ticket_out) return GPU_ERR_INVAL;

    /* Validate op_type on CPU side before queue round-trip (L-6) */
    if (item->op_type <= GPU_OP_INVALID || item->op_type >= GPU_OP_MAX)
        return GPU_ERR_INVAL;

    /* CAS loop to atomically claim a slot only if there is room.
     * This avoids the old __sync_fetch_and_sub rollback race (BUG-1). */
    uint64_t ticket;
    while (1) {
        uint64_t cur_head = *(volatile uint64_t *)engine->head;
        uint64_t cur_tail = *(volatile uint64_t *)engine->tail;

        /* Check if queue is full */
        if (cur_head - cur_tail >= GPU_QUEUE_SIZE) {
            __sync_fetch_and_add(&engine->queue_full_events, 1);
            return GPU_ERR_QUEUE_FULL; /* Distinct from invalid args (M-3) */
        }

        /* Attempt to claim this slot */
        if (__sync_bool_compare_and_swap((uint64_t *)engine->head,
                                         cur_head, cur_head + 1)) {
            ticket = cur_head;
            break;
        }
        /* CAS failed — another thread claimed it, retry */
    }

    __sync_fetch_and_add(&engine->items_submitted, 1);

    uint32_t slot = (uint32_t)(ticket & GPU_QUEUE_MASK);

    /* Copy work item into the queue slot */
    engine->results[slot].status = GPU_RESULT_PENDING;
    engine->results[slot].error_code = 0;
    engine->results[slot].ticket = 0; /* Clear generation counter */

    /* Copy payload first, leaving op_type as INVALID */
    gpu_op_type_t real_op = item->op_type;
    gpu_work_item_t temp = *item;
    temp.op_type = GPU_OP_INVALID;
    memcpy((void *)&engine->queue[slot], &temp, sizeof(gpu_work_item_t));

    __sync_synchronize(); /* Ensure item is visible before GPU reads it */

    /* Signal GPU by writing op_type */
    engine->queue[slot].op_type = real_op;

    *ticket_out = ticket;
    return 0;
}

int gpu_engine_poll(gpu_engine_t *engine, uint64_t ticket,
                    gpu_result_t *result)
{
    if (!engine) return -1;

    uint32_t slot = (uint32_t)(ticket & GPU_QUEUE_MASK);
    volatile gpu_result_t *r = &engine->results[slot];

    if (r->status != GPU_RESULT_READY)
        return 0; /* Still pending */

    /* Acquire fence: ensure we read results AFTER seeing READY (CQ-4) */
    __sync_synchronize();

    /* Validate generation counter to prevent stale reads (BUG-2) */
    if (r->ticket != ticket)
        return 0; /* Slot was reused by a newer submission; not our result */

    if (result) {
        result->error_code = r->error_code;
        result->status = r->status;
        result->actual_comp_size = r->actual_comp_size;
        result->crc32c_result = r->crc32c_result;
        memcpy((void *)result->sha256_result,
               (const void *)r->sha256_result, 32);
        result->ticket = r->ticket;
    }

    /* Only count first poll per ticket — CAS prevents double-counting (H-1).
     * We mark the slot as "reaped" by clearing the status back to PENDING. */
    if (__sync_bool_compare_and_swap((volatile int32_t *)&r->status,
                                     (int32_t)GPU_RESULT_READY,
                                     (int32_t)GPU_RESULT_PENDING)) {
        __sync_fetch_and_add(&engine->items_completed, 1);
    }
    return 1; /* Ready */
}

int gpu_engine_submit_and_wait(gpu_engine_t *engine, gpu_work_item_t *item,
                               gpu_result_t *result)
{
    uint64_t ticket;
    int rc = gpu_engine_submit(engine, item, &ticket);
    if (rc != 0) return rc;

    /* Spin-poll for completion with adaptive backoff:
     *   Phase 1 (spins 0-1000):   pure spin with CPU PAUSE hint
     *   Phase 2 (spins 1000-5000): sched_yield() to let other threads run
     *   Phase 3 (spins 5000+):     usleep(10) for long-running ops (1MB CRC, EC)
     */
    gpu_result_t res;
    int spins = 0;
    while (gpu_engine_poll(engine, ticket, &res) == 0) {
        if (spins < 1000) {
            /* Phase 1: tight spin with CPU pause hint to reduce power/contention */
#if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        } else if (spins < 5000) {
            /* Phase 2: yield to OS scheduler */
            sched_yield();
        } else {
            /* Phase 3: sleep for long-running operations */
            usleep(10);
        }

        if (++spins > 6000000) { /* ~60 sec timeout */
            fprintf(stderr, "gpu_engine_submit_and_wait: timeout polling ticket %lu\n", ticket);
            if (result) result->error_code = GPU_ERR_TIMEOUT;
            __sync_fetch_and_add(&engine->total_poll_spins, spins);
            return GPU_ERR_TIMEOUT;
        }
    }

    __sync_fetch_and_add(&engine->total_poll_spins, spins);

    if (result) *result = res;
    return res.error_code;
}

void gpu_engine_get_metrics(gpu_engine_t *engine, gpu_engine_metrics_t *metrics)
{
    if (!engine || !metrics) return;
    metrics->queue_full_events = engine->queue_full_events;
    metrics->items_submitted   = engine->items_submitted;
    metrics->items_completed   = engine->items_completed;
    metrics->total_poll_spins  = engine->total_poll_spins;
}

const gpu_work_item_t *gpu_engine_get_item(gpu_engine_t *engine,
                                            uint64_t ticket)
{
    if (!engine) return NULL;
    uint32_t slot = (uint32_t)(ticket & GPU_QUEUE_MASK);
    return &engine->queue[slot];
}

/* ── Error string helper ───────────────────────────────────────────────── */
const char *gpu_error_string(int32_t code)
{
    switch (code) {
    case GPU_SUCCESS:        return "success";
    case GPU_ERR_INVAL:      return "invalid argument";
    case GPU_ERR_NOMEM:      return "out of memory";
    case GPU_ERR_IO:         return "I/O error";
    case GPU_ERR_OVERFLOW:   return "buffer overflow";
    case GPU_ERR_CSUM:       return "checksum mismatch";
    case GPU_ERR_TIMEOUT:    return "operation timed out";
    case GPU_ERR_QUEUE_FULL: return "work queue full";
    case GPU_ERR_NOSYS:      return "not implemented";
    default:                 return "unknown error";
    }
}
