/**
 * gpu_engine.h — Persistent CUDA Kernel Engine Public API
 *
 * This is the main interface for the GPU storage acceleration engine.
 * The engine launches a persistent CUDA kernel at init time that runs
 * for the lifetime of the process, polling a shared-memory work queue.
 *
 * Thread safety: gpu_engine_submit() and gpu_engine_poll() are thread-safe
 * and can be called from multiple CPU threads (e.g., Argobots ULTs).
 *
 * IMPORTANT: While the engine is running, do NOT call cudaFree() or any
 * CUDA API that triggers implicit device synchronization. These will
 * deadlock because the persistent kernel never returns. Use gpu_mem_free()
 * only after gpu_engine_fini(). For dynamic GPU memory during engine
 * lifetime, use cudaMallocAsync/cudaFreeAsync on a separate stream.
 */
#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Work queue capacity (must be power of 2) ──────────────────────────── */
#define GPU_QUEUE_SIZE      4096
#define GPU_QUEUE_MASK      (GPU_QUEUE_SIZE - 1)

/* ── Operation types ───────────────────────────────────────────────────── */
typedef enum {
    GPU_OP_INVALID      = 0,   /* Slot is empty/invalid */
    GPU_OP_NOP          = 1,   /* No-op (for testing) */
    GPU_OP_CRC32C       = 2,   /* CRC32C checksum */
    GPU_OP_SHA256       = 3,   /* SHA256 hash */
    GPU_OP_EC_ENCODE    = 4,   /* EC parity generation (Encode) */
    GPU_OP_EC_DECODE    = 5,   /* EC data reconstruction (Decode) */
    GPU_OP_COMPRESS_LZ4 = 6,   /* Compress data using LZ4 */
    GPU_OP_DECOMPRESS_LZ4 = 7, /* Decompress data using LZ4 */
    GPU_OP_MAX
} gpu_op_type_t;

/* ── Result status ─────────────────────────────────────────────────────── */
typedef enum {
    GPU_RESULT_PENDING  = 0,   /* Not yet processed */
    GPU_RESULT_READY    = 1,   /* Result available */
} gpu_result_status_t;

/* ── Work item (submitted by CPU, consumed by GPU) ─────────────────────── */
typedef struct {
    gpu_op_type_t       op_type;        /* Operation to perform */
    volatile uint32_t   status;         /* 0=pending, written by CPU; read by GPU */
    void               *data_ptr;       /* GPU memory pointer to data */
    size_t              data_len;       /* Data length in bytes */

    /* Checksum output (DEPRECATED: results now in gpu_result_t) */
    uint32_t            crc32c_result;  /* CRC32C result (output) */
    uint8_t             sha256_result[32]; /* SHA256 result (output) */

    /* Compression / Decompression fields */
    void               *comp_out_ptr;   /* GPU memory pointer to output buffer */
    size_t              comp_max_size;  /* Maximum size of output buffer */
    size_t              actual_comp_size; /* Actual output size after operation */

    /* EC parity fields */
    void               *ec_ptrs[16];    /* Inline array of stripe data pointers (GPU mem) */
    void               *parity_ptrs[4]; /* Inline array of output parity buffers (GPU mem) */
    uint32_t            stripe_cnt;     /* Number of data stripes (k) */
    uint32_t            parity_cnt;     /* Number of parity stripes (p) */
    size_t              cell_size;      /* Size of each stripe cell in bytes */

    /* EC decode fields (used with GPU_OP_EC_DECODE) */
    uint32_t            failed_idx[2];  /* Indices of failed data stripes (max 2) */
    uint32_t            failed_cnt;     /* Number of failed stripes (1 or 2) */
} gpu_work_item_t;

/* ── Result slot (written by GPU, read by CPU) ─────────────────────────── */
typedef struct {
    volatile int32_t            error_code;       /* 0 = success, negative = error */
    volatile gpu_result_status_t status;           /* PENDING or READY */
    size_t                      actual_comp_size;  /* Actual output size after compression/decompression */
    uint32_t                    crc32c_result;     /* CRC32C output */
    uint8_t                     sha256_result[32]; /* SHA256 output (32 bytes) */
    uint64_t                    ticket;            /* Generation counter: matches submit ticket */
} gpu_result_t;

/* ── Engine Metrics ────────────────────────────────────────────────────── */
typedef struct {
    uint64_t queue_full_events;
    uint64_t items_submitted;
    uint64_t items_completed;
    uint64_t total_poll_spins;
} gpu_engine_metrics_t;

/* ── Opaque engine handle ──────────────────────────────────────────────── */
typedef struct gpu_engine gpu_engine_t;

/**
 * Initialize the GPU engine.
 * Allocates work queue in pinned memory, launches persistent CUDA kernel.
 *
 * @param engine_out  Pointer to receive the engine handle
 * @return 0 on success, negative error code on failure
 */
int gpu_engine_init(gpu_engine_t **engine_out);

/**
 * Shut down the GPU engine.
 * Signals the persistent kernel to exit, waits for completion, frees resources.
 *
 * @param engine  Engine handle (may be NULL)
 */
void gpu_engine_fini(gpu_engine_t *engine);

/**
 * Submit a work item to the GPU engine.
 * This is lock-free and can be called from multiple threads.
 *
 * @param engine     Engine handle
 * @param item       Work item to submit (copied into the queue)
 * @param ticket_out Receives a ticket number to poll for completion
 * @return 0 on success, -1 if queue is full
 */
int gpu_engine_submit(gpu_engine_t *engine, const gpu_work_item_t *item,
                      uint64_t *ticket_out);

/**
 * Poll for a work item's completion.
 *
 * @param engine   Engine handle
 * @param ticket   Ticket from gpu_engine_submit()
 * @param result   Receives the result (may be NULL if only checking status)
 * @return 1 if result is ready, 0 if still pending, negative on error
 */
int gpu_engine_poll(gpu_engine_t *engine, uint64_t ticket,
                    gpu_result_t *result);

/**
 * Submit and wait (blocking) for completion.
 * Convenience wrapper around submit + spin-poll.
 *
 * @param engine  Engine handle
 * @param item    Work item
 * @param result  Receives the result
 * @return 0 on success, negative on error
 */
int gpu_engine_submit_and_wait(gpu_engine_t *engine, gpu_work_item_t *item,
                               gpu_result_t *result);

/**
 * Retrieve current engine metrics.
 *
 * @param engine   Engine handle
 * @param metrics  Pointer to receive metrics
 */
void gpu_engine_get_metrics(gpu_engine_t *engine, gpu_engine_metrics_t *metrics);

/**
 * Register host memory for zero-copy GPU access.
 *
 * @param ptr   Host memory pointer (page-aligned)
 * @param size  Size in bytes
 * @return 0 on success, negative error code on failure
 */
int gpu_engine_register_host_mem(void *ptr, size_t size);

/**
 * Unregister host memory previously registered via gpu_engine_register_host_mem.
 *
 * @param ptr   Host memory pointer
 * @return 0 on success, negative error code on failure
 */
int gpu_engine_unregister_host_mem(void *ptr);

/**
 * Get device pointer for registered host memory.
 *
 * @param host_ptr   Registered host memory pointer
 * @param dev_ptr    Out parameter to receive the device pointer
 * @return 0 on success, negative error code on failure
 */
int gpu_engine_get_device_pointer(void *host_ptr, void **dev_ptr);

/**
 * Get the work item from the queue (for reading results like CRC).
 * Only valid after gpu_engine_poll() returns 1.
 *
 * DEPRECATED: All results (CRC32C, SHA256, compression size) are now
 * returned in gpu_result_t via gpu_engine_poll(). This function is
 * retained for backward compatibility only. The returned pointer
 * becomes invalid as soon as the slot is reused by a new submission.
 *
 * @param engine  Engine handle
 * @param ticket  Ticket number
 * @return Pointer to the work item in the queue (read-only), or NULL
 */
const gpu_work_item_t *gpu_engine_get_item(gpu_engine_t *engine,
                                            uint64_t ticket);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ENGINE_H */
