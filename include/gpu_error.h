/**
 * gpu_error.h — GPU-side Error Handling
 *
 * Non-fatal error macros for use inside CUDA kernels.
 * Errors are written to pinned shared memory and do NOT abort the GPU.
 * The CPU polls error slots to detect and handle errors.
 */
#ifndef GPU_ERROR_H
#define GPU_ERROR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes (matching DAOS DER_* numeric convention) ──────────────── */
#define GPU_SUCCESS         0
#define GPU_ERR_INVAL      -1001   /* Invalid argument */
#define GPU_ERR_NOMEM      -1002   /* Out of memory */
#define GPU_ERR_IO         -1003   /* I/O error */
#define GPU_ERR_OVERFLOW   -1004   /* Buffer overflow */
#define GPU_ERR_CSUM       -1005   /* Checksum mismatch */
#define GPU_ERR_UNKNOWN    -1099   /* Unknown error */

#ifdef __CUDACC__
/* ── GPU-side macros (only available in .cu files) ─────────────────────── */

/**
 * GPU_ASSERT: Non-fatal assertion. Records error and returns from the
 * current device function. Does NOT call __trap() or abort the kernel.
 */
#define GPU_ASSERT(cond, result_ptr, err_code) do {         \
    if (!(cond)) {                                           \
        if (result_ptr) {                                    \
            (result_ptr)->error_code = (err_code);           \
            __threadfence_system();                          \
        }                                                    \
        return;                                              \
    }                                                        \
} while(0)

/**
 * GPU_ERROR: Record an error code without returning.
 */
#define GPU_ERROR(result_ptr, err_code) do {                 \
    if (result_ptr) {                                        \
        (result_ptr)->error_code = (err_code);               \
        __threadfence_system();                              \
    }                                                        \
} while(0)

#endif /* __CUDACC__ */

/* ── CPU-side error checking (host code) ───────────────────────────────── */

/**
 * Check if an error code indicates failure.
 */
static inline int gpu_error_is_failure(int32_t code)
{
    return code < 0;
}

/**
 * Return a human-readable string for an error code.
 */
const char *gpu_error_string(int32_t code);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ERROR_H */
