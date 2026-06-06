/**
 * gpu_ec.h — GPU Erasure Coding Parity API
 *
 * Generates EC parity using Galois Field GF(2^8) arithmetic on GPU.
 * Supports multiple encoding modes for compatibility with external systems.
 */
#ifndef GPU_EC_H
#define GPU_EC_H

#include <stdint.h>
#include <stddef.h>

#if !defined(__CUDACC__) && !defined(CUDA_VERSION) && !defined(_CUDA_RUNTIME_H) && !defined(__CUDA_RUNTIME_H__)
typedef struct CUstream_st *cudaStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * EC encoding mode selection.
 *
 * GPU_EC_MODE_NATIVE: Original mode. Vandermonde g=2, GF(2^8)/0x11B.
 *   Self-consistent encode/decode but NOT byte-identical to ISA-L or Linux RAID-6.
 *
 * GPU_EC_MODE_RAID6: Linux RAID-6 compatible. Vandermonde g=2, GF(2^8)/0x11D.
 *   Q parity is byte-identical to Linux kernel RAID-6 output.
 *   P parity (XOR) is unchanged across all modes.
 *
 * GPU_EC_MODE_ISAL: ISA-L / DAOS compatible. Cauchy matrix, GF(2^8)/0x11D.
 *   Parity is byte-identical to ISA-L ec_encode_data() with gf_gen_cauchy1_matrix().
 *   Requires encode_matrix in the work item (set via gpu_ec_gen_cauchy_matrix).
 *   Decode requires a pre-computed decode matrix (set via gpu_ec_make_decode_matrix).
 */
typedef enum {
    GPU_EC_MODE_NATIVE = 0,  /* GF(2^8)/0x11B Vandermonde (default, backward compatible) */
    GPU_EC_MODE_RAID6  = 1,  /* GF(2^8)/0x11D Vandermonde (Linux RAID-6 compatible) */
    GPU_EC_MODE_ISAL   = 2,  /* GF(2^8)/0x11D Cauchy matrix (ISA-L/DAOS compatible) */
} gpu_ec_mode_t;

/**
 * Generate single parity (XOR) across data stripes on GPU.
 * This is equivalent to RAID-5 / EC k+1 parity.
 *
 * @param data_ptrs     Array of GPU pointers to data stripes
 * @param num_stripes   Number of data stripes (k)
 * @param stripe_len    Length of each stripe in bytes
 * @param parity_out    GPU pointer to output parity buffer (must be stripe_len bytes)
 * @return 0 on success
 */
int gpu_ec_xor_parity(void **data_ptrs, int num_stripes, size_t stripe_len,
                      void *parity_out);

/**
 * Direct multi-SM EC encode (bypasses persistent kernel).
 * Launches a dedicated kernel across all SMs for maximum VRAM bandwidth.
 * All pointers must be in GPU memory. Caller provides a CUDA stream.
 *
 * @param d_ec_ptrs     Array of GPU pointers to data stripes (host array)
 * @param d_parity_ptrs Array of GPU pointers to parity output (host array)
 * @param stripe_cnt    Number of data stripes (k)
 * @param parity_cnt    Number of parity stripes (p)
 * @param cell_size     Size of each stripe in bytes
 * @param stream        CUDA stream to launch on
 * @param ec_mode       Encoding mode (GPU_EC_MODE_NATIVE, GPU_EC_MODE_RAID6, etc.)
 * @return 0 on success
 */
int gpu_ec_encode_multi_sm(void **d_ec_ptrs, void **d_parity_ptrs,
                            int stripe_cnt, int parity_cnt, size_t cell_size,
                            cudaStream_t stream, int ec_mode);

/**
 * CPU reference XOR parity for verification.
 */
void cpu_ec_xor_parity(const void **data_ptrs, int num_stripes,
                       size_t stripe_len, void *parity_out);

/**
 * Initialize GF(2^8) exp/log/inverse tables on the GPU.
 * Initializes tables for both 0x11B and 0x11D polynomials.
 * Must be called before using EC decode operations.
 * Safe to call multiple times (idempotent).
 */
void gpu_ec_init_gf_tables(void);

/**
 * CPU reference EC decode for verification.
 * Reconstructs 1 or 2 failed data stripes from P and Q parity.
 *
 * @param ec_ptrs     Array of host pointers to data stripes (surviving data in place)
 * @param parity_ptrs Array of host pointers to parity data [P, Q]
 * @param stripe_cnt  Number of data stripes (k)
 * @param parity_cnt  Number of parity stripes (p, must be >= failed_cnt)
 * @param cell_size   Size of each stripe cell in bytes
 * @param failed_idx  Array of indices of failed stripes
 * @param failed_cnt  Number of failed stripes (1 or 2)
 * @param out_ptrs    Array of host pointers to receive reconstructed data
 * @param ec_mode     Encoding mode (default 0 = GPU_EC_MODE_NATIVE)
 */
void cpu_ec_decode(const void **ec_ptrs, const void **parity_ptrs,
                   int stripe_cnt, int parity_cnt, size_t cell_size,
                   const int *failed_idx, int failed_cnt,
                   void **out_ptrs, int ec_mode);

/**
 * Generate ISA-L compatible Cauchy encoding matrix.
 * Produces a (parity_cnt × stripe_cnt) matrix in GF(2^8)/0x11D.
 * Matrix element [r][j] = gf_inv_11d((stripe_cnt + r) ^ j).
 *
 * @param matrix      Output buffer (must be parity_cnt * stripe_cnt bytes)
 * @param stripe_cnt  Number of data stripes (k)
 * @param parity_cnt  Number of parity stripes (p)
 */
void gpu_ec_gen_cauchy_matrix(uint8_t *matrix, int stripe_cnt, int parity_cnt);

/**
 * Compute decode matrix for ISA-L mode given failed stripe indices.
 * The caller provides the encoding matrix (from gpu_ec_gen_cauchy_matrix).
 * This function extracts the surviving rows, inverts, and produces the
 * decode coefficients needed to reconstruct the failed stripes.
 *
 * @param encode_matrix  Encoding matrix (parity_cnt × stripe_cnt bytes)
 * @param stripe_cnt     Number of data stripes (k)
 * @param parity_cnt     Number of parity stripes (p)
 * @param failed_idx     Array of failed stripe indices
 * @param failed_cnt     Number of failed stripes
 * @param decode_matrix  Output decode matrix (failed_cnt × stripe_cnt bytes)
 * @return 0 on success, -1 if matrix is singular
 */
int gpu_ec_make_decode_matrix(const uint8_t *encode_matrix,
                              int stripe_cnt, int parity_cnt,
                              const int *failed_idx, int failed_cnt,
                              uint8_t *decode_matrix);

/**
 * Invert a square matrix over GF(2^8)/0x11D using Gaussian elimination.
 *
 * @param matrix   Input matrix (n × n, row-major), overwritten during computation
 * @param inverse  Output inverse matrix (n × n)
 * @param n        Matrix dimension
 * @return 0 on success, -1 if singular
 */
int gpu_ec_invert_matrix(uint8_t *matrix, uint8_t *inverse, int n);

#ifdef __cplusplus
}
#endif

#endif /* GPU_EC_H */
