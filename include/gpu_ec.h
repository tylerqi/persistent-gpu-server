/**
 * gpu_ec.h — GPU Erasure Coding Parity API
 *
 * Generates EC parity using Galois Field GF(2^8) arithmetic on GPU.
 */
#ifndef GPU_EC_H
#define GPU_EC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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
 * @return 0 on success
 */
int gpu_ec_encode_multi_sm(void **d_ec_ptrs, void **d_parity_ptrs,
                            int stripe_cnt, int parity_cnt, size_t cell_size,
                            cudaStream_t stream);

/**
 * CPU reference XOR parity for verification.
 */
void cpu_ec_xor_parity(const void **data_ptrs, int num_stripes,
                       size_t stripe_len, void *parity_out);

/**
 * Initialize GF(2^8) exp/log/inverse tables on the GPU.
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
 */
void cpu_ec_decode(const void **ec_ptrs, const void **parity_ptrs,
                   int stripe_cnt, int parity_cnt, size_t cell_size,
                   const int *failed_idx, int failed_cnt,
                   void **out_ptrs);

#ifdef __cplusplus
}
#endif

#endif /* GPU_EC_H */
