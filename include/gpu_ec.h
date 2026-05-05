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
 * CPU reference XOR parity for verification.
 */
void cpu_ec_xor_parity(const void **data_ptrs, int num_stripes,
                       size_t stripe_len, void *parity_out);

#ifdef __cplusplus
}
#endif

#endif /* GPU_EC_H */
