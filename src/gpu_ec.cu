/**
 * gpu_ec.cu — EC Parity on GPU
 *
 * P-Parity: Standard XOR across stripes (correct, interoperable).
 * Q-Parity: PLACEHOLDER — uses (val << 1) ^ val on 32-bit words.
 *   This is NOT real GF(2^8) multiplication. It operates on 32-bit words
 *   (crossing byte boundaries) and is non-invertible, so it CANNOT
 *   reconstruct data after a double-disk failure. For production RAID-6,
 *   replace with proper GF(2^8) tables (e.g., ISA-L compatible Cauchy matrix).
 */
#include "gpu_ec.h"
#include "gpu_comp.h"
#include <cuda_runtime.h>
#include <string.h>

/* ── GPU kernel: XOR parity across stripes ─────────────────────────────── */
__global__ void ec_xor_parity_kernel(
    uint8_t **data_ptrs, int num_stripes, size_t stripe_len, uint8_t *parity)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= stripe_len) return;

    uint8_t val = 0;
    for (int s = 0; s < num_stripes; s++) {
        val ^= data_ptrs[s][idx];
    }
    parity[idx] = val;
}

/* Host API */
int gpu_ec_xor_parity(void **data_ptrs, int num_stripes, size_t stripe_len,
                      void *parity_out)
{
    if (!data_ptrs || !parity_out || num_stripes <= 0 || stripe_len == 0)
        return -1;

    /* Copy pointer array to device */
    uint8_t **d_ptrs;
    cudaMalloc(&d_ptrs, sizeof(uint8_t *) * num_stripes);
    cudaMemcpy(d_ptrs, data_ptrs, sizeof(uint8_t *) * num_stripes,
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (int)((stripe_len + threads - 1) / threads);

    ec_xor_parity_kernel<<<blocks, threads>>>(
        d_ptrs, num_stripes, stripe_len, (uint8_t *)parity_out);

    cudaDeviceSynchronize();
    cudaFree(d_ptrs);
    return 0;
}

/* CPU reference */
void cpu_ec_xor_parity(const void **data_ptrs, int num_stripes,
                       size_t stripe_len, void *parity_out)
{
    uint8_t *parity = (uint8_t *)parity_out;
    memcpy(parity, data_ptrs[0], stripe_len);
    for (int s = 1; s < num_stripes; s++) {
        const uint8_t *src = (const uint8_t *)data_ptrs[s];
        for (size_t i = 0; i < stripe_len; i++) {
            parity[i] ^= src[i];
        }
    }
}

/* ── Cooperative Block EC Parity Generation ────────────────────────────── */
/* Computes P and Q parity for up to 16 data stripes.
 * Uses 16-byte vectorized loads (uint4) for massive VRAM throughput.
 */
__device__ void device_ec_encode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size)
{
    if (stripe_cnt == 0 || parity_cnt == 0 || cell_size == 0) return;

    size_t vec_len = cell_size / sizeof(uint4);

    /* P-Parity: Simple XOR */
    if (parity_cnt >= 1) {
        uint4 *p_out = (uint4*)parity_ptrs[0];
        for (size_t i = threadIdx.x; i < vec_len; i += blockDim.x) {
            uint4 p_val = ((const uint4*)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++) {
                uint4 s_val = ((const uint4*)ec_ptrs[s])[i];
                p_val.x ^= s_val.x;
                p_val.y ^= s_val.y;
                p_val.z ^= s_val.z;
                p_val.w ^= s_val.w;
            }
            p_out[i] = p_val;
        }
    }

    /* Q-Parity: Simulated GF(2^8) / Shift XOR for RAID-6 benchmark */
    if (parity_cnt >= 2) {
        uint4 *q_out = (uint4*)parity_ptrs[1];
        for (size_t i = threadIdx.x; i < vec_len; i += blockDim.x) {
            uint4 q_val = ((const uint4*)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++) {
                uint4 s_val = ((const uint4*)ec_ptrs[s])[i];
                /* Very simple placeholder GF multiplier (shift/XOR) for benchmarking throughput */
                q_val.x ^= (s_val.x << 1) ^ s_val.x;
                q_val.y ^= (s_val.y << 1) ^ s_val.y;
                q_val.z ^= (s_val.z << 1) ^ s_val.z;
                q_val.w ^= (s_val.w << 1) ^ s_val.w;
            }
            q_out[i] = q_val;
        }
    }
}
