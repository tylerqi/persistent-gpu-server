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

    /* Use a dedicated stream instead of cudaDeviceSynchronize() to avoid
     * deadlocking with the persistent kernel running on another stream. */
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    ec_xor_parity_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, num_stripes, stripe_len, (uint8_t *)parity_out);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
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

/* ── GF(2^8) Arithmetic ────────────────────────────────────────────────── */
/* Fast parallel GF(2^8) multiplication by 2 using irreducible polynomial 0x11B.
 * This is the standard polynomial used by ISA-L, DAOS, and Linux RAID-6.
 * Reduction constant = 0x1B (lower 8 bits of 0x11B).
 * Multiplies four bytes packed into a 32-bit word, executing on uint4 vectors.
 */
__device__ __forceinline__ uint4 gf_mul2(uint4 val)
{
    uint4 res;
    uint32_t mask;
    mask = (val.x & 0x80808080) >> 7; res.x = ((val.x << 1) & 0xFEFEFEFE) ^ (mask * 0x1B);
    mask = (val.y & 0x80808080) >> 7; res.y = ((val.y << 1) & 0xFEFEFEFE) ^ (mask * 0x1B);
    mask = (val.z & 0x80808080) >> 7; res.z = ((val.z << 1) & 0xFEFEFEFE) ^ (mask * 0x1B);
    mask = (val.w & 0x80808080) >> 7; res.w = ((val.w << 1) & 0xFEFEFEFE) ^ (mask * 0x1B);
    return res;
}

/* Scalar GF(2^8) multiply by 2 for tail bytes */
__device__ __forceinline__ uint8_t gf_mul2_byte(uint8_t val)
{
    return (uint8_t)(((val << 1) & 0xFE) ^ ((val >> 7) * 0x1B));
}

/* ── Cooperative Block EC Parity Generation ────────────────────────────── */
/* Computes P and Q parity for up to 16 data stripes.
 * Uses 16-byte vectorized loads (uint4) for massive VRAM throughput.
 * Handles non-16-aligned cell sizes with a scalar tail loop (C-3 fix).
 */
__device__ void device_ec_encode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size)
{
    if (stripe_cnt == 0 || parity_cnt == 0 || cell_size == 0) return;

    size_t vec_len = cell_size / sizeof(uint4);
    size_t tail_start = vec_len * sizeof(uint4); /* Byte offset where tail begins */
    size_t tail_len = cell_size - tail_start;     /* 0..15 bytes */

    /* P-Parity: Simple XOR */
    if (parity_cnt >= 1) {
        uint4 *p_out = (uint4*)parity_ptrs[0];
        /* Vectorized main loop */
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
        /* Scalar tail for non-16-aligned cell_size (C-3) */
        uint8_t *p_out_bytes = (uint8_t*)parity_ptrs[0];
        for (size_t i = tail_start + threadIdx.x; i < cell_size; i += blockDim.x) {
            uint8_t p_byte = ((const uint8_t*)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++) {
                p_byte ^= ((const uint8_t*)ec_ptrs[s])[i];
            }
            p_out_bytes[i] = p_byte;
        }
    }

    /* Q-Parity: GF(2^8) Horner's Method */
    if (parity_cnt >= 2) {
        uint4 *q_out = (uint4*)parity_ptrs[1];
        /* Vectorized main loop */
        for (size_t i = threadIdx.x; i < vec_len; i += blockDim.x) {
            uint4 q_val = ((const uint4*)ec_ptrs[stripe_cnt - 1])[i];
            for (int s = (int)stripe_cnt - 2; s >= 0; s--) {
                uint4 s_val = ((const uint4*)ec_ptrs[s])[i];
                q_val = gf_mul2(q_val);
                q_val.x ^= s_val.x;
                q_val.y ^= s_val.y;
                q_val.z ^= s_val.z;
                q_val.w ^= s_val.w;
            }
            q_out[i] = q_val;
        }
        /* Scalar tail for non-16-aligned cell_size (C-3) */
        uint8_t *q_out_bytes = (uint8_t*)parity_ptrs[1];
        for (size_t i = tail_start + threadIdx.x; i < cell_size; i += blockDim.x) {
            uint8_t q_byte = ((const uint8_t*)ec_ptrs[stripe_cnt - 1])[i];
            for (int s = (int)stripe_cnt - 2; s >= 0; s--) {
                q_byte = gf_mul2_byte(q_byte);
                q_byte ^= ((const uint8_t*)ec_ptrs[s])[i];
            }
            q_out_bytes[i] = q_byte;
        }
    }
}
