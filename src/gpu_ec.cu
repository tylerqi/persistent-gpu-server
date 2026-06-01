/**
 * gpu_ec.cu — EC Parity on GPU (Encode + Decode)
 *
 * P-Parity: Standard XOR across data stripes. Universally compatible with
 *   RAID-5, ISA-L, Linux RAID-6, and DAOS.
 *
 * Q-Parity: Vandermonde-style encoding using GF(2^8) Horner's method with
 *   generator g=2 and irreducible polynomial 0x11B (x^8+x^4+x^3+x+1) or 0x11D.
 *   Q[i] = 2^0·D0[i] ⊕ 2^1·D1[i] ⊕ … ⊕ 2^(k-1)·D(k-1)[i]
 *
 * Compatibility note:
 *   - ISA-L uses a Cauchy encoding matrix (not Vandermonde 2^s coefficients),
 *     so Q parity is NOT byte-identical to ISA-L output.
 *   - Linux RAID-6 uses irreducible polynomial 0x11D (not 0x11B),
 *     so Q parity is byte-identical to Linux RAID-6 output when MODE_RAID6 is selected.
 *   - P parity (XOR) is universally compatible.
 *   - The encode/decode pair in this file is self-consistent and correct
 *     for standalone use or as a matched GPU offload engine.
 */
#include "gpu_ec.h"
#include "gpu_comp.h"
#include <cuda_runtime.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>

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
    if (cudaMalloc(&d_ptrs, sizeof(uint8_t *) * num_stripes) != cudaSuccess)
        return -1;
    if (cudaMemcpy(d_ptrs, data_ptrs, sizeof(uint8_t *) * num_stripes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_ptrs);
        return -1;
    }

    int threads = 256;
    int blocks = (int)((stripe_len + threads - 1) / threads);

    /* Use a dedicated stream instead of cudaDeviceSynchronize() to avoid
     * deadlocking with the persistent kernel running on another stream. */
    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        cudaFree(d_ptrs);
        return -1;
    }

    ec_xor_parity_kernel<<<blocks, threads, 0, stream>>>(
        d_ptrs, num_stripes, stripe_len, (uint8_t *)parity_out);

    cudaError_t err = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_ptrs);
    return (err == cudaSuccess) ? 0 : -1;
}

/* GF(2^8) reduction constants */
#define GF_REDUCE_11B  0x1B   /* x^8+x^4+x^3+x+1 (native) */
#define GF_REDUCE_11D  0x1D   /* x^8+x^4+x^3+x^2+1 (RAID-6, ISA-L) */

/* Forward declarations of GF(2^8) helpers (defined below) */
__device__ __forceinline__ uint4 gf_mul2_r(uint4 val, uint32_t reduce);
__device__ __forceinline__ uint8_t gf_mul2_byte_r(uint8_t val, uint8_t reduce);
__device__ __forceinline__ uint4 gf_mul2(uint4 val);
__device__ __forceinline__ uint8_t gf_mul2_byte(uint8_t val);
__device__ __forceinline__ uint8_t gf_mul_bitwise_r(uint8_t a, uint8_t b, uint8_t reduce);
__device__ __forceinline__ uint8_t gf_inv_direct_r(uint8_t a, uint8_t reduce);

/* ── Multi-SM EC Encode kernel ─────────────────────────────────────────────
 * Unlike device_ec_encode (used by the persistent kernel on 1 SM),
 * this kernel distributes P and Q parity computation across ALL SMs.
 * Pointer arrays are passed as fixed-size kernel arguments (no cudaMalloc).
 */
__global__ void ec_encode_multi_sm_kernel(
    const void *ec0, const void *ec1, const void *ec2, const void *ec3,
    const void *ec4, const void *ec5, const void *ec6, const void *ec7,
    const void *ec8, const void *ec9, const void *ec10, const void *ec11,
    const void *ec12, const void *ec13, const void *ec14, const void *ec15,
    void *par0, void *par1, void *par2, void *par3,
    uint32_t stripe_cnt, uint32_t parity_cnt, size_t cell_size, uint32_t ec_mode)
{
    /* Reconstruct pointer arrays from arguments */
    const void *ec_ptrs[16] = {ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,
                               ec8,ec9,ec10,ec11,ec12,ec13,ec14,ec15};
    void *parity_ptrs[4] = {par0, par1, par2, par3};

    size_t vec_len = cell_size / sizeof(uint4);
    size_t tail_start = vec_len * sizeof(uint4);
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    uint32_t reduce = (ec_mode == 1 || ec_mode == 2) ? GF_REDUCE_11D : GF_REDUCE_11B;

    /* P-Parity: XOR across all stripes */
    if (parity_cnt >= 1) {
        uint4 *p_out = (uint4 *)parity_ptrs[0];
        for (size_t i = gid; i < vec_len; i += stride) {
            uint4 p_val = ((const uint4 *)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++) {
                uint4 s_val = ((const uint4 *)ec_ptrs[s])[i];
                p_val.x ^= s_val.x; p_val.y ^= s_val.y;
                p_val.z ^= s_val.z; p_val.w ^= s_val.w;
            }
            p_out[i] = p_val;
        }
        uint8_t *p_bytes = (uint8_t *)parity_ptrs[0];
        for (size_t i = tail_start + gid; i < cell_size; i += stride) {
            uint8_t p = ((const uint8_t *)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++)
                p ^= ((const uint8_t *)ec_ptrs[s])[i];
            p_bytes[i] = p;
        }
    }

    /* Q-Parity: GF(2^8) Horner's method */
    if (parity_cnt >= 2) {
        uint4 *q_out = (uint4 *)parity_ptrs[1];
        for (size_t i = gid; i < vec_len; i += stride) {
            uint4 q_val = ((const uint4 *)ec_ptrs[stripe_cnt - 1])[i];
            for (int s = (int)stripe_cnt - 2; s >= 0; s--) {
                uint4 s_val = ((const uint4 *)ec_ptrs[s])[i];
                q_val = gf_mul2_r(q_val, reduce);
                q_val.x ^= s_val.x; q_val.y ^= s_val.y;
                q_val.z ^= s_val.z; q_val.w ^= s_val.w;
            }
            q_out[i] = q_val;
        }
        uint8_t *q_bytes = (uint8_t *)parity_ptrs[1];
        for (size_t i = tail_start + gid; i < cell_size; i += stride) {
            uint8_t q = ((const uint8_t *)ec_ptrs[stripe_cnt - 1])[i];
            for (int s = (int)stripe_cnt - 2; s >= 0; s--) {
                q = gf_mul2_byte_r(q, reduce);
                q ^= ((const uint8_t *)ec_ptrs[s])[i];
            }
            q_bytes[i] = q;
        }
    }
}

/* Host API: Launch multi-SM EC encode on a given stream.
 * All pointers (ec_ptrs, parity_ptrs, and the data they point to)
 * must already be in GPU memory. */
int gpu_ec_encode_multi_sm(void **d_ec_ptrs, void **d_parity_ptrs,
                            int stripe_cnt, int parity_cnt, size_t cell_size,
                            cudaStream_t stream, int ec_mode)
{
    if (ec_mode == 2) // GPU_EC_MODE_ISAL
        return -1;
    if (!d_ec_ptrs || !d_parity_ptrs || stripe_cnt <= 0 || parity_cnt <= 0 || cell_size == 0)
        return -1;

    /* Pad pointer arrays to fixed kernel argument sizes */
    const void *ec[16] = {};
    void *par[4] = {};
    for (int i = 0; i < stripe_cnt && i < 16; i++) ec[i] = d_ec_ptrs[i];
    for (int i = 0; i < parity_cnt && i < 4; i++) par[i] = d_parity_ptrs[i];

    int threads = 256;
    size_t vec_len = cell_size / sizeof(uint4);
    int blocks = (int)((vec_len + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;

    ec_encode_multi_sm_kernel<<<blocks, threads, 0, stream>>>(
        ec[0],ec[1],ec[2],ec[3],ec[4],ec[5],ec[6],ec[7],
        ec[8],ec[9],ec[10],ec[11],ec[12],ec[13],ec[14],ec[15],
        par[0],par[1],par[2],par[3],
        stripe_cnt, parity_cnt, cell_size, ec_mode);

    /* Check for kernel launch errors */
    if (cudaGetLastError() != cudaSuccess) return -1;

    /* No cudaMalloc/cudaFree — all pointers passed as kernel arguments.
     * Caller must synchronize 'stream' before reading parity results. */
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
/* Fast parallel GF(2^8) multiplication by 2 using a parameterized irreducible polynomial.
 * Multiplies four bytes packed into a 32-bit word, executing on uint4 vectors.
 */
__device__ __forceinline__ uint4 gf_mul2_r(uint4 val, uint32_t reduce)
{
    uint4 res;
    uint32_t mask;
    mask = (val.x & 0x80808080) >> 7; res.x = ((val.x << 1) & 0xFEFEFEFE) ^ (mask * reduce);
    mask = (val.y & 0x80808080) >> 7; res.y = ((val.y << 1) & 0xFEFEFEFE) ^ (mask * reduce);
    mask = (val.z & 0x80808080) >> 7; res.z = ((val.z << 1) & 0xFEFEFEFE) ^ (mask * reduce);
    mask = (val.w & 0x80808080) >> 7; res.w = ((val.w << 1) & 0xFEFEFEFE) ^ (mask * reduce);
    return res;
}

/* Scalar GF(2^8) multiply by 2 using a parameterized irreducible polynomial */
__device__ __forceinline__ uint8_t gf_mul2_byte_r(uint8_t val, uint8_t reduce)
{
    return (uint8_t)(((val << 1) & 0xFE) ^ ((val >> 7) * reduce));
}

__device__ __forceinline__ uint4 gf_mul2(uint4 val)
{
    return gf_mul2_r(val, GF_REDUCE_11B);
}

__device__ __forceinline__ uint8_t gf_mul2_byte(uint8_t val)
{
    return gf_mul2_byte_r(val, GF_REDUCE_11B);
}

extern "C" {

/* ── Cooperative Block EC Parity Generation ────────────────────────────── */
/* Computes P and Q parity for up to 16 data stripes.
 * Uses 16-byte vectorized loads (uint4) for massive VRAM throughput.
 * Handles non-16-aligned cell sizes with a scalar tail loop (C-3 fix).
 */
__device__ void device_ec_encode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size, uint32_t ec_mode,
                                 const uint8_t *encode_matrix)
{
    if (stripe_cnt == 0 || parity_cnt == 0 || cell_size == 0) return;

    if (ec_mode == 2) {
        for (uint32_t r = 0; r < parity_cnt; r++) {
            uint8_t *out = (uint8_t *)parity_ptrs[r];
            for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
                uint8_t val = 0;
                for (uint32_t j = 0; j < stripe_cnt; j++) {
                    uint8_t coef = encode_matrix[r * stripe_cnt + j];
                    val ^= gf_mul_bitwise_r(coef, ((const uint8_t *)ec_ptrs[j])[i], GF_REDUCE_11D);
                }
                out[i] = val;
            }
        }
        return;
    }


    size_t vec_len = cell_size / sizeof(uint4);
    size_t tail_start = vec_len * sizeof(uint4); /* Byte offset where tail begins */

    uint32_t reduce = (ec_mode == 1 || ec_mode == 2) ? GF_REDUCE_11D : GF_REDUCE_11B;

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
                q_val = gf_mul2_r(q_val, reduce);
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
                q_byte = gf_mul2_byte_r(q_byte, reduce);
                q_byte ^= ((const uint8_t*)ec_ptrs[s])[i];
            }
            q_out_bytes[i] = q_byte;
        }
    }
}

} /* extern "C" — device_ec_encode */

/* ── GF(2^8) Exp/Log/Inverse Tables for EC Decode ─────────────────────── */
/* Required for EC decode (data reconstruction from P+Q parity).
 * Uses discrete logarithm approach:
 *   a * b = exp[log[a] + log[b]]
 * exp table extended to 512 entries to avoid modular arithmetic. */

__constant__ uint8_t gf_exp_table[512];
__constant__ uint8_t gf_log_table[256];
__constant__ uint8_t gf_inv_table[256];

static uint8_t h_gf_exp[512];
static uint8_t h_gf_log[256];
static uint8_t h_gf_inv[256];
static volatile int gf_host_ready = 0;
static pthread_once_t gf_init_once = PTHREAD_ONCE_INIT;

/* Host-side GF(2^8) multiply for table generation */
static uint8_t h_gf_mul_r(uint8_t a, uint8_t b, uint8_t reduce)
{
    uint8_t result = 0;
    uint8_t hi_bit;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        hi_bit = a & 0x80;
        a <<= 1;
        if (hi_bit) a ^= reduce;
        b >>= 1;
    }
    return result;
}

static uint8_t h_gf_mul(uint8_t a, uint8_t b)
{
    return h_gf_mul_r(a, b, GF_REDUCE_11B);
}

/* Pre-computed powers of 2 for Q-parity coefficients (encoding uses g=2 in Horner).
 * Only need up to 16 entries (max stripe count). */
__constant__ uint8_t gf_pow2_table[16];
static uint8_t h_gf_pow2[16];

__constant__ uint8_t gf_exp_table_11d[512];
__constant__ uint8_t gf_log_table_11d[256];
__constant__ uint8_t gf_inv_table_11d[256];
__constant__ uint8_t gf_pow2_table_11d[16];

static uint8_t h_gf_exp_11d[512];
static uint8_t h_gf_log_11d[256];
static uint8_t h_gf_inv_11d[256];
static uint8_t h_gf_pow2_11d[16];

/* Build host-side GF(2^8) tables only (no CUDA calls).
 * Thread-safe via atomic CAS — only one thread builds. */
static void ensure_host_gf_tables(void)
{
    if (gf_host_ready) return;

    static volatile int building = 0;
    if (__sync_bool_compare_and_swap(&building, 0, 1)) {
        /* Build exp/log tables using generator g=3 (PRIMITIVE root of GF(2^8)/0x11B).
         * Generator g=2 has order 51 and is NOT primitive — it only covers 51 of 255
         * non-zero elements, making the log table incomplete. */
        h_gf_exp[0] = 1;
        for (int i = 1; i < 512; i++) {
            h_gf_exp[i] = h_gf_mul(h_gf_exp[i - 1], 3);
        }
        memset(h_gf_log, 0, sizeof(h_gf_log));
        for (int i = 0; i < 255; i++) {
            h_gf_log[h_gf_exp[i]] = (uint8_t)i;
        }

        /* Build inverse table: inv[x] = x^254 = exp[254 - log[x]] */
        h_gf_inv[0] = 0;
        for (int i = 1; i < 256; i++) {
            h_gf_inv[i] = h_gf_exp[255 - h_gf_log[i]];
        }

        /* Build pow2 table: pow2[i] = 2^i in GF(2^8)/0x11B.
         * These are the encoding coefficients used by the Horner Q-parity method. */
        h_gf_pow2[0] = 1;
        for (int i = 1; i < 16; i++) {
            h_gf_pow2[i] = h_gf_mul(h_gf_pow2[i - 1], 2);
        }

        /* Build 11D tables dynamically using a primitive root generator */
        uint8_t gen_11d = 0;
        for (int candidate = 2; candidate < 256; candidate++) {
            h_gf_exp_11d[0] = 1;
            int visited[256] = {0};
            visited[1] = 1;
            int count = 1;
            for (int i = 1; i < 255; i++) {
                h_gf_exp_11d[i] = h_gf_mul_r(h_gf_exp_11d[i - 1], candidate, GF_REDUCE_11D);
                if (h_gf_exp_11d[i] != 0 && !visited[h_gf_exp_11d[i]]) {
                    visited[h_gf_exp_11d[i]] = 1;
                    count++;
                }
            }
            if (count == 255) {
                gen_11d = candidate;
                /* Extend to 512 entries */
                for (int i = 255; i < 512; i++) {
                    h_gf_exp_11d[i] = h_gf_mul_r(h_gf_exp_11d[i - 1], gen_11d, GF_REDUCE_11D);
                }
                break;
            }
        }

        memset(h_gf_log_11d, 0, sizeof(h_gf_log_11d));
        for (int i = 0; i < 255; i++) {
            h_gf_log_11d[h_gf_exp_11d[i]] = (uint8_t)i;
        }

        h_gf_inv_11d[0] = 0;
        for (int i = 1; i < 256; i++) {
            h_gf_inv_11d[i] = h_gf_exp_11d[255 - h_gf_log_11d[i]];
        }

        h_gf_pow2_11d[0] = 1;
        for (int i = 1; i < 16; i++) {
            h_gf_pow2_11d[i] = h_gf_mul_r(h_gf_pow2_11d[i - 1], 2, GF_REDUCE_11D);
        }

        __sync_synchronize();
        gf_host_ready = 1;
    } else {
        /* Another thread is building — spin until done */
        while (!gf_host_ready) __sync_synchronize();
    }
}

/* Initialize GF tables on GPU. Thread-safe, idempotent.
 * Calls ensure_host_gf_tables() first, then uploads to __constant__ memory. */
static void do_init_gf_tables(void)
{
    ensure_host_gf_tables();

    cudaError_t err;
    err = cudaMemcpyToSymbol(gf_exp_table, h_gf_exp, sizeof(h_gf_exp));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_exp_table) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
    err = cudaMemcpyToSymbol(gf_log_table, h_gf_log, sizeof(h_gf_log));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_log_table) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
    err = cudaMemcpyToSymbol(gf_inv_table, h_gf_inv, sizeof(h_gf_inv));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_inv_table) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
    err = cudaMemcpyToSymbol(gf_pow2_table, h_gf_pow2, sizeof(h_gf_pow2));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_pow2_table) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }

    /* 11D tables */
    err = cudaMemcpyToSymbol(gf_exp_table_11d, h_gf_exp_11d, sizeof(h_gf_exp_11d));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_exp_table_11d) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
    err = cudaMemcpyToSymbol(gf_log_table_11d, h_gf_log_11d, sizeof(h_gf_log_11d));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_log_table_11d) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
    err = cudaMemcpyToSymbol(gf_inv_table_11d, h_gf_inv_11d, sizeof(h_gf_inv_11d));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_inv_table_11d) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
    err = cudaMemcpyToSymbol(gf_pow2_table_11d, h_gf_pow2_11d, sizeof(h_gf_pow2_11d));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(gf_pow2_table_11d) failed: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
}

/* Initialize GF tables on GPU. Thread-safe, idempotent.
 * Calls ensure_host_gf_tables() first, then uploads to __constant__ memory. */
void gpu_ec_init_gf_tables(void)
{
    pthread_once(&gf_init_once, do_init_gf_tables);
}

/* Device-side GF(2^8) operations via exp/log/pow2 tables in __constant__ memory.
 * NOTE: Currently unused — device_ec_decode uses gf_mul_bitwise() and gf_inv_direct()
 * to avoid __constant__ memory dependency issues with separable compilation.
 * Retained for potential future optimization of non-persistent kernel decode paths. */
/* Device-side GF(2^8) multiply via exp/log tables (uses g=3 as generator) */
__device__ __forceinline__ uint8_t gf_mul_full(uint8_t a, uint8_t b)
{
    if (a == 0 || b == 0) return 0;
    int log_sum = (int)gf_log_table[a] + (int)gf_log_table[b];
    return gf_exp_table[log_sum]; /* exp table extended to 510 so no modulo needed */
}

/* Device-side GF(2^8) inverse */
__device__ __forceinline__ uint8_t gf_inv_val(uint8_t a)
{
    return gf_inv_table[a];
}

/* Device-side encoding coefficient: 2^n in GF(2^8)/0x11B.
 * NOT the same as exp_table[n] which uses generator 3. */
__device__ __forceinline__ uint8_t gf_pow2_val(int n)
{
    return gf_pow2_table[n];  /* n must be in [0, 15] */
}


/* ── Direct GF(2^8) Arithmetic (no tables, for EC decode) ──────────────── */
/* These bypass __constant__ memory and use bitwise operations only. */

__device__ __forceinline__ uint8_t gf_mul_bitwise_r(uint8_t a, uint8_t b, uint8_t reduce)
{
    uint8_t result = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= reduce;
        b >>= 1;
    }
    return result;
}

__device__ __forceinline__ uint8_t gf_inv_direct_r(uint8_t a, uint8_t reduce)
{
    if (a == 0) return 0;
    uint8_t a2   = gf_mul_bitwise_r(a, a, reduce);     /* a^2 */
    uint8_t a3   = gf_mul_bitwise_r(a2, a, reduce);    /* a^3 */
    uint8_t a6   = gf_mul_bitwise_r(a3, a3, reduce);   /* a^6 */
    uint8_t a7   = gf_mul_bitwise_r(a6, a, reduce);    /* a^7 */
    uint8_t a14  = gf_mul_bitwise_r(a7, a7, reduce);   /* a^14 */
    uint8_t a15  = gf_mul_bitwise_r(a14, a, reduce);   /* a^15 */
    uint8_t a30  = gf_mul_bitwise_r(a15, a15, reduce); /* a^30 */
    uint8_t a31  = gf_mul_bitwise_r(a30, a, reduce);   /* a^31 */
    uint8_t a62  = gf_mul_bitwise_r(a31, a31, reduce); /* a^62 */
    uint8_t a63  = gf_mul_bitwise_r(a62, a, reduce);   /* a^63 */
    uint8_t a126 = gf_mul_bitwise_r(a63, a63, reduce); /* a^126 */
    uint8_t a127 = gf_mul_bitwise_r(a126, a, reduce);  /* a^127 */
    uint8_t a254 = gf_mul_bitwise_r(a127, a127, reduce); /* a^254 */
    return a254;
}

__device__ __forceinline__ uint8_t gf_mul_bitwise(uint8_t a, uint8_t b)
{
    return gf_mul_bitwise_r(a, b, GF_REDUCE_11B);
}

__device__ __forceinline__ uint8_t gf_inv_direct(uint8_t a)
{
    return gf_inv_direct_r(a, GF_REDUCE_11B);
}

extern "C" {

/* ── Cooperative Block EC Data Reconstruction (Decode) ─────────────────── */
/*
 * Reconstructs failed data stripes from P and Q parity.
 *
 * Single failure (failed_cnt == 1):
 *   D_x = P ⊕ (all surviving data stripes)
 *   Only uses P parity.
 *
 * Double failure (failed_cnt == 2):
 *   Given failed stripes x and y (x < y), and syndromes:
 *     S_P = P ⊕ (all surviving data stripes)
 *     S_Q = Q ⊕ Σ(g^i · surviving[i])
 *   Solution:
 *     D_y = (S_Q ⊕ g^x · S_P) / (g^y ⊕ g^x)
 *     D_x = S_P ⊕ D_y
 *
 * The ec_ptrs array should have the surviving stripe data in place.
 * The failed stripe slots will be OVERWRITTEN with reconstructed data.
 * parity_ptrs[0] = P, parity_ptrs[1] = Q.
 */
__device__ void device_ec_decode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size,
                                 const uint32_t *failed_idx, uint32_t failed_cnt,
                                 uint32_t ec_mode,
                                 const uint8_t *decode_matrix)
{
    if (failed_cnt == 0 || cell_size == 0) return;

    uint32_t reduce = (ec_mode == 1 || ec_mode == 2) ? GF_REDUCE_11D : GF_REDUCE_11B;

    if (ec_mode == 2) {
        void *available_ptrs[16];
        uint32_t avail_cnt = 0;
        for (uint32_t s = 0; s < stripe_cnt; s++) {
            int failed = 0;
            for (uint32_t f = 0; f < failed_cnt; f++) {
                if (failed_idx[f] == s) {
                    failed = 1;
                    break;
                }
            }
            if (!failed) {
                available_ptrs[avail_cnt++] = ec_ptrs[s];
            }
        }
        for (uint32_t f = 0; f < failed_cnt; f++) {
            available_ptrs[avail_cnt++] = parity_ptrs[f];
        }

        for (uint32_t f_idx = 0; f_idx < failed_cnt; f_idx++) {
            uint32_t fid = failed_idx[f_idx];
            uint8_t *out = (uint8_t *)ec_ptrs[fid];
            for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
                uint8_t val = 0;
                for (uint32_t j = 0; j < stripe_cnt; j++) {
                    uint8_t coef = decode_matrix[f_idx * stripe_cnt + j];
                    val ^= gf_mul_bitwise_r(coef, ((const uint8_t *)available_ptrs[j])[i], GF_REDUCE_11D);
                }
                out[i] = val;
            }
        }
        return;
    }

    if (failed_cnt == 1) {
        /* ── Single failure: reconstruct from P parity (vectorized) ── */
        uint32_t fx = failed_idx[0];
        size_t vec_len = cell_size / sizeof(uint4);
        size_t tail_start = vec_len * sizeof(uint4);

        /* Vectorized main loop (16 bytes per iteration) */
        uint4 *out_vec = (uint4 *)ec_ptrs[fx];
        for (size_t i = threadIdx.x; i < vec_len; i += blockDim.x) {
            uint4 val = ((const uint4 *)parity_ptrs[0])[i];
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx) {
                    uint4 sv = ((const uint4 *)ec_ptrs[s])[i];
                    val.x ^= sv.x; val.y ^= sv.y;
                    val.z ^= sv.z; val.w ^= sv.w;
                }
            }
            out_vec[i] = val;
        }

        /* Scalar tail for non-16-aligned cell_size */
        uint8_t *out = (uint8_t *)ec_ptrs[fx];
        for (size_t i = tail_start + threadIdx.x; i < cell_size; i += blockDim.x) {
            uint8_t val = ((const uint8_t *)parity_ptrs[0])[i];
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx)
                    val ^= ((const uint8_t *)ec_ptrs[s])[i];
            }
            out[i] = val;
        }
    } else if (failed_cnt == 2 && parity_cnt >= 2) {
        /* ── Double failure: reconstruct from P + Q parity ──────── */
        uint32_t fx = failed_idx[0];
        uint32_t fy = failed_idx[1];
        if (fx > fy) { uint32_t t = fx; fx = fy; fy = t; }

        /* Precompute GF(2^8) power-of-2 coefficients once per decode
         * (eliminates O(k²) recomputation in the inner byte loop). */
        uint8_t gs_table[16];
        gs_table[0] = 1;
        for (uint32_t s = 1; s < stripe_cnt; s++)
            gs_table[s] = gf_mul2_byte_r(gs_table[s - 1], reduce);

        uint8_t g_x = gs_table[fx];
        uint8_t g_y = gs_table[fy];
        uint8_t coeff = gf_inv_direct_r(g_y ^ g_x, reduce);

        uint8_t *out_x = (uint8_t *)ec_ptrs[fx];
        uint8_t *out_y = (uint8_t *)ec_ptrs[fy];
        const uint8_t *p_data = (const uint8_t *)parity_ptrs[0];
        const uint8_t *q_data = (const uint8_t *)parity_ptrs[1];

        for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
            /* P syndrome: S_P = P ⊕ all surviving stripes */
            uint8_t s_p = p_data[i];
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy)
                    s_p ^= ((const uint8_t *)ec_ptrs[s])[i];
            }

            /* Q syndrome: S_Q = Q ⊕ Σ(2^s · surviving[s]) */
            uint8_t s_q = q_data[i];
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy)
                    s_q ^= gf_mul_bitwise_r(gs_table[s],
                                          ((const uint8_t *)ec_ptrs[s])[i], reduce);
            }

            /* Solve: D_y = (S_Q ⊕ g_x · S_P) * (g_y ⊕ g_x)^(-1) */
            uint8_t dy = gf_mul_bitwise_r(s_q ^ gf_mul_bitwise_r(g_x, s_p, reduce), coeff, reduce);
            uint8_t dx = s_p ^ dy;

            out_x[i] = dx;
            out_y[i] = dy;
        }
    }
    /* else: >2 failures not supported */
}

} /* extern "C" — device_ec_decode */

/* ── CPU Reference EC Decode for Verification ──────────────────────────── */

void cpu_ec_decode(const void **ec_ptrs, const void **parity_ptrs,
                   int stripe_cnt, int parity_cnt, size_t cell_size,
                   const int *failed_idx, int failed_cnt,
                   void **out_ptrs, int ec_mode)
{
    ensure_host_gf_tables();

    if (ec_mode == 2) {
        uint8_t encode_matrix[64];
        uint8_t decode_matrix[32];
        gpu_ec_gen_cauchy_matrix(encode_matrix, stripe_cnt, parity_cnt);
        if (gpu_ec_make_decode_matrix(encode_matrix, stripe_cnt, parity_cnt, failed_idx, failed_cnt, decode_matrix) != 0) {
            return;
        }
        
        const uint8_t *available[16];
        int avail_cnt = 0;
        for (int s = 0; s < stripe_cnt; s++) {
            int failed = 0;
            for (int f = 0; f < failed_cnt; f++) {
                if (failed_idx[f] == s) {
                    failed = 1;
                    break;
                }
            }
            if (!failed) {
                available[avail_cnt++] = (const uint8_t *)ec_ptrs[s];
            }
        }
        for (int f = 0; f < failed_cnt; f++) {
            available[avail_cnt++] = (const uint8_t *)parity_ptrs[f];
        }
        
        for (int f_idx = 0; f_idx < failed_cnt; f_idx++) {
            uint8_t *out = (uint8_t *)out_ptrs[f_idx];
            for (size_t i = 0; i < cell_size; i++) {
                uint8_t val = 0;
                for (int j = 0; j < stripe_cnt; j++) {
                    uint8_t coef = decode_matrix[f_idx * stripe_cnt + j];
                    val ^= h_gf_mul_r(coef, available[j][i], GF_REDUCE_11D);
                }
                out[i] = val;
            }
        }
        return;
    }

    const uint8_t *inv_tbl = (ec_mode == 1 || ec_mode == 2) ? h_gf_inv_11d : h_gf_inv;
    const uint8_t *pow2_tbl = (ec_mode == 1 || ec_mode == 2) ? h_gf_pow2_11d : h_gf_pow2;
    uint8_t reduce = (ec_mode == 1 || ec_mode == 2) ? GF_REDUCE_11D : GF_REDUCE_11B;

    if (failed_cnt == 1) {
        int fx = failed_idx[0];
        uint8_t *out = (uint8_t *)out_ptrs[0];
        const uint8_t *p_data = (const uint8_t *)parity_ptrs[0];

        for (size_t i = 0; i < cell_size; i++) {
            uint8_t val = p_data[i];
            for (int s = 0; s < stripe_cnt; s++) {
                if (s != fx) val ^= ((const uint8_t *)ec_ptrs[s])[i];
            }
            out[i] = val;
        }
    } else if (failed_cnt == 2 && parity_cnt >= 2) {
        int fx = failed_idx[0];
        int fy = failed_idx[1];
        /* Track output pointers BEFORE swapping indices so they follow
         * the caller's failed_idx order, not the sorted order. */
        uint8_t *out_x = (uint8_t *)out_ptrs[0];
        uint8_t *out_y = (uint8_t *)out_ptrs[1];
        if (fx > fy) {
            int t = fx; fx = fy; fy = t;
            /* Swap outputs to match: out_x still receives D_fx data */
            uint8_t *tp = out_x; out_x = out_y; out_y = tp;
        }

        const uint8_t *p_data = (const uint8_t *)parity_ptrs[0];
        const uint8_t *q_data = (const uint8_t *)parity_ptrs[1];

        uint8_t g_x = pow2_tbl[fx];
        uint8_t g_y = pow2_tbl[fy];
        uint8_t coeff = inv_tbl[g_y ^ g_x];

        for (size_t i = 0; i < cell_size; i++) {
            uint8_t s_p = p_data[i];
            for (int s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy)
                    s_p ^= ((const uint8_t *)ec_ptrs[s])[i];
            }
            uint8_t s_q = q_data[i];
            for (int s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy)
                    s_q ^= h_gf_mul_r(pow2_tbl[s],
                                      ((const uint8_t *)ec_ptrs[s])[i], reduce);
            }
            uint8_t dy = h_gf_mul_r(s_q ^ h_gf_mul_r(g_x, s_p, reduce), coeff, reduce);
            uint8_t dx = s_p ^ dy;
            out_x[i] = dx;
            out_y[i] = dy;
        }
    }
}

static inline uint8_t host_gf_mul_11d(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    int log_sum = (int)h_gf_log_11d[a] + (int)h_gf_log_11d[b];
    return h_gf_exp_11d[log_sum];
}

static inline uint8_t host_gf_inv_11d(uint8_t a) {
    return h_gf_inv_11d[a];
}

extern "C" {

void gpu_ec_gen_cauchy_matrix(uint8_t *matrix, int stripe_cnt, int parity_cnt)
{
    ensure_host_gf_tables();
    for (int r = 0; r < parity_cnt; r++) {
        for (int j = 0; j < stripe_cnt; j++) {
            uint8_t val = (uint8_t)((stripe_cnt + r) ^ j);
            matrix[r * stripe_cnt + j] = host_gf_inv_11d(val);
        }
    }
}

int gpu_ec_invert_matrix(uint8_t *matrix, uint8_t *inverse, int n)
{
    ensure_host_gf_tables();
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i * n + j] = (i == j) ? 1 : 0;
        }
    }
    
    for (int col = 0; col < n; col++) {
        int pivot_row = -1;
        for (int r = col; r < n; r++) {
            if (matrix[r * n + col] != 0) {
                pivot_row = r;
                break;
            }
        }
        if (pivot_row == -1) {
            return -1;
        }
        
        if (pivot_row != col) {
            for (int j = 0; j < n; j++) {
                uint8_t tmp = matrix[col * n + j];
                matrix[col * n + j] = matrix[pivot_row * n + j];
                matrix[pivot_row * n + j] = tmp;
                
                tmp = inverse[col * n + j];
                inverse[col * n + j] = inverse[pivot_row * n + j];
                inverse[pivot_row * n + j] = tmp;
            }
        }
        
        uint8_t pivot = matrix[col * n + col];
        uint8_t inv_pivot = host_gf_inv_11d(pivot);
        for (int j = 0; j < n; j++) {
            matrix[col * n + j] = host_gf_mul_11d(matrix[col * n + j], inv_pivot);
            inverse[col * n + j] = host_gf_mul_11d(inverse[col * n + j], inv_pivot);
        }
        
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            uint8_t coef = matrix[r * n + col];
            if (coef == 0) continue;
            for (int j = 0; j < n; j++) {
                matrix[r * n + j] ^= host_gf_mul_11d(matrix[col * n + j], coef);
                inverse[r * n + j] ^= host_gf_mul_11d(inverse[col * n + j], coef);
            }
        }
    }
    return 0;
}

int gpu_ec_make_decode_matrix(const uint8_t *encode_matrix,
                              int stripe_cnt, int parity_cnt,
                              const int *failed_idx, int failed_cnt,
                              uint8_t *decode_matrix)
{
    if (stripe_cnt <= 0 || stripe_cnt > 16 || failed_cnt <= 0 || failed_cnt > stripe_cnt) {
        return -1;
    }
    
    int surv_idx[16];
    int surv_cnt = 0;
    for (int i = 0; i < stripe_cnt; i++) {
        int failed = 0;
        for (int f = 0; f < failed_cnt; f++) {
            if (failed_idx[f] == i) {
                failed = 1;
                break;
            }
        }
        if (!failed) {
            if (surv_cnt >= stripe_cnt) return -1;
            surv_idx[surv_cnt++] = i;
        }
    }
    
    for (int i = 0; i < failed_cnt; i++) {
        if (i >= parity_cnt) return -1;
        if (surv_cnt >= stripe_cnt) return -1;
        surv_idx[surv_cnt++] = stripe_cnt + i;
    }
    
    if (surv_cnt != stripe_cnt) {
        return -1;
    }
    
    uint8_t a_surv[256];
    memset(a_surv, 0, sizeof(a_surv));
    for (int r = 0; r < stripe_cnt; r++) {
        int idx = surv_idx[r];
        if (idx < stripe_cnt) {
            a_surv[r * stripe_cnt + idx] = 1;
        } else {
            int p_idx = idx - stripe_cnt;
            for (int c = 0; c < stripe_cnt; c++) {
                a_surv[r * stripe_cnt + c] = encode_matrix[p_idx * stripe_cnt + c];
            }
        }
    }
    
    uint8_t inverse[256];
    if (gpu_ec_invert_matrix(a_surv, inverse, stripe_cnt) != 0) {
        return -1;
    }
    
    for (int f_idx = 0; f_idx < failed_cnt; f_idx++) {
        int fid = failed_idx[f_idx];
        for (int c = 0; c < stripe_cnt; c++) {
            decode_matrix[f_idx * stripe_cnt + c] = inverse[fid * stripe_cnt + c];
        }
    }
    
    return 0;
}

}
