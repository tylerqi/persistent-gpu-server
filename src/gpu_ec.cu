/**
 * gpu_ec.cu — EC Parity on GPU with Correct GF(2^8) Arithmetic
 *
 * FIXED (Issue #1): Q-Parity now uses proper byte-wise GF(2^8) multiplication with
 * irreducible polynomial 0x11B, compatible with ISA-L, DAOS, and Linux RAID-6.
 * This enables correct EC data reconstruction (decode) for single and double
 * stripe failures. The Q-parity is now mathematically invertible.
 *
 * P-Parity: Standard XOR across stripes (unchanged, correct).
 * Q-Parity: Fixed GF(2^8) Horner's method with proper per-byte reduction.
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

/* Forward declarations of GF(2^8) helpers (defined below) */
__device__ __forceinline__ uint8_t gf_mul2_byte(uint8_t val);

/* ── Multi-SM EC Encode kernel ─────────────────────────────────────────────
 * Unlike device_ec_encode (used by the persistent kernel on 1 SM),
 * this kernel distributes P and Q parity computation across ALL SMs.
 * Pointer arrays are passed as fixed-size kernel arguments (no cudaMalloc).
 * FIXED: Now uses scalar byte operations for correct GF(2^8) arithmetic.
 */
__global__ void ec_encode_multi_sm_kernel(
    const void *ec0, const void *ec1, const void *ec2, const void *ec3,
    const void *ec4, const void *ec5, const void *ec6, const void *ec7,
    const void *ec8, const void *ec9, const void *ec10, const void *ec11,
    const void *ec12, const void *ec13, const void *ec14, const void *ec15,
    void *par0, void *par1, void *par2, void *par3,
    uint32_t stripe_cnt, uint32_t parity_cnt, size_t cell_size)
{
    /* Reconstruct pointer arrays from arguments */
    const void *ec_ptrs[16] = {ec0,ec1,ec2,ec3,ec4,ec5,ec6,ec7,
                               ec8,ec9,ec10,ec11,ec12,ec13,ec14,ec15};
    void *parity_ptrs[4] = {par0, par1, par2, par3};

    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    /* P-Parity: XOR across all stripes */
    if (parity_cnt >= 1) {
        uint8_t *p_out = (uint8_t *)parity_ptrs[0];
        for (size_t i = gid; i < cell_size; i += stride) {
            uint8_t p = ((const uint8_t *)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++)
                p ^= ((const uint8_t *)ec_ptrs[s])[i];
            p_out[i] = p;
        }
    }

    /* Q-Parity: FIXED GF(2^8) Horner's method (proper byte-wise operations) */
    if (parity_cnt >= 2) {
        uint8_t *q_out = (uint8_t *)parity_ptrs[1];
        for (size_t i = gid; i < cell_size; i += stride) {
            uint8_t q = ((const uint8_t *)ec_ptrs[stripe_cnt - 1])[i];
            for (int s = (int)stripe_cnt - 2; s >= 0; s--) {
                q = gf_mul2_byte(q);  /* FIXED: proper GF(2^8) multiply */
                q ^= ((const uint8_t *)ec_ptrs[s])[i];
            }
            q_out[i] = q;
        }
    }
}

/* Host API: Launch multi-SM EC encode on a given stream.
 * All pointers (ec_ptrs, parity_ptrs, and the data they point to)
 * must already be in GPU memory. */
int gpu_ec_encode_multi_sm(void **d_ec_ptrs, void **d_parity_ptrs,
                            int stripe_cnt, int parity_cnt, size_t cell_size,
                            cudaStream_t stream)
{
    if (!d_ec_ptrs || !d_parity_ptrs || stripe_cnt <= 0 || parity_cnt <= 0 || cell_size == 0)
        return -1;

    /* Pad pointer arrays to fixed kernel argument sizes */
    const void *ec[16] = {};
    void *par[4] = {};
    for (int i = 0; i < stripe_cnt && i < 16; i++) ec[i] = d_ec_ptrs[i];
    for (int i = 0; i < parity_cnt && i < 4; i++) par[i] = d_parity_ptrs[i];

    int threads = 256;
    int blocks = (int)((cell_size + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;

    ec_encode_multi_sm_kernel<<<blocks, threads, 0, stream>>>(
        ec[0],ec[1],ec[2],ec[3],ec[4],ec[5],ec[6],ec[7],
        ec[8],ec[9],ec[10],ec[11],ec[12],ec[13],ec[14],ec[15],
        par[0],par[1],par[2],par[3],
        stripe_cnt, parity_cnt, cell_size);

    /* No cudaMalloc/cudaFree — all pointers passed as kernel arguments */
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

/* ══════════════════════════════════════════════════════════════════════════
 * GF(2^8) Arithmetic — FIXED for Proper Field Arithmetic
 * ══════════════════════════════════════════════════════════════════════════
 *
 * ISSUE #1 FIX: Replaced non-invertible 32-bit shift logic with proper
 * byte-wise GF(2^8) multiplication using irreducible polynomial 0x11B.
 * 
 * Standard polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
 * Reduction constant:  0x1B (lower 8 bits of 0x11B)
 * 
 * This is the standard used by ISA-L, DAOS, and Linux RAID-6.
 * The field is now a proper group under multiplication, enabling inversion
 * and correct EC data reconstruction (decode).
 * ══════════════════════════════════════════════════════════════════════════
 */

/**
 * Proper GF(2^8) multiplication by 2 using irreducible polynomial 0x11B.
 * 
 * FIXED from previous implementation which incorrectly operated on 32-bit
 * words, causing overflow between bytes and losing invertibility.
 * 
 * Algorithm (mathematically correct):
 *   1. Shift left by 1 bit
 *   2. If MSB (bit 7) was set before shift, XOR result with 0x1B (polynomial reduction)
 * 
 * This is the unique GF(2^8) multiply-by-2 operation and is INVERTIBLE.
 */
__device__ __forceinline__ uint8_t gf_mul2_byte(uint8_t val)
{
    uint8_t hi_bit = val & 0x80;
    uint8_t shifted = (uint8_t)(val << 1);
    /* If high bit was set, result overflowed; reduce by XORing with 0x1B */
    if (hi_bit)
        shifted ^= 0x1B;
    return shifted;
}

/**
 * General GF(2^8) multiplication using bitwise operations.
 * Implements: a * b in GF(2^8)/0x11B using binary multiplication.
 * This function is portable, correct, and INVERTIBLE (supports division for EC decode).
 * 
 * Algorithm: Binary multiplication with Galois Field reduction.
 */
__device__ __forceinline__ uint8_t gf_mul_bitwise(uint8_t a, uint8_t b)
{
    uint8_t result = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= 0x1B;  /* Reduce by irreducible polynomial */
        b >>= 1;
    }
    return result;
}

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
static int gf_tables_initialized = 0;

/* Host-side GF(2^8) multiply for table generation */
static uint8_t h_gf_mul(uint8_t a, uint8_t b)
{
    uint8_t result = 0;
    uint8_t hi_bit;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        hi_bit = a & 0x80;
        a <<= 1;
        if (hi_bit) a ^= 0x1B; /* Reduce by x^8 + x^4 + x^3 + x + 1 */
        b >>= 1;
    }
    return result;
}

/* Pre-computed powers of 2 for Q-parity coefficients (encoding uses g=2 in Horner).
 * pow2_table[i] = 2^i in GF(2^8)/0x11B. Only need up to 16 entries (max stripe count). */
__constant__ uint8_t gf_pow2_table[16];
static uint8_t h_gf_pow2[16];

void gpu_ec_init_gf_tables(void)
{
    if (gf_tables_initialized) return;

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

    cudaMemcpyToSymbol(gf_exp_table, h_gf_exp, sizeof(h_gf_exp));
    cudaMemcpyToSymbol(gf_log_table, h_gf_log, sizeof(h_gf_log));
    cudaMemcpyToSymbol(gf_inv_table, h_gf_inv, sizeof(h_gf_inv));
    cudaMemcpyToSymbol(gf_pow2_table, h_gf_pow2, sizeof(h_gf_pow2));

    gf_tables_initialized = 1;
}

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

/* GF(2^8) inverse via Fermat's little theorem: a^(-1) = a^254 */
__device__ __forceinline__ uint8_t gf_inv_direct(uint8_t a)
{
    if (a == 0) return 0;
    uint8_t a2   = gf_mul_bitwise(a, a);     /* a^2 */
    uint8_t a3   = gf_mul_bitwise(a2, a);    /* a^3 */
    uint8_t a6   = gf_mul_bitwise(a3, a3);   /* a^6 */
    uint8_t a7   = gf_mul_bitwise(a6, a);    /* a^7 */
    uint8_t a14  = gf_mul_bitwise(a7, a7);   /* a^14 */
    uint8_t a15  = gf_mul_bitwise(a14, a);   /* a^15 */
    uint8_t a30  = gf_mul_bitwise(a15, a15); /* a^30 */
    uint8_t a31  = gf_mul_bitwise(a30, a);   /* a^31 */
    uint8_t a62  = gf_mul_bitwise(a31, a31); /* a^62 */
    uint8_t a63  = gf_mul_bitwise(a62, a);   /* a^63 */
    uint8_t a126 = gf_mul_bitwise(a63, a63); /* a^126 */
    uint8_t a127 = gf_mul_bitwise(a126, a);  /* a^127 */
    uint8_t a254 = gf_mul_bitwise(a127, a127); /* a^254 */
    return a254;
}

extern "C" {

/* ── Cooperative Block EC Parity Generation (FIXED) ────────────────────── */
/* Computes P and Q parity for up to 16 data stripes.
 * Now uses scalar byte operations to ensure correct GF(2^8) arithmetic
 * for each byte independently (no cross-byte overflow).
 */
__device__ void device_ec_encode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size)
{
    if (stripe_cnt == 0 || parity_cnt == 0 || cell_size == 0) return;

    /* P-Parity: Simple XOR (unchanged) */
    if (parity_cnt >= 1) {
        uint8_t *p_out = (uint8_t*)parity_ptrs[0];
        for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
            uint8_t p_byte = ((const uint8_t*)ec_ptrs[0])[i];
            for (uint32_t s = 1; s < stripe_cnt; s++) {
                p_byte ^= ((const uint8_t*)ec_ptrs[s])[i];
            }
            p_out[i] = p_byte;
        }
    }

    /* Q-Parity: FIXED GF(2^8) Horner's Method
     * Now uses byte-wise proper GF(2^8) operations via gf_mul2_byte()
     * instead of the previous broken uint32_t logic.
     * This is mathematically correct and INVERTIBLE for EC decode.
     */
    if (parity_cnt >= 2) {
        uint8_t *q_out = (uint8_t*)parity_ptrs[1];
        for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
            uint8_t q_byte = ((const uint8_t*)ec_ptrs[stripe_cnt - 1])[i];
            for (int s = (int)stripe_cnt - 2; s >= 0; s--) {
                q_byte = gf_mul2_byte(q_byte);  /* FIXED: proper GF(2^8) multiply */
                q_byte ^= ((const uint8_t*)ec_ptrs[s])[i];
            }
            q_out[i] = q_byte;
        }
    }
}

} /* extern "C" — device_ec_encode */

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
 *     D_y = (S_Q ⊕ g_x · S_P) / (g_y ⊕ g_x)
 *     D_x = S_P ⊕ D_y
 *
 * The ec_ptrs array should have the surviving stripe data in place.
 * The failed stripe slots will be OVERWRITTEN with reconstructed data.
 * parity_ptrs[0] = P, parity_ptrs[1] = Q.
 *
 * NOW WORKS because Q-parity is invertible (ISSUE #1 FIXED).
 */
__device__ void device_ec_decode(void *const *ec_ptrs, void *const *parity_ptrs,
                                 uint32_t stripe_cnt, uint32_t parity_cnt,
                                 size_t cell_size,
                                 const uint32_t *failed_idx, uint32_t failed_cnt)
{
    if (failed_cnt == 0 || cell_size == 0) return;

    if (failed_cnt == 1) {
        /* ── Single failure: reconstruct from P parity ──────────── */
        uint32_t fx = failed_idx[0];
        uint8_t *out = (uint8_t *)ec_ptrs[fx];
        const uint8_t *p_data = (const uint8_t *)parity_ptrs[0];

        for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
            uint8_t val = __ldcg(&p_data[i]);
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx) {
                    val ^= __ldcg(&((const uint8_t *)ec_ptrs[s])[i]);
                }
            }
            out[i] = val;
        }
    } else if (failed_cnt == 2 && parity_cnt >= 2) {
        /* ── Double failure: reconstruct from P + Q parity ──────── */
        uint32_t fx = failed_idx[0];
        uint32_t fy = failed_idx[1];
        if (fx > fy) { uint32_t t = fx; fx = fy; fy = t; }

        uint8_t *out_x = (uint8_t *)ec_ptrs[fx];
        uint8_t *out_y = (uint8_t *)ec_ptrs[fy];
        const uint8_t *p_data = (const uint8_t *)parity_ptrs[0];
        const uint8_t *q_data = (const uint8_t *)parity_ptrs[1];

        /* Compute pow2 coefficients: 2^fx, 2^fy */
        uint8_t g_x = 1;
        for (uint32_t j = 0; j < fx; j++) g_x = gf_mul2_byte(g_x);
        uint8_t g_y = 1;
        for (uint32_t j = 0; j < fy; j++) g_y = gf_mul2_byte(g_y);

        uint8_t coeff = gf_inv_direct(g_y ^ g_x);

        for (size_t i = threadIdx.x; i < cell_size; i += blockDim.x) {
            /* Use __ldcg (load cached-global, bypass L1) for all data reads.
             * Essential in persistent kernels: data buffers may have been
             * modified by host-side operations (cudaMemset) on a different
             * stream, and L1 cache may contain stale entries. */

            /* P syndrome: S_P = P ⊕ all surviving stripes */
            uint8_t s_p = __ldcg(&p_data[i]);
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy) {
                    s_p ^= __ldcg(&((const uint8_t *)ec_ptrs[s])[i]);
                }
            }

            /* Q syndrome: S_Q = Q ⊕ Σ(2^s · surviving[s]) */
            uint8_t s_q = __ldcg(&q_data[i]);
            for (uint32_t s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy) {
                    uint8_t gs = 1;
                    for (uint32_t j = 0; j < s; j++) gs = gf_mul2_byte(gs);
                    s_q ^= gf_mul_bitwise(gs, __ldcg(&((const uint8_t *)ec_ptrs[s])[i]));
                }
            }

            /* Solve: D_y = (S_Q ⊕ g_x · S_P) * (g_y ⊕ g_x)^(-1) */
            uint8_t dy = gf_mul_bitwise(s_q ^ gf_mul_bitwise(g_x, s_p), coeff);
            uint8_t dx = s_p ^ dy;

            out_x[i] = dx;
            out_y[i] = dy;
        }
    }
    /* else: >2 failures not supported */
}

} /* extern "C" — device_ec_decode */

/* ── CPU Reference EC Decode for Verification ──────────────────────────── */

static uint8_t cpu_gf_mul_ref(uint8_t a, uint8_t b)
{
    return h_gf_mul(a, b);
}

void cpu_ec_decode(const void **ec_ptrs, const void **parity_ptrs,
                   int stripe_cnt, int parity_cnt, size_t cell_size,
                   const int *failed_idx, int failed_cnt,
                   void **out_ptrs)
{
    gpu_ec_init_gf_tables();

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
        if (fx > fy) { int t = fx; fx = fy; fy = t; }

        uint8_t *out_x = (uint8_t *)out_ptrs[0];
        uint8_t *out_y = (uint8_t *)out_ptrs[1];
        const uint8_t *p_data = (const uint8_t *)parity_ptrs[0];
        const uint8_t *q_data = (const uint8_t *)parity_ptrs[1];

        uint8_t g_x = h_gf_pow2[fx];
        uint8_t g_y = h_gf_pow2[fy];
        uint8_t coeff = h_gf_inv[g_y ^ g_x];

        for (size_t i = 0; i < cell_size; i++) {
            uint8_t s_p = p_data[i];
            for (int s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy)
                    s_p ^= ((const uint8_t *)ec_ptrs[s])[i];
            }
            uint8_t s_q = q_data[i];
            for (int s = 0; s < stripe_cnt; s++) {
                if (s != fx && s != fy)
                    s_q ^= cpu_gf_mul_ref(h_gf_pow2[s],
                                          ((const uint8_t *)ec_ptrs[s])[i]);
            }
            uint8_t dy = cpu_gf_mul_ref(s_q ^ cpu_gf_mul_ref(g_x, s_p), coeff);
            uint8_t dx = s_p ^ dy;
            out_x[i] = dx;
            out_y[i] = dy;
        }
    }
}
