/**
 * gpu_csum.cu — CRC32C and SHA256 implementations for GPU
 */
#include "gpu_csum.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * CRC32C (Castagnoli) — iSCSI polynomial 0x1EDC6F41
 * ═══════════════════════════════════════════════════════════════════════════ */

/* CRC32C lookup table — generated for polynomial 0x82F63B78 (bit-reversed)
 * Non-static: required for cudaMemcpyToSymbol with CUDA_SEPARABLE_COMPILATION */
__constant__ uint32_t crc32c_table[256];
static uint32_t h_crc32c_table[256];

/* Thread-safe init using pthread_once (H-3) */
static pthread_once_t crc32c_init_once = PTHREAD_ONCE_INIT;

static void do_init_crc32c_table(void)
{
    const uint32_t poly = 0x82F63B78u; /* Bit-reversed CRC32C polynomial */
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? poly : 0);
        }
        h_crc32c_table[i] = crc;
    }
    cudaError_t err = cudaMemcpyToSymbol(crc32c_table, h_crc32c_table, sizeof(h_crc32c_table));
    if (err != cudaSuccess) {
        fprintf(stderr, "CRITICAL: cudaMemcpyToSymbol(crc32c_table) failed: %s\n"
                        "  CRC32C computations will produce WRONG results!\n",
                cudaGetErrorString(err));
        cudaGetLastError(); /* consume error so it doesn't poison subsequent calls */
    }
}

static int init_crc32c_table(void)
{
    pthread_once(&crc32c_init_once, do_init_crc32c_table);
    return 0;
}

/* Device-side CRC32C — called by persistent kernel */
__device__ uint32_t device_crc32c(const uint8_t *data, size_t len)
{
    uint32_t crc = 0xFFFFFFFF;
    
    size_t i = 0;

    /* Only use vectorized loads if data is 16-byte aligned */
    if (((uintptr_t)data & 0xF) == 0) {
        const uint4 *data_u4 = (const uint4 *)data;
        size_t len_u4 = len / 16;
        
        for (size_t j = 0; j < len_u4; j++) {
            uint4 val = data_u4[j];
            
            /* Process 16 bytes sequentially but loaded in one 128-bit transaction */
            uint8_t *bytes = (uint8_t *)&val;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                uint8_t idx = (uint8_t)((crc ^ bytes[k]) & 0xFF);
                crc = (crc >> 8) ^ crc32c_table[idx];
            }
        }
        
        i = len_u4 * 16;
    }
    
    /* Tail bytes (or all bytes if unaligned) */
    for (; i < len; i++) {
        uint8_t idx = (uint8_t)((crc ^ data[i]) & 0xFF);
        crc = (crc >> 8) ^ crc32c_table[idx];
    }
    
    return crc ^ 0xFFFFFFFF;
}

/* Standalone kernel for direct API calls */
__global__ void crc32c_kernel(const uint8_t *data, size_t len, uint32_t *result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = device_crc32c(data, len);
    }
}

/* Host API: compute CRC32C on GPU data */
int gpu_crc32c(const void *gpu_data, size_t len, uint32_t *crc_out)
{
    if (!gpu_data || !crc_out || len == 0) return -1;

    init_crc32c_table();

    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));

    crc32c_kernel<<<1, 1>>>((const uint8_t *)gpu_data, len, d_result);

    cudaMemcpy(crc_out, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return 0;
}

/* Public init — called by gpu_engine_init to pre-populate __constant__ tables */
void gpu_csum_init(void)
{
    init_crc32c_table();
}

/* CPU reference CRC32C for verification */
static uint32_t cpu_crc32c_table[256];
static pthread_once_t cpu_crc32c_init_once = PTHREAD_ONCE_INIT;

static void do_init_cpu_crc32c_table(void)
{
    const uint32_t poly = 0x82F63B78u;
    for (int i = 0; i < 256; i++) {
        uint32_t crc = (uint32_t)i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? poly : 0);
        }
        cpu_crc32c_table[i] = crc;
    }
}

uint32_t cpu_crc32c(const void *data, size_t len)
{
    /* Thread-safe one-time table init (H-4) */
    pthread_once(&cpu_crc32c_init_once, do_init_cpu_crc32c_table);

    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ cpu_crc32c_table[(crc ^ p[i]) & 0xFF];
    }
    return crc ^ 0xFFFFFFFF;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SHA256
 * ═══════════════════════════════════════════════════════════════════════════ */

/* SHA256 constants */
static __constant__ uint32_t sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ static uint32_t sha256_rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ static uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ static uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ static uint32_t sha256_sig0(uint32_t x) { return sha256_rotr(x,2) ^ sha256_rotr(x,13) ^ sha256_rotr(x,22); }
__device__ static uint32_t sha256_sig1(uint32_t x) { return sha256_rotr(x,6) ^ sha256_rotr(x,11) ^ sha256_rotr(x,25); }
__device__ static uint32_t sha256_gam0(uint32_t x) { return sha256_rotr(x,7) ^ sha256_rotr(x,18) ^ (x >> 3); }
__device__ static uint32_t sha256_gam1(uint32_t x) { return sha256_rotr(x,17) ^ sha256_rotr(x,19) ^ (x >> 10); }

__device__ void device_sha256(const uint8_t *data, size_t len, uint8_t *out)
{
    uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

    /* Calculate padded length */
    size_t bit_len = len * 8;
    size_t padded = ((len + 8) / 64 + 1) * 64;

    /* Process 64-byte blocks */
    for (size_t block = 0; block < padded; block += 64) {
        uint32_t w[64];

        /* Build message schedule */
        for (int i = 0; i < 16; i++) {
            size_t pos = block + i * 4;
            uint32_t val = 0;
            for (int b = 0; b < 4; b++) {
                size_t idx = pos + b;
                uint8_t byte_val;
                if (idx < len)
                    byte_val = data[idx];
                else if (idx == len)
                    byte_val = 0x80;
                else if (idx >= padded - 8) {
                    int shift = (int)(7 - (idx - (padded - 8))) * 8;
                    byte_val = (uint8_t)((bit_len >> shift) & 0xFF);
                } else
                    byte_val = 0;
                val = (val << 8) | byte_val;
            }
            w[i] = val;
        }
        for (int i = 16; i < 64; i++) {
            w[i] = sha256_gam1(w[i-2]) + w[i-7] + sha256_gam0(w[i-15]) + w[i-16];
        }

        uint32_t a=h0, b=h1, c=h2, d=h3, e=h4, f=h5, g=h6, hh=h7;
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + sha256_sig1(e) + sha256_ch(e,f,g) + sha256_k[i] + w[i];
            uint32_t t2 = sha256_sig0(a) + sha256_maj(a,b,c);
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=hh;
    }

    /* Output */
    uint32_t hash[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (hash[i] >> 24) & 0xFF;
        out[i*4+1] = (hash[i] >> 16) & 0xFF;
        out[i*4+2] = (hash[i] >> 8)  & 0xFF;
        out[i*4+3] =  hash[i]        & 0xFF;
    }
}

/* Standalone SHA256 kernel */
__global__ void sha256_kernel(const uint8_t *data, size_t len, uint8_t *result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        device_sha256(data, len, result);
    }
}

int gpu_sha256(const void *gpu_data, size_t len, uint8_t hash_out[32])
{
    if (!gpu_data || !hash_out || len == 0) return -1;

    uint8_t *d_result;
    cudaMalloc(&d_result, 32);

    sha256_kernel<<<1, 1>>>((const uint8_t *)gpu_data, len, d_result);

    cudaMemcpy(hash_out, d_result, 32, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    return 0;
}
