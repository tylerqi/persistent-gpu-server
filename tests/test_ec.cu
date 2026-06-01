/**
 * test_ec.cu — EC parity correctness tests: GPU vs CPU reference
 */
#include "gpu_ec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Helper: allocate GPU stripes and fill with pattern */
static void **alloc_gpu_stripes(int n, size_t len, uint8_t **h_data)
{
    void **d_ptrs = (void **)malloc(sizeof(void *) * n);
    for (int i = 0; i < n; i++) {
        h_data[i] = (uint8_t *)malloc(len);
        for (size_t j = 0; j < len; j++)
            h_data[i][j] = (uint8_t)((i * 37 + j * 13 + 7) & 0xFF);
        cudaMalloc(&d_ptrs[i], len);
        cudaMemcpy(d_ptrs[i], h_data[i], len, cudaMemcpyHostToDevice);
    }
    return d_ptrs;
}

static void free_gpu_stripes(void **d_ptrs, uint8_t **h_data, int n)
{
    for (int i = 0; i < n; i++) {
        cudaFree(d_ptrs[i]);
        free(h_data[i]);
    }
    free(d_ptrs);
}

/* Test: 2+1 (2 data stripes, 1 parity) */
static int test_ec_2plus1(void)
{
    const int k = 2;
    const size_t len = 4096;
    uint8_t *h_data[2];

    void **d_ptrs = alloc_gpu_stripes(k, len, h_data);

    /* GPU parity */
    void *d_parity;
    cudaMalloc(&d_parity, len);
    int rc = gpu_ec_xor_parity(d_ptrs, k, len, d_parity);
    if (rc != 0) TEST_FAIL("ec_2plus1", "gpu_ec_xor_parity failed");

    uint8_t *gpu_parity = (uint8_t *)malloc(len);
    cudaMemcpy(gpu_parity, d_parity, len, cudaMemcpyDeviceToHost);

    /* CPU parity */
    uint8_t *cpu_parity = (uint8_t *)malloc(len);
    cpu_ec_xor_parity((const void **)h_data, k, len, cpu_parity);

    /* Compare */
    if (memcmp(gpu_parity, cpu_parity, len) != 0) {
        /* Find first mismatch */
        for (size_t i = 0; i < len; i++) {
            if (gpu_parity[i] != cpu_parity[i]) {
                char msg[128];
                snprintf(msg, sizeof(msg), "mismatch at byte %zu: GPU=0x%02X CPU=0x%02X",
                         i, gpu_parity[i], cpu_parity[i]);
                free(gpu_parity); free(cpu_parity);
                cudaFree(d_parity);
                free_gpu_stripes(d_ptrs, h_data, k);
                TEST_FAIL("ec_2plus1", msg);
            }
        }
    }

    free(gpu_parity); free(cpu_parity);
    cudaFree(d_parity);
    free_gpu_stripes(d_ptrs, h_data, k);
    TEST_PASS("ec_2plus1");
    return 0;
}

/* Test: 4+1 (4 data stripes) */
static int test_ec_4plus1(void)
{
    const int k = 4;
    const size_t len = 65536;
    uint8_t *h_data[4];

    void **d_ptrs = alloc_gpu_stripes(k, len, h_data);

    void *d_parity;
    cudaMalloc(&d_parity, len);
    gpu_ec_xor_parity(d_ptrs, k, len, d_parity);

    uint8_t *gpu_parity = (uint8_t *)malloc(len);
    cudaMemcpy(gpu_parity, d_parity, len, cudaMemcpyDeviceToHost);

    uint8_t *cpu_parity = (uint8_t *)malloc(len);
    cpu_ec_xor_parity((const void **)h_data, k, len, cpu_parity);

    if (memcmp(gpu_parity, cpu_parity, len) != 0)
        TEST_FAIL("ec_4plus1", "parity mismatch");

    free(gpu_parity); free(cpu_parity);
    cudaFree(d_parity);
    free_gpu_stripes(d_ptrs, h_data, k);
    TEST_PASS("ec_4plus1");
    return 0;
}

/* Test: 8+1 (8 data stripes) */
static int test_ec_8plus1(void)
{
    const int k = 8;
    const size_t len = 131072; /* 128 KB */
    uint8_t *h_data[8];

    void **d_ptrs = alloc_gpu_stripes(k, len, h_data);

    void *d_parity;
    cudaMalloc(&d_parity, len);
    gpu_ec_xor_parity(d_ptrs, k, len, d_parity);

    uint8_t *gpu_parity = (uint8_t *)malloc(len);
    cudaMemcpy(gpu_parity, d_parity, len, cudaMemcpyDeviceToHost);

    uint8_t *cpu_parity = (uint8_t *)malloc(len);
    cpu_ec_xor_parity((const void **)h_data, k, len, cpu_parity);

    if (memcmp(gpu_parity, cpu_parity, len) != 0)
        TEST_FAIL("ec_8plus1", "parity mismatch");

    free(gpu_parity); free(cpu_parity);
    cudaFree(d_parity);
    free_gpu_stripes(d_ptrs, h_data, k);
    TEST_PASS("ec_8plus1");
    return 0;
}

/* Test: verify parity correctness (XOR of all stripes + parity == 0) */
static int test_ec_verify(void)
{
    const int k = 4;
    const size_t len = 4096;
    uint8_t *h_data[4];

    void **d_ptrs = alloc_gpu_stripes(k, len, h_data);

    void *d_parity;
    cudaMalloc(&d_parity, len);
    gpu_ec_xor_parity(d_ptrs, k, len, d_parity);

    uint8_t *gpu_parity = (uint8_t *)malloc(len);
    cudaMemcpy(gpu_parity, d_parity, len, cudaMemcpyDeviceToHost);

    /* XOR all data + parity should be zero */
    uint8_t *check = (uint8_t *)calloc(1, len);
    for (int s = 0; s < k; s++) {
        for (size_t i = 0; i < len; i++)
            check[i] ^= h_data[s][i];
    }
    for (size_t i = 0; i < len; i++)
        check[i] ^= gpu_parity[i];

    for (size_t i = 0; i < len; i++) {
        if (check[i] != 0) TEST_FAIL("ec_verify", "XOR of data+parity != 0");
    }

    free(check); free(gpu_parity);
    cudaFree(d_parity);
    free_gpu_stripes(d_ptrs, h_data, k);
    TEST_PASS("ec_verify");
    return 0;
}

static inline uint8_t cpu_gf_mul2_byte(uint8_t val, int ec_mode)
{
    uint8_t reduce = (ec_mode == 1) ? 0x1D : 0x1B;
    return (uint8_t)(((val << 1) & 0xFE) ^ ((val >> 7) * reduce));
}

static void cpu_ec_encode_2(const void **data_ptrs, int stripe_cnt, size_t stripe_len, void **parity_ptrs, int ec_mode)
{
    uint8_t **parity_ptrs_u8 = (uint8_t **)parity_ptrs;
    const uint8_t **data_ptrs_u8 = (const uint8_t **)data_ptrs;

    // P parity (XOR)
    memset(parity_ptrs_u8[0], 0, stripe_len);
    for (int s = 0; s < stripe_cnt; s++) {
        for (size_t i = 0; i < stripe_len; i++) {
            parity_ptrs_u8[0][i] ^= data_ptrs_u8[s][i];
        }
    }

    // Q parity (Vandermonde Horner)
    for (size_t i = 0; i < stripe_len; i++) {
        uint8_t q = data_ptrs_u8[stripe_cnt - 1][i];
        for (int s = stripe_cnt - 2; s >= 0; s--) {
            q = cpu_gf_mul2_byte(q, ec_mode) ^ data_ptrs_u8[s][i];
        }
        parity_ptrs_u8[1][i] = q;
    }
}

static int test_ec_n_plus_2(int k, size_t len, int ec_mode, const char *name)
{
    uint8_t **h_data = (uint8_t **)malloc(sizeof(uint8_t *) * k);
    void **d_ptrs = alloc_gpu_stripes(k, len, h_data);

    void *d_parity[2];
    cudaMalloc(&d_parity[0], len);
    cudaMalloc(&d_parity[1], len);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int rc = gpu_ec_encode_multi_sm(d_ptrs, d_parity, k, 2, len, stream, ec_mode);
    cudaStreamSynchronize(stream);

    if (rc != 0) {
        cudaFree(d_parity[0]); cudaFree(d_parity[1]);
        cudaStreamDestroy(stream);
        free_gpu_stripes(d_ptrs, h_data, k);
        TEST_FAIL(name, "gpu_ec_encode_multi_sm failed");
    }

    uint8_t *gpu_parity[2];
    gpu_parity[0] = (uint8_t *)malloc(len);
    gpu_parity[1] = (uint8_t *)malloc(len);
    cudaMemcpy(gpu_parity[0], d_parity[0], len, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_parity[1], d_parity[1], len, cudaMemcpyDeviceToHost);

    uint8_t *cpu_parity[2];
    cpu_parity[0] = (uint8_t *)malloc(len);
    cpu_parity[1] = (uint8_t *)malloc(len);
    
    cpu_ec_encode_2((const void **)h_data, k, len, (void **)cpu_parity, ec_mode);

    if (memcmp(gpu_parity[0], cpu_parity[0], len) != 0 ||
        memcmp(gpu_parity[1], cpu_parity[1], len) != 0) {
        free(gpu_parity[0]); free(gpu_parity[1]);
        free(cpu_parity[0]); free(cpu_parity[1]);
        cudaFree(d_parity[0]); cudaFree(d_parity[1]);
        cudaStreamDestroy(stream);
        free_gpu_stripes(d_ptrs, h_data, k);
        TEST_FAIL(name, "parity mismatch");
    }

    free(gpu_parity[0]); free(gpu_parity[1]);
    free(cpu_parity[0]); free(cpu_parity[1]);
    cudaFree(d_parity[0]); cudaFree(d_parity[1]);
    cudaStreamDestroy(stream);
    free_gpu_stripes(d_ptrs, h_data, k);
    TEST_PASS(name);
    return 0;
}

static int test_ec_2plus2(void)
{
    return test_ec_n_plus_2(2, 4096, 0, "ec_2plus2");
}

static int test_ec_4plus2(void)
{
    return test_ec_n_plus_2(4, 65536, 0, "ec_4plus2");
}

static int test_ec_8plus2(void)
{
    return test_ec_n_plus_2(8, 131072, 0, "ec_8plus2");
}

int main(void)
{
    printf("=== test_ec ===\n");
    int failures = 0;
    failures += test_ec_2plus1();
    failures += test_ec_4plus1();
    failures += test_ec_8plus1();
    failures += test_ec_verify();
    failures += test_ec_2plus2();
    failures += test_ec_4plus2();
    failures += test_ec_8plus2();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
