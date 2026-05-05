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

int main(void)
{
    printf("=== test_ec ===\n");
    int failures = 0;
    failures += test_ec_2plus1();
    failures += test_ec_4plus1();
    failures += test_ec_8plus1();
    failures += test_ec_verify();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
