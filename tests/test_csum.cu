/**
 * test_csum.cu — Checksum correctness tests: GPU vs CPU reference
 */
#include "gpu_csum.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Test CRC32C: known test vector */
static int test_crc32c_known(void)
{
    /* "123456789" should produce CRC32C = 0xE3069283 */
    const char *test_str = "123456789";
    size_t len = 9;

    uint32_t cpu_crc = cpu_crc32c(test_str, len);
    if (cpu_crc != 0xE3069283) {
        char msg[128];
        snprintf(msg, sizeof(msg), "CPU CRC32C mismatch: got 0x%08X, expected 0xE3069283", cpu_crc);
        TEST_FAIL("crc32c_known", msg);
    }

    /* GPU */
    void *d_data;
    cudaMalloc(&d_data, len);
    cudaMemcpy(d_data, test_str, len, cudaMemcpyHostToDevice);

    uint32_t gpu_crc;
    gpu_crc32c(d_data, len, &gpu_crc);
    cudaFree(d_data);

    if (gpu_crc != cpu_crc) {
        char msg[128];
        snprintf(msg, sizeof(msg), "GPU vs CPU mismatch: GPU=0x%08X CPU=0x%08X", gpu_crc, cpu_crc);
        TEST_FAIL("crc32c_known", msg);
    }

    TEST_PASS("crc32c_known");
    return 0;
}

/* Test CRC32C: various buffer sizes */
static int test_crc32c_sizes(void)
{
    size_t sizes[] = {1, 7, 64, 256, 1024, 4096, 65536};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < num_sizes; s++) {
        size_t len = sizes[s];
        uint8_t *h_data = (uint8_t *)malloc(len);
        /* Fill with deterministic pattern */
        for (size_t i = 0; i < len; i++) h_data[i] = (uint8_t)((i * 31 + 17) & 0xFF);

        uint32_t cpu_crc = cpu_crc32c(h_data, len);

        void *d_data;
        cudaMalloc(&d_data, len);
        cudaMemcpy(d_data, h_data, len, cudaMemcpyHostToDevice);

        uint32_t gpu_crc;
        gpu_crc32c(d_data, len, &gpu_crc);
        cudaFree(d_data);
        free(h_data);

        if (gpu_crc != cpu_crc) {
            char msg[128];
            snprintf(msg, sizeof(msg), "size=%zu: GPU=0x%08X CPU=0x%08X", len, gpu_crc, cpu_crc);
            TEST_FAIL("crc32c_sizes", msg);
        }
    }

    TEST_PASS("crc32c_sizes");
    return 0;
}

/* Test CRC32C: zeros */
static int test_crc32c_zeros(void)
{
    size_t len = 4096;
    uint8_t *h_data = (uint8_t *)calloc(1, len);

    uint32_t cpu_crc = cpu_crc32c(h_data, len);

    void *d_data;
    cudaMalloc(&d_data, len);
    cudaMemcpy(d_data, h_data, len, cudaMemcpyHostToDevice);

    uint32_t gpu_crc;
    gpu_crc32c(d_data, len, &gpu_crc);
    cudaFree(d_data);
    free(h_data);

    if (gpu_crc != cpu_crc) {
        char msg[128];
        snprintf(msg, sizeof(msg), "zeros: GPU=0x%08X CPU=0x%08X", gpu_crc, cpu_crc);
        TEST_FAIL("crc32c_zeros", msg);
    }

    TEST_PASS("crc32c_zeros");
    return 0;
}

/* Test SHA256: known test vector */
static int test_sha256_known(void)
{
    /* SHA256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad */
    const char *test_str = "abc";
    size_t len = 3;
    uint8_t expected[32] = {
        0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,
        0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
        0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,
        0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad
    };

    void *d_data;
    cudaMalloc(&d_data, len);
    cudaMemcpy(d_data, test_str, len, cudaMemcpyHostToDevice);

    uint8_t result[32];
    gpu_sha256(d_data, len, result);
    cudaFree(d_data);

    if (memcmp(result, expected, 32) != 0) {
        printf("  SHA256 result: ");
        for (int i = 0; i < 32; i++) printf("%02x", result[i]);
        printf("\n  Expected:      ");
        for (int i = 0; i < 32; i++) printf("%02x", expected[i]);
        printf("\n");
        TEST_FAIL("sha256_known", "hash mismatch");
    }

    TEST_PASS("sha256_known");
    return 0;
}

/* Test SHA256: single zero byte (gpu_sha256 rejects len=0) */
static int test_sha256_single_byte(void)
{
    /* SHA256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 */
    uint8_t expected[32] = {
        0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,
        0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
        0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,
        0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55
    };

    /* gpu_sha256 requires non-zero length, so skip empty test */
    /* Test with single byte instead */
    uint8_t byte = 0x00;
    void *d_data;
    cudaMalloc(&d_data, 1);
    cudaMemcpy(d_data, &byte, 1, cudaMemcpyHostToDevice);

    uint8_t result[32];
    gpu_sha256(d_data, 1, result);
    cudaFree(d_data);

    /* Just verify it doesn't crash and produces 32 bytes */
    int all_zero = 1;
    for (int i = 0; i < 32; i++) {
        if (result[i] != 0) { all_zero = 0; break; }
    }
    if (all_zero) TEST_FAIL("sha256_single_byte", "result is all zeros");

    TEST_PASS("sha256_single_byte");
    return 0;
}

int main(void)
{
    printf("=== test_csum ===\n");
    int failures = 0;
    failures += test_crc32c_known();
    failures += test_crc32c_sizes();
    failures += test_crc32c_zeros();
    failures += test_sha256_known();
    failures += test_sha256_single_byte();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
