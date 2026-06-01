/**
 * test_ec_isal_compat.cu — Erasure Coding Compatibility Tests (ISA-L / Cauchy Mode)
 */
#include "gpu_engine.h"
#include "gpu_ec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Helper: allocate and fill host buffer with pseudo-random data */
static uint8_t *alloc_host_buffer(size_t len, int seed)
{
    uint8_t *buf = (uint8_t *)malloc(len);
    for (size_t i = 0; i < len; i++) {
        buf[i] = (uint8_t)((i * seed + 17) & 0xFF);
    }
    return buf;
}

/* Host GF(2^8)/0x11D bitwise multiplication for verification reference */
static uint8_t host_gf_mul_bitwise_11d(uint8_t a, uint8_t b)
{
    uint8_t result = 0;
    uint8_t reduce = 0x1D;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= reduce;
        b >>= 1;
    }
    return result;
}

/* CPU reference encode for ISA-L Cauchy mode */
static void cpu_isal_encode(const uint8_t *encode_matrix, const uint8_t **h_data, uint8_t **h_parity, int k, int p, size_t len)
{
    for (int r = 0; r < p; r++) {
        for (size_t i = 0; i < len; i++) {
            uint8_t val = 0;
            for (int j = 0; j < k; j++) {
                uint8_t coef = encode_matrix[r * k + j];
                val ^= host_gf_mul_bitwise_11d(coef, h_data[j][i]);
            }
            h_parity[r][i] = val;
        }
    }
}

/* Pre-allocated GPU buffers for test suite */
struct test_buffers {
    void *ec_data[16];      /* Up to 16 data stripes */
    void *ec_parity[4];     /* Up to 4 parity stripes */
    size_t cell_size;
};

static int alloc_test_buffers(struct test_buffers *buf, int k, int p, size_t cell_size)
{
    buf->cell_size = cell_size;
    for (int i = 0; i < k; i++) {
        if (cudaMalloc(&buf->ec_data[i], cell_size) != cudaSuccess)
            return -1;
    }
    for (int i = 0; i < p; i++) {
        if (cudaMalloc(&buf->ec_parity[i], cell_size) != cudaSuccess)
            return -1;
    }
    return 0;
}

static void free_test_buffers(struct test_buffers *buf, int k, int p)
{
    for (int i = 0; i < k; i++) cudaFree(buf->ec_data[i]);
    for (int i = 0; i < p; i++) cudaFree(buf->ec_parity[i]);
}

/* ── Test 1: Cauchy Matrix Generation and Invertibility ──────── */
static int test_cauchy_invertibility(void)
{
    uint8_t encode_matrix[64];
    uint8_t decode_matrix[32];
    int failed_idx[2];

    gpu_ec_init_gf_tables();

    /* Try a variety of k, p configurations */
    int test_configs[][2] = {
        {4, 2}, {6, 3}, {8, 4}, {10, 4}, {12, 4}, {16, 4}
    };

    for (size_t c = 0; c < sizeof(test_configs) / sizeof(test_configs[0]); c++) {
        int k = test_configs[c][0];
        int p = test_configs[c][1];

        gpu_ec_gen_cauchy_matrix(encode_matrix, k, p);

        /* Test all single failure cases */
        for (int i = 0; i < k; i++) {
            failed_idx[0] = i;
            int rc = gpu_ec_make_decode_matrix(encode_matrix, k, p, failed_idx, 1, decode_matrix);
            if (rc != 0) {
                printf("  [DIAG] Single failure invert failed for k=%d, p=%d, failed=%d\n", k, p, i);
                TEST_FAIL("cauchy_invertibility", "Make decode matrix failed for single failure");
            }
        }

        /* Test a subset of double failure cases */
        for (int i = 0; i < k; i++) {
            for (int j = i + 1; j < k; j++) {
                failed_idx[0] = i;
                failed_idx[1] = j;
                int rc = gpu_ec_make_decode_matrix(encode_matrix, k, p, failed_idx, 2, decode_matrix);
                if (rc != 0) {
                    printf("  [DIAG] Double failure invert failed for k=%d, p=%d, failed={%d,%d}\n", k, p, i, j);
                    TEST_FAIL("cauchy_invertibility", "Make decode matrix failed for double failure");
                }
            }
        }
    }

    TEST_PASS("cauchy_invertibility");
    return 0;
}

/* ── Test 2: GPU Encode vs CPU reference ────────────────────── */
static int test_isal_encode_vs_cpu(gpu_engine_t *eng, struct test_buffers *buf)
{
    const int k = 4, p = 2;
    const size_t len = 4096;

    /* Generate data on host */
    uint8_t *h_orig[4];
    const uint8_t *h_orig_const[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 19 + 7);
        h_orig_const[i] = h_orig[i];
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);

    /* Generate Cauchy Matrix */
    uint8_t encode_matrix[64];
    gpu_ec_gen_cauchy_matrix(encode_matrix, k, p);

    /* Run CPU encode */
    uint8_t *h_parity_cpu[2];
    for (int i = 0; i < p; i++) {
        h_parity_cpu[i] = (uint8_t *)malloc(len);
    }
    cpu_isal_encode(encode_matrix, h_orig_const, h_parity_cpu, k, p, len);

    /* Run GPU encode */
    cudaMemset(buf->ec_parity[0], 0, len);
    cudaMemset(buf->ec_parity[1], 0, len);
    cudaStreamSynchronize(0);

    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k;
    enc_item.parity_cnt = p;
    enc_item.cell_size = len;
    enc_item.ec_mode = GPU_EC_MODE_ISAL;
    memcpy(enc_item.ec_encode_matrix, encode_matrix, k * p);

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_FAIL("isal_encode_vs_cpu", "GPU encode failed");
    }

    /* Compare outputs */
    uint8_t *h_parity_gpu[2];
    for (int i = 0; i < p; i++) {
        h_parity_gpu[i] = (uint8_t *)malloc(len);
        cudaMemcpy(h_parity_gpu[i], buf->ec_parity[i], len, cudaMemcpyDeviceToHost);
        if (memcmp(h_parity_gpu[i], h_parity_cpu[i], len) != 0) {
            printf("  [DIAG] Parity mismatch on row %d\n", i);
            TEST_FAIL("isal_encode_vs_cpu", "GPU parity does not match CPU reference");
        }
    }

    /* Cleanup */
    for (int i = 0; i < k; i++) free(h_orig[i]);
    for (int i = 0; i < p; i++) {
        free(h_parity_cpu[i]);
        free(h_parity_gpu[i]);
    }

    TEST_PASS("isal_encode_vs_cpu");
    return 0;
}

/* ── Test 3: Cauchy decode roundtrip (single/double failures) ── */
static int test_isal_roundtrip(gpu_engine_t *eng, struct test_buffers *buf)
{
    const int k = 6, p = 4;
    const size_t len = buf->cell_size;

    /* Generate data on host */
    uint8_t *h_orig[6];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 23 + 11);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);

    /* Generate Cauchy Matrix */
    uint8_t encode_matrix[64];
    gpu_ec_gen_cauchy_matrix(encode_matrix, k, p);

    /* GPU encode */
    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k;
    enc_item.parity_cnt = p;
    enc_item.cell_size = len;
    enc_item.ec_mode = GPU_EC_MODE_ISAL;
    memcpy(enc_item.ec_encode_matrix, encode_matrix, k * p);

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("isal_roundtrip", "encode failed");
    }

    /* Test all single-failure cases */
    for (int failed = 0; failed < k; failed++) {
        /* backup */
        uint8_t *backup = (uint8_t *)malloc(len);
        cudaMemcpy(backup, buf->ec_data[failed], len, cudaMemcpyDeviceToHost);

        /* simulate failure */
        cudaMemset(buf->ec_data[failed], 0, len);
        cudaStreamSynchronize(0);

        /* Make decode matrix */
        uint8_t decode_matrix[32];
        int failed_arr[1] = { failed };
        rc = gpu_ec_make_decode_matrix(encode_matrix, k, p, failed_arr, 1, decode_matrix);
        if (rc != 0) {
            free(backup);
            for (int i = 0; i < k; i++) free(h_orig[i]);
            TEST_FAIL("isal_roundtrip", "make single-failure decode matrix failed");
        }

        /* Submit decode */
        gpu_work_item_t dec_item = {};
        dec_item.op_type = GPU_OP_EC_DECODE;
        for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
        for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
        dec_item.stripe_cnt = k;
        dec_item.parity_cnt = p;
        dec_item.cell_size = len;
        dec_item.failed_idx[0] = failed;
        dec_item.failed_cnt = 1;
        dec_item.ec_mode = GPU_EC_MODE_ISAL;
        memcpy(dec_item.ec_decode_matrix, decode_matrix, k);

        rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
        if (rc != 0 || res.error_code != 0) {
            free(backup);
            for (int i = 0; i < k; i++) free(h_orig[i]);
            TEST_FAIL("isal_roundtrip", "single-failure GPU decode failed");
        }

        /* Verify */
        uint8_t *recon = (uint8_t *)malloc(len);
        cudaMemcpy(recon, buf->ec_data[failed], len, cudaMemcpyDeviceToHost);
        if (memcmp(recon, backup, len) != 0) {
            free(recon); free(backup);
            for (int i = 0; i < k; i++) free(h_orig[i]);
            TEST_FAIL("isal_roundtrip", "single-failure verification failed");
        }
        free(recon);
        free(backup);
    }

    /* Test all double-failure cases */
    for (int fx = 0; fx < k; fx++) {
        for (int fy = fx + 1; fy < k; fy++) {
            /* Restore original data */
            for (int i = 0; i < k; i++) {
                cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
            }
            cudaStreamSynchronize(0);

            /* Re-encode to clean parities */
            rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
            if (rc != 0 || res.error_code != 0) {
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("isal_roundtrip", "re-encode failed");
            }

            /* Simulate failures */
            cudaMemset(buf->ec_data[fx], 0, len);
            cudaMemset(buf->ec_data[fy], 0, len);
            cudaStreamSynchronize(0);

            /* Make decode matrix */
            uint8_t decode_matrix[32];
            int failed_arr[2] = { fx, fy };
            rc = gpu_ec_make_decode_matrix(encode_matrix, k, p, failed_arr, 2, decode_matrix);
            if (rc != 0) {
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("isal_roundtrip", "make double-failure decode matrix failed");
            }

            /* Submit decode */
            gpu_work_item_t dec_item = {};
            dec_item.op_type = GPU_OP_EC_DECODE;
            for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
            for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
            dec_item.stripe_cnt = k;
            dec_item.parity_cnt = p;
            dec_item.cell_size = len;
            dec_item.failed_idx[0] = fx;
            dec_item.failed_idx[1] = fy;
            dec_item.failed_cnt = 2;
            dec_item.ec_mode = GPU_EC_MODE_ISAL;
            memcpy(dec_item.ec_decode_matrix, decode_matrix, k * 2);

            rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
            if (rc != 0 || res.error_code != 0) {
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("isal_roundtrip", "double-failure GPU decode failed");
            }

            /* Verify fx */
            uint8_t *recon_x = (uint8_t *)malloc(len);
            cudaMemcpy(recon_x, buf->ec_data[fx], len, cudaMemcpyDeviceToHost);
            if (memcmp(recon_x, h_orig[fx], len) != 0) {
                free(recon_x);
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("isal_roundtrip", "double-failure fx verification failed");
            }
            free(recon_x);

            /* Verify fy */
            uint8_t *recon_y = (uint8_t *)malloc(len);
            cudaMemcpy(recon_y, buf->ec_data[fy], len, cudaMemcpyDeviceToHost);
            if (memcmp(recon_y, h_orig[fy], len) != 0) {
                free(recon_y);
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("isal_roundtrip", "double-failure fy verification failed");
            }
            free(recon_y);
        }
    }

    for (int i = 0; i < k; i++) free(h_orig[i]);
    TEST_PASS("isal_roundtrip");
    return 0;
}

/* ── Test 4: Mode Isolation (ISAL vs Native) ─────────────────── */
static int test_isal_mode_isolation(gpu_engine_t *eng, struct test_buffers *buf)
{
    const int k = 4, p = 2;
    size_t len = buf->cell_size;

    /* Generate data on host */
    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 29 + 3);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);

    /* Generate Cauchy Matrix */
    uint8_t encode_matrix[64];
    gpu_ec_gen_cauchy_matrix(encode_matrix, k, p);

    /* Encode in NATIVE mode */
    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k;
    enc_item.parity_cnt = p;
    enc_item.cell_size = len;
    enc_item.ec_mode = GPU_EC_MODE_NATIVE;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("isal_mode_isolation", "encode failed");
    }

    /* Zero out failed stripe 1 */
    cudaMemset(buf->ec_data[1], 0, len);
    cudaStreamSynchronize(0);

    /* Decode in ISAL mode with incorrect coefficients (should fail to match original data) */
    uint8_t decode_matrix[32];
    int failed_arr[1] = { 1 };
    rc = gpu_ec_make_decode_matrix(encode_matrix, k, p, failed_arr, 1, decode_matrix);
    if (rc != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("isal_mode_isolation", "make decode matrix failed");
    }

    gpu_work_item_t dec_item = {};
    dec_item.op_type = GPU_OP_EC_DECODE;
    for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
    dec_item.stripe_cnt = k;
    dec_item.parity_cnt = p;
    dec_item.cell_size = len;
    dec_item.failed_idx[0] = 1;
    dec_item.failed_cnt = 1;
    dec_item.ec_mode = GPU_EC_MODE_ISAL;
    memcpy(dec_item.ec_decode_matrix, decode_matrix, k);

    rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("isal_mode_isolation", "decode failed unexpectedly");
    }

    /* Verify reconstructed data does NOT match original */
    uint8_t *recon = (uint8_t *)malloc(len);
    cudaMemcpy(recon, buf->ec_data[1], len, cudaMemcpyDeviceToHost);
    int matches = (memcmp(recon, h_orig[1], len) == 0);
    free(recon);
    for (int i = 0; i < k; i++) free(h_orig[i]);

    if (matches) {
        TEST_FAIL("isal_mode_isolation", "Native-encoded data was correctly reconstructed by ISA-L Cauchy decoder (should be incorrect)");
    }

    TEST_PASS("isal_mode_isolation");
    return 0;
}

int main(void)
{
    printf("=== test_ec_isal_compat ===\n");

    /* Allocate GPU memory BEFORE engine init to avoid deadlock */
    struct test_buffers buf;
    if (alloc_test_buffers(&buf, 16, 4, 4096) != 0) {
        fprintf(stderr, "GPU buffer allocation failed\n");
        return 1;
    }

    /* Initialize GPU engine and GF tables */
    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Engine init failed\n");
        free_test_buffers(&buf, 16, 4);
        return 1;
    }
    gpu_ec_init_gf_tables();

    int failures = 0;
    failures += test_cauchy_invertibility();
    failures += test_isal_encode_vs_cpu(eng, &buf);
    failures += test_isal_roundtrip(eng, &buf);
    failures += test_isal_mode_isolation(eng, &buf);

    gpu_engine_fini(eng);
    free_test_buffers(&buf, 16, 4);

    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
