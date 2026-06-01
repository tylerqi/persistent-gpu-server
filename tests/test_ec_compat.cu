/**
 * test_ec_compat.cu — Erasure Coding Compatibility Tests (Linux RAID-6)
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

/* Pre-allocated GPU buffers for all tests */
struct decode_buffers {
    void *ec_data[8];      /* Up to 8 data stripes */
    void *ec_parity[2];    /* P + Q parity */
    size_t cell_size;
};

static int alloc_gpu_buffers(struct decode_buffers *buf, int k, size_t cell_size)
{
    buf->cell_size = cell_size;
    for (int i = 0; i < k; i++) {
        if (cudaMalloc(&buf->ec_data[i], cell_size) != cudaSuccess)
            return -1;
    }
    for (int i = 0; i < 2; i++) {
        if (cudaMalloc(&buf->ec_parity[i], cell_size) != cudaSuccess)
            return -1;
    }
    return 0;
}

static void free_gpu_buffers(struct decode_buffers *buf, int k)
{
    for (int i = 0; i < k; i++) cudaFree(buf->ec_data[i]);
    for (int i = 0; i < 2; i++) cudaFree(buf->ec_parity[i]);
}

/* ── Test: Golden Vectors for Linux RAID-6 and Native modes ────────────── */
static int test_raid6_golden_vectors(gpu_engine_t *eng, struct decode_buffers *buf)
{
    const int k = 4, p = 2;
    const size_t len = 4;

    /* 4-byte data patterns */
    uint8_t h_d0[4] = {0x01, 0x80, 0xA5, 0x12};
    uint8_t h_d1[4] = {0x05, 0x80, 0x5A, 0x34};
    uint8_t h_d2[4] = {0x09, 0x80, 0xFF, 0x56};
    uint8_t h_d3[4] = {0x0D, 0x80, 0x00, 0x78};

    /* Expected outputs */
    uint8_t expected_p[4] = {0x00, 0x00, 0x00, 0x08};
    uint8_t expected_q_native[4] = {0x47, 0xC1, 0xC0, 0xD4};
    uint8_t expected_q_raid6[4] = {0x47, 0xD3, 0xCA, 0xD8};

    /* Copy data to device */
    cudaMemcpy(buf->ec_data[0], h_d0, len, cudaMemcpyHostToDevice);
    cudaMemcpy(buf->ec_data[1], h_d1, len, cudaMemcpyHostToDevice);
    cudaMemcpy(buf->ec_data[2], h_d2, len, cudaMemcpyHostToDevice);
    cudaMemcpy(buf->ec_data[3], h_d3, len, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);

    /* 1. Test Native Mode (0x11B) */
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
        TEST_FAIL("raid6_golden_vectors", "Native mode encode failed");
    }

    uint8_t out_p[4], out_q[4];
    cudaMemcpy(out_p, buf->ec_parity[0], len, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_q, buf->ec_parity[1], len, cudaMemcpyDeviceToHost);

    if (memcmp(out_p, expected_p, len) != 0) {
        printf("  [DIAG] Native P mismatch: got [%02X %02X %02X %02X] expected [%02X %02X %02X %02X]\n",
               out_p[0], out_p[1], out_p[2], out_p[3],
               expected_p[0], expected_p[1], expected_p[2], expected_p[3]);
        TEST_FAIL("raid6_golden_vectors", "Native P parity mismatch");
    }
    if (memcmp(out_q, expected_q_native, len) != 0) {
        printf("  [DIAG] Native Q mismatch: got [%02X %02X %02X %02X] expected [%02X %02X %02X %02X]\n",
               out_q[0], out_q[1], out_q[2], out_q[3],
               expected_q_native[0], expected_q_native[1], expected_q_native[2], expected_q_native[3]);
        TEST_FAIL("raid6_golden_vectors", "Native Q parity mismatch");
    }

    /* 2. Test RAID-6 Mode (0x11D) */
    cudaMemset(buf->ec_parity[0], 0, len);
    cudaMemset(buf->ec_parity[1], 0, len);
    cudaStreamSynchronize(0);

    enc_item.ec_mode = GPU_EC_MODE_RAID6;
    rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_FAIL("raid6_golden_vectors", "RAID6 mode encode failed");
    }

    cudaMemcpy(out_p, buf->ec_parity[0], len, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_q, buf->ec_parity[1], len, cudaMemcpyDeviceToHost);

    if (memcmp(out_p, expected_p, len) != 0) {
        TEST_FAIL("raid6_golden_vectors", "RAID6 P parity mismatch");
    }
    if (memcmp(out_q, expected_q_raid6, len) != 0) {
        printf("  [DIAG] RAID6 Q mismatch: got [%02X %02X %02X %02X] expected [%02X %02X %02X %02X]\n",
               out_q[0], out_q[1], out_q[2], out_q[3],
               expected_q_raid6[0], expected_q_raid6[1], expected_q_raid6[2], expected_q_raid6[3]);
        TEST_FAIL("raid6_golden_vectors", "RAID6 Q parity mismatch");
    }

    TEST_PASS("raid6_golden_vectors");
    return 0;
}

/* ── Test: RAID-6 roundtrip (all single & double failures) ─────────────── */
static int test_raid6_roundtrip(gpu_engine_t *eng, struct decode_buffers *buf)
{
    const int k = 4, p = 2;
    size_t len = buf->cell_size;

    /* Generate original data */
    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 7 + 13);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);

    /* Encode in RAID6 mode */
    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k;
    enc_item.parity_cnt = p;
    enc_item.cell_size = len;
    enc_item.ec_mode = GPU_EC_MODE_RAID6;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("raid6_roundtrip", "encode failed");
    }

    /* Test all single-failure cases */
    for (int failed = 0; failed < k; failed++) {
        /* Backup failed stripe */
        uint8_t *backup = (uint8_t *)malloc(len);
        cudaMemcpy(backup, buf->ec_data[failed], len, cudaMemcpyDeviceToHost);

        /* Simulate failure */
        cudaMemset(buf->ec_data[failed], 0, len);
        cudaStreamSynchronize(0);

        /* Decode */
        gpu_work_item_t dec_item = {};
        dec_item.op_type = GPU_OP_EC_DECODE;
        for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
        for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
        dec_item.stripe_cnt = k;
        dec_item.parity_cnt = p;
        dec_item.cell_size = len;
        dec_item.failed_idx[0] = failed;
        dec_item.failed_cnt = 1;
        dec_item.ec_mode = GPU_EC_MODE_RAID6;

        rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
        if (rc != 0 || res.error_code != 0) {
            free(backup);
            for (int i = 0; i < k; i++) free(h_orig[i]);
            TEST_FAIL("raid6_roundtrip", "single-failure decode failed");
        }

        /* Verify */
        uint8_t *h_reconstructed = (uint8_t *)malloc(len);
        cudaMemcpy(h_reconstructed, buf->ec_data[failed], len, cudaMemcpyDeviceToHost);
        if (memcmp(h_reconstructed, backup, len) != 0) {
            free(h_reconstructed); free(backup);
            for (int i = 0; i < k; i++) free(h_orig[i]);
            TEST_FAIL("raid6_roundtrip", "single-failure data mismatch");
        }

        free(h_reconstructed);
        free(backup);
    }

    /* Test all double-failure cases */
    for (int fx = 0; fx < k; fx++) {
        for (int fy = fx + 1; fy < k; fy++) {
            /* Restore all data to original */
            for (int i = 0; i < k; i++) {
                cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
            }
            cudaStreamSynchronize(0);

            /* Re-encode to get clean parity */
            rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
            if (rc != 0 || res.error_code != 0) {
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("raid6_roundtrip", "re-encode failed");
            }

            /* Simulate failures */
            cudaMemset(buf->ec_data[fx], 0, len);
            cudaMemset(buf->ec_data[fy], 0, len);
            cudaStreamSynchronize(0);

            /* Decode */
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
            dec_item.ec_mode = GPU_EC_MODE_RAID6;

            rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
            if (rc != 0 || res.error_code != 0) {
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("raid6_roundtrip", "double-failure decode failed");
            }

            /* Verify fx */
            uint8_t *h_recon_x = (uint8_t *)malloc(len);
            cudaMemcpy(h_recon_x, buf->ec_data[fx], len, cudaMemcpyDeviceToHost);
            if (memcmp(h_recon_x, h_orig[fx], len) != 0) {
                free(h_recon_x);
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("raid6_roundtrip", "double-failure fx data mismatch");
            }
            free(h_recon_x);

            /* Verify fy */
            uint8_t *h_recon_y = (uint8_t *)malloc(len);
            cudaMemcpy(h_recon_y, buf->ec_data[fy], len, cudaMemcpyDeviceToHost);
            if (memcmp(h_recon_y, h_orig[fy], len) != 0) {
                free(h_recon_y);
                for (int i = 0; i < k; i++) free(h_orig[i]);
                TEST_FAIL("raid6_roundtrip", "double-failure fy data mismatch");
            }
            free(h_recon_y);
        }
    }

    for (int i = 0; i < k; i++) free(h_orig[i]);
    TEST_PASS("raid6_roundtrip");
    return 0;
}

/* ── Test: Mode Isolation (Native encode, RAID-6 decode fails to match) ── */
static int test_mode_isolation(gpu_engine_t *eng, struct decode_buffers *buf)
{
    const int k = 4, p = 2;
    size_t len = buf->cell_size;

    /* Generate original data */
    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 11 + 5);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);

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
        TEST_FAIL("mode_isolation", "encode failed");
    }

    /* Zero out stripe 1 and 3 */
    cudaMemset(buf->ec_data[1], 0, len);
    cudaMemset(buf->ec_data[3], 0, len);
    cudaStreamSynchronize(0);

    /* Decode in RAID6 mode (incorrect!) */
    gpu_work_item_t dec_item = {};
    dec_item.op_type = GPU_OP_EC_DECODE;
    for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
    dec_item.stripe_cnt = k;
    dec_item.parity_cnt = p;
    dec_item.cell_size = len;
    dec_item.failed_idx[0] = 1;
    dec_item.failed_idx[1] = 3;
    dec_item.failed_cnt = 2;
    dec_item.ec_mode = GPU_EC_MODE_RAID6;

    rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("mode_isolation", "decode failed unexpectedly");
    }

    /* Verify that data is WRONG (does not match original) */
    uint8_t *h_reconstructed = (uint8_t *)malloc(len);
    cudaMemcpy(h_reconstructed, buf->ec_data[1], len, cudaMemcpyDeviceToHost);
    
    int matches = (memcmp(h_reconstructed, h_orig[1], len) == 0);
    free(h_reconstructed);
    for (int i = 0; i < k; i++) free(h_orig[i]);

    if (matches) {
        TEST_FAIL("mode_isolation", "Native-encoded data was correctly decoded by RAID-6 (should be incorrect!)");
    }

    TEST_PASS("mode_isolation");
    return 0;
}

int main(void)
{
    printf("=== test_ec_compat ===\n");

    /* Allocate GPU memory BEFORE engine init to avoid deadlock */
    struct decode_buffers buf;
    if (alloc_gpu_buffers(&buf, 4, 4096) != 0) {
        fprintf(stderr, "GPU buffer allocation failed\n");
        return 1;
    }

    /* Initialize GPU engine and GF tables */
    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Engine init failed\n");
        free_gpu_buffers(&buf, 4);
        return 1;
    }
    gpu_ec_init_gf_tables();

    int failures = 0;
    failures += test_raid6_golden_vectors(eng, &buf);
    failures += test_raid6_roundtrip(eng, &buf);
    failures += test_mode_isolation(eng, &buf);

    gpu_engine_fini(eng);
    free_gpu_buffers(&buf, 4);

    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
