/**
 * test_ec_decode.cu — EC Decode (Data Reconstruction) Tests
 *
 * Tests single-failure and double-failure data reconstruction
 * via the persistent kernel dispatch path, cross-verified against
 * CPU reference implementation.
 *
 * Pipeline:
 *   1. Generate random data stripes
 *   2. Compute P+Q parity via GPU_OP_EC_ENCODE
 *   3. Zero out "failed" stripe(s)
 *   4. Reconstruct via GPU_OP_EC_DECODE
 *   5. Compare reconstructed data against original
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

/* ── Test: Single failure reconstruction (4+2, fail stripe 1) ──────────── */
static int test_single_failure(gpu_engine_t *eng, struct decode_buffers *buf)
{
    const int k = 4, p = 2;
    size_t len = buf->cell_size;

    /* Generate original data */
    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 7 + 3);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }

    /* Step 1: Encode P+Q parity */
    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k;
    enc_item.parity_cnt = p;
    enc_item.cell_size = len;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("single_failure", "encode failed");
    }

    /* Step 2: Simulate failure — zero stripe 1 */
    int failed = 1;
    cudaMemset(buf->ec_data[failed], 0, len);

    /* Step 3: Reconstruct via EC_DECODE */
    gpu_work_item_t dec_item = {};
    dec_item.op_type = GPU_OP_EC_DECODE;
    for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
    dec_item.stripe_cnt = k;
    dec_item.parity_cnt = p;
    dec_item.cell_size = len;
    dec_item.failed_idx[0] = failed;
    dec_item.failed_cnt = 1;

    rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
    printf("  [DIAG] decode rc=%d error_code=%d\n", rc, res.error_code);
    if (rc != 0 || res.error_code != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "decode failed rc=%d err=%d", rc, res.error_code);
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("single_failure", msg);
    }

    /* Step 4: Verify reconstructed data */
    uint8_t *h_reconstructed = (uint8_t *)malloc(len);
    cudaMemcpy(h_reconstructed, buf->ec_data[failed], len, cudaMemcpyDeviceToHost);
    
    /* Also dump P parity and surviving stripes for byte 0 */
    uint8_t p0, d0_0, d2_0, d3_0;
    cudaMemcpy(&p0, buf->ec_parity[0], 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&d0_0, buf->ec_data[0], 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&d2_0, buf->ec_data[2], 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&d3_0, buf->ec_data[3], 1, cudaMemcpyDeviceToHost);
    printf("  [DIAG] P[0]=0x%02X D0[0]=0x%02X D2[0]=0x%02X D3[0]=0x%02X recon_D1[0]=0x%02X\n",
           p0, d0_0, d2_0, d3_0, h_reconstructed[0]);
    printf("  [DIAG] Expected: P^D0^D2^D3 = 0x%02X\n", p0 ^ d0_0 ^ d2_0 ^ d3_0);

    if (memcmp(h_reconstructed, h_orig[failed], len) != 0) {
        /* Find first mismatch */
        for (size_t i = 0; i < len; i++) {
            if (h_reconstructed[i] != h_orig[failed][i]) {
                char msg[128];
                snprintf(msg, sizeof(msg), "mismatch at byte %zu: got 0x%02X expected 0x%02X",
                         i, h_reconstructed[i], h_orig[failed][i]);
                free(h_reconstructed);
                for (int j = 0; j < k; j++) free(h_orig[j]);
                TEST_FAIL("single_failure", msg);
            }
        }
    }

    free(h_reconstructed);
    for (int i = 0; i < k; i++) free(h_orig[i]);
    TEST_PASS("single_failure (4+2, fail D1)");
    return 0;
}

/* ── Test: Double failure reconstruction (4+2, fail stripes 0 and 3) ───── */
static int test_double_failure(gpu_engine_t *eng, struct decode_buffers *buf)
{
    const int k = 4, p = 2;
    size_t len = buf->cell_size;

    /* Generate original data */
    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 13 + 5);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }

    /* Step 1: Encode P+Q parity */
    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k;
    enc_item.parity_cnt = p;
    enc_item.cell_size = len;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("double_failure", "encode failed");
    }

    /* Step 2: Simulate double failure — zero stripes 0 and 3 */
    int fx = 0, fy = 3;
    cudaMemset(buf->ec_data[fx], 0, len);
    cudaMemset(buf->ec_data[fy], 0, len);

    /* Step 3: Reconstruct via EC_DECODE */
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

    rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
    printf("  [DIAG] double decode rc=%d error_code=%d\n", rc, res.error_code);
    if (rc != 0 || res.error_code != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "decode failed rc=%d err=%d", rc, res.error_code);
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("double_failure", msg);
    }

    /* Step 4: Verify both reconstructed stripes */
    uint8_t *h_recon_x = (uint8_t *)malloc(len);
    uint8_t *h_recon_y = (uint8_t *)malloc(len);
    cudaMemcpy(h_recon_x, buf->ec_data[fx], len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_recon_y, buf->ec_data[fy], len, cudaMemcpyDeviceToHost);
    
    /* Dump diagnostics */
    uint8_t p0, q0, d1_0, d2_0;
    cudaMemcpy(&p0, buf->ec_parity[0], 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&q0, buf->ec_parity[1], 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&d1_0, buf->ec_data[1], 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&d2_0, buf->ec_data[2], 1, cudaMemcpyDeviceToHost);
    printf("  [DIAG] P[0]=0x%02X Q[0]=0x%02X D1[0]=0x%02X D2[0]=0x%02X\n",
           p0, q0, d1_0, d2_0);
    printf("  [DIAG] recon D0[0]=0x%02X D3[0]=0x%02X (expected 0x%02X 0x%02X)\n",
           h_recon_x[0], h_recon_y[0], h_orig[fx][0], h_orig[fy][0]);

    if (memcmp(h_recon_x, h_orig[fx], len) != 0) {
        for (size_t i = 0; i < len; i++) {
            if (h_recon_x[i] != h_orig[fx][i]) {
                char msg[128];
                snprintf(msg, sizeof(msg), "D%d mismatch at byte %zu: got 0x%02X expected 0x%02X",
                         fx, i, h_recon_x[i], h_orig[fx][i]);
                free(h_recon_x); free(h_recon_y);
                for (int j = 0; j < k; j++) free(h_orig[j]);
                TEST_FAIL("double_failure", msg);
            }
        }
    }
    if (memcmp(h_recon_y, h_orig[fy], len) != 0) {
        for (size_t i = 0; i < len; i++) {
            if (h_recon_y[i] != h_orig[fy][i]) {
                char msg[128];
                snprintf(msg, sizeof(msg), "D%d mismatch at byte %zu: got 0x%02X expected 0x%02X",
                         fy, i, h_recon_y[i], h_orig[fy][i]);
                free(h_recon_x); free(h_recon_y);
                for (int j = 0; j < k; j++) free(h_orig[j]);
                TEST_FAIL("double_failure", msg);
            }
        }
    }

    free(h_recon_x); free(h_recon_y);
    for (int i = 0; i < k; i++) free(h_orig[i]);
    TEST_PASS("double_failure (4+2, fail D0+D3)");
    return 0;
}

/* ── Test: Double failure with adjacent stripes (4+2, fail D2 and D3) ──── */
static int test_double_failure_adjacent(gpu_engine_t *eng, struct decode_buffers *buf)
{
    const int k = 4, p = 2;
    size_t len = buf->cell_size;

    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 29 + 11);
        cudaMemcpy(buf->ec_data[i], h_orig[i], len, cudaMemcpyHostToDevice);
    }

    /* Encode */
    gpu_work_item_t enc_item = {};
    enc_item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc_item.parity_ptrs[i] = buf->ec_parity[i];
    enc_item.stripe_cnt = k; enc_item.parity_cnt = p; enc_item.cell_size = len;

    gpu_result_t res;
    gpu_engine_submit_and_wait(eng, &enc_item, &res);

    /* Fail D2 and D3 (adjacent) */
    cudaMemset(buf->ec_data[2], 0, len);
    cudaMemset(buf->ec_data[3], 0, len);

    /* Decode */
    gpu_work_item_t dec_item = {};
    dec_item.op_type = GPU_OP_EC_DECODE;
    for (int i = 0; i < k; i++) dec_item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) dec_item.parity_ptrs[i] = buf->ec_parity[i];
    dec_item.stripe_cnt = k; dec_item.parity_cnt = p; dec_item.cell_size = len;
    dec_item.failed_idx[0] = 2; dec_item.failed_idx[1] = 3;
    dec_item.failed_cnt = 2;

    int rc = gpu_engine_submit_and_wait(eng, &dec_item, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        TEST_FAIL("double_adjacent", "decode failed");
    }

    /* Verify */
    uint8_t *h_r2 = (uint8_t *)malloc(len);
    uint8_t *h_r3 = (uint8_t *)malloc(len);
    cudaMemcpy(h_r2, buf->ec_data[2], len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r3, buf->ec_data[3], len, cudaMemcpyDeviceToHost);

    printf("  [DIAG] adjacent: rc=%d err=%d D2[0]=0x%02X(exp 0x%02X) D3[0]=0x%02X(exp 0x%02X)\n",
           rc, res.error_code, h_r2[0], h_orig[2][0], h_r3[0], h_orig[3][0]);

    int fail = 0;
    if (memcmp(h_r2, h_orig[2], len) != 0) {
        printf("  D2 mismatch!\n"); fail = 1;
    }
    if (memcmp(h_r3, h_orig[3], len) != 0) {
        printf("  D3 mismatch!\n"); fail = 1;
    }

    free(h_r2); free(h_r3);
    for (int i = 0; i < k; i++) free(h_orig[i]);
    if (fail) TEST_FAIL("double_adjacent", "data mismatch after reconstruction");
    TEST_PASS("double_failure_adjacent (4+2, fail D2+D3)");
    return 0;
}

/* ── Test: CPU reference decode cross-check ────────────────────────────── */
static int test_cpu_decode_crosscheck(void)
{
    const int k = 4, p = 2;
    const size_t len = 1024;

    /* Generate data */
    uint8_t *h_data[4];
    for (int i = 0; i < k; i++) h_data[i] = alloc_host_buffer(len, i * 11 + 7);

    /* Compute P and Q on CPU */
    uint8_t *p_parity = (uint8_t *)calloc(1, len);
    uint8_t *q_parity = (uint8_t *)calloc(1, len);

    /* P parity: XOR all stripes */
    for (size_t i = 0; i < len; i++) {
        uint8_t val = 0;
        for (int s = 0; s < k; s++) val ^= h_data[s][i];
        p_parity[i] = val;
    }

    /* Q parity: Horner's method with GF(2^8) - must match encode */
    gpu_ec_init_gf_tables();
    cpu_ec_xor_parity((const void **)h_data, k, len, p_parity); /* Use actual function */

    /* For Q, we need to match the encode Horner's method */
    /* Q[i] = gf_mul2(gf_mul2(...(gf_mul2(D[k-1]) ^ D[k-2])...) ^ D[0]) */
    for (size_t i = 0; i < len; i++) {
        uint8_t q = h_data[k - 1][i];
        for (int s = k - 2; s >= 0; s--) {
            /* gf_mul2_byte equivalent */
            q = (uint8_t)(((q << 1) & 0xFE) ^ ((q >> 7) * 0x1B));
            q ^= h_data[s][i];
        }
        q_parity[i] = q;
    }

    /* Save originals */
    uint8_t *orig_0 = (uint8_t *)malloc(len);
    memcpy(orig_0, h_data[0], len);

    /* Single-failure CPU decode */
    int failed_idx[1] = {0};
    void *out_ptrs[1] = {h_data[0]};
    memset(h_data[0], 0, len); /* "fail" stripe 0 */

    const void *parity_ptrs[2] = {p_parity, q_parity};
    cpu_ec_decode((const void **)h_data, parity_ptrs,
                  k, p, len, failed_idx, 1, out_ptrs);

    if (memcmp(h_data[0], orig_0, len) != 0) {
        free(orig_0); free(p_parity); free(q_parity);
        for (int i = 0; i < k; i++) free(h_data[i]);
        TEST_FAIL("cpu_decode_crosscheck", "CPU single-failure decode mismatch");
    }

    /* Double-failure CPU decode test */
    uint8_t *orig_1 = (uint8_t *)malloc(len);
    uint8_t *orig_3 = (uint8_t *)malloc(len);
    memcpy(orig_1, h_data[1], len);
    memcpy(orig_3, h_data[3], len);
    
    /* Recompute P and Q fresh from all data */
    for (size_t i = 0; i < len; i++) {
        uint8_t val = 0;
        for (int s = 0; s < k; s++) val ^= h_data[s][i];
        p_parity[i] = val;
    }
    for (size_t i = 0; i < len; i++) {
        uint8_t q = h_data[k - 1][i];
        for (int s = k - 2; s >= 0; s--) {
            q = (uint8_t)(((q << 1) & 0xFE) ^ ((q >> 7) * 0x1B));
            q ^= h_data[s][i];
        }
        q_parity[i] = q;
    }
    
    memset(h_data[1], 0, len);
    memset(h_data[3], 0, len);
    
    int failed2[2] = {1, 3};
    void *out2[2] = {h_data[1], h_data[3]};
    cpu_ec_decode((const void **)h_data, parity_ptrs,
                  k, p, len, failed2, 2, out2);
    
    printf("  [DIAG] CPU double decode: D1[0]=0x%02X (expected 0x%02X) D3[0]=0x%02X (expected 0x%02X)\n",
           h_data[1][0], orig_1[0], h_data[3][0], orig_3[0]);
    
    if (memcmp(h_data[1], orig_1, len) != 0 || memcmp(h_data[3], orig_3, len) != 0) {
        free(orig_0); free(orig_1); free(orig_3); free(p_parity); free(q_parity);
        for (int i = 0; i < k; i++) free(h_data[i]);
        TEST_FAIL("cpu_decode_crosscheck", "CPU double-failure decode mismatch");
    }

    free(orig_0); free(orig_1); free(orig_3); free(p_parity); free(q_parity);
    for (int i = 0; i < k; i++) free(h_data[i]);
    TEST_PASS("cpu_decode_crosscheck (single + double failure)");
    return 0;
}

/* ── Test: EC_DECODE with invalid parameters ───────────────────────────── */
static int test_decode_invalid_params(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_DECODE;
    item.stripe_cnt = 4;
    item.parity_cnt = 2;
    item.cell_size = 4096;
    item.failed_cnt = 0; /* Invalid: must be 1 or 2 */

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    /* Either the host-side submit rejects it (rc != 0) or
     * the GPU-side dispatch sets error_code != 0 — both are valid. */
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("decode_invalid_params (failed_cnt=0)");
        return 0;
    }
    TEST_FAIL("decode_invalid_params", "expected error for failed_cnt=0");
}

int main(void)
{
    printf("=== test_ec_decode ===\n");

    /* Allocate GPU memory BEFORE engine init */
    struct decode_buffers buf = {};
    const size_t cell_size = 64 * 1024; /* 64KB */
    if (alloc_gpu_buffers(&buf, 4, cell_size) != 0) {
        fprintf(stderr, "Failed to allocate GPU buffers\n");
        return 1;
    }

    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Failed to init GPU engine\n");
        free_gpu_buffers(&buf, 4);
        return 1;
    }

    int failures = 0;
    failures += test_single_failure(eng, &buf);
    failures += test_double_failure(eng, &buf);
    failures += test_double_failure_adjacent(eng, &buf);
    failures += test_decode_invalid_params(eng);
    failures += test_cpu_decode_crosscheck();

    gpu_engine_fini(eng);
    free_gpu_buffers(&buf, 4);

    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
