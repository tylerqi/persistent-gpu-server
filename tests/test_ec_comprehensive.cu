/**
 * test_ec_comprehensive.cu — Comprehensive EC Test Suite
 */
#include "gpu_engine.h"
#include "gpu_ec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

enum PatternType {
    PATTERN_SEED,
    PATTERN_ZEROS,
    PATTERN_ONES,
    PATTERN_AA,
    PATTERN_SEQ
};

static void fill_host_buffer(uint8_t *buf, size_t len, enum PatternType type, int seed)
{
    for (size_t i = 0; i < len; i++) {
        switch (type) {
            case PATTERN_ZEROS: buf[i] = 0x00; break;
            case PATTERN_ONES:  buf[i] = 0xFF; break;
            case PATTERN_AA:    buf[i] = 0xAA; break;
            case PATTERN_SEQ:   buf[i] = (uint8_t)(i & 0xFF); break;
            case PATTERN_SEED:
            default:            buf[i] = (uint8_t)((i * seed + 17) & 0xFF); break;
        }
    }
}

static uint8_t *alloc_host_buffer(size_t len, int seed)
{
    uint8_t *buf = (uint8_t *)malloc(len);
    if (buf) {
        fill_host_buffer(buf, len, PATTERN_SEED, seed);
    }
    return buf;
}

struct test_buffers {
    void *ec_data[16];
    void *ec_parity[4];
    size_t max_size;
};

static int alloc_test_buffers(struct test_buffers *buf, size_t max_size)
{
    buf->max_size = max_size;
    for (int i = 0; i < 16; i++) {
        if (cudaMalloc(&buf->ec_data[i], max_size) != cudaSuccess)
            return -1;
    }
    for (int i = 0; i < 4; i++) {
        if (cudaMalloc(&buf->ec_parity[i], max_size) != cudaSuccess)
            return -1;
    }
    return 0;
}

static void free_test_buffers(struct test_buffers *buf)
{
    for (int i = 0; i < 16; i++) cudaFree(buf->ec_data[i]);
    for (int i = 0; i < 4; i++) cudaFree(buf->ec_parity[i]);
}

/* Category 1, 3, 5, 7 helper */
static int run_roundtrip_test(gpu_engine_t *eng, struct test_buffers *buf,
                               int k, int p, size_t cell_size,
                               const int *failed_idx, int failed_cnt,
                               enum PatternType pat_type, const char *test_name)
{
    uint8_t *h_orig[16];
    for (int i = 0; i < k; i++) {
        h_orig[i] = (uint8_t *)malloc(cell_size);
        fill_host_buffer(h_orig[i], cell_size, pat_type, i * 17 + 5);
        cudaMemcpy(buf->ec_data[i], h_orig[i], cell_size, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);
    
    gpu_work_item_t enc = {};
    enc.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < k; i++) enc.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) enc.parity_ptrs[i] = buf->ec_parity[i];
    enc.stripe_cnt = k;
    enc.parity_cnt = p;
    enc.cell_size = cell_size;
    
    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &enc, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "encode failed: rc=%d, err=%d", rc, res.error_code);
        TEST_FAIL(test_name, msg);
    }
    
    for (int i = 0; i < failed_cnt; i++) {
        cudaMemset(buf->ec_data[failed_idx[i]], 0, cell_size);
    }
    cudaStreamSynchronize(0);
    
    gpu_work_item_t dec = {};
    dec.op_type = GPU_OP_EC_DECODE;
    for (int i = 0; i < k; i++) dec.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < p; i++) dec.parity_ptrs[i] = buf->ec_parity[i];
    dec.stripe_cnt = k;
    dec.parity_cnt = p;
    dec.cell_size = cell_size;
    dec.failed_cnt = failed_cnt;
    for (int i = 0; i < failed_cnt; i++) dec.failed_idx[i] = failed_idx[i];
    
    rc = gpu_engine_submit_and_wait(eng, &dec, &res);
    if (rc != 0 || res.error_code != 0) {
        for (int i = 0; i < k; i++) free(h_orig[i]);
        char msg[128];
        snprintf(msg, sizeof(msg), "decode failed: rc=%d, err=%d", rc, res.error_code);
        TEST_FAIL(test_name, msg);
    }
    
    for (int fi = 0; fi < failed_cnt; fi++) {
        int idx = failed_idx[fi];
        uint8_t *h_recon = (uint8_t *)malloc(cell_size);
        cudaMemcpy(h_recon, buf->ec_data[idx], cell_size, cudaMemcpyDeviceToHost);
        
        if (memcmp(h_recon, h_orig[idx], cell_size) != 0) {
            printf("  [DEBUG] %s: mismatch on stripe %d:\n", test_name, idx);
            int printed = 0;
            for (size_t i = 0; i < cell_size && printed < 16; i++) {
                if (h_recon[i] != h_orig[idx][i]) {
                    printf("    byte %zu: got 0x%02X expected 0x%02X\n", i, h_recon[i], h_orig[idx][i]);
                    printed++;
                }
            }
            free(h_recon);
            for (int j = 0; j < k; j++) free(h_orig[j]);
            TEST_FAIL(test_name, "mismatch in reconstructed stripe");
        }
        free(h_recon);
    }
    
    for (int i = 0; i < k; i++) free(h_orig[i]);
    TEST_PASS(test_name);
    return 0;
}

/* Local GF multiply for tables */
static uint8_t local_gf_mul(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    uint8_t hi_bit;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        hi_bit = a & 0x80;
        a <<= 1;
        if (hi_bit) a ^= 0x1B;
        b >>= 1;
    }
    return result;
}

static inline uint8_t cpu_gf_mul2_byte(uint8_t val) {
    return (uint8_t)(((val << 1) & 0xFE) ^ ((val >> 7) * 0x1B));
}

static void cpu_encode_pq(const uint8_t **h_data, int k, size_t len, uint8_t *p_out, uint8_t *q_out) {
    for (size_t i = 0; i < len; i++) {
        uint8_t p = h_data[0][i];
        for (int s = 1; s < k; s++) {
            p ^= h_data[s][i];
        }
        p_out[i] = p;
    }
    for (size_t i = 0; i < len; i++) {
        uint8_t q = h_data[k - 1][i];
        for (int s = k - 2; s >= 0; s--) {
            q = cpu_gf_mul2_byte(q) ^ h_data[s][i];
        }
        q_out[i] = q;
    }
}

/* Category 2: GF(2^8) algebraic property verification */
static int test_category_2_gf_properties(void)
{
    uint8_t h_gf_exp[512];
    uint8_t h_gf_log[256];
    uint8_t h_gf_inv[256];
    uint8_t h_gf_pow2[16];

    /* Build exp/log tables */
    h_gf_exp[0] = 1;
    for (int i = 1; i < 512; i++) {
        h_gf_exp[i] = local_gf_mul(h_gf_exp[i - 1], 3);
    }
    memset(h_gf_log, 0, sizeof(h_gf_log));
    for (int i = 0; i < 255; i++) {
        h_gf_log[h_gf_exp[i]] = (uint8_t)i;
    }
    h_gf_inv[0] = 0;
    for (int i = 1; i < 256; i++) {
        h_gf_inv[i] = h_gf_exp[255 - h_gf_log[i]];
    }
    h_gf_pow2[0] = 1;
    for (int i = 1; i < 16; i++) {
        h_gf_pow2[i] = local_gf_mul(h_gf_pow2[i - 1], 2);
    }

    /* 1. gf_exp_log_inverse */
    for (int i = 0; i < 255; i++) {
        if (h_gf_log[h_gf_exp[i]] != i) {
            TEST_FAIL("gf_exp_log_inverse", "log[exp[i]] != i");
        }
    }
    TEST_PASS("gf_exp_log_inverse");

    /* 2. gf_inv_self_inverse */
    for (int x = 1; x < 256; x++) {
        if (h_gf_inv[h_gf_inv[x]] != x) {
            TEST_FAIL("gf_inv_self_inverse", "inv[inv[x]] != x");
        }
    }
    TEST_PASS("gf_inv_self_inverse");

    /* 3. gf_mul_identity */
    for (int a = 0; a < 256; a++) {
        if (local_gf_mul(a, 1) != a) {
            TEST_FAIL("gf_mul_identity", "a * 1 != a");
        }
    }
    TEST_PASS("gf_mul_identity");

    /* 4. gf_mul_zero */
    for (int a = 0; a < 256; a++) {
        if (local_gf_mul(a, 0) != 0) {
            TEST_FAIL("gf_mul_zero", "a * 0 != 0");
        }
    }
    TEST_PASS("gf_mul_zero");

    /* 5. gf_mul_inverse */
    for (int a = 1; a < 256; a++) {
        if (local_gf_mul(a, h_gf_inv[a]) != 1) {
            TEST_FAIL("gf_mul_inverse", "a * inv[a] != 1");
        }
    }
    TEST_PASS("gf_mul_inverse");

    /* 6. gf_pow2_consistency */
    uint8_t expected = 1;
    for (int i = 0; i < 16; i++) {
        if (h_gf_pow2[i] != expected) {
            TEST_FAIL("gf_pow2_consistency", "pow2[i] != 2^i");
        }
        expected = local_gf_mul(expected, 2);
    }
    TEST_PASS("gf_pow2_consistency");

    /* 7. gf_mul2_byte_vs_table */
    for (int x = 0; x < 256; x++) {
        if (cpu_gf_mul2_byte(x) != local_gf_mul(x, 2)) {
            TEST_FAIL("gf_mul2_byte_vs_table", "mul2_byte != mul(x,2)");
        }
    }
    TEST_PASS("gf_mul2_byte_vs_table");

    return 0;
}

/* Category 4: CPU Decode Descending Index Test */
static int test_cpu_decode_descending(const int *failed_idx, const char *test_name)
{
    const int k = 4;
    const int p = 2;
    const size_t len = 1024;
    
    uint8_t *h_orig[4];
    for (int i = 0; i < k; i++) {
        h_orig[i] = alloc_host_buffer(len, i * 19 + 7);
    }
    
    uint8_t *p_parity = (uint8_t *)malloc(len);
    uint8_t *q_parity = (uint8_t *)malloc(len);
    const uint8_t *data_ptrs[4] = {h_orig[0], h_orig[1], h_orig[2], h_orig[3]};
    cpu_encode_pq(data_ptrs, k, len, p_parity, q_parity);
    
    uint8_t *ec_ptrs_copy[4];
    for (int i = 0; i < k; i++) {
        ec_ptrs_copy[i] = (uint8_t *)malloc(len);
        memcpy(ec_ptrs_copy[i], h_orig[i], len);
    }
    memset(ec_ptrs_copy[failed_idx[0]], 0, len);
    memset(ec_ptrs_copy[failed_idx[1]], 0, len);
    
    const void *ec_ptrs[4] = {ec_ptrs_copy[0], ec_ptrs_copy[1], ec_ptrs_copy[2], ec_ptrs_copy[3]};
    const void *parity_ptrs[2] = {p_parity, q_parity};
    
    uint8_t *out_ptrs[2];
    out_ptrs[0] = (uint8_t *)malloc(len);
    out_ptrs[1] = (uint8_t *)malloc(len);
    
    cpu_ec_decode(ec_ptrs, parity_ptrs, k, p, len, failed_idx, 2, (void **)out_ptrs, GPU_EC_MODE_NATIVE);
    
    int failed = 0;
    if (memcmp(out_ptrs[0], h_orig[failed_idx[0]], len) != 0) {
        failed = 1;
    }
    if (memcmp(out_ptrs[1], h_orig[failed_idx[1]], len) != 0) {
        failed = 2;
    }
    
    for (int i = 0; i < k; i++) {
        free(h_orig[i]);
        free(ec_ptrs_copy[i]);
    }
    free(p_parity);
    free(q_parity);
    free(out_ptrs[0]);
    free(out_ptrs[1]);
    
    if (failed == 1) TEST_FAIL(test_name, "out_ptrs[0] mismatch");
    if (failed == 2) TEST_FAIL(test_name, "out_ptrs[1] mismatch");
    
    TEST_PASS(test_name);
    return 0;
}

/* Category 6: API Robustness / Negative tests */
static int test_decode_failed_cnt_3(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_DECODE;
    item.stripe_cnt = 4;
    item.parity_cnt = 2;
    item.cell_size = 4096;
    item.failed_cnt = 3;
    item.failed_idx[0] = 0;
    item.failed_idx[1] = 1;

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("decode_failed_cnt_3");
        return 0;
    }
    TEST_FAIL("decode_failed_cnt_3", "expected error for failed_cnt=3");
}

static int test_decode_failed_idx_oob(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_DECODE;
    item.stripe_cnt = 4;
    item.parity_cnt = 2;
    item.cell_size = 4096;
    item.failed_cnt = 1;
    item.failed_idx[0] = 99;

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("decode_failed_idx_oob");
        return 0;
    }
    TEST_FAIL("decode_failed_idx_oob", "expected error for failed_idx=99");
}

static int test_decode_parity_insufficient(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_DECODE;
    item.stripe_cnt = 4;
    item.parity_cnt = 1;
    item.cell_size = 4096;
    item.failed_cnt = 2;
    item.failed_idx[0] = 0;
    item.failed_idx[1] = 1;

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("decode_parity_insufficient");
        return 0;
    }
    TEST_FAIL("decode_parity_insufficient", "expected error for failed_cnt=2 with parity_cnt=1");
}

static int test_encode_stripe_cnt_0(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_ENCODE;
    item.stripe_cnt = 0;
    item.parity_cnt = 2;
    item.cell_size = 4096;

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("encode_stripe_cnt_0");
        return 0;
    }
    TEST_FAIL("encode_stripe_cnt_0", "expected error for stripe_cnt=0");
}

static int test_encode_stripe_cnt_17(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_ENCODE;
    item.stripe_cnt = 17;
    item.parity_cnt = 2;
    item.cell_size = 4096;

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("encode_stripe_cnt_17");
        return 0;
    }
    TEST_FAIL("encode_stripe_cnt_17", "expected error for stripe_cnt=17");
}

static int test_encode_cell_size_0(gpu_engine_t *eng)
{
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_ENCODE;
    item.stripe_cnt = 4;
    item.parity_cnt = 2;
    item.cell_size = 0;

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        TEST_PASS("encode_cell_size_0");
        return 0;
    }
    TEST_FAIL("encode_cell_size_0", "expected error for cell_size=0");
}

static int test_xor_parity_null_ptrs(void) {
    void *dummy = (void *)0x1234;
    int rc = gpu_ec_xor_parity(NULL, 4, 4096, dummy);
    if (rc != 0) {
        TEST_PASS("xor_parity_null_ptrs");
        return 0;
    }
    TEST_FAIL("xor_parity_null_ptrs", "expected error for null ptr");
}

static int test_xor_parity_zero_stripes(void) {
    void *dummy = (void *)0x1234;
    void *d_ptrs[4] = {};
    int rc = gpu_ec_xor_parity(d_ptrs, 0, 4096, dummy);
    if (rc != 0) {
        TEST_PASS("xor_parity_zero_stripes");
        return 0;
    }
    TEST_FAIL("xor_parity_zero_stripes", "expected error for zero stripes");
}

static int test_xor_parity_zero_len(void) {
    void *dummy = (void *)0x1234;
    void *d_ptrs[4] = {};
    int rc = gpu_ec_xor_parity(d_ptrs, 4, 0, dummy);
    if (rc != 0) {
        TEST_PASS("xor_parity_zero_len");
        return 0;
    }
    TEST_FAIL("xor_parity_zero_len", "expected error for zero stripe len");
}

static int test_multi_sm_null_ptrs(void) {
    void *d_parity_ptrs[2] = {};
    int rc = gpu_ec_encode_multi_sm(NULL, d_parity_ptrs, 4, 2, 4096, 0, GPU_EC_MODE_NATIVE);
    if (rc != 0) {
        TEST_PASS("multi_sm_null_ptrs");
        return 0;
    }
    TEST_FAIL("multi_sm_null_ptrs", "expected error for null ptrs");
}

int main(void)
{
    printf("=== test_ec_comprehensive ===\n");
    int failures = 0;

    /* Initialize GF tables for host tests (can be done before engine init) */
    gpu_ec_init_gf_tables();

    /* Allocate GPU memory BEFORE engine init to avoid deadlock */
    struct test_buffers buf = {};
    const size_t max_size = 1024 * 1024; /* 1MB */
    if (alloc_test_buffers(&buf, max_size) != 0) {
        fprintf(stderr, "Failed to allocate GPU buffers\n");
        return 1;
    }

    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Failed to init GPU engine\n");
        free_test_buffers(&buf);
        return 1;
    }

    /* ── Category 1: Boundary & Alignment Tests ───────────────────────── */
    printf("\nCategory 1: Boundary & Alignment Tests\n");
    int f_1[] = {1};
    int f_2[] = {2};
    int f_03[] = {0, 3};
    int f_12[] = {1, 2};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 1, f_1, 1, PATTERN_SEED, "encode_decode_1byte");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 15, f_1, 1, PATTERN_SEED, "encode_decode_15bytes");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 17, f_2, 1, PATTERN_SEED, "encode_decode_17bytes");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4087, f_03, 2, PATTERN_SEED, "encode_decode_non_aligned");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_03, 2, PATTERN_SEED, "encode_decode_exact_aligned");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 1048576, f_12, 2, PATTERN_SEED, "encode_decode_1MB");

    /* ── Category 2: GF(2^8) Algebraic Property Verification ───────────── */
    printf("\nCategory 2: GF(2^8) Algebraic Property Verification\n");
    failures += test_category_2_gf_properties();

    /* ── Category 3: Exhaustive Failure Combination Coverage ───────────── */
    printf("\nCategory 3: Exhaustive Failure Combination Coverage\n");
    int f_0[] = {0};
    int f_3[] = {3};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_0, 1, PATTERN_SEED, "single_fail_D0");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_1, 1, PATTERN_SEED, "single_fail_D1");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_2, 1, PATTERN_SEED, "single_fail_D2");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_3, 1, PATTERN_SEED, "single_fail_D3");
    
    int f_01[] = {0, 1};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_01, 2, PATTERN_SEED, "double_fail_01");
    int f_02[] = {0, 2};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_02, 2, PATTERN_SEED, "double_fail_02");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_03, 2, PATTERN_SEED, "double_fail_03");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_12, 2, PATTERN_SEED, "double_fail_12");
    int f_13[] = {1, 3};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_13, 2, PATTERN_SEED, "double_fail_13");
    int f_23[] = {2, 3};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_23, 2, PATTERN_SEED, "double_fail_23");

    int f_30[] = {3, 0};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_30, 2, PATTERN_SEED, "double_fail_30");
    int f_21[] = {2, 1};
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_21, 2, PATTERN_SEED, "double_fail_21");

    /* ── Category 4: CPU Decode Descending Index Test ───────────────────── */
    printf("\nCategory 4: CPU Decode Descending Index Test\n");
    int f_cpu_30[] = {3, 0};
    int f_cpu_31[] = {3, 1};
    failures += test_cpu_decode_descending(f_cpu_30, "cpu_decode_descending_30");
    failures += test_cpu_decode_descending(f_cpu_31, "cpu_decode_descending_31");

    /* ── Category 5: Data Pattern Stress Tests ─────────────────────────── */
    printf("\nCategory 5: Data Pattern Stress Tests\n");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_12, 2, PATTERN_ZEROS, "pattern_all_zeros");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_12, 2, PATTERN_ONES, "pattern_all_ones");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_12, 2, PATTERN_AA, "pattern_alternating_0xAA");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_12, 2, PATTERN_SEQ, "pattern_sequential_bytes");
    failures += run_roundtrip_test(eng, &buf, 4, 2, 4096, f_12, 2, PATTERN_SEED, "pattern_random_seeded");

    /* ── Category 6: API Robustness & Negative Tests ───────────────────── */
    printf("\nCategory 6: API Robustness & Negative Tests\n");
    failures += test_decode_failed_cnt_3(eng);
    failures += test_decode_failed_idx_oob(eng);
    failures += test_decode_parity_insufficient(eng);
    failures += test_encode_stripe_cnt_0(eng);
    failures += test_encode_stripe_cnt_17(eng);
    failures += test_encode_cell_size_0(eng);
    failures += test_xor_parity_null_ptrs();
    failures += test_xor_parity_zero_stripes();
    failures += test_xor_parity_zero_len();
    failures += test_multi_sm_null_ptrs();

    /* ── Category 7: Encode-Decode Round-Trip with Varying k ───────────── */
    printf("\nCategory 7: Encode-Decode Round-Trip with Varying k\n");
    int f_01_var[] = {0, 1};
    int f_25_var[] = {2, 5};
    int f_015_var[] = {0, 15};
    int f_2_var[] = {2};
    failures += run_roundtrip_test(eng, &buf, 2, 2, 4096, f_01_var, 2, PATTERN_SEED, "roundtrip_2plus2");
    failures += run_roundtrip_test(eng, &buf, 8, 2, 4096, f_25_var, 2, PATTERN_SEED, "roundtrip_8plus2");
    failures += run_roundtrip_test(eng, &buf, 16, 2, 4096, f_015_var, 2, PATTERN_SEED, "roundtrip_16plus2");
    failures += run_roundtrip_test(eng, &buf, 4, 1, 4096, f_2_var, 1, PATTERN_SEED, "roundtrip_ponly_4plus1");

    /* Clean up */
    gpu_engine_fini(eng);
    free_test_buffers(&buf);

    printf("\n=== Comprehensive Test Summary ===\n");
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
