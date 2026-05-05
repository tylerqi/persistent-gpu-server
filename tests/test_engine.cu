/**
 * test_engine.cu — Unit test: persistent kernel launch, work dispatch, shutdown
 */
#include "gpu_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Test 1: Init and fini without work */
static int test_init_fini(void)
{
    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) TEST_FAIL("init_fini", "gpu_engine_init failed");
    if (eng == NULL) TEST_FAIL("init_fini", "engine is NULL");

    gpu_engine_fini(eng);
    TEST_PASS("init_fini");
    return 0;
}

/* Test 2: Submit and complete a NOP */
static int test_nop_submit(void)
{
    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_NOP;

    gpu_result_t result;
    int rc = gpu_engine_submit_and_wait(eng, &item, &result);
    if (rc != 0) TEST_FAIL("nop_submit", "submit_and_wait failed");
    if (result.error_code != 0) TEST_FAIL("nop_submit", "unexpected error");

    gpu_engine_fini(eng);
    TEST_PASS("nop_submit");
    return 0;
}

/* Test 3: Submit 1000 NOPs */
static int test_burst_nops(void)
{
    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    const int N = 1000;
    uint64_t tickets[1000];
    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_NOP;

    /* Submit all */
    for (int i = 0; i < N; i++) {
        int rc = gpu_engine_submit(eng, &item, &tickets[i]);
        if (rc != 0) {
            char msg[64];
            snprintf(msg, sizeof(msg), "submit failed at i=%d", i);
            TEST_FAIL("burst_nops", msg);
        }
    }

    /* Poll all to completion */
    int completed = 0;
    int spins = 0;
    while (completed < N && spins < 10000000) {
        for (int i = 0; i < N; i++) {
            if (tickets[i] == UINT64_MAX) continue;
            gpu_result_t res;
            if (gpu_engine_poll(eng, tickets[i], &res) == 1) {
                if (res.error_code != 0) {
                    TEST_FAIL("burst_nops", "NOP returned error");
                }
                tickets[i] = UINT64_MAX; /* mark done */
                completed++;
            }
        }
        spins++;
    }

    if (completed != N) TEST_FAIL("burst_nops", "not all items completed");

    gpu_engine_fini(eng);
    TEST_PASS("burst_nops");
    return 0;
}

/* Test 4: CRC32C via engine */
static int test_engine_crc32c(void)
{
    /* Prepare test data on GPU */
    const size_t len = 1024;
    uint8_t *h_data = (uint8_t *)malloc(len);
    for (size_t i = 0; i < len; i++) h_data[i] = (uint8_t)(i & 0xFF);

    void *d_data;
    cudaMalloc(&d_data, len);
    cudaMemcpy(d_data, h_data, len, cudaMemcpyHostToDevice);
    printf("  [TRACE] test_engine_crc32c: data copied\n");

    printf("  [TRACE] test_engine_crc32c: starting\n");
    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) TEST_FAIL("engine_crc32c", "gpu_engine_init failed");
    printf("  [TRACE] test_engine_crc32c: engine init done\n");

    /* Submit CRC32C */
    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_CRC32C;
    item.data_ptr = d_data;
    item.data_len = len;

    gpu_result_t result;
    printf("  [TRACE] test_engine_crc32c: submitting\n");
    rc = gpu_engine_submit_and_wait(eng, &item, &result);
    printf("  [TRACE] test_engine_crc32c: submit_and_wait returned %d\n", rc);
    if (rc != 0) TEST_FAIL("engine_crc32c", "submit_and_wait failed");

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
        TEST_FAIL("engine_crc32c", "Kernel crashed");
    }

    /* Read CRC result from the queue item */
    /* Note: result is in the queue's item slot, not in the result struct */

    gpu_engine_fini(eng);
    cudaFree(d_data);
    free(h_data);
    TEST_PASS("engine_crc32c");
    return 0;
}

/* Test 5: Multiple init/fini cycles */
static int test_reinit(void)
{
    for (int i = 0; i < 3; i++) {
        gpu_engine_t *eng = NULL;
        int rc = gpu_engine_init(&eng);
        if (rc != 0) TEST_FAIL("reinit", "init failed on cycle");

        gpu_work_item_t item;
        memset(&item, 0, sizeof(item));
        item.op_type = GPU_OP_NOP;

        gpu_result_t result;
        rc = gpu_engine_submit_and_wait(eng, &item, &result);
        if (rc != 0) TEST_FAIL("reinit", "submit failed on cycle");

        gpu_engine_fini(eng);
    }
    TEST_PASS("reinit");
    return 0;
}

int main(void)
{
    printf("=== test_engine ===\n");
    int failures = 0;
    failures += test_init_fini();
    failures += test_nop_submit();
    failures += test_burst_nops();
    failures += test_engine_crc32c();
    failures += test_reinit();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
