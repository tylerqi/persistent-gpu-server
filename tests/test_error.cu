/**
 * test_error.cu — GPU error handling tests
 */
#include "gpu_engine.h"
#include "gpu_error.h"
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Test: NULL data pointer should produce GPU_ERR_INVAL */
static int test_null_data_error(void)
{
    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_CRC32C;
    item.data_ptr = NULL;  /* Invalid! */
    item.data_len = 1024;

    gpu_result_t result;
    gpu_engine_submit_and_wait(eng, &item, &result);

    if (result.error_code != GPU_ERR_INVAL) {
        char msg[128];
        snprintf(msg, sizeof(msg), "expected GPU_ERR_INVAL(%d), got %d",
                 GPU_ERR_INVAL, result.error_code);
        gpu_engine_fini(eng);
        TEST_FAIL("null_data_error", msg);
    }

    gpu_engine_fini(eng);
    TEST_PASS("null_data_error");
    return 0;
}

/* Test: zero length should produce GPU_ERR_INVAL */
static int test_zero_len_error(void)
{
    void *d_data;
    cudaMalloc(&d_data, 1024);

    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_CRC32C;
    item.data_ptr = d_data;
    item.data_len = 0;  /* Invalid! */

    gpu_result_t result;
    gpu_engine_submit_and_wait(eng, &item, &result);

    gpu_engine_fini(eng);
    cudaFree(d_data);
    if (result.error_code != GPU_ERR_INVAL) {
        TEST_FAIL("zero_len_error", "expected GPU_ERR_INVAL");
    }

    TEST_PASS("zero_len_error");
    return 0;
}

/* Test: error on one item doesn't affect the next */
static int test_error_isolation(void)
{
    void *d_data;
    cudaMalloc(&d_data, 1024);
    uint8_t h_data[1024];
    memset(h_data, 0xAB, 1024);
    cudaMemcpy(d_data, h_data, 1024, cudaMemcpyHostToDevice);

    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    /* Submit bad item */
    gpu_work_item_t bad_item;
    memset(&bad_item, 0, sizeof(bad_item));
    bad_item.op_type = GPU_OP_CRC32C;
    bad_item.data_ptr = NULL;
    bad_item.data_len = 1024;

    gpu_result_t bad_result;
    gpu_engine_submit_and_wait(eng, &bad_item, &bad_result);
    if (bad_result.error_code != GPU_ERR_INVAL) {
        gpu_engine_fini(eng);
        cudaFree(d_data);
        TEST_FAIL("error_isolation", "bad item should return GPU_ERR_INVAL");
    }

    /* Submit good item after the bad one */
    gpu_work_item_t good_item;
    memset(&good_item, 0, sizeof(good_item));
    good_item.op_type = GPU_OP_NOP;

    gpu_result_t good_result;
    gpu_engine_submit_and_wait(eng, &good_item, &good_result);
    if (good_result.error_code != GPU_SUCCESS) {
        gpu_engine_fini(eng);
        cudaFree(d_data);
        TEST_FAIL("error_isolation", "good item after bad should succeed");
    }

    gpu_engine_fini(eng);
    cudaFree(d_data);
    TEST_PASS("error_isolation");
    return 0;
}

/* Test: error string helper */
static int test_error_strings(void)
{
    if (strcmp(gpu_error_string(GPU_SUCCESS), "success") != 0)
        TEST_FAIL("error_strings", "GPU_SUCCESS string wrong");
    if (strcmp(gpu_error_string(GPU_ERR_INVAL), "invalid argument") != 0)
        TEST_FAIL("error_strings", "GPU_ERR_INVAL string wrong");
    if (strcmp(gpu_error_string(-9999), "unknown error") != 0)
        TEST_FAIL("error_strings", "unknown error string wrong");

    TEST_PASS("error_strings");
    return 0;
}

int main(void)
{
    printf("=== test_error ===\n");
    int failures = 0;
    failures += test_null_data_error();
    failures += test_zero_len_error();
    failures += test_error_isolation();
    failures += test_error_strings();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
