/**
 * test_metrics.cu — Unit test: telemetry/metrics counters
 *
 * Validates that gpu_engine_get_metrics() returns correct counters
 * after submitting known work via the persistent kernel.
 */
#include "gpu_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Test 1: Metrics after submitting N NOPs */
static int test_metrics_after_nops(void)
{
    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) TEST_FAIL("metrics_after_nops", "init failed");

    const int N = 100;
    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_NOP;

    for (int i = 0; i < N; i++) {
        gpu_result_t res;
        rc = gpu_engine_submit_and_wait(eng, &item, &res);
        if (rc != 0) {
            gpu_engine_fini(eng);
            TEST_FAIL("metrics_after_nops", "submit_and_wait failed");
        }
    }

    gpu_engine_metrics_t metrics;
    gpu_engine_get_metrics(eng, &metrics);

    gpu_engine_fini(eng);

    /* items_submitted must be >= N (other tests might have run concurrently,
     * but this is a standalone test so it should be exactly N) */
    if (metrics.items_submitted < (uint64_t)N) {
        char msg[128];
        snprintf(msg, sizeof(msg), "items_submitted=%lu, expected >= %d",
                 (unsigned long)metrics.items_submitted, N);
        TEST_FAIL("metrics_after_nops", msg);
    }

    if (metrics.items_completed < (uint64_t)N) {
        char msg[128];
        snprintf(msg, sizeof(msg), "items_completed=%lu, expected >= %d",
                 (unsigned long)metrics.items_completed, N);
        TEST_FAIL("metrics_after_nops", msg);
    }

    if (metrics.total_poll_spins == 0) {
        TEST_FAIL("metrics_after_nops", "total_poll_spins should be > 0 after blocking waits");
    }

    TEST_PASS("metrics_after_nops");
    return 0;
}

/* Test 2: Queue full events should be 0 under light load */
static int test_metrics_no_queue_full(void)
{
    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) TEST_FAIL("metrics_no_queue_full", "init failed");

    /* Submit just a few items — queue depth is 4096, this should never overflow */
    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_NOP;

    for (int i = 0; i < 10; i++) {
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
    }

    gpu_engine_metrics_t metrics;
    gpu_engine_get_metrics(eng, &metrics);

    gpu_engine_fini(eng);

    if (metrics.queue_full_events != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "queue_full_events=%lu, expected 0",
                 (unsigned long)metrics.queue_full_events);
        TEST_FAIL("metrics_no_queue_full", msg);
    }

    TEST_PASS("metrics_no_queue_full");
    return 0;
}

/* Test 3: Metrics counters start at zero on fresh engine */
static int test_metrics_initial_zero(void)
{
    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) TEST_FAIL("metrics_initial_zero", "init failed");

    gpu_engine_metrics_t metrics;
    gpu_engine_get_metrics(eng, &metrics);

    gpu_engine_fini(eng);

    if (metrics.items_submitted != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "items_submitted=%lu, expected 0",
                 (unsigned long)metrics.items_submitted);
        TEST_FAIL("metrics_initial_zero", msg);
    }

    if (metrics.items_completed != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg), "items_completed=%lu, expected 0",
                 (unsigned long)metrics.items_completed);
        TEST_FAIL("metrics_initial_zero", msg);
    }

    TEST_PASS("metrics_initial_zero");
    return 0;
}

int main(void)
{
    printf("=== test_metrics ===\n");

    int failures = 0;
    failures += test_metrics_initial_zero();
    failures += test_metrics_after_nops();
    failures += test_metrics_no_queue_full();

    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
