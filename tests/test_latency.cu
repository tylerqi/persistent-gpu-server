/**
 * test_latency.cu — Dispatch latency measurement
 */
#include "gpu_engine.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

static inline double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

/* Measure NOP dispatch latency (submit → poll ready) */
static int test_nop_latency(void)
{
    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    const int N = 10000;
    const int WARMUP = 100;
    double *latencies = (double *)malloc(sizeof(double) * N);

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        gpu_work_item_t item;
        memset(&item, 0, sizeof(item));
        item.op_type = GPU_OP_NOP;
        gpu_result_t result;
        gpu_engine_submit_and_wait(eng, &item, &result);
    }

    /* Measure */
    for (int i = 0; i < N; i++) {
        gpu_work_item_t item;
        memset(&item, 0, sizeof(item));
        item.op_type = GPU_OP_NOP;

        double t0 = now_us();

        uint64_t ticket;
        gpu_engine_submit(eng, &item, &ticket);
        gpu_result_t result;
        while (gpu_engine_poll(eng, ticket, &result) == 0) { /* spin */ }

        double t1 = now_us();
        latencies[i] = t1 - t0;
    }

    /* Sort for percentiles */
    qsort(latencies, N, sizeof(double), cmp_double);

    double sum = 0;
    for (int i = 0; i < N; i++) sum += latencies[i];

    printf("  NOP dispatch latency (%d samples):\n", N);
    printf("    min   = %.2f µs\n", latencies[0]);
    printf("    p50   = %.2f µs\n", latencies[N/2]);
    printf("    p90   = %.2f µs\n", latencies[(int)(N*0.9)]);
    printf("    p99   = %.2f µs\n", latencies[(int)(N*0.99)]);
    printf("    p99.9 = %.2f µs\n", latencies[(int)(N*0.999)]);
    printf("    max   = %.2f µs\n", latencies[N-1]);
    printf("    avg   = %.2f µs\n", sum / N);

    /* Sanity check: p50 should be under 100 µs for basic functionality */
    if (latencies[N/2] > 100.0) {
        free(latencies);
        gpu_engine_fini(eng);
        TEST_FAIL("nop_latency", "p50 latency > 100 µs");
    }

    free(latencies);
    gpu_engine_fini(eng);
    TEST_PASS("nop_latency");
    return 0;
}

/* Measure CRC32C latency for 4KB data */
static int test_crc32c_latency(void)
{
    const size_t len = 4096;
    void *d_data;
    cudaMalloc(&d_data, len);
    cudaMemset(d_data, 0xAB, len);

    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    const int N = 5000;
    const int WARMUP = 50;
    double *latencies = (double *)malloc(sizeof(double) * N);

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        gpu_work_item_t item;
        memset(&item, 0, sizeof(item));
        item.op_type = GPU_OP_CRC32C;
        item.data_ptr = d_data;
        item.data_len = len;
        gpu_result_t result;
        gpu_engine_submit_and_wait(eng, &item, &result);
    }

    /* Measure */
    for (int i = 0; i < N; i++) {
        gpu_work_item_t item;
        memset(&item, 0, sizeof(item));
        item.op_type = GPU_OP_CRC32C;
        item.data_ptr = d_data;
        item.data_len = len;

        double t0 = now_us();
        uint64_t ticket;
        gpu_engine_submit(eng, &item, &ticket);
        gpu_result_t result;
        while (gpu_engine_poll(eng, ticket, &result) == 0) {}
        double t1 = now_us();

        latencies[i] = t1 - t0;
    }

    qsort(latencies, N, sizeof(double), cmp_double);

    double sum = 0;
    for (int i = 0; i < N; i++) sum += latencies[i];

    printf("  CRC32C 4KB dispatch latency (%d samples):\n", N);
    printf("    min   = %.2f µs\n", latencies[0]);
    printf("    p50   = %.2f µs\n", latencies[N/2]);
    printf("    p99   = %.2f µs\n", latencies[(int)(N*0.99)]);
    printf("    avg   = %.2f µs\n", sum / N);

    free(latencies);
    gpu_engine_fini(eng);
    cudaFree(d_data);
    TEST_PASS("crc32c_latency");
    return 0;
}

int main(void)
{
    printf("=== test_latency ===\n");
    int failures = 0;
    failures += test_nop_latency();
    failures += test_crc32c_latency();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
