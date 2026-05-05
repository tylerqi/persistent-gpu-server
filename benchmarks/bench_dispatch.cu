/**
 * bench_dispatch.cu — Persistent kernel dispatch latency histogram
 */
#include "gpu_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <thread>
#include <vector>
#include <atomic>

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

static void print_histogram(double *latencies, int n, const char *label)
{
    qsort(latencies, n, sizeof(double), cmp_double);

    double sum = 0;
    for (int i = 0; i < n; i++) sum += latencies[i];

    printf("  %s (%d samples):\n", label, n);
    printf("    min   = %8.2f µs\n", latencies[0]);
    printf("    p50   = %8.2f µs\n", latencies[n/2]);
    printf("    p90   = %8.2f µs\n", latencies[(int)(n*0.9)]);
    printf("    p99   = %8.2f µs\n", latencies[(int)(n*0.99)]);
    printf("    p99.9 = %8.2f µs\n", latencies[(int)(n*0.999)]);
    printf("    max   = %8.2f µs\n", latencies[n-1]);
    printf("    avg   = %8.2f µs\n", sum / n);

    /* Histogram buckets */
    int buckets[] = {1, 2, 5, 10, 20, 50, 100, 500};
    int bucket_count = sizeof(buckets) / sizeof(buckets[0]);
    printf("    distribution:\n");
    int prev = 0;
    for (int b = 0; b < bucket_count; b++) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (latencies[i] >= prev && latencies[i] < buckets[b])
                count++;
        }
        if (count > 0) {
            printf("      %3d-%3d µs: %5d (%5.1f%%)\n",
                   prev, buckets[b], count, 100.0 * count / n);
        }
        prev = buckets[b];
    }
    int tail = 0;
    for (int i = 0; i < n; i++) {
        if (latencies[i] >= prev) tail++;
    }
    if (tail > 0) {
        printf("      %3d+   µs: %5d (%5.1f%%)\n", prev, tail, 100.0 * tail / n);
    }
}

int main(void)
{
    printf("=== Persistent Kernel Dispatch Latency ===\n\n");

    /* Allocate data buffers before engine init to prevent deadlocks */
    const size_t size_4k = 4096;
    const size_t size_1m = 1048576;
    const size_t size_4m = 4 * 1048576;


    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) {
        fprintf(stderr, "Failed to init engine\n");
        return 1;
    }

    const int N = 50000;
    const int WARMUP = 100;
    double *latencies = (double *)malloc(sizeof(double) * N);

    /* ── Helper macro for running a benchmark ─────────────────────── */
    /* ── Helper macro for running a benchmark ─────────────────────── */
#define BENCHMARK_WORKLOAD(name_label, op, d_data_arr, d_len, d_comp_out_arr, comp_out_size, test_N) do { \
    const int num_threads = 128; \
    /* Latency test */ \
    int local_N = test_N / num_threads; \
    int actual_N = local_N * num_threads; \
    std::vector<std::thread> lat_threads; \
    for (int t = 0; t < num_threads; t++) { \
        lat_threads.emplace_back([&, t]() { \
            for (int i = 0; i < WARMUP / num_threads; i++) { \
                gpu_work_item_t item = {}; \
                item.op_type = op; \
                item.data_ptr = d_data_arr[0]; \
                item.data_len = d_len; \
                if (op == GPU_OP_EC_ENCODE) { \
                    for (int s = 0; s < 4; s++) item.ec_ptrs[s] = d_data_arr[0]; \
                    for (int s = 0; s < 2; s++) item.parity_ptrs[s] = d_comp_out_arr[0]; \
                    item.stripe_cnt = 4; item.parity_cnt = 2; item.cell_size = d_len; \
                } \
                gpu_result_t result; \
                gpu_engine_submit_and_wait(eng, &item, &result); \
            } \
            for (int i = 0; i < local_N; i++) { \
                gpu_work_item_t item = {}; \
                item.op_type = op; \
                item.data_ptr = d_data_arr[0]; \
                item.data_len = d_len; \
                if (op == GPU_OP_EC_ENCODE) { \
                    for (int s = 0; s < 4; s++) item.ec_ptrs[s] = d_data_arr[0]; \
                    for (int s = 0; s < 2; s++) item.parity_ptrs[s] = d_comp_out_arr[0]; \
                    item.stripe_cnt = 4; item.parity_cnt = 2; item.cell_size = d_len; \
                } \
                double t0 = now_us(); \
                uint64_t ticket; \
                gpu_engine_submit(eng, &item, &ticket); \
                gpu_result_t result; \
                while (gpu_engine_poll(eng, ticket, &result) == 0) {} \
                latencies[t * local_N + i] = now_us() - t0; \
            } \
        }); \
    } \
    for (auto& t : lat_threads) t.join(); \
    print_histogram(latencies, actual_N, name_label " Latency"); \
    \
    /* Throughput test */ \
    double t_start = now_us(); \
    std::atomic<int> total_ops(0); \
    std::vector<std::atomic<int>> sec_ops(120); \
    for (int i = 0; i < 120; i++) sec_ops[i] = 0; \
    std::vector<std::thread> tp_threads; \
    for (int t = 0; t < num_threads; t++) { \
        tp_threads.emplace_back([&, t]() { \
            int ops = 0; \
            while (true) { \
                double now = now_us(); \
                int sec = (int)((now - t_start) / 1e6); \
                if (sec >= 120) break; \
                gpu_work_item_t item = {}; \
                item.op_type = op; \
                item.data_ptr = d_data_arr[t]; \
                item.data_len = d_len; \
                if (op == GPU_OP_COMPRESS_LZ4 || op == GPU_OP_DECOMPRESS_LZ4) { \
                    item.comp_out_ptr = d_comp_out_arr[t]; \
                    item.comp_max_size = comp_out_size; \
                } else if (op == GPU_OP_EC_ENCODE) { \
                    for (int s = 0; s < 4; s++) item.ec_ptrs[s] = d_data_arr[t]; \
                    for (int s = 0; s < 2; s++) item.parity_ptrs[s] = d_comp_out_arr[t]; \
                    item.stripe_cnt = 4; item.parity_cnt = 2; item.cell_size = d_len; \
                } \
                gpu_result_t result; \
                gpu_engine_submit_and_wait(eng, &item, &result); \
                sec_ops[sec]++; \
                if (sec >= 60) ops++; \
            } \
            total_ops += ops; \
        }); \
    } \
    for (auto& t : tp_threads) t.join(); \
    double elapsed = 60.0; \
    double iops = total_ops / elapsed; \
    double bw_gbps = (iops * d_len) / (1024.0 * 1024.0 * 1024.0); \
    printf("\n  %s Throughput (%d threads):\n", name_label, num_threads); \
    printf("    Average IOPS (60s-120s):      %.0f ops/sec\n", iops); \
    if (d_len > 0) printf("    Average Bandwidth (60s-120s): %.3f GB/s\n", bw_gbps); \
    printf("    Per-second Histogram (60s-120s):\n"); \
    int max_ops = 0; \
    for (int i = 60; i < 120; i++) { \
        if (sec_ops[i] > max_ops) max_ops = sec_ops[i]; \
    } \
    for (int i = 60; i < 120; i++) { \
        int ops_this_sec = sec_ops[i]; \
        double bw_this_sec = (ops_this_sec * d_len) / (1024.0 * 1024.0 * 1024.0); \
        int bar_len = (max_ops > 0) ? (int)((double)ops_this_sec / max_ops * 40) : 0; \
        printf("      [%3ds]: %7d IOPS ", i+1, ops_this_sec); \
        if (d_len > 0) printf("(%5.3f GB/s) ", bw_this_sec); \
        else printf("              "); \
        printf("|"); \
        for (int b = 0; b < bar_len; b++) printf("="); \
        printf("\n"); \
    } \
    printf("\n"); \
} while(0)

    void *d_data_4k[128];
    void *d_data_1m[128];
    void *d_data_4m[128];
    void *d_comp_out[128];
    void *d_comp_out_4m[128];
    for (int i = 0; i < 128; i++) {
        cudaMalloc(&d_data_4k[i], size_4k);
        cudaMalloc(&d_data_1m[i], size_1m);
        cudaMalloc(&d_data_4m[i], size_4m);
        cudaMalloc(&d_comp_out[i], size_1m * 2);
        cudaMalloc(&d_comp_out_4m[i], size_4m * 2);
    }
    void *d_null[128] = {0};

    /* Run the benchmarks */
    BENCHMARK_WORKLOAD("NOP", GPU_OP_NOP, d_null, 0, d_null, 0, 50000);
    BENCHMARK_WORKLOAD("4KB CRC32C", GPU_OP_CRC32C, d_data_4k, size_4k, d_null, 0, 10000);
    BENCHMARK_WORKLOAD("1MB CRC32C", GPU_OP_CRC32C, d_data_1m, size_1m, d_null, 0, 1000);
    BENCHMARK_WORKLOAD("1MB LZ4 Compress", GPU_OP_COMPRESS_LZ4, d_data_1m, size_1m, d_comp_out, size_1m*2, 1000);
    BENCHMARK_WORKLOAD("1MB LZ4 Decompress", GPU_OP_DECOMPRESS_LZ4, d_comp_out, size_1m*2, d_data_1m, size_1m, 1000);
    BENCHMARK_WORKLOAD("4MB LZ4 Compress", GPU_OP_COMPRESS_LZ4, d_data_4m, size_4m, d_comp_out_4m, size_4m*2, 500);
    BENCHMARK_WORKLOAD("4MB LZ4 Decompress", GPU_OP_DECOMPRESS_LZ4, d_comp_out_4m, size_4m*2, d_data_4m, size_4m, 500);
    BENCHMARK_WORKLOAD("1MB EC 4+2 Encode", GPU_OP_EC_ENCODE, d_data_1m, size_1m, d_comp_out, size_1m*2, 1000);

    free(latencies);
    gpu_engine_fini(eng);
    printf("=== Done ===\n");
    return 0;
}
