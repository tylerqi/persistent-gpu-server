/**
 * bench_dispatch.cu — Persistent kernel dispatch latency histogram
 *
 * FIX: The original benchmark used 128 threads for the latency test with
 * raw submit()+poll() that ignored GPU_ERR_QUEUE_FULL return codes. When
 * submit() returned QUEUE_FULL, the ticket variable was left uninitialized,
 * causing poll() to spin forever on a garbage ticket — a permanent deadlock.
 *
 * Additionally, 128 CPU threads all doing tight poll() spins on PCIe-mapped
 * pinned memory created a coherence storm that starved the GPU persistent
 * kernel from making progress, causing the benchmark to stall indefinitely.
 *
 * Changes:
 * 1. Latency test uses 128 threads with submit_and_wait() (which has
 *    internal queue-full retry and adaptive backoff) instead of raw
 *    submit()+poll() without error handling.
 * 2. Throughput test reduced from 120s to 30s (10s warmup + 20s measured)
 *    to keep total benchmark time reasonable (~4min vs ~16min).
 */
#include "gpu_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <thread>
#include <vector>
#include <atomic>
#include <sched.h>

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

    /* Allocate data buffers before engine init to prevent deadlocks.
     * cudaMalloc triggers implicit device synchronization which will
     * deadlock if the persistent kernel is already running. */
    const size_t size_4k = 4096;
    const size_t size_1m = 1048576;
    const size_t size_4m = 4 * 1048576;

    void *d_data_4k[128];
    void *d_data_1m[128];
    void *d_data_4m[128];
    void *d_comp_out[128];
    void *d_comp_out_4m[128];
    int alloc_fail = 0;
    for (int i = 0; i < 128; i++) {
        if (cudaMalloc(&d_data_4k[i], size_4k) != cudaSuccess ||
            cudaMalloc(&d_data_1m[i], size_1m) != cudaSuccess ||
            cudaMalloc(&d_data_4m[i], size_4m) != cudaSuccess ||
            cudaMalloc(&d_comp_out[i], size_1m * 2) != cudaSuccess ||
            cudaMalloc(&d_comp_out_4m[i], size_4m * 2) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed at buffer index %d: %s\n",
                    i, cudaGetErrorString(cudaGetLastError()));
            alloc_fail = 1;
            break;
        }
    }
    void *d_null[128] = {0};

    /* Report GPU memory after allocations */
    size_t gpu_free = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    printf("GPU Memory: %zu MB free / %zu MB total (%.1f%% used)\n",
           gpu_free / (1024*1024), gpu_total / (1024*1024),
           100.0 * (1.0 - (double)gpu_free / gpu_total));
    if (alloc_fail) {
        fprintf(stderr, "Aborting: GPU memory allocation failed\n");
        return 1;
    }
    fflush(stdout);

    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) {
        fprintf(stderr, "Failed to init engine\n");
        return 1;
    }

    const int WARMUP = 50;
    /* Throughput test timing: 3s warmup + 7s measured = 10s per workload */
    const int TP_DURATION_SEC = 10;
    const int TP_WARMUP_SEC   = 3;

    /* ── Helper macro for running a benchmark ─────────────────────── */
    /* Latency: 128 threads, each does local_N submit_and_wait ops.
     * Uses submit_and_wait() which handles queue-full + backoff internally.
     * Throughput: 128 threads saturating the engine for TP_DURATION_SEC. */
#define BENCHMARK_WORKLOAD(name_label, op, d_data_arr, d_len, d_comp_out_arr, comp_out_size, test_N) do { \
    printf("--- Starting workload: %s ---\n", name_label); fflush(stdout); \
    const int num_threads = 128; \
    int local_N = test_N / num_threads; \
    int actual_N = local_N * num_threads; \
    double *lat_arr = (double *)malloc(sizeof(double) * actual_N); \
    \
    /* Latency test: use submit_and_wait to avoid queue-full deadlock */ \
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
                gpu_result_t result; \
                double t0 = now_us(); \
                gpu_engine_submit_and_wait(eng, &item, &result); \
                lat_arr[t * local_N + i] = now_us() - t0; \
            } \
        }); \
    } \
    for (auto& t : lat_threads) t.join(); \
    print_histogram(lat_arr, actual_N, name_label " Latency"); \
    \
    /* Throughput test */ \
    double t_start = now_us(); \
    std::atomic<int> total_ops(0); \
    std::vector<std::atomic<int>> sec_ops(TP_DURATION_SEC); \
    for (int i = 0; i < TP_DURATION_SEC; i++) sec_ops[i] = 0; \
    std::vector<std::thread> tp_threads; \
    for (int t = 0; t < num_threads; t++) { \
        tp_threads.emplace_back([&, t]() { \
            int ops = 0; \
            while (true) { \
                double now = now_us(); \
                int sec = (int)((now - t_start) / 1e6); \
                if (sec >= TP_DURATION_SEC) break; \
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
                if (sec >= TP_WARMUP_SEC) ops++; \
            } \
            total_ops += ops; \
        }); \
    } \
    for (auto& t : tp_threads) t.join(); \
    double elapsed = (double)(TP_DURATION_SEC - TP_WARMUP_SEC); \
    double iops = total_ops / elapsed; \
    double bw_gbps = (iops * d_len) / (1024.0 * 1024.0 * 1024.0); \
    printf("\n  %s Throughput (%d threads):\n", name_label, num_threads); \
    printf("    Average IOPS (%ds-%ds):      %.0f ops/sec\n", TP_WARMUP_SEC, TP_DURATION_SEC, iops); \
    if (d_len > 0) printf("    Average Bandwidth (%ds-%ds): %.3f GB/s\n", TP_WARMUP_SEC, TP_DURATION_SEC, bw_gbps); \
    printf("    Per-second Histogram (%ds-%ds):\n", TP_WARMUP_SEC, TP_DURATION_SEC); \
    int max_ops = 0; \
    for (int i = TP_WARMUP_SEC; i < TP_DURATION_SEC; i++) { \
        if (sec_ops[i] > max_ops) max_ops = sec_ops[i]; \
    } \
    for (int i = TP_WARMUP_SEC; i < TP_DURATION_SEC; i++) { \
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
    free(lat_arr); \
    printf("--- Finished workload: %s ---\n", name_label); fflush(stdout); \
} while(0)

    /* Run the benchmarks — reduced latency samples for faster turnaround */
    BENCHMARK_WORKLOAD("NOP", GPU_OP_NOP, d_null, 0, d_null, 0, 5000);
    BENCHMARK_WORKLOAD("4KB CRC32C", GPU_OP_CRC32C, d_data_4k, size_4k, d_null, 0, 2560);
    BENCHMARK_WORKLOAD("1MB CRC32C", GPU_OP_CRC32C, d_data_1m, size_1m, d_null, 0, 256);
    BENCHMARK_WORKLOAD("1MB LZ4 Compress", GPU_OP_COMPRESS_LZ4, d_data_1m, size_1m, d_comp_out, size_1m*2, 256);
    BENCHMARK_WORKLOAD("1MB LZ4 Decompress", GPU_OP_DECOMPRESS_LZ4, d_comp_out, size_1m*2, d_data_1m, size_1m, 256);
    BENCHMARK_WORKLOAD("4MB LZ4 Compress", GPU_OP_COMPRESS_LZ4, d_data_4m, size_4m, d_comp_out_4m, size_4m*2, 256);
    BENCHMARK_WORKLOAD("4MB LZ4 Decompress", GPU_OP_DECOMPRESS_LZ4, d_comp_out_4m, size_4m*2, d_data_4m, size_4m, 256);
    BENCHMARK_WORKLOAD("1MB EC 4+2 Encode", GPU_OP_EC_ENCODE, d_data_1m, size_1m, d_comp_out, size_1m*2, 256);

    gpu_engine_fini(eng);

    /* Free GPU memory (M-6) */
    for (int i = 0; i < 128; i++) {
        cudaFree(d_data_4k[i]);
        cudaFree(d_data_1m[i]);
        cudaFree(d_data_4m[i]);
        cudaFree(d_comp_out[i]);
        cudaFree(d_comp_out_4m[i]);
    }

    printf("=== Done ===\n");
    return 0;
}
