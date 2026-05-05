/**
 * bench_csum.cu — CRC32C throughput benchmark: GPU vs CPU
 */
#include "gpu_csum.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void bench_size(size_t len, int iterations)
{
    /* Prepare data */
    uint8_t *h_data = (uint8_t *)malloc(len);
    for (size_t i = 0; i < len; i++) h_data[i] = (uint8_t)(i & 0xFF);

    void *d_data;
    cudaMalloc(&d_data, len);
    cudaMemcpy(d_data, h_data, len, cudaMemcpyHostToDevice);

    /* ── CPU benchmark ──────────────────────────────────────────────── */
    double t0 = now_sec();
    uint32_t cpu_crc = 0;
    for (int i = 0; i < iterations; i++) {
        cpu_crc = cpu_crc32c(h_data, len);
    }
    double cpu_time = now_sec() - t0;
    double cpu_gbps = ((double)len * iterations) / cpu_time / 1e9;

    /* ── GPU benchmark ──────────────────────────────────────────────── */
    /* Warmup */
    uint32_t gpu_crc;
    for (int i = 0; i < 10; i++) gpu_crc32c(d_data, len, &gpu_crc);

    t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        gpu_crc32c(d_data, len, &gpu_crc);
    }
    cudaDeviceSynchronize();
    double gpu_time = now_sec() - t0;
    double gpu_gbps = ((double)len * iterations) / gpu_time / 1e9;

    /* Verify correctness */
    const char *match = (gpu_crc == cpu_crc) ? "MATCH" : "MISMATCH";

    printf("  %8zuB: CPU=%.3f GB/s  GPU=%.3f GB/s  speedup=%.1fx  %s\n",
           len, cpu_gbps, gpu_gbps, gpu_gbps/cpu_gbps, match);

    cudaFree(d_data);
    free(h_data);
}

int main(void)
{
    printf("=== CRC32C Throughput Benchmark ===\n");
    printf("  (Standalone kernel, not persistent kernel)\n\n");

    bench_size(1024,      50000);
    bench_size(4096,      20000);
    bench_size(16384,     10000);
    bench_size(65536,      5000);
    bench_size(131072,     2000);
    bench_size(1048576,     500);

    printf("\n=== Done ===\n");
    return 0;
}
