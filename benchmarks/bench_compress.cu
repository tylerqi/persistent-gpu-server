/**
 * bench_compress.cu — LZ4 Compression Throughput Benchmark
 *
 * Measures standalone compression/decompression throughput at various
 * data sizes and patterns. Uses the host-callable gpu_lz4_compress()
 * API (not the persistent kernel).
 *
 * NOTE: Without nvCOMPDx, this benchmarks the memcpy stub (1:1 ratio).
 */
#include "gpu_comp.h"
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

static void fill_pattern(uint8_t *buf, size_t len, const char *label)
{
    if (strcmp(label, "zeros") == 0) {
        memset(buf, 0, len);
    } else if (strcmp(label, "repeating") == 0) {
        for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)(i % 16);
    } else { /* "random" */
        for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)((i * 2654435761u) & 0xFF);
    }
}

static void bench_compress(size_t len, const char *pattern, int iterations)
{
    uint8_t *h_data = (uint8_t *)malloc(len);
    fill_pattern(h_data, len, pattern);

    void *d_in, *d_out, *d_decomp;
    size_t out_max = len * 2; /* Worst case for LZ4 */
    cudaMalloc(&d_in, len);
    cudaMalloc(&d_out, out_max);
    cudaMalloc(&d_decomp, len);
    cudaMemcpy(d_in, h_data, len, cudaMemcpyHostToDevice);

    /* Warmup */
    size_t comp_len = 0;
    for (int i = 0; i < 5; i++) {
        gpu_lz4_compress(d_in, len, d_out, out_max, &comp_len);
    }

    /* Benchmark compress */
    double t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        gpu_lz4_compress(d_in, len, d_out, out_max, &comp_len);
    }
    double comp_time = now_sec() - t0;
    double comp_gbps = ((double)len * iterations) / comp_time / 1e9;
    double ratio = (comp_len > 0) ? (double)len / comp_len : 0;

    /* Benchmark decompress */
    size_t decomp_len = 0;
    for (int i = 0; i < 5; i++) {
        gpu_lz4_decompress(d_out, comp_len, d_decomp, len, &decomp_len);
    }

    t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        gpu_lz4_decompress(d_out, comp_len, d_decomp, len, &decomp_len);
    }
    double decomp_time = now_sec() - t0;
    double decomp_gbps = ((double)len * iterations) / decomp_time / 1e9;

    /* Verify roundtrip */
    uint8_t *h_out = (uint8_t *)malloc(len);
    cudaMemcpy(h_out, d_decomp, len, cudaMemcpyDeviceToHost);
    const char *verify = (memcmp(h_data, h_out, len) == 0) ? "OK" : "MISMATCH";

    printf("  %8zuB %-10s: comp=%.3f GB/s  decomp=%.3f GB/s  ratio=%.2fx  %s\n",
           len, pattern, comp_gbps, decomp_gbps, ratio, verify);

    free(h_data); free(h_out);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_decomp);
}

int main(void)
{
    printf("=== LZ4 Compression Throughput Benchmark ===\n");
#ifndef USE_NVCOMPDX
    printf("  (Memcpy stub mode — no real compression)\n");
#endif
    printf("  (Standalone kernel, not persistent kernel)\n\n");

    const char *patterns[] = {"zeros", "repeating", "random"};
    size_t sizes[] = {4096, 65536, 262144, 1048576};
    int iters[] = {5000, 1000, 500, 200};

    for (int p = 0; p < 3; p++) {
        printf("Pattern: %s\n", patterns[p]);
        for (int s = 0; s < 4; s++) {
            bench_compress(sizes[s], patterns[p], iters[s]);
        }
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
