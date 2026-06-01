/**
 * bench_ec.cu — EC Parity Generation Throughput Benchmark
 *
 * Measures standalone EC P and P+Q parity generation throughput
 * at various stripe counts and cell sizes. Uses the host-callable
 * gpu_ec_xor_parity() API (not the persistent kernel).
 */
#include "gpu_ec.h"
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

static void bench_ec_parity(int num_stripes, size_t cell_size, int iterations)
{
    /* Allocate host data */
    uint8_t **h_data = (uint8_t **)malloc(sizeof(uint8_t *) * num_stripes);
    for (int i = 0; i < num_stripes; i++) {
        h_data[i] = (uint8_t *)malloc(cell_size);
        for (size_t j = 0; j < cell_size; j++)
            h_data[i][j] = (uint8_t)((i * 37 + j * 13 + 7) & 0xFF);
    }

    /* Allocate GPU stripes + parity */
    void **d_ptrs = (void **)malloc(sizeof(void *) * num_stripes);
    for (int i = 0; i < num_stripes; i++) {
        cudaMalloc(&d_ptrs[i], cell_size);
        cudaMemcpy(d_ptrs[i], h_data[i], cell_size, cudaMemcpyHostToDevice);
    }

    void *d_parity;
    cudaMalloc(&d_parity, cell_size);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        gpu_ec_xor_parity(d_ptrs, num_stripes, cell_size, d_parity);
    }

    /* GPU benchmark */
    double t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        gpu_ec_xor_parity(d_ptrs, num_stripes, cell_size, d_parity);
    }
    double gpu_time = now_sec() - t0;
    /* Total data processed = num_stripes * cell_size per iteration (reads all stripes) */
    double total_bytes = (double)num_stripes * cell_size * iterations;
    double gpu_gbps = total_bytes / gpu_time / 1e9;
    double iops = iterations / gpu_time;

    /* CPU benchmark */
    uint8_t *cpu_parity = (uint8_t *)calloc(1, cell_size);
    t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        cpu_ec_xor_parity((const void **)h_data, num_stripes, cell_size, cpu_parity);
    }
    double cpu_time = now_sec() - t0;
    double cpu_gbps = total_bytes / cpu_time / 1e9;

    /* Verify */
    uint8_t *gpu_parity_h = (uint8_t *)malloc(cell_size);
    cudaMemcpy(gpu_parity_h, d_parity, cell_size, cudaMemcpyDeviceToHost);
    const char *match = (memcmp(gpu_parity_h, cpu_parity, cell_size) == 0) ? "MATCH" : "MISMATCH";

    printf("  %d+1 %8zuB: CPU=%.3f GB/s  GPU=%.3f GB/s  IOPS=%.0f  speedup=%.1fx  %s\n",
           num_stripes, cell_size, cpu_gbps, gpu_gbps, iops,
           cpu_gbps > 0 ? gpu_gbps / cpu_gbps : 0, match);

    /* Cleanup */
    free(gpu_parity_h); free(cpu_parity);
    for (int i = 0; i < num_stripes; i++) {
        cudaFree(d_ptrs[i]);
        free(h_data[i]);
    }
    free(d_ptrs); free(h_data);
    cudaFree(d_parity);
}

static void cpu_ec_encode_2(const uint8_t **data_ptrs, uint8_t **parity_ptrs, int stripe_cnt, size_t stripe_len, int ec_mode)
{
    // P parity (XOR)
    memset(parity_ptrs[0], 0, stripe_len);
    for (int s = 0; s < stripe_cnt; s++) {
        for (size_t i = 0; i < stripe_len; i++) {
            parity_ptrs[0][i] ^= data_ptrs[s][i];
        }
    }

    // Q parity (Vandermonde Horner)
    uint8_t reduce = (ec_mode == 1) ? 0x1D : 0x1B;
    for (size_t i = 0; i < stripe_len; i++) {
        uint8_t q = data_ptrs[stripe_cnt - 1][i];
        for (int s = stripe_cnt - 2; s >= 0; s--) {
            q = (uint8_t)(((q << 1) & 0xFE) ^ ((q >> 7) * reduce));
            q ^= data_ptrs[s][i];
        }
        parity_ptrs[1][i] = q;
    }
}

static void bench_ec_parity_2(int num_stripes, size_t cell_size, int iterations, int ec_mode)
{
    /* Allocate host data */
    uint8_t **h_data = (uint8_t **)malloc(sizeof(uint8_t *) * num_stripes);
    for (int i = 0; i < num_stripes; i++) {
        h_data[i] = (uint8_t *)malloc(cell_size);
        for (size_t j = 0; j < cell_size; j++)
            h_data[i][j] = (uint8_t)((i * 37 + j * 13 + 7) & 0xFF);
    }

    /* Allocate GPU stripes + parity */
    void **d_ptrs = (void **)malloc(sizeof(void *) * num_stripes);
    for (int i = 0; i < num_stripes; i++) {
        cudaMalloc(&d_ptrs[i], cell_size);
        cudaMemcpy(d_ptrs[i], h_data[i], cell_size, cudaMemcpyHostToDevice);
    }

    void *d_parity[2];
    cudaMalloc(&d_parity[0], cell_size);
    cudaMalloc(&d_parity[1], cell_size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        gpu_ec_encode_multi_sm(d_ptrs, d_parity, num_stripes, 2, cell_size, stream, ec_mode);
        cudaStreamSynchronize(stream);
    }

    /* GPU benchmark */
    double t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        gpu_ec_encode_multi_sm(d_ptrs, d_parity, num_stripes, 2, cell_size, stream, ec_mode);
        cudaStreamSynchronize(stream);
    }
    double gpu_time = now_sec() - t0;
    double total_bytes = (double)num_stripes * cell_size * iterations;
    double gpu_gbps = total_bytes / gpu_time / 1e9;
    double iops = iterations / gpu_time;

    /* CPU benchmark */
    uint8_t *cpu_parity[2];
    cpu_parity[0] = (uint8_t *)calloc(1, cell_size);
    cpu_parity[1] = (uint8_t *)calloc(1, cell_size);
    t0 = now_sec();
    for (int i = 0; i < iterations; i++) {
        cpu_ec_encode_2((const uint8_t **)h_data, cpu_parity, num_stripes, cell_size, ec_mode);
    }
    double cpu_time = now_sec() - t0;
    double cpu_gbps = total_bytes / cpu_time / 1e9;

    /* Verify */
    uint8_t *gpu_parity_h[2];
    gpu_parity_h[0] = (uint8_t *)malloc(cell_size);
    gpu_parity_h[1] = (uint8_t *)malloc(cell_size);
    cudaMemcpy(gpu_parity_h[0], d_parity[0], cell_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_parity_h[1], d_parity[1], cell_size, cudaMemcpyDeviceToHost);
    
    bool match_ok = (memcmp(gpu_parity_h[0], cpu_parity[0], cell_size) == 0) &&
                    (memcmp(gpu_parity_h[1], cpu_parity[1], cell_size) == 0);
    const char *match = match_ok ? "MATCH" : "MISMATCH";

    printf("  %d+2 %8zuB: CPU=%.3f GB/s  GPU=%.3f GB/s  IOPS=%.0f  speedup=%.1fx  %s\n",
           num_stripes, cell_size, cpu_gbps, gpu_gbps, iops,
           cpu_gbps > 0 ? gpu_gbps / cpu_gbps : 0, match);

    /* Cleanup */
    free(gpu_parity_h[0]); free(gpu_parity_h[1]);
    free(cpu_parity[0]); free(cpu_parity[1]);
    for (int i = 0; i < num_stripes; i++) {
        cudaFree(d_ptrs[i]);
        free(h_data[i]);
    }
    free(d_ptrs); free(h_data);
    cudaFree(d_parity[0]); cudaFree(d_parity[1]);
    cudaStreamDestroy(stream);
}

int main(void)
{
    printf("=== EC Parity Generation Benchmark ===\n");
    printf("  (Standalone kernel, XOR P-parity and P+Q parity)\n\n");

    int stripe_counts[] = {2, 4, 8, 16};
    size_t cell_sizes[] = {4096, 65536, 262144, 1048576};
    int iters[] = {5000, 2000, 500, 200};

    for (int sc = 0; sc < 4; sc++) {
        printf("Cell size: %zuB\n", cell_sizes[sc]);
        for (int k = 0; k < 4; k++) {
            bench_ec_parity(stripe_counts[k], cell_sizes[sc], iters[sc]);
        }
        for (int k = 0; k < 4; k++) {
            bench_ec_parity_2(stripe_counts[k], cell_sizes[sc], iters[sc], 1); // ec_mode = 1 (RAID6)
        }
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
