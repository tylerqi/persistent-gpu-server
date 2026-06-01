/**
 * bench_ec_e2e_quick.cu — Quick EC E2E test with multi-SM kernel.
 */
#include "gpu_engine.h"
#include "gpu_ec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    printf("=== Quick EC E2E (multi-SM) ===\n\n");
    fflush(stdout);

    const size_t SZ = 1048576;
    const int K = 4, P = 2, ITERS = 200;

    void *h_data, *h_parity, *d_data, *d_parity;
    cudaHostAlloc(&h_data, K * SZ, cudaHostAllocDefault);
    cudaHostAlloc(&h_parity, P * SZ, cudaHostAllocDefault);
    cudaMalloc(&d_data, K * SZ);
    cudaMalloc(&d_parity, P * SZ);
    memset(h_data, 0xAB, K * SZ);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    void *d_stripes[4], *d_par[2];
    for (int i = 0; i < K; i++) d_stripes[i] = (uint8_t*)d_data + i * SZ;
    for (int i = 0; i < P; i++) d_par[i] = (uint8_t*)d_parity + i * SZ;

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cudaMemcpyAsync(d_data, h_data, K * SZ, cudaMemcpyHostToDevice, stream);
        gpu_ec_encode_multi_sm(d_stripes, d_par, K, P, SZ, stream, GPU_EC_MODE_NATIVE);
        cudaMemcpyAsync(h_parity, d_parity, P * SZ, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    /* Latency test */
    double t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        cudaMemcpyAsync(d_data, h_data, K * SZ, cudaMemcpyHostToDevice, stream);
        gpu_ec_encode_multi_sm(d_stripes, d_par, K, P, SZ, stream, GPU_EC_MODE_NATIVE);
        cudaMemcpyAsync(h_parity, d_parity, P * SZ, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    double avg_us = (now_us() - t0) / ITERS;

    /* Verify */
    uint8_t *cpu_p = (uint8_t *)calloc(1, SZ);
    for (size_t j = 0; j < SZ; j++) {
        uint8_t v = ((uint8_t*)h_data)[j];
        for (int s = 1; s < K; s++) v ^= ((uint8_t*)h_data)[s * SZ + j];
        cpu_p[j] = v;
    }
    int match = (memcmp(h_parity, cpu_p, SZ) == 0);

    printf("EC 4+2 E2E (H2D+compute+D2H), 1MB cell, %d iters:\n", ITERS);
    printf("  avg latency: %.1f µs\n", avg_us);
    printf("  IOPS:        %.0f\n", 1e6 / avg_us);
    printf("  PCIe BW:     %.3f GB/s (6MB per op)\n",
           (6.0 * 1048576 / 1e9) / (avg_us / 1e6));
    printf("  verify:      %s\n", match ? "MATCH" : "MISMATCH");

    free(cpu_p);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_data); cudaFreeHost(h_parity);
    cudaFree(d_data); cudaFree(d_parity);
    printf("\n=== Done ===\n");
    return 0;
}
