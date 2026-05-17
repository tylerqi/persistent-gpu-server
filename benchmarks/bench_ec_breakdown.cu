/**
 * bench_ec_breakdown.cu — Measure each phase of EC E2E separately.
 * Isolates: H2D, dispatch overhead, EC compute, D2H.
 */
#include "gpu_engine.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    printf("=== EC E2E Phase Breakdown ===\n\n");

    const size_t SZ = 1048576; /* 1MB */
    const int K = 4, P = 2;
    const int ITERS = 200;

    /* Allocate contiguous pinned host + device buffers */
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

    /* Start engine */
    gpu_engine_t *eng = NULL;
    gpu_engine_init(&eng);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        cudaMemcpyAsync(d_data, h_data, K * SZ, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_EC_ENCODE;
        for (int s = 0; s < K; s++) item.ec_ptrs[s] = d_stripes[s];
        for (int s = 0; s < P; s++) item.parity_ptrs[s] = d_par[s];
        item.stripe_cnt = K; item.parity_cnt = P; item.cell_size = SZ;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
        cudaMemcpyAsync(h_parity, d_parity, P * SZ, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    /* Phase 1: H2D only */
    double t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        cudaMemcpyAsync(d_data, h_data, K * SZ, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }
    double h2d_us = (now_us() - t0) / ITERS;

    /* Phase 2: Dispatch + EC compute only (data already on GPU) */
    t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_EC_ENCODE;
        for (int s = 0; s < K; s++) item.ec_ptrs[s] = d_stripes[s];
        for (int s = 0; s < P; s++) item.parity_ptrs[s] = d_par[s];
        item.stripe_cnt = K; item.parity_cnt = P; item.cell_size = SZ;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
    }
    double dispatch_ec_us = (now_us() - t0) / ITERS;

    /* Phase 3: NOP dispatch only (measure pure dispatch overhead) */
    t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_NOP;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
    }
    double nop_us = (now_us() - t0) / ITERS;

    /* Phase 4: D2H only */
    t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        cudaMemcpyAsync(h_parity, d_parity, P * SZ, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    double d2h_us = (now_us() - t0) / ITERS;

    /* Phase 5: Full E2E */
    t0 = now_us();
    for (int i = 0; i < ITERS; i++) {
        cudaMemcpyAsync(d_data, h_data, K * SZ, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_EC_ENCODE;
        for (int s = 0; s < K; s++) item.ec_ptrs[s] = d_stripes[s];
        for (int s = 0; s < P; s++) item.parity_ptrs[s] = d_par[s];
        item.stripe_cnt = K; item.parity_cnt = P; item.cell_size = SZ;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
        cudaMemcpyAsync(h_parity, d_parity, P * SZ, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    double e2e_us = (now_us() - t0) / ITERS;

    double ec_compute_us = dispatch_ec_us - nop_us;
    double sum = h2d_us + nop_us + ec_compute_us + d2h_us;
    double unaccounted = e2e_us - sum;

    printf("Phase Breakdown (avg of %d iterations, 4+2 EC, 1MB cell):\n\n", ITERS);
    printf("  %-30s %10.1f µs  (%5.1f%%)\n", "H2D (4MB pinned async):", h2d_us, h2d_us/e2e_us*100);
    printf("  %-30s %10.1f µs  (%5.1f%%)\n", "Dispatch overhead (NOP):", nop_us, nop_us/e2e_us*100);
    printf("  %-30s %10.1f µs  (%5.1f%%)\n", "EC compute (XOR):", ec_compute_us, ec_compute_us/e2e_us*100);
    printf("  %-30s %10.1f µs  (%5.1f%%)\n", "D2H (2MB pinned async):", d2h_us, d2h_us/e2e_us*100);
    printf("  %-30s %10.1f µs  (%5.1f%%)\n", "Unaccounted (sync/overhead):", unaccounted, unaccounted/e2e_us*100);
    printf("  %s\n", "──────────────────────────────────────────");
    printf("  %-30s %10.1f µs  (100%%)\n", "Total E2E:", e2e_us);

    double pcie_total_mb = (double)(K + P) * SZ / (1024.0 * 1024.0);
    printf("\n  PCIe payload per op: %.1f MB\n", pcie_total_mb);
    printf("  Effective PCIe BW:   %.3f GB/s (of 7.88 GB/s theoretical)\n",
           (pcie_total_mb / 1024.0) / (e2e_us / 1e6));
    printf("  H2D BW:              %.3f GB/s\n", ((double)K*SZ/1e9) / (h2d_us/1e6));
    printf("  D2H BW:              %.3f GB/s\n", ((double)P*SZ/1e9) / (d2h_us/1e6));

    gpu_engine_fini(eng);
    cudaStreamDestroy(stream);
    cudaFreeHost(h_data); cudaFreeHost(h_parity);
    cudaFree(d_data); cudaFree(d_parity);

    printf("\n=== Done ===\n");
    return 0;
}
