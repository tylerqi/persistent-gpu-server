/**
 * bench_e2e.cu — End-to-End (Full Path) Benchmark
 *
 * Measures the REAL end-to-end latency and throughput including:
 *   1. Host → Device memory copy (PCIe)
 *   2. GPU computation (via persistent kernel dispatch)
 *   3. Device → Host memory copy (PCIe)
 *
 * This is the benchmark that reflects actual DAOS integration performance,
 * where data originates from host memory and results must be returned.
 *
 * Contrast with bench_dispatch.cu which only measures GPU-internal VRAM ops.
 */
#include "gpu_engine.h"
#include "gpu_csum.h"
#include "gpu_ec.h"
#include <cuda_runtime.h>
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

static void print_stats(double *lat, int n, const char *label)
{
    qsort(lat, n, sizeof(double), cmp_double);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += lat[i];

    printf("  %s (%d samples):\n", label, n);
    printf("    min   = %10.2f µs\n", lat[0]);
    printf("    p50   = %10.2f µs\n", lat[n / 2]);
    printf("    p90   = %10.2f µs\n", lat[(int)(n * 0.9)]);
    printf("    p99   = %10.2f µs\n", lat[(int)(n * 0.99)]);
    printf("    max   = %10.2f µs\n", lat[n - 1]);
    printf("    avg   = %10.2f µs\n", sum / n);
    printf("    IOPS  = %10.0f ops/sec\n", n / (sum / 1e6));
}

/* ── Pinned memory helper ────────────────────────────────────────────────── */
static void *pinned_alloc(size_t sz) {
    void *p = NULL;
    cudaHostAlloc(&p, sz, cudaHostAllocDefault);
    return p;
}

/* ── Per-workload context ────────────────────────────────────────────────── */
struct e2e_ctx {
    /* GPU-side pre-allocated buffers (allocated before engine start) */
    void *d_data;        /* input data on GPU */
    void *d_out;         /* output data on GPU (for compress/EC) */
    /* Host-side pinned memory */
    void *h_data;        /* input data on host */
    void *h_result;      /* result buffer on host */
    size_t data_len;
    size_t out_len;
};

/* ── CRC32C full path ─────────────────────────────────────────────────────
 * H2D: copy data → GPU
 * Compute: CRC32C via persistent kernel
 * D2H: read 4-byte CRC result from gpu_result_t (already on host, zero-copy)
 */
static void bench_crc32c_e2e(gpu_engine_t *eng, struct e2e_ctx *ctx,
                              int iterations, const char *label)
{
    double *lat = (double *)malloc(sizeof(double) * iterations);

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        cudaMemcpy(ctx->d_data, ctx->h_data, ctx->data_len, cudaMemcpyHostToDevice);
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_CRC32C;
        item.data_ptr = ctx->d_data;
        item.data_len = ctx->data_len;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
    }

    for (int i = 0; i < iterations; i++) {
        double t0 = now_us();

        /* 1. H2D: copy data to GPU */
        cudaMemcpy(ctx->d_data, ctx->h_data, ctx->data_len, cudaMemcpyHostToDevice);

        /* 2. Compute: dispatch CRC32C */
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_CRC32C;
        item.data_ptr = ctx->d_data;
        item.data_len = ctx->data_len;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);

        /* 3. D2H: CRC result is in res.crc32c_result (already on host) */
        *(uint32_t *)ctx->h_result = res.crc32c_result;

        lat[i] = now_us() - t0;
    }

    print_stats(lat, iterations, label);

    /* Verify against CPU */
    uint32_t cpu_crc = cpu_crc32c((const char *)ctx->h_data, ctx->data_len);
    uint32_t gpu_crc = *(uint32_t *)ctx->h_result;
    printf("    verify: %s (GPU=0x%08X CPU=0x%08X)\n",
           gpu_crc == cpu_crc ? "MATCH" : "MISMATCH", gpu_crc, cpu_crc);

    free(lat);
}

/* ── SHA256 full path ─────────────────────────────────────────────────────
 * H2D: copy data → GPU
 * Compute: SHA256 via persistent kernel
 * D2H: read 32-byte hash from gpu_result_t (already on host)
 */
static void bench_sha256_e2e(gpu_engine_t *eng, struct e2e_ctx *ctx,
                              int iterations, const char *label)
{
    double *lat = (double *)malloc(sizeof(double) * iterations);

    for (int i = 0; i < 10; i++) {
        cudaMemcpy(ctx->d_data, ctx->h_data, ctx->data_len, cudaMemcpyHostToDevice);
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_SHA256;
        item.data_ptr = ctx->d_data;
        item.data_len = ctx->data_len;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
    }

    for (int i = 0; i < iterations; i++) {
        double t0 = now_us();
        cudaMemcpy(ctx->d_data, ctx->h_data, ctx->data_len, cudaMemcpyHostToDevice);

        gpu_work_item_t item = {};
        item.op_type = GPU_OP_SHA256;
        item.data_ptr = ctx->d_data;
        item.data_len = ctx->data_len;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);

        memcpy(ctx->h_result, res.sha256_result, 32);
        lat[i] = now_us() - t0;
    }

    print_stats(lat, iterations, label);
    free(lat);
}

/* ── Compress full path ───────────────────────────────────────────────────
 * H2D: copy uncompressed data → GPU
 * Compute: LZ4 compress via persistent kernel
 * D2H: copy compressed output back to host
 */
static void bench_compress_e2e(gpu_engine_t *eng, struct e2e_ctx *ctx,
                                int iterations, const char *label)
{
    double *lat = (double *)malloc(sizeof(double) * iterations);
    size_t last_comp_size = 0;

    for (int i = 0; i < 10; i++) {
        cudaMemcpy(ctx->d_data, ctx->h_data, ctx->data_len, cudaMemcpyHostToDevice);
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_COMPRESS_LZ4;
        item.data_ptr = ctx->d_data;
        item.data_len = ctx->data_len;
        item.comp_out_ptr = ctx->d_out;
        item.comp_max_size = ctx->out_len;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
    }

    for (int i = 0; i < iterations; i++) {
        double t0 = now_us();

        /* 1. H2D */
        cudaMemcpy(ctx->d_data, ctx->h_data, ctx->data_len, cudaMemcpyHostToDevice);

        /* 2. Compress */
        gpu_work_item_t item = {};
        item.op_type = GPU_OP_COMPRESS_LZ4;
        item.data_ptr = ctx->d_data;
        item.data_len = ctx->data_len;
        item.comp_out_ptr = ctx->d_out;
        item.comp_max_size = ctx->out_len;
        gpu_result_t res;
        gpu_engine_submit_and_wait(eng, &item, &res);
        last_comp_size = res.actual_comp_size;

        /* 3. D2H: copy compressed output */
        cudaMemcpy(ctx->h_result, ctx->d_out, last_comp_size, cudaMemcpyDeviceToHost);

        lat[i] = now_us() - t0;
    }

    double ratio = (last_comp_size > 0) ? (double)ctx->data_len / last_comp_size : 0;
    print_stats(lat, iterations, label);
    printf("    ratio : %.2fx (in=%zu out=%zu)\n", ratio, ctx->data_len, last_comp_size);

    free(lat);
}

/* ── EC Encode full path (DIRECT MULTI-SM) ───────────────────────────────
 * Bypasses persistent kernel — launches EC directly across all SMs.
 *   H2D: single contiguous async copy
 *   Compute: multi-SM EC kernel (all 34 SMs)
 *   D2H: single contiguous async copy
 */
static void bench_ec_encode_e2e(gpu_engine_t *eng,
                                 void *d_data_contig, void *d_parity_contig,
                                 void *h_data_contig, void *h_parity_contig,
                                 size_t cell_size, int iterations,
                                 const char *label)
{
    const int k = 4, p = 2;
    double *lat = (double *)malloc(sizeof(double) * iterations);

    /* Set up per-stripe pointers into contiguous GPU memory */
    void *d_stripes[4], *d_parity[2];
    for (int s = 0; s < k; s++) d_stripes[s] = (uint8_t *)d_data_contig + s * cell_size;
    for (int s = 0; s < p; s++) d_parity[s] = (uint8_t *)d_parity_contig + s * cell_size;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cudaMemcpyAsync(d_data_contig, h_data_contig, k * cell_size, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        gpu_ec_encode_multi_sm(d_stripes, d_parity, k, p, cell_size, stream);
    }

    for (int i = 0; i < iterations; i++) {
        double t0 = now_us();

        /* 1. H2D: single contiguous async copy */
        cudaMemcpyAsync(d_data_contig, h_data_contig, k * cell_size,
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        /* 2. Compute: direct multi-SM EC encode (all SMs) */
        gpu_ec_encode_multi_sm(d_stripes, d_parity, k, p, cell_size, stream);

        /* 3. D2H: single contiguous async copy */
        cudaMemcpyAsync(h_parity_contig, d_parity_contig, p * cell_size,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        lat[i] = now_us() - t0;
    }

    print_stats(lat, iterations, label);

    /* Verify P parity against CPU */
    uint8_t *cpu_p = (uint8_t *)calloc(1, cell_size);
    for (size_t j = 0; j < cell_size; j++) {
        uint8_t val = ((uint8_t *)h_data_contig)[j];  /* stripe 0 */
        for (int s = 1; s < k; s++) val ^= ((uint8_t *)h_data_contig)[s * cell_size + j];
        cpu_p[j] = val;
    }
    const char *match = (memcmp(h_parity_contig, cpu_p, cell_size) == 0) ? "MATCH" : "MISMATCH";
    printf("    verify: %s (P parity vs CPU)\n", match);
    free(cpu_p);

    cudaStreamDestroy(stream);
    free(lat);
}

/* ── Multi-threaded throughput helper ─────────────────────────────────────
 * Runs N threads each doing full-path operations for a fixed duration.
 * Uses per-thread CUDA streams to avoid default-stream serialization.
 */
static void bench_e2e_throughput(gpu_engine_t *eng, gpu_op_type_t op,
                                  void **d_data_arr, void **d_out_arr,
                                  void **h_data_arr, void **h_out_arr,
                                  size_t data_len, size_t out_len,
                                  int num_threads, int duration_sec,
                                  int warmup_sec, const char *label,
                                  cudaStream_t *streams)
{
    printf("\n  %s Throughput (%d threads, %ds measured):\n", label, num_threads, duration_sec - warmup_sec);

    std::atomic<int> total_ops(0);
    double t_start = now_us();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            cudaStream_t s = streams[t];
            int ops = 0;
            while (true) {
                double now = now_us();
                int sec = (int)((now - t_start) / 1e6);
                if (sec >= duration_sec) break;

                /* 1. H2D: async on per-thread stream */
                cudaMemcpyAsync(d_data_arr[t], h_data_arr[t], data_len,
                                cudaMemcpyHostToDevice, s);
                cudaStreamSynchronize(s);

                /* 2. Compute: dispatch via persistent kernel */
                gpu_work_item_t item = {};
                item.op_type = op;
                item.data_ptr = d_data_arr[t];
                item.data_len = data_len;
                if (op == GPU_OP_COMPRESS_LZ4 || op == GPU_OP_DECOMPRESS_LZ4) {
                    item.comp_out_ptr = d_out_arr[t];
                    item.comp_max_size = out_len;
                }
                gpu_result_t res;
                gpu_engine_submit_and_wait(eng, &item, &res);

                /* 3. D2H: async on per-thread stream */
                if (op == GPU_OP_CRC32C) {
                    *(uint32_t *)h_out_arr[t] = res.crc32c_result;
                } else if (op == GPU_OP_SHA256) {
                    memcpy(h_out_arr[t], res.sha256_result, 32);
                } else if (op == GPU_OP_COMPRESS_LZ4) {
                    cudaMemcpyAsync(h_out_arr[t], d_out_arr[t],
                                    res.actual_comp_size,
                                    cudaMemcpyDeviceToHost, s);
                    cudaStreamSynchronize(s);
                }

                if (sec >= warmup_sec) ops++;
            }
            total_ops += ops;
        });
    }
    for (auto &t : threads) t.join();

    double measured = (double)(duration_sec - warmup_sec);
    double iops = total_ops / measured;
    double bw_in = (iops * data_len) / (1024.0 * 1024.0 * 1024.0);
    printf("    IOPS:          %.0f ops/sec\n", iops);
    printf("    H2D+Compute:   %.3f GB/s (input data rate)\n", bw_in);
    printf("    PCIe payload:  %.3f GB/s (H2D transfers)\n", bw_in);
}

/* ── Multi-threaded EC encode throughput (OPTIMIZED) ─────────────────────
 * Uses contiguous buffers: single H2D for all k stripes, single D2H for
 * all p parity stripes.  cudaMemcpyAsync on per-thread streams.
 */
struct ec_thread_bufs {
    void *d_data_contig;    /* GPU: k stripes contiguous */
    void *d_parity_contig;  /* GPU: p parity contiguous */
    void *h_data_contig;    /* Host: k stripes contiguous (pinned) */
    void *h_parity_contig;  /* Host: p parity contiguous (pinned) */
};

static void bench_ec_e2e_throughput(gpu_engine_t *eng,
                                    struct ec_thread_bufs *bufs,
                                    size_t cell_size,
                                    int num_threads, int duration_sec,
                                    int warmup_sec, const char *label,
                                    cudaStream_t *streams)
{
    const int k = 4, p = 2;
    int measured_sec = duration_sec - warmup_sec;
    printf("\n  %s Throughput (%d threads, %ds measured):\n", label, num_threads, measured_sec);

    std::atomic<int> total_ops(0);
    double t_start = now_us();

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            int ops = 0;
            struct ec_thread_bufs *b = &bufs[t];
            cudaStream_t s = streams[t];
            /* Per-stripe GPU pointers into contiguous memory */
            void *d_stripes[4], *d_par[2];
            for (int i = 0; i < k; i++) d_stripes[i] = (uint8_t *)b->d_data_contig + i * cell_size;
            for (int i = 0; i < p; i++) d_par[i] = (uint8_t *)b->d_parity_contig + i * cell_size;

            while (true) {
                double now = now_us();
                int sec = (int)((now - t_start) / 1e6);
                if (sec >= duration_sec) break;

                /* 1. H2D: single contiguous async copy (k * cell_size) */
                cudaMemcpyAsync(b->d_data_contig, b->h_data_contig,
                                k * cell_size, cudaMemcpyHostToDevice, s);
                cudaStreamSynchronize(s);

                /* 2. Compute: direct multi-SM EC encode */
                gpu_ec_encode_multi_sm(d_stripes, d_par, k, p, cell_size, s);

                /* 3. D2H: single contiguous async copy (p * cell_size) */
                cudaMemcpyAsync(b->h_parity_contig, b->d_parity_contig,
                                p * cell_size, cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);

                if (sec >= warmup_sec) ops++;
            }
            total_ops += ops;
        });
    }
    for (auto &t : threads) t.join();

    double measured = (double)measured_sec;
    double iops = total_ops / measured;
    double pcie_per_op = (double)(k + p) * cell_size;
    double bw_in = (iops * k * cell_size) / (1024.0 * 1024.0 * 1024.0);
    double bw_pcie = (iops * pcie_per_op) / (1024.0 * 1024.0 * 1024.0);
    printf("    IOPS:          %.0f ops/sec\n", iops);
    printf("    H2D+Compute:   %.3f GB/s (input data rate, %d×%zuB H2D)\n", bw_in, k, cell_size);
    printf("    PCIe payload:  %.3f GB/s (H2D + D2H total)\n", bw_pcie);
}

int main(void)
{
    printf("=== End-to-End Full Path Benchmark ===\n");
    printf("  Measures: Host→GPU copy + GPU compute + GPU→Host copy\n");
    printf("  This reflects real-world DAOS integration performance.\n\n");

    /* Query PCIe info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %zu MB)\n", prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));

    size_t gpu_free, gpu_total;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    printf("GPU Memory: %zu MB free / %zu MB total\n\n",
           gpu_free / (1024 * 1024), gpu_total / (1024 * 1024));

#ifndef USE_NVCOMPDX
    printf("NOTE: LZ4 in memcpy stub mode (no actual compression).\n\n");
#endif

    /* ── Pre-allocate everything ─────────────────────────────────────── */
    const size_t SZ_4K  = 4096;
    const size_t SZ_64K = 65536;
    const size_t SZ_1M  = 1048576;
    const int NT = 16; /* threads for throughput test */

    /* Single-thread latency contexts */
    struct e2e_ctx ctx_crc_4k, ctx_crc_1m, ctx_sha_4k, ctx_sha_64k;
    struct e2e_ctx ctx_comp_4k, ctx_comp_1m;

    auto alloc_ctx = [](struct e2e_ctx *c, size_t dlen, size_t olen) {
        c->data_len = dlen;
        c->out_len = olen;
        c->h_data = pinned_alloc(dlen);
        c->h_result = pinned_alloc(olen > 0 ? olen : 64);
        cudaMalloc(&c->d_data, dlen);
        if (olen > 0) cudaMalloc(&c->d_out, olen);
        else c->d_out = NULL;
        /* Fill with test data */
        for (size_t i = 0; i < dlen; i++)
            ((uint8_t *)c->h_data)[i] = (uint8_t)((i * 37 + 13) & 0xFF);
    };

    alloc_ctx(&ctx_crc_4k,  SZ_4K,  0);
    alloc_ctx(&ctx_crc_1m,  SZ_1M,  0);
    alloc_ctx(&ctx_sha_4k,  SZ_4K,  0);
    alloc_ctx(&ctx_sha_64k, SZ_64K, 0);
    alloc_ctx(&ctx_comp_4k, SZ_4K,  SZ_4K * 2);
    alloc_ctx(&ctx_comp_1m, SZ_1M,  SZ_1M * 2);

    /* EC: contiguous buffers — 4 data stripes + 2 parity, 1MB each */
    void *d_ec_data_contig, *d_ec_parity_contig;
    void *h_ec_data_contig, *h_ec_parity_contig;
    cudaMalloc(&d_ec_data_contig, 4 * SZ_1M);
    cudaMalloc(&d_ec_parity_contig, 2 * SZ_1M);
    h_ec_data_contig = pinned_alloc(4 * SZ_1M);
    h_ec_parity_contig = pinned_alloc(2 * SZ_1M);
    for (int i = 0; i < 4; i++)
        for (size_t j = 0; j < SZ_1M; j++)
            ((uint8_t *)h_ec_data_contig)[i * SZ_1M + j] = (uint8_t)((i * 7 + j * 11 + 3) & 0xFF);

    /* Multi-thread throughput buffers */
    void *d_mt_data[NT], *d_mt_out[NT];
    void *h_mt_data[NT], *h_mt_out[NT];
    for (int i = 0; i < NT; i++) {
        cudaMalloc(&d_mt_data[i], SZ_4K);
        cudaMalloc(&d_mt_out[i], SZ_4K * 2);
        h_mt_data[i] = pinned_alloc(SZ_4K);
        h_mt_out[i] = pinned_alloc(SZ_4K * 2);
        for (size_t j = 0; j < SZ_4K; j++)
            ((uint8_t *)h_mt_data[i])[j] = (uint8_t)((i + j) & 0xFF);
    }
    /* 1MB multi-thread buffers */
    void *d_mt_1m[NT], *d_mt_1m_out[NT];
    void *h_mt_1m[NT], *h_mt_1m_out[NT];
    for (int i = 0; i < NT; i++) {
        cudaMalloc(&d_mt_1m[i], SZ_1M);
        cudaMalloc(&d_mt_1m_out[i], SZ_1M * 2);
        h_mt_1m[i] = pinned_alloc(SZ_1M);
        h_mt_1m_out[i] = pinned_alloc(SZ_1M * 2);
        for (size_t j = 0; j < SZ_1M; j++)
            ((uint8_t *)h_mt_1m[i])[j] = (uint8_t)((i + j * 3) & 0xFF);
    }
    /* Multi-thread EC: contiguous buffers per thread */
    struct ec_thread_bufs mt_ec[NT];
    for (int t = 0; t < NT; t++) {
        cudaMalloc(&mt_ec[t].d_data_contig, 4 * SZ_1M);
        cudaMalloc(&mt_ec[t].d_parity_contig, 2 * SZ_1M);
        mt_ec[t].h_data_contig = pinned_alloc(4 * SZ_1M);
        mt_ec[t].h_parity_contig = pinned_alloc(2 * SZ_1M);
        for (int s = 0; s < 4; s++)
            for (size_t j = 0; j < SZ_1M; j++)
                ((uint8_t *)mt_ec[t].h_data_contig)[s * SZ_1M + j] = (uint8_t)((t * 7 + s * 11 + j * 3) & 0xFF);
    }

    printf("GPU Memory after alloc: ");
    cudaMemGetInfo(&gpu_free, &gpu_total);
    printf("%zu MB free / %zu MB total\n\n", gpu_free / (1024 * 1024), gpu_total / (1024 * 1024));

    /* Per-thread CUDA streams for concurrent PCIe DMA */
    cudaStream_t mt_streams[NT];
    for (int i = 0; i < NT; i++)
        cudaStreamCreateWithFlags(&mt_streams[i], cudaStreamNonBlocking);

    /* ── Start engine ────────────────────────────────────────────────── */
    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Engine init failed\n");
        return 1;
    }

    /* ═══════════════════════════════════════════════════════════════════
     * Section 1: Single-Thread Latency (full path)
     * ═══════════════════════════════════════════════════════════════════ */
    printf("─── Single-Thread Latency (H2D + Compute + D2H) ───\n\n");

    bench_crc32c_e2e(eng, &ctx_crc_4k,  2000, "4KB CRC32C E2E");
    printf("\n");
    bench_crc32c_e2e(eng, &ctx_crc_1m,   500, "1MB CRC32C E2E");
    printf("\n");
    bench_sha256_e2e(eng, &ctx_sha_4k,  2000, "4KB SHA256 E2E");
    printf("\n");
    bench_sha256_e2e(eng, &ctx_sha_64k, 1000, "64KB SHA256 E2E");
    printf("\n");
    bench_compress_e2e(eng, &ctx_comp_4k, 2000, "4KB LZ4 Compress E2E");
    printf("\n");
    bench_compress_e2e(eng, &ctx_comp_1m,  500, "1MB LZ4 Compress E2E");
    printf("\n");

    /* ═══════════════════════════════════════════════════════════════════
     * Section 2: Multi-Thread Throughput (full path)
     * ═══════════════════════════════════════════════════════════════════ */
    printf("\n─── Multi-Thread Throughput (H2D + Compute + D2H) ───\n");

    bench_e2e_throughput(eng, GPU_OP_CRC32C,
                         d_mt_data, NULL, h_mt_data, h_mt_out,
                         SZ_4K, 0, NT, 10, 3,
                         "4KB CRC32C E2E", mt_streams);

    bench_e2e_throughput(eng, GPU_OP_CRC32C,
                         d_mt_1m, NULL, h_mt_1m, h_mt_1m_out,
                         SZ_1M, 0, NT, 10, 3,
                         "1MB CRC32C E2E", mt_streams);

    bench_e2e_throughput(eng, GPU_OP_COMPRESS_LZ4,
                         d_mt_data, d_mt_out, h_mt_data, h_mt_out,
                         SZ_4K, SZ_4K * 2, NT, 10, 3,
                         "4KB LZ4 Compress E2E", mt_streams);

    bench_e2e_throughput(eng, GPU_OP_COMPRESS_LZ4,
                         d_mt_1m, d_mt_1m_out, h_mt_1m, h_mt_1m_out,
                         SZ_1M, SZ_1M * 2, NT, 10, 3,
                         "1MB LZ4 Compress E2E", mt_streams);

    /* ── Cleanup Engine (Stop persistent kernel before direct multi-SM EC benchmarks) ── */
    gpu_engine_fini(eng);

    /* ═══════════════════════════════════════════════════════════════════
     * Section 3: Standalone Multi-SM EC Benchmarks (no engine needed)
     * ═══════════════════════════════════════════════════════════════════ */
    printf("\n─── Standalone Multi-SM EC Benchmarks (Direct Launch) ───\n\n");

    bench_ec_encode_e2e(NULL, d_ec_data_contig, d_ec_parity_contig,
                        h_ec_data_contig, h_ec_parity_contig, SZ_1M, 200,
                        "1MB EC 4+2 Encode E2E");

    bench_ec_e2e_throughput(NULL, mt_ec, SZ_1M, NT, 10, 3,
                            "1MB EC 4+2 Encode E2E", mt_streams);

    printf("\n");

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    for (int i = 0; i < NT; i++)
        cudaStreamDestroy(mt_streams[i]);

    auto free_ctx = [](struct e2e_ctx *c) {
        cudaFreeHost(c->h_data);
        cudaFreeHost(c->h_result);
        cudaFree(c->d_data);
        if (c->d_out) cudaFree(c->d_out);
    };
    free_ctx(&ctx_crc_4k);  free_ctx(&ctx_crc_1m);
    free_ctx(&ctx_sha_4k);  free_ctx(&ctx_sha_64k);
    free_ctx(&ctx_comp_4k); free_ctx(&ctx_comp_1m);

    cudaFree(d_ec_data_contig); cudaFreeHost(h_ec_data_contig);
    cudaFree(d_ec_parity_contig); cudaFreeHost(h_ec_parity_contig);
    for (int i = 0; i < NT; i++) {
        cudaFree(d_mt_data[i]); cudaFree(d_mt_out[i]);
        cudaFreeHost(h_mt_data[i]); cudaFreeHost(h_mt_out[i]);
        cudaFree(d_mt_1m[i]); cudaFree(d_mt_1m_out[i]);
        cudaFreeHost(h_mt_1m[i]); cudaFreeHost(h_mt_1m_out[i]);
        cudaFree(mt_ec[i].d_data_contig); cudaFreeHost(mt_ec[i].h_data_contig);
        cudaFree(mt_ec[i].d_parity_contig); cudaFreeHost(mt_ec[i].h_parity_contig);
    }

    printf("=== Done ===\n");
    return 0;
}
