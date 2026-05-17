/**
 * bench_h2d.cu — Micro-benchmark: raw H2D/D2H PCIe transfer latency.
 * Measures cudaMemcpy and cudaMemcpyAsync for various sizes.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

static double now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    printf("=== Raw PCIe H2D / D2H Transfer Benchmark ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    size_t sizes[] = {4096, 65536, 262144, 1048576, 4194304, 16777216};
    const char *labels[] = {"4KB", "64KB", "256KB", "1MB", "4MB", "16MB"};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    /* Test both pageable and pinned memory */
    for (int pinned = 0; pinned <= 1; pinned++) {
        printf("─── %s Host Memory ───\n", pinned ? "PINNED" : "PAGEABLE");
        printf("%-8s  %12s  %12s  %10s  %12s  %12s  %10s\n",
               "Size", "H2D sync(µs)", "H2D async(µs)", "H2D GB/s",
               "D2H sync(µs)", "D2H async(µs)", "D2H GB/s");

        for (int si = 0; si < nsizes; si++) {
            size_t sz = sizes[si];
            void *h_buf, *d_buf;

            if (pinned)
                cudaHostAlloc(&h_buf, sz, cudaHostAllocDefault);
            else
                h_buf = malloc(sz);

            cudaMalloc(&d_buf, sz);
            memset(h_buf, 0xAB, sz);

            cudaStream_t stream;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

            const int WARMUP = 10, ITERS = 100;

            /* Warmup */
            for (int i = 0; i < WARMUP; i++) {
                cudaMemcpy(d_buf, h_buf, sz, cudaMemcpyHostToDevice);
                cudaMemcpy(h_buf, d_buf, sz, cudaMemcpyDeviceToHost);
            }

            /* H2D sync */
            double t0 = now_us();
            for (int i = 0; i < ITERS; i++)
                cudaMemcpy(d_buf, h_buf, sz, cudaMemcpyHostToDevice);
            double h2d_sync = (now_us() - t0) / ITERS;

            /* H2D async */
            t0 = now_us();
            for (int i = 0; i < ITERS; i++) {
                cudaMemcpyAsync(d_buf, h_buf, sz, cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            double h2d_async = (now_us() - t0) / ITERS;

            double h2d_gbps = (sz / (h2d_async * 1e-6)) / (1024.0*1024*1024);

            /* D2H sync */
            t0 = now_us();
            for (int i = 0; i < ITERS; i++)
                cudaMemcpy(h_buf, d_buf, sz, cudaMemcpyDeviceToHost);
            double d2h_sync = (now_us() - t0) / ITERS;

            /* D2H async */
            t0 = now_us();
            for (int i = 0; i < ITERS; i++) {
                cudaMemcpyAsync(h_buf, d_buf, sz, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
            double d2h_async = (now_us() - t0) / ITERS;

            double d2h_gbps = (sz / (d2h_async * 1e-6)) / (1024.0*1024*1024);

            printf("%-8s  %12.1f  %12.1f  %10.3f  %12.1f  %12.1f  %10.3f\n",
                   labels[si], h2d_sync, h2d_async, h2d_gbps,
                   d2h_sync, d2h_async, d2h_gbps);

            cudaStreamDestroy(stream);
            cudaFree(d_buf);
            if (pinned) cudaFreeHost(h_buf); else free(h_buf);
        }
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
