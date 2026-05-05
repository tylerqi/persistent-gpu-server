/**
 * test_compress.cu — Test LZ4 Compression and Decompression on GPU
 */
#include "gpu_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

int main(void)
{
    printf("=== Test LZ4 Compress / Decompress ===\n");

    gpu_engine_t *eng = NULL;
    int rc = gpu_engine_init(&eng);
    if (rc != 0) {
        fprintf(stderr, "Failed to init engine\n");
        return 1;
    }

    const size_t raw_size = 4096;
    const size_t comp_max_size = 8192; // Max possible size for LZ4
    
    uint8_t *h_raw = (uint8_t *)malloc(raw_size);
    uint8_t *h_out = (uint8_t *)malloc(comp_max_size);
    uint8_t *h_decomp = (uint8_t *)malloc(raw_size);

    // Generate highly compressible data (repeating pattern)
    for (size_t i = 0; i < raw_size; i++) {
        h_raw[i] = (uint8_t)(i % 16);
    }

    void *d_raw, *d_comp, *d_decomp;
    cudaMalloc(&d_raw, raw_size);
    cudaMalloc(&d_comp, comp_max_size);
    cudaMalloc(&d_decomp, raw_size);

    cudaMemcpy(d_raw, h_raw, raw_size, cudaMemcpyHostToDevice);

    /* ── Compress ────────────────────────────────────────────── */
    gpu_work_item_t item_comp = {};
    item_comp.op_type = GPU_OP_COMPRESS_LZ4;
    item_comp.data_ptr = d_raw;
    item_comp.data_len = raw_size;
    item_comp.comp_out_ptr = d_comp;
    item_comp.comp_max_size = comp_max_size;

    gpu_result_t result_comp;
    rc = gpu_engine_submit_and_wait(eng, &item_comp, &result_comp);
    if (rc != 0 || result_comp.error_code != 0) {
        fprintf(stderr, "Compression failed! err=%d\n", result_comp.error_code);
        return 1;
    }
    
    size_t compressed_size = result_comp.actual_comp_size;
    printf("Original Size:   %zu bytes\n", raw_size);
    printf("Compressed Size: %zu bytes\n", compressed_size);

    /* ── Decompress ──────────────────────────────────────────── */
    gpu_work_item_t item_decomp = {};
    item_decomp.op_type = GPU_OP_DECOMPRESS_LZ4;
    item_decomp.data_ptr = d_comp;
    item_decomp.data_len = compressed_size;
    item_decomp.comp_out_ptr = d_decomp;
    item_decomp.comp_max_size = raw_size;

    gpu_result_t result_decomp;
    rc = gpu_engine_submit_and_wait(eng, &item_decomp, &result_decomp);
    if (rc != 0 || result_decomp.error_code != 0) {
        fprintf(stderr, "Decompression failed! err=%d\n", result_decomp.error_code);
        return 1;
    }

    size_t decompressed_size = result_decomp.actual_comp_size;
    printf("Decompressed Size: %zu bytes\n", decompressed_size);

    if (decompressed_size != raw_size) {
        fprintf(stderr, "Size mismatch after decompression!\n");
        return 1;
    }

    cudaMemcpy(h_decomp, d_decomp, raw_size, cudaMemcpyDeviceToHost);

    if (memcmp(h_raw, h_decomp, raw_size) != 0) {
        fprintf(stderr, "Data corruption! Decompressed data does not match original.\n");
        return 1;
    }

    printf("SUCCESS! Compression and Decompression verified.\n");

    /* Must shut down engine BEFORE cudaFree (avoids deadlock with persistent kernel) */
    gpu_engine_fini(eng);

    cudaFree(d_raw);
    cudaFree(d_comp);
    cudaFree(d_decomp);
    free(h_raw);
    free(h_out);
    free(h_decomp);

    return 0;
}
