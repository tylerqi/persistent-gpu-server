/**
 * simple_server.cu — Minimal example of the persistent GPU engine
 *
 * Demonstrates: init → allocate GPU buffer → submit CRC32C → read result → fini
 */
#include "gpu_engine.h"
#include "gpu_csum.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    printf("=== Persistent GPU Engine: Simple Example ===\n\n");

    /* 1. Prepare test data */
    const char *test_data = "Hello, GPU-accelerated DAOS!";
    size_t data_len = strlen(test_data);

    void *d_data;
    cudaMalloc(&d_data, data_len);
    cudaMemcpy(d_data, test_data, data_len, cudaMemcpyHostToDevice);
    printf("[1] Data uploaded to GPU memory: \"%s\" (%zu bytes)\n", test_data, data_len);

    /* 2. Initialize the persistent GPU engine */
    gpu_engine_t *engine = NULL;
    int rc = gpu_engine_init(&engine);
    if (rc != 0) {
        fprintf(stderr, "gpu_engine_init failed: %d\n", rc);
        return rc;
    }
    printf("[2] GPU engine initialized successfully.\n");

    /* 3. Compute CRC32C via the persistent kernel */
    gpu_work_item_t item;
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_CRC32C;
    item.data_ptr = d_data;
    item.data_len = data_len;

    gpu_result_t result;
    rc = gpu_engine_submit_and_wait(engine, &item, &result);
    if (rc != 0) {
        fprintf(stderr, "CRC32C computation failed: %d\n", rc);
    } else {
        /* CRC result is now in gpu_result_t (unified API) */
        printf("[3] GPU CRC32C result: 0x%08X (error_code=%d)\n",
               result.crc32c_result, result.error_code);
    }

    /* 4. Verify against CPU reference */
    uint32_t cpu_crc = cpu_crc32c(test_data, data_len);
    printf("[4] CPU CRC32C reference: 0x%08X\n", cpu_crc);

    /* 5. Compute SHA256 via the engine */
    memset(&item, 0, sizeof(item));
    item.op_type = GPU_OP_SHA256;
    item.data_ptr = d_data;
    item.data_len = data_len;

    rc = gpu_engine_submit_and_wait(engine, &item, &result);
    if (rc != 0) {
        fprintf(stderr, "SHA256 computation failed: %d\n", rc);
    } else {
        printf("[5] SHA256 done (error_code=%d)\n", result.error_code);
    }

    /* 4. Shutdown */
    gpu_engine_fini(engine);
    cudaFree(d_data);
    printf("[4] Engine shutdown complete. Exiting.\n");

    return 0;
}
