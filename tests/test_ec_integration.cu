/**
 * test_ec_integration.cu — EC File-Level Integration Test Suite
 *
 * Verifies end-to-end data integrity by:
 *   1. Generating N data files on disk with random data.
 *   2. Computing P and Q parity via CPU reference EC.
 *   3. Deleting M files (M <= P) from disk.
 *   4. Reconstructing the deleted files via GPU EC persistent engine.
 *   5. Verifying reconstructed files against original content.
 */
#include "gpu_engine.h"
#include "gpu_ec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <string>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Helper: generate deterministic pseudorandom data for a stripe */
static void generate_stripe_data(uint8_t *buf, size_t len, int stripe_idx, size_t size_param)
{
    uint32_t seed = stripe_idx * 12345 + size_param + 7;
    uint32_t state = seed;
    for (size_t i = 0; i < len; i++) {
        state = state * 1103515245 + 12345;
        buf[i] = (uint8_t)((state >> 16) & 0xFF);
    }
}

/* Helper: write buffer to file */
static bool write_file(const std::string &path, const uint8_t *buf, size_t len)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    size_t written = fwrite(buf, 1, len, f);
    fclose(f);
    return written == len;
}

/* Helper: read file to buffer */
static bool read_file(const std::string &path, uint8_t *buf, size_t len)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    size_t read_bytes = fread(buf, 1, len, f);
    fclose(f);
    return read_bytes == len;
}

/* Helper: CPU reference EC encode (P & Q parity) */
static void cpu_ec_encode(const uint8_t **data_ptrs, uint8_t **parity_ptrs, int stripe_cnt, int parity_cnt, size_t stripe_len)
{
    // 1. Compute P parity (simple XOR)
    memset(parity_ptrs[0], 0, stripe_len);
    for (int s = 0; s < stripe_cnt; s++) {
        for (size_t i = 0; i < stripe_len; i++) {
            parity_ptrs[0][i] ^= data_ptrs[s][i];
        }
    }

    // 2. Compute Q parity using Horner's method with GF(2^8)
    if (parity_cnt > 1) {
        for (size_t i = 0; i < stripe_len; i++) {
            uint8_t q = data_ptrs[stripe_cnt - 1][i];
            for (int s = stripe_cnt - 2; s >= 0; s--) {
                q = (uint8_t)(((q << 1) & 0xFE) ^ ((q >> 7) * 0x1B));
                q ^= data_ptrs[s][i];
            }
            parity_ptrs[1][i] = q;
        }
    }
}

/* GPU pre-allocated memory buffers */
struct test_buffers {
    void *d_data[4];
    void *d_parity[2];
    size_t max_size;
};

static int alloc_test_buffers(struct test_buffers *buf, size_t max_size)
{
    buf->max_size = max_size;
    for (int i = 0; i < 4; i++) {
        if (cudaMalloc(&buf->d_data[i], max_size) != cudaSuccess) return -1;
    }
    for (int i = 0; i < 2; i++) {
        if (cudaMalloc(&buf->d_parity[i], max_size) != cudaSuccess) return -1;
    }
    return 0;
}

static void free_test_buffers(struct test_buffers *buf)
{
    for (int i = 0; i < 4; i++) cudaFree(buf->d_data[i]);
    for (int i = 0; i < 2; i++) cudaFree(buf->d_parity[i]);
}

static void cleanup_temp_files()
{
    for (int i = 0; i < 4; i++) {
        std::string path = "test_ec_integration_temp/data_" + std::to_string(i) + ".bin";
        unlink(path.c_str());
    }
    for (int i = 0; i < 2; i++) {
        std::string path = "test_ec_integration_temp/parity_" + std::to_string(i) + ".bin";
        unlink(path.c_str());
    }
    rmdir("test_ec_integration_temp");
}

static int run_integration_case(gpu_engine_t *eng, struct test_buffers *buf, size_t S, const std::vector<int> &deleted_indices)
{
    const int k = 4;
    const int p = 2;

    printf("  Running case: size=%zu, deleting %zu stripe(s) {", S, deleted_indices.size());
    for (size_t i = 0; i < deleted_indices.size(); i++) {
        printf("%s%d", (i > 0 ? ", " : ""), deleted_indices[i]);
    }
    printf("} ... ");
    fflush(stdout);

    // Create temp directory
    mkdir("test_ec_integration_temp", 0777);

    // 1. Generate N=4 data files on disk with random data
    std::vector<uint8_t*> h_data(k);
    for (int i = 0; i < k; i++) {
        h_data[i] = (uint8_t*)malloc(S);
        generate_stripe_data(h_data[i], S, i, S);
        std::string path = "test_ec_integration_temp/data_" + std::to_string(i) + ".bin";
        if (!write_file(path, h_data[i], S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to write data file %s\n", path.c_str());
            for (int j = 0; j <= i; j++) free(h_data[j]);
            cleanup_temp_files();
            return 1;
        }
    }

    // 2. Compute parities via CPU EC
    std::vector<uint8_t*> h_parity(p);
    for (int i = 0; i < p; i++) {
        h_parity[i] = (uint8_t*)malloc(S);
    }
    cpu_ec_encode((const uint8_t**)h_data.data(), h_parity.data(), k, p, S);

    // Write parity files to disk
    for (int i = 0; i < p; i++) {
        std::string path = "test_ec_integration_temp/parity_" + std::to_string(i) + ".bin";
        if (!write_file(path, h_parity[i], S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to write parity file %s\n", path.c_str());
            for (int j = 0; j < k; j++) free(h_data[j]);
            for (int j = 0; j < p; j++) free(h_parity[j]);
            cleanup_temp_files();
            return 1;
        }
    }

    // Free initial host buffers to save memory during GPU phase
    for (int i = 0; i < k; i++) free(h_data[i]);
    for (int i = 0; i < p; i++) free(h_parity[i]);

    // 3. Simulating file deletion on disk
    for (int idx : deleted_indices) {
        std::string path = "test_ec_integration_temp/data_" + std::to_string(idx) + ".bin";
        if (unlink(path.c_str()) != 0) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to unlink file %s\n", path.c_str());
            cleanup_temp_files();
            return 1;
        }
    }

    // 4. GPU Reconstruction setup
    std::vector<uint32_t> failed_idx;
    uint8_t *io_buf = (uint8_t*)malloc(S);

    for (int i = 0; i < k; i++) {
        std::string path = "test_ec_integration_temp/data_" + std::to_string(i) + ".bin";
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) {
            failed_idx.push_back(i);
            cudaMemset(buf->d_data[i], 0, S);
        } else {
            size_t r = fread(io_buf, 1, S, f);
            fclose(f);
            if (r != S) {
                printf("[FAIL]\n");
                fprintf(stderr, "Error: Short read on %s\n", path.c_str());
                free(io_buf);
                cleanup_temp_files();
                return 1;
            }
            cudaMemcpy(buf->d_data[i], io_buf, S, cudaMemcpyHostToDevice);
        }
    }

    for (int i = 0; i < p; i++) {
        std::string path = "test_ec_integration_temp/parity_" + std::to_string(i) + ".bin";
        if (!read_file(path, io_buf, S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to read parity file %s\n", path.c_str());
            free(io_buf);
            cleanup_temp_files();
            return 1;
        }
        cudaMemcpy(buf->d_parity[i], io_buf, S, cudaMemcpyHostToDevice);
    }

    // 5. Submit reconstruction task
    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_DECODE;
    for (int i = 0; i < k; i++) item.ec_ptrs[i] = buf->d_data[i];
    for (int i = 0; i < p; i++) item.parity_ptrs[i] = buf->d_parity[i];
    item.stripe_cnt = k;
    item.parity_cnt = p;
    item.cell_size = S;
    item.failed_cnt = failed_idx.size();
    for (size_t i = 0; i < failed_idx.size(); i++) {
        item.failed_idx[i] = failed_idx[i];
    }

    gpu_result_t res = {};
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0 || res.error_code != 0) {
        printf("[FAIL]\n");
        fprintf(stderr, "Error: Reconstruction engine submit/wait failed: rc=%d err=%d\n", rc, res.error_code);
        free(io_buf);
        cleanup_temp_files();
        return 1;
    }

    // 6. Write reconstructed files back to disk
    for (int idx : deleted_indices) {
        cudaMemcpy(io_buf, buf->d_data[idx], S, cudaMemcpyDeviceToHost);
        std::string path = "test_ec_integration_temp/data_" + std::to_string(idx) + ".bin";
        if (!write_file(path, io_buf, S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to write back reconstructed file %s\n", path.c_str());
            free(io_buf);
            cleanup_temp_files();
            return 1;
        }
    }

    // 7. Verify byte-by-byte
    bool verified = true;
    uint8_t *expected_buf = (uint8_t*)malloc(S);

    for (int idx : deleted_indices) {
        std::string path = "test_ec_integration_temp/data_" + std::to_string(idx) + ".bin";
        if (!read_file(path, io_buf, S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to read reconstructed file %s for verify\n", path.c_str());
            verified = false;
            break;
        }
        generate_stripe_data(expected_buf, S, idx, S);
        if (memcmp(io_buf, expected_buf, S) != 0) {
            size_t mismatch_pos = 0;
            for (size_t j = 0; j < S; j++) {
                if (io_buf[j] != expected_buf[j]) {
                    mismatch_pos = j;
                    break;
                }
            }
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Mismatch in data_%d.bin at offset %zu: Got 0x%02X, Expected 0x%02X\n",
                    idx, mismatch_pos, io_buf[mismatch_pos], expected_buf[mismatch_pos]);
            verified = false;
            break;
        }
    }

    free(io_buf);
    free(expected_buf);
    cleanup_temp_files();

    if (!verified) return 1;

    printf("[PASS]\n");
    return 0;
}

int main(void)
{
    printf("=== Starting EC File-Level Integration Tests ===\n");

    const size_t max_size = 128 * 1024 * 1024; // 128MB
    struct test_buffers buf = {};
    if (alloc_test_buffers(&buf, max_size) != 0) {
        fprintf(stderr, "Error: Failed to pre-allocate GPU memory buffers\n");
        return 1;
    }

    // Initialize engine
    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Error: Failed to initialize persistent engine\n");
        free_test_buffers(&buf);
        return 1;
    }
    gpu_ec_init_gf_tables();

    // List of sizes to test
    std::vector<size_t> test_sizes = {
        4 * 1024,          // 4KB
        256 * 1024,        // 256KB
        1024 * 1024,       // 1MB
        4 * 1024 * 1024,   // 4MB
        16 * 1024 * 1024,  // 16MB
        64 * 1024 * 1024,  // 64MB
        128 * 1024 * 1024  // 128MB
    };

    int total_failures = 0;

    for (size_t S : test_sizes) {
        // Scenario A: Single failure (delete stripe 1)
        total_failures += run_integration_case(eng, &buf, S, {1});

        // Scenario B: Double failure (delete stripes 0 and 3)
        total_failures += run_integration_case(eng, &buf, S, {0, 3});
    }

    // Cleanup engine and GPU buffers
    gpu_engine_fini(eng);
    free_test_buffers(&buf);

    printf("=== Integration Tests Completed. %s (%d failures) ===\n",
           total_failures ? "FAILED" : "PASSED", total_failures);

    return total_failures;
}
