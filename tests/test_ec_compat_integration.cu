/**
 * test_ec_compat_integration.cu — EC File-Level Compatibility Integration Test Suite
 *
 * Verifies end-to-end compatibility for RAID-6 and ISA-L modes by:
 *   1. Generating N data files on disk with random data.
 *   2. Computing parity files via CPU compatibility reference models (RAID-6 / ISA-L Cauchy).
 *   3. Deleting M data files (M <= parity count) from disk.
 *   4. Reconstructing the deleted files via GPU EC persistent engine using the matching mode.
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
    uint32_t seed = stripe_idx * 54321 + size_param + 13;
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

/* Host GF(2^8)/0x11D bitwise multiplication helper */
static uint8_t host_gf_mul_11d(uint8_t a, uint8_t b)
{
    uint8_t result = 0;
    uint8_t reduce = 0x1D;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t hi = a & 0x80;
        a <<= 1;
        if (hi) a ^= reduce;
        b >>= 1;
    }
    return result;
}

/* Helper: CPU reference RAID-6 Vandermonde encode */
static void cpu_raid6_encode(const uint8_t **data_ptrs, uint8_t **parity_ptrs, int stripe_cnt, int parity_cnt, size_t stripe_len)
{
    // P is simple XOR
    memset(parity_ptrs[0], 0, stripe_len);
    for (int s = 0; s < stripe_cnt; s++) {
        for (size_t i = 0; i < stripe_len; i++) {
            parity_ptrs[0][i] ^= data_ptrs[s][i];
        }
    }

    // Q is 0x11D Vandermonde with Horner
    if (parity_cnt > 1) {
        for (size_t i = 0; i < stripe_len; i++) {
            uint8_t q = data_ptrs[stripe_cnt - 1][i];
            for (int s = stripe_cnt - 2; s >= 0; s--) {
                q = (uint8_t)(((q << 1) & 0xFE) ^ ((q >> 7) * 0x1D));
                q ^= data_ptrs[s][i];
            }
            parity_ptrs[1][i] = q;
        }
    }
}

/* Helper: CPU reference ISA-L Cauchy encode */
static void cpu_isal_encode(const uint8_t *encode_matrix, const uint8_t **data_ptrs, uint8_t **parity_ptrs, int stripe_cnt, int parity_cnt, size_t stripe_len)
{
    for (int r = 0; r < parity_cnt; r++) {
        memset(parity_ptrs[r], 0, stripe_len);
        for (size_t i = 0; i < stripe_len; i++) {
            uint8_t val = 0;
            for (int j = 0; j < stripe_cnt; j++) {
                uint8_t coef = encode_matrix[r * stripe_cnt + j];
                val ^= host_gf_mul_11d(coef, data_ptrs[j][i]);
            }
            parity_ptrs[r][i] = val;
        }
    }
}

/* GPU pre-allocated memory buffers */
struct test_buffers {
    void *d_data[16];       /* Up to 16 data stripes */
    void *d_parity[4];      /* Up to 4 parity stripes */
    size_t max_size;
};

static int alloc_test_buffers(struct test_buffers *buf, size_t max_size)
{
    buf->max_size = max_size;
    for (int i = 0; i < 16; i++) {
        if (cudaMalloc(&buf->d_data[i], max_size) != cudaSuccess) return -1;
    }
    for (int i = 0; i < 4; i++) {
        if (cudaMalloc(&buf->d_parity[i], max_size) != cudaSuccess) return -1;
    }
    return 0;
}

static void free_test_buffers(struct test_buffers *buf)
{
    for (int i = 0; i < 16; i++) cudaFree(buf->d_data[i]);
    for (int i = 0; i < 4; i++) cudaFree(buf->d_parity[i]);
}

static void cleanup_temp_files()
{
    for (int i = 0; i < 16; i++) {
        std::string path = "test_ec_compat_integration_temp/data_" + std::to_string(i) + ".bin";
        unlink(path.c_str());
    }
    for (int i = 0; i < 4; i++) {
        std::string path = "test_ec_compat_integration_temp/parity_" + std::to_string(i) + ".bin";
        unlink(path.c_str());
    }
    rmdir("test_ec_compat_integration_temp");
}

static int run_compat_case(gpu_engine_t *eng, struct test_buffers *buf, size_t S, int k, int p,
                           gpu_ec_mode_t ec_mode, const std::vector<int> &deleted_indices)
{
    std::string mode_str = (ec_mode == GPU_EC_MODE_RAID6) ? "RAID6" : "ISAL";
    printf("  Running case: mode=%s, size=%zu, deleting %zu stripe(s) {", mode_str.c_str(), S, deleted_indices.size());
    for (size_t i = 0; i < deleted_indices.size(); i++) {
        printf("%s%d", (i > 0 ? ", " : ""), deleted_indices[i]);
    }
    printf("} ... ");
    fflush(stdout);

    // Create temp directory
    mkdir("test_ec_compat_integration_temp", 0777);

    // 1. Generate N data files on disk
    std::vector<uint8_t*> h_data(k);
    std::vector<const uint8_t*> h_data_const(k);
    for (int i = 0; i < k; i++) {
        h_data[i] = (uint8_t*)malloc(S);
        generate_stripe_data(h_data[i], S, i, S);
        h_data_const[i] = h_data[i];
        std::string path = "test_ec_compat_integration_temp/data_" + std::to_string(i) + ".bin";
        if (!write_file(path, h_data[i], S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to write data file %s\n", path.c_str());
            for (int j = 0; j <= i; j++) free(h_data[j]);
            cleanup_temp_files();
            return 1;
        }
    }

    // Generate encode matrix if ISAL mode
    uint8_t encode_matrix[64] = {};
    if (ec_mode == GPU_EC_MODE_ISAL) {
        gpu_ec_gen_cauchy_matrix(encode_matrix, k, p);
    }

    // 2. Compute parities via CPU compatibility reference models
    std::vector<uint8_t*> h_parity(p);
    for (int i = 0; i < p; i++) {
        h_parity[i] = (uint8_t*)malloc(S);
    }

    if (ec_mode == GPU_EC_MODE_RAID6) {
        cpu_raid6_encode(h_data_const.data(), h_parity.data(), k, p, S);
    } else {
        cpu_isal_encode(encode_matrix, h_data_const.data(), h_parity.data(), k, p, S);
    }

    // Write parity files to disk
    for (int i = 0; i < p; i++) {
        std::string path = "test_ec_compat_integration_temp/parity_" + std::to_string(i) + ".bin";
        if (!write_file(path, h_parity[i], S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to write parity file %s\n", path.c_str());
            for (int j = 0; j < k; j++) free(h_data[j]);
            for (int j = 0; j < p; j++) free(h_parity[j]);
            cleanup_temp_files();
            return 1;
        }
    }

    // Free initial host buffers to save memory
    for (int i = 0; i < k; i++) free(h_data[i]);
    for (int i = 0; i < p; i++) free(h_parity[i]);

    // 3. Simulating file deletion
    for (int idx : deleted_indices) {
        std::string path = "test_ec_compat_integration_temp/data_" + std::to_string(idx) + ".bin";
        if (unlink(path.c_str()) != 0) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to unlink file %s\n", path.c_str());
            cleanup_temp_files();
            return 1;
        }
    }

    // 4. GPU Reconstruction setup
    std::vector<int> failed_idx;
    uint8_t *io_buf = (uint8_t*)malloc(S);

    for (int i = 0; i < k; i++) {
        std::string path = "test_ec_compat_integration_temp/data_" + std::to_string(i) + ".bin";
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
        std::string path = "test_ec_compat_integration_temp/parity_" + std::to_string(i) + ".bin";
        if (!read_file(path, io_buf, S)) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Failed to read parity file %s\n", path.c_str());
            free(io_buf);
            cleanup_temp_files();
            return 1;
        }
        cudaMemcpy(buf->d_parity[i], io_buf, S, cudaMemcpyHostToDevice);
    }
    cudaStreamSynchronize(0);

    // Compute decode matrix if ISAL mode
    uint8_t decode_matrix[32] = {};
    if (ec_mode == GPU_EC_MODE_ISAL) {
        int rc = gpu_ec_make_decode_matrix(encode_matrix, k, p, failed_idx.data(), failed_idx.size(), decode_matrix);
        if (rc != 0) {
            printf("[FAIL]\n");
            fprintf(stderr, "Error: Host solver failed to compute decode matrix\n");
            free(io_buf);
            cleanup_temp_files();
            return 1;
        }
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
    item.ec_mode = ec_mode;
    if (ec_mode == GPU_EC_MODE_ISAL) {
        memcpy(item.ec_decode_matrix, decode_matrix, k * failed_idx.size());
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
        std::string path = "test_ec_compat_integration_temp/data_" + std::to_string(idx) + ".bin";
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
        std::string path = "test_ec_compat_integration_temp/data_" + std::to_string(idx) + ".bin";
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
    printf("=== Starting EC Compatibility File-Level Integration Tests ===\n");

    const size_t max_size = 16 * 1024 * 1024; // 16MB max size to run fast
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
        16 * 1024 * 1024   // 16MB
    };

    int total_failures = 0;

    for (size_t S : test_sizes) {
        // --- 1. Test RAID-6 Compatibility (k=4, p=2) ---
        total_failures += run_compat_case(eng, &buf, S, 4, 2, GPU_EC_MODE_RAID6, {1});
        total_failures += run_compat_case(eng, &buf, S, 4, 2, GPU_EC_MODE_RAID6, {0, 3});

        // --- 2. Test ISA-L Compatibility (k=6, p=4) ---
        total_failures += run_compat_case(eng, &buf, S, 6, 4, GPU_EC_MODE_ISAL, {2});
        total_failures += run_compat_case(eng, &buf, S, 6, 4, GPU_EC_MODE_ISAL, {1, 4});
    }

    // Cleanup engine and GPU buffers
    gpu_engine_fini(eng);
    free_test_buffers(&buf);

    printf("=== Compatibility Integration Tests Completed. %s (%d failures) ===\n",
           total_failures ? "FAILED" : "PASSED", total_failures);

    return total_failures;
}
