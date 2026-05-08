/**
 * test_verify_all.cu — Comprehensive data verification for persistent GPU engine
 * Runs every operation type via the lock-free queue and cross-checks with CPU logic.
 *
 * This is the ONLY test that validates the full persistent kernel dispatch path
 * end-to-end. All other test_*.cu files use standalone kernel APIs.
 *
 * NOTE: All GPU memory is allocated BEFORE the engine starts, because cudaMalloc
 * triggers implicit device synchronization which deadlocks with the persistent kernel.
 */
#include "gpu_engine.h"
#include "gpu_csum.h"
#include "gpu_ec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

/* Helper: alloc and populate host buffer */
static uint8_t *alloc_host_buffer(size_t len, int seed) {
    uint8_t *buf = (uint8_t *)malloc(len);
    for (size_t i = 0; i < len; i++) {
        buf[i] = (uint8_t)((i * seed + 17) & 0xFF);
    }
    return buf;
}

/* Helper: simple SHA256 string test vector */
static const uint8_t expected_sha256_abc[32] = {
    0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,
    0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
    0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,
    0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad
};

/* Helper: CPU GF(2^8) multiply by 2 for packed 32-bit words.
 * Uses polynomial 0x11B (reduction constant 0x1B) — ISA-L compatible. */
static inline uint32_t cpu_gf_mul2_32(uint32_t val) {
    uint32_t mask = (val & 0x80808080) >> 7;
    return ((val << 1) & 0xFEFEFEFE) ^ (mask * 0x1B);
}

/* ── Pre-allocated GPU buffers ──────────────────────────────────────────── */
/* All allocated before engine init to avoid cudaMalloc deadlock */
struct gpu_buffers {
    /* CRC test */
    void *crc_data;     /* 1MB */
    /* SHA test */
    void *sha_data;     /* 3 bytes */
    /* Compress test */
    void *comp_data;    /* 4KB */
    void *comp_out;     /* 8KB */
    void *decomp_out;   /* 4KB */
    /* EC test (4 data + 2 parity) */
    void *ec_data[4];   /* 1MB each */
    void *ec_parity[2]; /* 1MB each */
};

/* ── Verification Tests ─────────────────────────────────────────────────── */

static int verify_crc32c(gpu_engine_t *eng, struct gpu_buffers *buf) {
    size_t len = 1024 * 1024;
    uint8_t *h_data = alloc_host_buffer(len, 31);
    cudaMemcpy(buf->crc_data, h_data, len, cudaMemcpyHostToDevice);

    gpu_work_item_t item = {};
    item.op_type = GPU_OP_CRC32C;
    item.data_ptr = buf->crc_data;
    item.data_len = len;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0) { free(h_data); TEST_FAIL("verify_crc32c", "submit_and_wait failed"); }

    uint32_t gpu_crc = res.crc32c_result;
    uint32_t cpu_crc = cpu_crc32c((const char*)h_data, len);
    free(h_data);

    if (gpu_crc != cpu_crc) {
        char msg[128];
        snprintf(msg, sizeof(msg), "Mismatch: GPU=0x%08X CPU=0x%08X", gpu_crc, cpu_crc);
        TEST_FAIL("verify_crc32c", msg);
    }
    TEST_PASS("verify_crc32c");
    return 0;
}

static int verify_sha256(gpu_engine_t *eng, struct gpu_buffers *buf) {
    const char *test_str = "abc";
    cudaMemcpy(buf->sha_data, test_str, 3, cudaMemcpyHostToDevice);

    gpu_work_item_t item = {};
    item.op_type = GPU_OP_SHA256;
    item.data_ptr = buf->sha_data;
    item.data_len = 3;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0) TEST_FAIL("verify_sha256", "submit_and_wait failed");

    if (memcmp(res.sha256_result, expected_sha256_abc, 32) != 0)
        TEST_FAIL("verify_sha256", "SHA256 hash mismatch vs known CPU vector");

    TEST_PASS("verify_sha256");
    return 0;
}

static int verify_compress_decompress(gpu_engine_t *eng, struct gpu_buffers *buf) {
    size_t len = 4096;
    uint8_t *h_orig = alloc_host_buffer(len, 13);
    uint8_t *h_out = (uint8_t *)malloc(len);
    cudaMemcpy(buf->comp_data, h_orig, len, cudaMemcpyHostToDevice);

    /* Compress */
    gpu_work_item_t item_c = {};
    item_c.op_type = GPU_OP_COMPRESS_LZ4;
    item_c.data_ptr = buf->comp_data;
    item_c.data_len = len;
    item_c.comp_out_ptr = buf->comp_out;
    item_c.comp_max_size = len * 2;
    gpu_result_t res_c;
    int rc = gpu_engine_submit_and_wait(eng, &item_c, &res_c);
    if (rc != 0) { free(h_orig); free(h_out); TEST_FAIL("verify_compress", "compress submit failed"); }

    /* Decompress */
    gpu_work_item_t item_d = {};
    item_d.op_type = GPU_OP_DECOMPRESS_LZ4;
    item_d.data_ptr = buf->comp_out;
    item_d.data_len = res_c.actual_comp_size;
    item_d.comp_out_ptr = buf->decomp_out;
    item_d.comp_max_size = len;
    gpu_result_t res_d;
    rc = gpu_engine_submit_and_wait(eng, &item_d, &res_d);
    if (rc != 0) { free(h_orig); free(h_out); TEST_FAIL("verify_compress", "decompress submit failed"); }

    cudaMemcpy(h_out, buf->decomp_out, len, cudaMemcpyDeviceToHost);

    if (memcmp(h_orig, h_out, len) != 0) {
        free(h_orig); free(h_out);
        TEST_FAIL("verify_compress", "Data corruption after roundtrip");
    }

    free(h_orig); free(h_out);
    TEST_PASS("verify_compress_decompress");
    return 0;
}

static int verify_ec_encode(gpu_engine_t *eng, struct gpu_buffers *buf) {
    const uint32_t k = 4;
    const uint32_t p = 2;
    const size_t len = 1024 * 1024;

    uint8_t *h_data[4];
    uint8_t *h_gpu_parity[2];
    uint8_t *h_cpu_parity[2];

    for (int i = 0; i < (int)k; i++) {
        h_data[i] = alloc_host_buffer(len, i * 7 + 1);
        cudaMemcpy(buf->ec_data[i], h_data[i], len, cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < (int)p; i++) {
        h_gpu_parity[i] = (uint8_t *)calloc(1, len);
        h_cpu_parity[i] = (uint8_t *)calloc(1, len);
    }

    gpu_work_item_t item = {};
    item.op_type = GPU_OP_EC_ENCODE;
    for (int i = 0; i < (int)k; i++) item.ec_ptrs[i] = buf->ec_data[i];
    for (int i = 0; i < (int)p; i++) item.parity_ptrs[i] = buf->ec_parity[i];
    item.stripe_cnt = k;
    item.parity_cnt = p;
    item.cell_size = len;

    gpu_result_t res;
    int rc = gpu_engine_submit_and_wait(eng, &item, &res);
    if (rc != 0) TEST_FAIL("verify_ec_encode", "submit failed");

    for (int i = 0; i < (int)p; i++) {
        cudaMemcpy(h_gpu_parity[i], buf->ec_parity[i], len, cudaMemcpyDeviceToHost);
    }

    /* CPU P Parity: XOR */
    for (size_t i = 0; i < len; i++) {
        uint8_t val = h_data[0][i];
        for (int s = 1; s < (int)k; s++) val ^= h_data[s][i];
        h_cpu_parity[0][i] = val;
    }

    /* CPU Q Parity: GF(2^8) Horner's Method */
    uint32_t *cpu_q_32 = (uint32_t *)h_cpu_parity[1];
    for (size_t i = 0; i < len / 4; i++) {
        uint32_t val = ((uint32_t *)h_data[k - 1])[i];
        for (int s = (int)k - 2; s >= 0; s--) {
            uint32_t s_val = ((uint32_t *)h_data[s])[i];
            val = cpu_gf_mul2_32(val);
            val ^= s_val;
        }
        cpu_q_32[i] = val;
    }

    /* Verify both P and Q parity against CPU reference */
    for (int pi = 0; pi < (int)p; pi++) {
        int mismatch = memcmp(h_gpu_parity[pi], h_cpu_parity[pi], len);
        if (mismatch != 0) {
            const char *label = (pi == 0) ? "P" : "Q";
            for (int j = 0; j < 16; j++) {
                printf("%s byte %d: GPU %02x, CPU %02x\n", label, j,
                       h_gpu_parity[pi][j], h_cpu_parity[pi][j]);
            }
            TEST_FAIL("verify_ec_encode", "GPU EC parity mismatch with CPU");
        }
    }

    for (int i = 0; i < (int)k; i++) free(h_data[i]);
    for (int i = 0; i < (int)p; i++) { free(h_gpu_parity[i]); free(h_cpu_parity[i]); }

    TEST_PASS("verify_ec_encode");
    return 0;
}

int main(void) {
    printf("=== Comprehensive Data Verification ===\n");

    /* Allocate ALL GPU memory BEFORE engine init.
     * cudaMalloc triggers implicit device synchronization which will
     * deadlock if the persistent kernel is already running. */
    struct gpu_buffers buf = {};
    cudaMalloc(&buf.crc_data, 1024 * 1024);
    cudaMalloc(&buf.sha_data, 64);
    cudaMalloc(&buf.comp_data, 4096);
    cudaMalloc(&buf.comp_out, 8192);
    cudaMalloc(&buf.decomp_out, 4096);
    for (int i = 0; i < 4; i++) cudaMalloc(&buf.ec_data[i], 1024 * 1024);
    for (int i = 0; i < 2; i++) cudaMalloc(&buf.ec_parity[i], 1024 * 1024);

    /* Now init engine (launches persistent kernel) */
    gpu_engine_t *eng = NULL;
    if (gpu_engine_init(&eng) != 0) {
        fprintf(stderr, "Failed to init GPU engine\n");
        return 1;
    }

    int failures = 0;
    failures += verify_compress_decompress(eng, &buf);
    failures += verify_crc32c(eng, &buf);
    failures += verify_sha256(eng, &buf);
    failures += verify_ec_encode(eng, &buf);

    /* Shut down engine BEFORE freeing GPU memory */
    gpu_engine_fini(eng);

    /* Now safe to free GPU memory */
    cudaFree(buf.crc_data);
    cudaFree(buf.sha_data);
    cudaFree(buf.comp_data);
    cudaFree(buf.comp_out);
    cudaFree(buf.decomp_out);
    for (int i = 0; i < 4; i++) cudaFree(buf.ec_data[i]);
    for (int i = 0; i < 2; i++) cudaFree(buf.ec_parity[i]);

    printf("\n=== Verification %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
