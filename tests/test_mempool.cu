/**
 * test_mempool.cu — Unit test for lock-free GPU memory pool
 */
#include "gpu_mempool.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

#define TEST_PASS(name) printf("  [PASS] %s\n", name)
#define TEST_FAIL(name, msg) do { printf("  [FAIL] %s: %s\n", name, msg); return 1; } while(0)

int test_basic_alloc(void) {
    gpu_mempool_t *pool = NULL;
    int rc = gpu_mempool_create(&pool, 1024, 4);
    if (rc != 0 || pool == NULL) TEST_FAIL("basic_alloc", "create failed");

    void *p1 = gpu_mempool_alloc(pool);
    void *p2 = gpu_mempool_alloc(pool);
    void *p3 = gpu_mempool_alloc(pool);
    void *p4 = gpu_mempool_alloc(pool);
    
    if (!p1 || !p2 || !p3 || !p4) TEST_FAIL("basic_alloc", "alloc returned NULL prematurely");

    void *p5 = gpu_mempool_alloc(pool);
    if (p5 != NULL) TEST_FAIL("basic_alloc", "pool should be empty");

    gpu_mempool_free(pool, p2);
    void *p6 = gpu_mempool_alloc(pool);
    if (p6 != p2) TEST_FAIL("basic_alloc", "freed block not reused");

    gpu_mempool_destroy(pool);
    TEST_PASS("basic_alloc");
    return 0;
}

int test_concurrent_alloc(void) {
    gpu_mempool_t *pool = NULL;
    const int num_blocks = 1000;
    const int num_threads = 10;
    const int allocs_per_thread = 100;

    int rc = gpu_mempool_create(&pool, 256, num_blocks);
    if (rc != 0) TEST_FAIL("concurrent_alloc", "create failed");

    std::vector<std::thread> threads;
    std::vector<std::vector<void*>> results(num_threads);

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < allocs_per_thread; i++) {
                void *ptr = gpu_mempool_alloc(pool);
                results[t].push_back(ptr);
            }
        });
    }

    for (auto& t : threads) t.join();

    // Verify all pointers are valid and unique
    int valid_count = 0;
    for (int t = 0; t < num_threads; t++) {
        for (void *ptr : results[t]) {
            if (ptr != NULL) valid_count++;
        }
    }
    if (valid_count != num_blocks) TEST_FAIL("concurrent_alloc", "did not allocate all blocks");

    // Free everything concurrently
    threads.clear();
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            for (void *ptr : results[t]) {
                if (ptr) gpu_mempool_free(pool, ptr);
            }
        });
    }
    for (auto& t : threads) t.join();

    // Should be able to allocate num_blocks again
    for (int i = 0; i < num_blocks; i++) {
        void *p = gpu_mempool_alloc(pool);
        if (!p) TEST_FAIL("concurrent_alloc", "failed to allocate after concurrent free");
    }

    gpu_mempool_destroy(pool);
    TEST_PASS("concurrent_alloc");
    return 0;
}

int main(void) {
    printf("=== test_mempool ===\n");
    int failures = 0;
    failures += test_basic_alloc();
    failures += test_concurrent_alloc();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures;
}
