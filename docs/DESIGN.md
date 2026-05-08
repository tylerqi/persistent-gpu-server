# Design Document: Persistent GPU Storage Engine

## 1. Problem Statement

Modern storage systems like DAOS perform intensive compute on the data path — checksumming every I/O for integrity, generating erasure coding parity for fault tolerance, and compressing data for capacity efficiency. On high-throughput NVMe systems processing millions of IOPS, these operations consume significant CPU resources.

GPUs offer massive parallel throughput for these operations, but traditional CUDA usage patterns — launch a kernel, wait for completion, launch the next — incur per-launch overhead (~5-10µs) that negates the benefit for small, latency-sensitive storage I/Os.

### Design Goals

1. **Sub-millisecond dispatch latency** for individual operations
2. **>100K IOPS** aggregate throughput from 128+ concurrent submitters
3. **Zero kernel launch overhead** on the hot path
4. **Lock-free submission** compatible with DAOS Argobots user-level threads
5. **No CUDA API calls** on the data path (no implicit synchronization)

## 2. Persistent Kernel Architecture

### 2.1 Why Persistent Kernels?

A traditional CUDA workflow:
```
for each I/O:
    cudaMalloc(...)          // ~1µs, triggers sync
    cudaMemcpy(H2D, ...)    // ~2-5µs
    kernel<<<1,1>>>(...)     // ~5-10µs launch overhead
    cudaDeviceSynchronize()  // blocks CPU
    cudaMemcpy(D2H, ...)    // ~2-5µs
    cudaFree(...)            // ~1µs, triggers sync
```

Total overhead per operation: **~15-25µs minimum**, regardless of actual compute time. At 100K IOPS, this is 1.5-2.5 seconds of overhead per second — more than the available time budget.

The persistent kernel approach:
```
// Init (once):
engine = gpu_engine_init()    // launches kernel forever

// Per I/O (hot path):
item.op_type = CRC32C
item.data_ptr = d_buf
gpu_engine_submit(engine, &item, &ticket)  // CAS on head, memcpy to slot
gpu_engine_poll(engine, ticket, &result)   // read volatile pinned mem

// Shutdown (once):
gpu_engine_fini(engine)
```

Hot-path operations involve **zero CUDA driver calls** — only CPU memory operations on pinned memory and atomic compare-and-swap.

### 2.2 Kernel Lifecycle

```
gpu_engine_init()
        │
        ├── cudaHostAlloc(queue, results, head, tail, shutdown)
        ├── cudaStreamCreate(NonBlocking)
        ├── persistent_kernel<<<N_SMs, 128>>>(queue, results, head, tail, shutdown)
        │
        │   ┌──────────────────────────────────────────────┐
        │   │ persistent_kernel() runs INDEFINITELY        │
        │   │                                              │
        │   │   while (!*shutdown) {                       │
        │   │       // leader thread polls tail < head     │
        │   │       // atomicCAS to claim work item        │
        │   │       // dispatch operation (all 128 thrds)  │
        │   │       // write result + threadfence_system    │
        │   │   }                                          │
        │   └──────────────────────────────────────────────┘
        │
gpu_engine_fini()
        │
        ├── *shutdown = 1
        ├── __sync_synchronize()
        └── cudaStreamSynchronize(stream)  // waits for kernel exit
```

### 2.3 Block-Level Work Stealing

Each GPU SM runs one block of 128 threads. Work stealing between blocks uses `atomicCAS` on the `tail` pointer:

```cuda
// Only leader thread (tid 0) attempts to claim work
if (tid == 0) {
    uint64_t my_tail = *tail;
    if (my_tail < *head) {
        old = atomicCAS(tail, my_tail, my_tail + 1);
        if (old == my_tail) {
            // Successfully claimed item at index my_tail
            shared_tail = my_tail;
        } else {
            shared_tail = -1;  // Another block stole it
        }
    }
}
__syncthreads();  // Broadcast result to all 128 threads
```

This ensures exactly one block processes each work item, with no duplicate execution.

## 3. Work Queue Design

### 3.1 Ring Buffer Structure

```
Queue: 4096 slots (power of 2 for modulo via bitmask)

  head (CPU atomic)                        tail (GPU atomic)
    │                                        │
    ▼                                        ▼
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│Slot 0 │Slot 1 │Slot 2 │ ...   │Slot N │ ...   │ 4095  │
│INVALID│ READY │CRC32C │       │PENDING│       │INVALID│
└───────┴───────┴───────┴───────┴───────┴───────┴───────┘
         ◄─── completed ──►◄── in-flight ──►◄── empty ──►
```

- **Slot index**: `ticket & 0xFFF` (GPU_QUEUE_MASK)
- **Full condition**: `head - tail >= 4096`
- **Empty condition**: `head == tail`
- **Wraparound**: Ticket values are monotonically increasing 64-bit integers; slot reuse is managed by the generation counter in results

### 3.2 Slot Lifecycle

```
State transitions for queue slot S:

  CPU submit()              GPU persistent_kernel         CPU poll()
  ─────────────              ────────────────────         ──────────
  
  1. CAS head++              
  2. result[S].status=PENDING
  3. queue[S] = item (op=INVALID)
  4. __sync_synchronize
  5. queue[S].op = real_op ──────► 6. Wait for op != INVALID
                                   7. Copy to shared memory
                                   8. Execute operation
                                   9. Write result fields
                                  10. result[S].ticket = tail
                                  11. __threadfence_system
                                  12. result[S].status = READY
                                  13. __threadfence_system ──► 14. Read status == READY
                                  14. queue[S].op = INVALID      15. Verify ticket match
                                                                 16. Copy result
                                                                 17. CAS status READY→PENDING
```

### 3.3 Generation Counter (ABA Prevention)

When the queue wraps around (after 4096 submissions), slot indices repeat. Without protection, a CPU thread polling for ticket `T` could see a result from ticket `T + 4096` in the same slot.

**Solution**: Each result slot stores the exact 64-bit ticket number. `gpu_engine_poll()` validates `result[slot].ticket == expected_ticket` before accepting the result.

## 4. Synchronization Details

### 4.1 CPU → GPU Signaling

The two-phase write protocol ensures the GPU never reads a partially-written work item:

```c
// Phase 1: Write all fields except op_type
temp = *item;
temp.op_type = GPU_OP_INVALID;  // Sentinel
memcpy(&queue[slot], &temp, sizeof(temp));

// Fence: all writes above visible before the signal write below
__sync_synchronize();

// Phase 2: Signal GPU by writing the real op_type
queue[slot].op_type = real_op;
```

The GPU's polling loop checks `item->op_type != GPU_OP_INVALID` — this acts as the "data ready" signal.

### 4.2 GPU → CPU Signaling

```cuda
// Write all result fields first
result->error_code = ...;
result->crc32c_result = ...;
result->ticket = my_tail;

// System-wide fence: ensures all result fields are visible
__threadfence_system();

// Signal CPU by writing READY
result->status = GPU_RESULT_READY;

// Second fence: ensures READY is visible
__threadfence_system();
```

The CPU's `gpu_engine_poll()` reads `status` first, then acquires via `__sync_synchronize()` before reading result fields — guaranteeing it sees the completed data.

### 4.3 Backoff Strategy

| Phase | CPU (submit_and_wait) | GPU (idle poll) |
|-------|----------------------|-----------------|
| Phase 1 | `__builtin_ia32_pause()` (0-1000 spins) | `__nanosleep(1000)` (1µs) |
| Phase 2 | `sched_yield()` (1000-5000 spins) | — |
| Phase 3 | `usleep(10)` (5000+ spins) | — |
| Timeout | 6M spins (~60s) → `GPU_ERR_TIMEOUT` | — |

## 5. Operation Implementations

### 5.1 CRC32C

- **Algorithm**: Castagnoli CRC32C with bit-reversed polynomial `0x82F63B78`
- **Table**: 256-entry lookup table in CUDA `__constant__` memory (1 KB, cached in constant cache)
- **Vectorization**: Aligned data uses `uint4` loads (128-bit) to reduce memory transactions; processes 16 bytes per load in a sequential loop through the CRC table
- **Initialization**: `pthread_once` ensures the table is copied to GPU exactly once, even with multiple threads calling `gpu_csum_init()`

### 5.2 SHA256

- **Algorithm**: FIPS 180-4 SHA-256, single-threaded per block
- **Constants**: 64-entry K table in `__constant__` memory
- **Padding**: Inline padding computation (no pre-processing buffer needed)
- **Limitation**: Single-threaded execution (only `tid == 0` computes); future optimization could parallelize across threads for large payloads

### 5.3 EC Parity (P + Q)

**P-Parity** (XOR):
```
P[i] = D0[i] ⊕ D1[i] ⊕ D2[i] ⊕ ... ⊕ Dk-1[i]
```

**Q-Parity** (GF(2^8) Horner's method):
```
Q[i] = g^(k-1) · D0[i] ⊕ g^(k-2) · D1[i] ⊕ ... ⊕ g^0 · Dk-1[i]

Computed as:  Q = ((...((Dk-1 · g ⊕ Dk-2) · g ⊕ Dk-3) · g) ... ⊕ D0)
```

Where `g = 2` in GF(2^8) with irreducible polynomial `x^8 + x^4 + x^3 + x + 1` (`0x11B`).

**GF(2^8) multiplication by 2** (vectorized across 4 bytes packed in uint32_t):
```cuda
__device__ uint4 gf_mul2(uint4 val) {
    // For each 32-bit word containing 4 packed GF(2^8) elements:
    // mask = high bits (overflow indicators)
    // result = (val << 1) & 0xFEFEFEFE  (shift each byte left)
    //        ^ (mask * 0x1B)             (reduce overflows with polynomial)
}
```

This processes 16 bytes (4 × uint32_t in uint4) per instruction, achieving high memory throughput.

### 5.4 LZ4 Compression

Two code paths controlled by `USE_NVCOMPDX`:

| Path | Mechanism | Compression Ratio |
|------|-----------|-------------------|
| nvCOMPDx | Block-cooperative via cooperative groups | Real LZ4 |
| Stub | Vectorized `uint4` memcpy | 1:1 (no compression) |

The stub path enables full system testing without the nvCOMPDx dependency. Both paths participate in the block-cooperative execution model (all 128 threads active).

> **⚠ Bandwidth caveat**: In stub mode, LZ4 "compression" is a GPU VRAM-to-VRAM memcpy (~448 GB/s peak on RTX 2060 SUPER). Benchmark bandwidth numbers like "107 GB/s" or "272 GB/s" reflect this VRAM copy speed and **do not represent actual compression throughput or PCIe transfer bandwidth** (PCIe 3.0 x16 ≈ 15.75 GB/s). Only IOPS numbers are meaningful as a measure of dispatch overhead.

## 6. Memory Pool Design

### 6.1 Problem

During engine lifetime, `cudaMalloc()` triggers implicit device synchronization → deadlock with persistent kernel. Applications need a way to dynamically allocate GPU memory.

### 6.2 Solution: Lock-Free Treiber Stack

```
Pool structure:
  gpu_base: [Block 0 | Block 1 | Block 2 | ... | Block N-1]
                ↕         ↕         ↕               ↕
  next_idx:  [  1    |   2    |   3    | ... | NULL    ]  (free-list links)
  bitmap:    [  0    |   0    |   0    | ... |   0     ]  (1=allocated)

  head: [32-bit generation | 32-bit index]
        ────────────────────────────────────
        ABA-safe 64-bit atomic variable
```

**Alloc**:
```c
old_head = pool->head;  // [gen | idx]
next = pool->next_idx_array[idx];
new_head = [(gen+1) | next];
CAS(&pool->head, old_head, new_head);
// Set bitmap bit for double-free detection
```

**Free**:
```c
// Check bitmap — if bit already 0, this is a double-free
old_bits = atomic_and(&bitmap[word], ~bit);
if (!(old_bits & bit)) { ERROR: double free! }

// Push back onto stack
next_idx_array[idx] = current_head_idx;
new_head = [(gen+1) | idx];
CAS(&pool->head, old_head, new_head);
```

The 32-bit generation counter prevents the ABA problem where a freed-and-reallocated block could cause the CAS to spuriously succeed.

## 7. Error Handling Philosophy

The persistent kernel must **never crash**. A `__trap()` or unhandled exception would kill the entire engine, losing all in-flight operations.

**Design rules**:
1. All errors are written to the result slot's `error_code` field
2. `__threadfence_system()` ensures errors are visible to CPU before `status = READY`
3. Invalid operations return `GPU_ERR_INVAL`; unimplemented ops return `GPU_ERR_NOSYS`
4. CPU-side `gpu_engine_submit()` validates `op_type` range before writing to queue (fail fast)
5. The `GPU_ASSERT` macro records errors and returns from `__device__` functions without trapping

## 8. Testing Strategy

### Unit Tests (8 tests, 31 sub-tests)

| Test | Coverage |
|------|----------|
| `test_engine` | Init/fini lifecycle, NOP dispatch, CRC32C via engine |
| `test_csum` | CRC32C and SHA256 correctness against CPU reference |
| `test_ec` | EC parity (P + Q) correctness against CPU reference |
| `test_error` | Invalid operations, error code propagation |
| `test_latency` | Dispatch round-trip timing |
| `test_compress` | LZ4 compress→decompress roundtrip integrity |
| `test_verify_all` | Cross-verification of all operations vs. CPU golden reference |
| `test_mempool` | Alloc/free, exhaustion, double-free detection |

### Benchmarks (2 benchmarks)

| Benchmark | Measures |
|-----------|----------|
| `bench_csum` | Standalone CRC32C kernel throughput (CPU vs GPU) |
| `bench_dispatch` | Persistent kernel dispatch: latency histogram + sustained throughput at 128 threads |

## 9. Known Limitations and Future Work

### Current Limitations

1. **SHA256 is single-threaded**: Only thread 0 in the block computes SHA256. For large payloads, this underutilizes the block's 128 threads.

2. **EC decode not implemented**: `GPU_OP_EC_DECODE` returns `GPU_ERR_NOSYS`. Data reconstruction from P+Q parity requires syndrome calculation and Galois field inversion.

3. **LZ4 stub mode**: Without nvCOMPDx, compression is a memcpy (1:1 ratio). The API contract is preserved but no actual compression occurs.

4. **Single GPU**: The engine assumes one GPU device. Multi-GPU support would require per-device engine instances.

5. **Fixed queue size**: 4096 slots is hardcoded. Optimal size depends on workload concurrency.

### Future Work

- **DAOS VOS integration**: Wire `gpu_engine_submit()` into DAOS's checksum and EC paths
- **Async pipeline**: Use `submit()` + `poll()` for overlapped I/O and compute
- **Multi-queue**: Separate high-priority (checksum) and low-priority (compression) queues
- **RDMA integration**: GPUDirect RDMA for zero-copy from NVMe to GPU memory
- **EC decode**: Complete RAID-6 data reconstruction for single and double failures

## 10. File Reference

```
persistent-gpu-server/
├── include/
│   ├── gpu_engine.h      # Public API: init, submit, poll, fini
│   ├── gpu_error.h       # Error codes and GPU-side assert macros
│   ├── gpu_csum.h        # Standalone CRC32C/SHA256 API
│   ├── gpu_ec.h          # Standalone EC parity API
│   ├── gpu_mem.h         # GPU memory alloc/free wrappers
│   ├── gpu_mempool.h     # Lock-free GPU memory pool API
│   └── gpu_comp.h        # Internal: device-side function declarations
├── src/
│   ├── gpu_engine.cu     # Persistent kernel + submit/poll/wait (504 lines)
│   ├── gpu_csum.cu       # CRC32C + SHA256 device implementations (254 lines)
│   ├── gpu_ec.cu         # EC parity with GF(2^8) (166 lines)
│   ├── gpu_comp.cu       # LZ4 compress/decompress (139 lines)
│   ├── gpu_mem.cu        # cudaMalloc/Free wrappers (38 lines)
│   └── gpu_mempool.cu    # Lock-free Treiber stack pool (142 lines)
├── tests/                # 8 unit test files
├── benchmarks/           # bench_csum, bench_dispatch
├── examples/
│   └── simple_server.cu  # Minimal usage example
├── scripts/
│   └── download_nvcompdx.sh
└── CMakeLists.txt        # Build system (107 lines)
```

**Total**: ~3,465 lines of CUDA/C++ across 23 source files.
