# Architecture: Persistent GPU Storage Engine

## Overview

The Persistent GPU Storage Engine is a GPU-accelerated storage data-path library designed for integration with [DAOS](https://github.com/daos-stack/daos) (Distributed Asynchronous Object Storage). It offloads computationally intensive storage operations — checksum, erasure coding, and compression — to a CUDA GPU using a **persistent kernel** architecture.

Unlike traditional GPU computing where kernels are launched and retired per operation, this engine launches a single long-lived CUDA kernel at initialization that polls a shared-memory work queue for the entire process lifetime. This eliminates kernel launch overhead (~5–10µs per launch) and enables sub-millisecond dispatch latency at scale.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Host CPU (Multi-threaded)                    │
│                                                                     │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐        ┌──────────┐    │
│   │ DAOS ULT │  │ DAOS ULT │  │ DAOS ULT │  ...   │ DAOS ULT │    │
│   │ Thread 0 │  │ Thread 1 │  │ Thread 2 │        │Thread N-1│    │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘        └────┬─────┘    │
│        │             │             │                    │           │
│        └──────┬──────┴──────┬──────┘                    │           │
│               ▼             ▼                           ▼           │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │              gpu_engine_submit() / submit_and_wait()          │ │
│   │              Lock-free CAS on head pointer                    │ │
│   └───────────────────────┬───────────────────────────────────────┘ │
│                           ▼                                         │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │           Pinned Memory Work Queue (4096 slots)               │ │
│   │                                                               │ │
│   │   ┌──────┬──────┬──────┬─────┬──────────────────┬──────┐     │ │
│   │   │Slot 0│Slot 1│Slot 2│ ... │                  │ 4095 │     │ │
│   │   └──────┴──────┴──────┴─────┴──────────────────┴──────┘     │ │
│   │                                                               │ │
│   │   head ─────────────────────────────► tail                    │ │
│   │   (CPU writes, GPU reads)  (GPU advances via atomicCAS)       │ │
│   └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                               │ PCIe Bus
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU (Persistent Kernel)                          │
│                                                                     │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐          ┌─────────┐       │
│   │  SM  0  │ │  SM  1  │ │  SM  2  │   ...    │  SM N-1 │       │
│   │ Block 0 │ │ Block 1 │ │ Block 2 │          │Block N-1│       │
│   │128 thrds│ │128 thrds│ │128 thrds│          │128 thrds│       │
│   └────┬────┘ └────┬────┘ └────┬────┘          └────┬────┘       │
│        │           │           │                    │              │
│        └─────┬─────┴─────┬─────┘                    │              │
│              ▼           ▼                          ▼              │
│   ┌───────────────────────────────────────────────────────────────┐│
│   │                   Operation Dispatch                          ││
│   │  NOP │ CRC32C │ SHA256 │ EC Encode │ LZ4 Comp │ LZ4 Decomp  ││
│   └───────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Zero-launch overhead**: The persistent kernel eliminates CUDA launch latency entirely. Work items are dispatched by writing to pinned memory and advancing an atomic pointer — no driver calls on the hot path.

2. **Lock-free concurrency**: The CPU-side submit path uses compare-and-swap (CAS) on the queue head pointer. Multiple DAOS user-level threads (ULTs) or OS threads can submit work concurrently without any mutex.

3. **No implicit synchronization**: All GPU memory for data buffers must be allocated **before** engine initialization or via the lock-free `gpu_mempool`. Calling `cudaMalloc()` / `cudaFree()` while the persistent kernel is running triggers implicit device synchronization and will **deadlock**.

4. **Block-cooperative execution**: Each GPU block (128 threads) processes one work item at a time. Operations like EC parity and LZ4 compression use all threads in the block cooperatively for high throughput.

5. **Adaptive backoff**: Both CPU poll loops and GPU idle loops use progressive backoff (CPU: pause → yield → usleep; GPU: `__nanosleep`) to reduce PCIe coherence traffic and power consumption during idle periods.

## System Components

### Core Engine (`gpu_engine.cu`)

The central component. Manages the persistent kernel lifecycle and provides the submit/poll API.

| Component | Description |
|-----------|-------------|
| `gpu_engine_init()` | Allocates pinned memory, launches persistent kernel (1 block per SM) |
| `gpu_engine_fini()` | Sets shutdown flag, waits for kernel exit, frees resources |
| `gpu_engine_submit()` | Lock-free CAS to claim queue slot, copies work item, returns ticket |
| `gpu_engine_poll()` | Checks result slot for completion, validates generation counter |
| `gpu_engine_submit_and_wait()` | Blocking wrapper with 3-phase adaptive backoff |
| `persistent_kernel()` | GPU-side: leader-thread polls `tail`, claims work via `atomicCAS`, dispatches |

### Checksum (`gpu_csum.cu`)

Implements CRC32C (Castagnoli polynomial, iSCSI-compatible) and SHA256 as CUDA `__device__` functions callable from the persistent kernel.

- **CRC32C**: Table-driven with `__constant__` memory lookup table. Uses vectorized `uint4` loads for 16-byte aligned data.
- **SHA256**: Full FIPS 180-4 implementation with 64-round message schedule, constants in `__constant__` memory.
- **Thread-safe init**: CRC32C table population uses `pthread_once` to prevent race conditions during first use.

### Erasure Coding (`gpu_ec.cu`)

Implements RAID-6 compatible parity generation and data reconstruction using Galois Field GF(2^8) arithmetic.

#### EC Encode (Parity Generation)

- **P-Parity**: Standard XOR across data stripes (RAID-5 compatible)
- **Q-Parity**: Horner's method with GF(2^8) multiplication by generator `g=2`, using irreducible polynomial `0x11B` (ISA-L / Linux RAID-6 compatible)
- **Vectorized**: Uses `uint4` (128-bit) loads/stores for memory throughput. Scalar tail loop handles non-16-aligned cell sizes.
- **Block-cooperative**: All 128 threads in a block participate in parallel stripe processing.

#### EC Decode (Data Reconstruction)

Reconstructs 1 or 2 failed data stripes from P and Q parity:

- **Single failure**: Reconstructed from P parity via XOR syndrome: `D_failed = P ⊕ Σ(surviving stripes)`
- **Double failure**: Solves a 2x2 system over GF(2^8) using both P and Q syndromes:
  - `S_P = P ⊕ Σ(surviving)` — XOR syndrome
  - `S_Q = Q ⊕ Σ(2^s · surviving[s])` — weighted syndrome
  - `D_y = (S_Q ⊕ g_x · S_P) · (g_y ⊕ g_x)^(-1)` — GF(2^8) division
  - `D_x = S_P ⊕ D_y`
- **GF(2^8) arithmetic**: Uses direct bitwise multiplication (`gf_mul_bitwise`) and Fermat's little theorem inversion (`gf_inv_direct`) — no lookup tables needed, avoiding constant memory dependency
- **L1 cache bypass**: All data reads use `__ldcg()` (load cached-global) to bypass L1 cache. This is **essential** in persistent kernels where host-side `cudaMemset` operations on failed stripes may not be visible through stale L1 entries

### Compression (`gpu_comp.cu`)

LZ4 compression/decompression with two implementations:

- **nvCOMPDx path** (`USE_NVCOMPDX`): Uses NVIDIA's MathDx library for production-grade block-cooperative LZ4 via cooperative groups.
- **Stub path** (default): Vectorized memcpy fallback that preserves the API contract (1:1 compression ratio). Enables full pipeline testing without the nvCOMPDx dependency.

### Memory Pool (`gpu_mempool.cu`)

Lock-free GPU memory allocator for use during engine lifetime (when `cudaMalloc` would deadlock).

- **Treiber stack**: Lock-free free-list using 64-bit CAS with ABA-safe generation counter `[32-bit gen | 32-bit index]`
- **Double-free detection**: Atomic bitmap tracks allocated blocks; frees against already-free blocks are caught and rejected
- **Fixed-size blocks**: Pre-allocates a contiguous GPU memory region, divided into equal-sized blocks

### Memory Utilities (`gpu_mem.cu`)

Thin wrappers around `cudaMalloc` / `cudaFree` / `cudaMemcpy` for host code. **Must not be used while the engine is running.**

### Error Handling (`gpu_error.h`)

Non-fatal error system designed for persistent kernel use:

- Errors are written to pinned memory result slots (never `__trap()` or abort)
- CPU polls error codes alongside results
- Error codes follow DAOS `DER_*` numeric convention

## Memory Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Host (CPU) Memory                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │          Pinned Memory (cudaHostAllocMapped)        │  │
│  │                                                    │  │
│  │  ┌─────────────┐  ┌──────────────┐                │  │
│  │  │ Work Queue  │  │ Result Slots │                │  │
│  │  │ 4096 items  │  │ 4096 results │                │  │
│  │  │ (~1.1 MB)   │  │ (~280 KB)    │                │  │
│  │  └─────────────┘  └──────────────┘                │  │
│  │                                                    │  │
│  │  ┌──────┐ ┌──────┐ ┌──────────┐                   │  │
│  │  │ head │ │ tail │ │ shutdown │                   │  │
│  │  │ 8B   │ │ 8B   │ │ 4B       │                   │  │
│  │  └──────┘ └──────┘ └──────────┘                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │          GPU Mempool (host-side metadata)           │  │
│  │  Free-list array, alloc bitmap, base pointer       │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   GPU Device Memory (VRAM)                │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         __constant__ Memory                        │  │
│  │  CRC32C table (1 KB) │ SHA256 constants (256 B)    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         User Data Buffers (pre-allocated)          │  │
│  │  Allocated via cudaMalloc BEFORE engine init       │  │
│  │  Or via gpu_mempool DURING engine lifetime         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Mempool Backing Store                      │  │
│  │  Contiguous region: block_size × block_count       │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Pinned Memory Coherence Model

The work queue and result slots reside in **pinned host memory** mapped into the GPU's address space (`cudaHostAllocMapped`). This provides:

1. **Direct CPU writes**: CPU threads write work items without any `cudaMemcpy` call
2. **Direct GPU reads**: GPU reads items directly over PCIe (zero-copy)
3. **Coherence**: `__threadfence_system()` on GPU and `__sync_synchronize()` on CPU ensure cross-device visibility

**Critical constraint**: With 128+ CPU threads and 34 GPU blocks all accessing the same pinned memory, PCIe coherence traffic can become a bottleneck. The engine uses progressive backoff (`__nanosleep` on GPU, `sched_yield`/`usleep` on CPU) to mitigate this.

## Work Queue Protocol

The work queue is a single-producer-multiple-consumer ring buffer (CPU produces, GPU blocks consume) with the following protocol:

### Submit (CPU side)
```
1. CAS loop: atomically increment head if (head - tail) < 4096
2. Copy work item payload with op_type = INVALID
3. Memory fence (__sync_synchronize)
4. Write real op_type (signals GPU)
5. Return ticket = claimed head value
```

### Consume (GPU side)
```
1. Leader thread reads head and tail
2. If tail < head: atomicCAS to advance tail
3. Wait for op_type != INVALID (CPU may still be writing)
4. Copy item to shared memory for block-wide access
5. Dispatch operation (all 128 threads participate)
6. Write result fields + generation ticket
7. __threadfence_system() to flush to CPU
8. Set status = READY
```

### Poll (CPU side)
```
1. Read result[slot].status
2. If READY: validate generation counter (ticket match)
3. Copy result to caller
4. CAS status READY→PENDING (prevents double-counting)
```

## Concurrency Model

| Resource | Mechanism | Notes |
|----------|-----------|-------|
| Queue `head` | CAS (CPU) | Multiple CPU threads compete; losers retry |
| Queue `tail` | `atomicCAS` (GPU) | Multiple GPU blocks compete; losers retry |
| Result slots | `__threadfence_system` + volatile | GPU writes, CPU reads; generation counter prevents stale reads |
| Mempool free-list | CAS (CPU) | 64-bit `[gen|index]` prevents ABA problem |
| Mempool bitmap | `__sync_fetch_and_or/and` | Atomic bit manipulation for double-free detection |
| CRC32C table | `pthread_once` | One-time initialization, safe from multiple threads |

## Build System

The project uses CMake 3.18+ with CUDA language support.

**Key build decisions**:
- **Static library** (`libdaos_gpu_engine.a`): Avoids CUDA fatbin registration conflicts that occur with shared libraries when multiple `.cu` executables link against the same library.
- **Separable compilation** (`CUDA_SEPARABLE_COMPILATION ON`): Required for `__device__` functions in separate translation units (CRC32C in `gpu_csum.cu` called from `gpu_engine.cu`).
- **Architecture**: Defaults to `sm_75` (Turing/RTX 2060); configurable via `-DCUDA_ARCH=XX`.

## Integration with DAOS

The engine is designed to plug into DAOS's data path as a GPU-accelerated offload engine:

```
DAOS Server Target
    │
    ├── VOS (Versioning Object Store)
    │       │
    │       ├── Checksum verification ──────► gpu_engine: GPU_OP_CRC32C
    │       ├── EC parity generation ───────► gpu_engine: GPU_OP_EC_ENCODE
    │       └── Inline compression ─────────► gpu_engine: GPU_OP_COMPRESS_LZ4
    │
    ├── Object I/O (Argobots ULTs)
    │       │
    │       └── Multiple ULTs submit concurrently via gpu_engine_submit()
    │
    └── SWIM / Management
            │
            └── gpu_engine_get_metrics() for telemetry
```

Each DAOS I/O target thread (Argobots ULT) calls `gpu_engine_submit_and_wait()` for blocking operations or `gpu_engine_submit()` + `gpu_engine_poll()` for asynchronous pipelining.

## Performance Characteristics

Measured on NVIDIA RTX 2060 SUPER (8 GB, 34 SMs, sm_75):

| Metric | Value |
|--------|-------|
| NOP dispatch (p50) | ~714 µs |
| NOP throughput (128 threads) | 176K IOPS |
| 4KB CRC32C throughput | 172K IOPS |
| 1MB LZ4 compress throughput* | 110K IOPS |
| 1MB LZ4 decompress throughput* | 109K IOPS |
| 4MB LZ4 decompress throughput* | 35K IOPS |
| 1MB EC 4+2 encode throughput | 52K IOPS |
| Queue depth | 4096 |
| GPU memory overhead | ~1.4 MB (queue + results) |

> **Note**: GPU Data Rate numbers (e.g., 107 GB/s for LZ4 compress) represent
> GPU-internal VRAM-to-VRAM processing speed, **not** PCIe transfer bandwidth.
> All data is pre-resident on the GPU. PCIe 3.0 x16 is ~15.75 GB/s; end-to-end
> throughput including host↔device transfer will be PCIe-limited.
>
> **\*** LZ4 compress/decompress running in **memcpy stub mode**. The nvCOMPDx
> LZ4 device-side API is defined in headers (MathDx 25.06+), but the
> pre-compiled `libnvcompdx.fatbin` only ships **ANS** algorithm
> implementations — no LZ4 symbols are present.  IOPS numbers reflect
> GPU VRAM copy speed with zero actual compression. Real LZ4 requires a
> future MathDx release that includes LZ4 in the fatbin, or use of
> `nvJitLink` for runtime LTO compilation.
