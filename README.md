# Persistent GPU Storage Engine

A GPU-accelerated storage data-path engine that offloads checksum, erasure coding, and compression operations to CUDA GPUs using a **persistent kernel** architecture. Designed for integration with [DAOS](https://github.com/daos-stack/daos) distributed storage.

## Key Features

- **Persistent kernel** — eliminates CUDA launch overhead; work is dispatched by writing to pinned memory
- **Lock-free submission** — CAS-based work queue supports 128+ concurrent CPU threads (compatible with Argobots ULTs)
- **176K IOPS dispatch** — sub-millisecond latency for NOP operations
- **GPU-accelerated operations**:
  - CRC32C checksumming (Castagnoli, iSCSI-compatible)
  - SHA256 hashing
  - Erasure Coding parity (RAID-6 P+Q with GF(2^8), ISA-L compatible)
  - LZ4 compression/decompression (via nvCOMPDx or stub)
- **Lock-free GPU memory pool** — allocate device memory without deadlocking the persistent kernel

## Architecture

```
  CPU Threads (DAOS ULTs)           GPU Persistent Kernel
  ┌──────────────────────┐         ┌──────────────────────┐
  │ submit() ──► CAS ────┼── ──── ─┤─► poll tail          │
  │    (lock-free)        │  PCIe  │   atomicCAS claim     │
  │ poll() ◄── read ─────┼── ──── ─┤─◄ dispatch + result  │
  └──────────────────────┘         └──────────────────────┘
        Pinned Memory Work Queue (4096 slots)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system architecture and [docs/DESIGN.md](docs/DESIGN.md) for detailed design decisions.

## Quick Start

### Prerequisites

- CUDA Toolkit 12.x+
- CMake 3.18+
- NVIDIA GPU (sm_75+ recommended; tested on RTX 2060 SUPER)
- Linux (tested on x86_64)

### Build

```bash
mkdir build && cd build
cmake .. -DCUDA_ARCH=75    # Set to your GPU's compute capability
make -j$(nproc)
```

### Run Tests

```bash
cd build
ctest --output-on-failure     # Run all 8 unit tests
./bench_csum                  # CRC32C benchmark
./bench_dispatch              # Full dispatch benchmark (128 threads, ~90s)
```

### Example Usage

```c
#include "gpu_engine.h"

// 1. Allocate GPU buffers BEFORE engine init
void *d_data;
cudaMalloc(&d_data, data_len);
cudaMemcpy(d_data, host_data, data_len, cudaMemcpyHostToDevice);

// 2. Initialize engine (launches persistent kernel)
gpu_engine_t *engine;
gpu_engine_init(&engine);

// 3. Submit work (lock-free, thread-safe)
gpu_work_item_t item = {0};
item.op_type = GPU_OP_CRC32C;
item.data_ptr = d_data;
item.data_len = data_len;

gpu_result_t result;
gpu_engine_submit_and_wait(engine, &item, &result);
printf("CRC32C: 0x%08X\n", result.crc32c_result);

// 4. Shutdown
gpu_engine_fini(engine);
cudaFree(d_data);  // Safe to free AFTER engine shutdown
```

## Performance

Measured on NVIDIA RTX 2060 SUPER (8 GB, 34 SMs, sm_75) with 128 concurrent threads:

| Operation | IOPS | GPU Data Rate† | Latency (p50) |
|-----------|------|----------------|---------------|
| NOP dispatch | 175,819 | — | 714 µs |
| 4KB CRC32C | 171,676 | 0.66 GB/s | 738 µs |
| 1MB CRC32C | 985 | 0.96 GB/s | 102 ms |
| 1MB LZ4 Compress* | 109,725 | 107 GB/s | 238 µs |
| 1MB LZ4 Decompress* | 109,211 | 213 GB/s | 188 µs |
| 4MB LZ4 Compress* | 34,826 | 136 GB/s | 196 µs |
| 4MB LZ4 Decompress* | 34,780 | 272 GB/s | 185 µs |
| 1MB EC 4+2 Encode | 51,962 | 51 GB/s | 761 µs |

> **†** GPU Data Rate = `IOPS × data_size`. Measures GPU-internal (VRAM-to-VRAM) processing throughput, **not** PCIe transfer bandwidth. PCIe 3.0 x16 is ~15.75 GB/s; these numbers are higher because data is pre-resident on the GPU.
>
> **\*** LZ4 running in **memcpy stub mode** (nvCOMPDx not installed). Numbers reflect GPU VRAM copy speed with zero actual compression. With nvCOMPDx, expect real LZ4 ratios but lower throughput due to compute overhead.

## Project Structure

```
persistent-gpu-server/
├── include/              # Public headers
│   ├── gpu_engine.h      # Core API: init, submit, poll, fini
│   ├── gpu_error.h       # Error codes and GPU-side assert macros
│   ├── gpu_csum.h        # CRC32C / SHA256 standalone API
│   ├── gpu_ec.h          # Erasure Coding parity API
│   ├── gpu_mem.h         # GPU memory alloc/free wrappers
│   └── gpu_mempool.h     # Lock-free GPU memory pool
├── src/                  # Implementation
│   ├── gpu_engine.cu     # Persistent kernel + work queue
│   ├── gpu_csum.cu       # CRC32C + SHA256
│   ├── gpu_ec.cu         # EC parity (P + Q) with GF(2^8)
│   ├── gpu_comp.cu       # LZ4 compress/decompress
│   ├── gpu_mem.cu        # Memory utilities
│   └── gpu_mempool.cu    # Lock-free Treiber stack allocator
├── tests/                # 8 unit tests (31 sub-tests)
├── benchmarks/           # Throughput + latency benchmarks
├── examples/             # simple_server.cu
├── docs/                 # Architecture & design documents
│   ├── ARCHITECTURE.md   # System architecture
│   └── DESIGN.md         # Detailed design decisions
└── CMakeLists.txt        # Build system
```

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — System overview, component diagram, memory model, concurrency model
- **[Design](docs/DESIGN.md)** — Design rationale, work queue protocol, synchronization details, algorithm specifications

## API Reference

### Core Engine

| Function | Description |
|----------|-------------|
| `gpu_engine_init(engine_out)` | Initialize engine, launch persistent kernel |
| `gpu_engine_fini(engine)` | Shutdown engine, wait for kernel exit |
| `gpu_engine_submit(engine, item, ticket_out)` | Lock-free submit, returns ticket |
| `gpu_engine_poll(engine, ticket, result)` | Non-blocking poll for completion |
| `gpu_engine_submit_and_wait(engine, item, result)` | Blocking submit with adaptive backoff |
| `gpu_engine_get_metrics(engine, metrics)` | Read telemetry counters |

### Operations

| `gpu_op_type_t` | Description |
|-----------------|-------------|
| `GPU_OP_NOP` | No-op (latency testing) |
| `GPU_OP_CRC32C` | CRC32C checksum |
| `GPU_OP_SHA256` | SHA256 hash |
| `GPU_OP_EC_ENCODE` | EC parity generation (P + Q) |
| `GPU_OP_COMPRESS_LZ4` | LZ4 compression |
| `GPU_OP_DECOMPRESS_LZ4` | LZ4 decompression |

### Memory Pool

| Function | Description |
|----------|-------------|
| `gpu_mempool_create(pool_out, block_size, count)` | Create fixed-size block pool |
| `gpu_mempool_destroy(pool)` | Free pool and backing GPU memory |
| `gpu_mempool_alloc(pool)` | Lock-free allocate a block |
| `gpu_mempool_free(pool, ptr)` | Lock-free free with double-free detection |

## Important Constraints

> ⚠️ **Do NOT call `cudaMalloc()` or `cudaFree()` while the engine is running.** These trigger implicit device synchronization which will deadlock with the persistent kernel. Allocate all GPU buffers before `gpu_engine_init()` or use `gpu_mempool` for dynamic allocation.

## License

See [LICENSE](LICENSE) for details.
