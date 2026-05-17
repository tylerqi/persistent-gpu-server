#!/usr/bin/env bash
#
# run_benchmarks.sh — Run all GPU engine benchmarks and generate a Markdown report.
#
# Usage:
#   ./run_benchmarks.sh              # Build, run, generate report
#   ./run_benchmarks.sh --no-build   # Skip build, just run benchmarks
#   ./run_benchmarks.sh --quick      # Quick mode: skip bench_dispatch (longest)
#
# Output:
#   benchmark_report_YYYYMMDD_HHMMSS.md  (in project root)
#
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$PROJECT_DIR/benchmark_report_${TIMESTAMP}.md"
RAW_DIR="$PROJECT_DIR/benchmark_raw_${TIMESTAMP}"

SKIP_BUILD=0
QUICK_MODE=0

for arg in "$@"; do
    case "$arg" in
        --no-build) SKIP_BUILD=1 ;;
        --quick)    QUICK_MODE=1 ;;
        -h|--help)
            echo "Usage: $0 [--no-build] [--quick]"
            echo "  --no-build  Skip cmake/make, use existing build"
            echo "  --quick     Skip bench_dispatch (saves ~2 min)"
            exit 0
            ;;
    esac
done

mkdir -p "$RAW_DIR"

# ── Helper Functions ─────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_bench() {
    local name="$1"
    local binary="$2"
    local outfile="$RAW_DIR/${name}.txt"

    if [[ ! -x "$BUILD_DIR/$binary" ]]; then
        log "SKIP: $binary not found"
        echo "SKIPPED" > "$outfile"
        return
    fi

    log "Running $name ..."
    local t0
    t0=$(date +%s)

    timeout 600 "$BUILD_DIR/$binary" > "$outfile" 2>&1 || true

    local t1
    t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    log "  Done: $name (${elapsed}s)"
    echo "$elapsed" > "$RAW_DIR/${name}.elapsed"
}

# ── System Information ───────────────────────────────────────────────────

collect_sysinfo() {
    local info="$RAW_DIR/sysinfo.txt"
    {
        echo "=== GPU ==="
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,clocks.current.graphics,clocks.current.memory,pcie.link.gen.current,pcie.link.width.current,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
        echo ""
        echo "=== CUDA ==="
        nvcc --version 2>/dev/null | grep release || echo "nvcc unavailable"
        echo ""
        echo "=== CPU ==="
        lscpu 2>/dev/null | grep -E "Model name|CPU\(s\)|Thread|MHz|Cache" || echo "lscpu unavailable"
        echo ""
        echo "=== Memory ==="
        free -h 2>/dev/null | head -2 || echo "free unavailable"
        echo ""
        echo "=== Kernel ==="
        uname -r 2>/dev/null || echo "unknown"
        echo ""
        echo "=== Build Mode ==="
        if grep -q "USE_NVCOMPDX" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
            grep "BUILD_WITH_NVCOMPDX" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null || echo "nvCOMPDx status unknown"
        else
            echo "nvCOMPDx: OFF (memcpy stubs)"
        fi
    } > "$info"
}

# ── Build ────────────────────────────────────────────────────────────────

if [[ $SKIP_BUILD -eq 0 ]]; then
    log "Building project ..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake "$PROJECT_DIR" -DCUDA_ARCH=75 -DBUILD_WITH_NVCOMPDX=ON > "$RAW_DIR/build.log" 2>&1
    make -j"$(nproc)" >> "$RAW_DIR/build.log" 2>&1
    log "Build complete"
    cd "$PROJECT_DIR"
else
    log "Skipping build (--no-build)"
fi

# ── Collect System Info ──────────────────────────────────────────────────

log "Collecting system information ..."
collect_sysinfo

# ── Run Tests First (sanity check) ───────────────────────────────────────

log "Running unit tests (sanity check) ..."
cd "$BUILD_DIR"
TEST_RESULT=""
if ctest --output-on-failure --timeout 120 > "$RAW_DIR/test_results.txt" 2>&1; then
    TEST_RESULT="✅ All tests passed"
else
    TEST_RESULT="⚠️ Some tests failed (see raw output)"
fi
TESTS_PASSED=$(grep -c "Passed" "$RAW_DIR/test_results.txt" 2>/dev/null || echo "0")
TESTS_TOTAL=$(grep "tests passed" "$RAW_DIR/test_results.txt" 2>/dev/null | grep -oP '\d+ tests' | head -1 || echo "?")
cd "$PROJECT_DIR"

# ── Run Benchmarks ───────────────────────────────────────────────────────

log "Starting benchmarks ..."
BENCH_START=$(date +%s)

run_bench "csum"      "bench_csum"
run_bench "ec"        "bench_ec"
run_bench "compress"  "bench_compress"

if [[ $QUICK_MODE -eq 0 ]]; then
    run_bench "dispatch" "bench_dispatch"
else
    log "SKIP: bench_dispatch (--quick mode)"
    echo "SKIPPED (--quick mode)" > "$RAW_DIR/dispatch.txt"
fi

run_bench "e2e" "bench_e2e"

BENCH_END=$(date +%s)
BENCH_TOTAL=$(( BENCH_END - BENCH_START ))
log "All benchmarks complete (total: ${BENCH_TOTAL}s)"

# ── Generate Report ──────────────────────────────────────────────────────

log "Generating report: $REPORT"

# Parse system info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_PCIE_GEN=$(nvidia-smi --query-gpu=pcie.link.gen.current --format=csv,noheader 2>/dev/null | head -1 || echo "?")
GPU_PCIE_W=$(nvidia-smi --query-gpu=pcie.link.width.current --format=csv,noheader 2>/dev/null | head -1 || echo "?")
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader 2>/dev/null | head -1 || echo "?")
CUDA_VER=$(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' | sed 's/,.*//' || echo "Unknown")
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | sed 's/.*:\s*//' || echo "Unknown")
KERNEL_VER=$(uname -r 2>/dev/null || echo "Unknown")

# Compression mode
COMP_MODE="memcpy stub (no real compression)"
if grep -q "BUILD_WITH_NVCOMPDX:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    COMP_MODE="nvCOMPDx (real LZ4)"
fi

# Parse benchmark results into extractable numbers
extract_csum_table() {
    local f="$RAW_DIR/csum.txt"
    [[ -f "$f" ]] || return
    echo "| Data Size | CPU (GB/s) | GPU (GB/s) | Speedup | Verify |"
    echo "|-----------|------------|------------|---------|--------|"
    grep -E "^\s+[0-9]+B:" "$f" | while read -r line; do
        local size cpu gpu speedup match
        size=$(echo "$line" | grep -oP '^\s*\K[0-9]+B')
        cpu=$(echo "$line" | grep -oP 'CPU=\K[0-9.]+')
        gpu=$(echo "$line" | grep -oP 'GPU=\K[0-9.]+')
        speedup=$(echo "$line" | grep -oP 'speedup=\K[0-9.]+x')
        match=$(echo "$line" | grep -oP '(MATCH|MISMATCH)$')
        echo "| $size | $cpu | $gpu | $speedup | $match |"
    done
}

extract_ec_table() {
    local f="$RAW_DIR/ec.txt"
    [[ -f "$f" ]] || return
    echo "| Config | Cell Size | CPU (GB/s) | GPU (GB/s) | IOPS | Speedup | Verify |"
    echo "|--------|-----------|------------|------------|------|---------|--------|"
    grep -E "^\s+[0-9]+\+1" "$f" | while read -r line; do
        local config size cpu gpu iops speedup match
        config=$(echo "$line" | grep -oP '^\s*\K[0-9]+\+1')
        size=$(echo "$line" | grep -oP '[0-9]+B')
        cpu=$(echo "$line" | grep -oP 'CPU=\K[0-9.]+')
        gpu=$(echo "$line" | grep -oP 'GPU=\K[0-9.]+')
        iops=$(echo "$line" | grep -oP 'IOPS=\K[0-9]+')
        speedup=$(echo "$line" | grep -oP 'speedup=\K[0-9.]+x')
        match=$(echo "$line" | grep -oP '(MATCH|MISMATCH)$')
        echo "| $config | $size | $cpu | $gpu | $iops | $speedup | $match |"
    done
}

extract_compress_section() {
    local f="$RAW_DIR/compress.txt"
    [[ -f "$f" ]] || return
    local pattern="$1"
    echo "| Data Size | Comp (GB/s) | Decomp (GB/s) | Ratio | Verify |"
    echo "|-----------|-------------|---------------|-------|--------|"
    # Get lines between "Pattern: $pattern" and next empty line
    sed -n "/Pattern: $pattern/,/^$/p" "$f" | grep -E "^\s+[0-9]+B" | while read -r line; do
        local size comp decomp ratio match
        size=$(echo "$line" | grep -oP '^\s*\K[0-9]+B')
        # Use word boundary: match ' comp=' not 'decomp='
        comp=$(echo "$line" | grep -oP '\bcomp=\K[0-9.]+')
        decomp=$(echo "$line" | grep -oP 'decomp=\K[0-9.]+')
        ratio=$(echo "$line" | grep -oP 'ratio=\K[0-9.]+x')
        match=$(echo "$line" | grep -oP '(OK|MISMATCH)$')
        echo "| $size | $comp | $decomp | $ratio | $match |"
    done
}

extract_dispatch_latency() {
    local f="$RAW_DIR/dispatch.txt"
    [[ -f "$f" ]] || return
    echo "| Workload | p50 (µs) | p99 (µs) | p99.9 (µs) | avg (µs) |"
    echo "|----------|----------|----------|------------|----------|"
    # Parse latency sections
    local current_workload=""
    while IFS= read -r line; do
        if echo "$line" | grep -qP "^\s+.+ Latency \("; then
            current_workload=$(echo "$line" | sed 's/^\s*//' | sed 's/ Latency (.*//')
        elif echo "$line" | grep -qP "^\s+p50\s+="; then
            local p50 p99 p999 avg
            p50=$(echo "$line" | grep -oP '[0-9.]+(?= µs)')
            # Read next lines for other percentiles
            read -r l90; read -r l99; read -r l999; read -r lmax; read -r lavg
            p99=$(echo "$l99" | grep -oP '[0-9.]+(?= µs)')
            p999=$(echo "$l999" | grep -oP '[0-9.]+(?= µs)')
            avg=$(echo "$lavg" | grep -oP '[0-9.]+(?= µs)')
            echo "| $current_workload | $p50 | $p99 | $p999 | $avg |"
        fi
    done < "$f"
}

extract_dispatch_throughput() {
    local f="$RAW_DIR/dispatch.txt"
    [[ -f "$f" ]] || return
    echo "| Workload | IOPS | GPU Data Rate |"
    echo "|----------|------|---------------|"
    # Find each "<Workload> Throughput (128 threads):" section and extract IOPS + bandwidth
    local current_workload=""
    while IFS= read -r line; do
        if echo "$line" | grep -qP 'Throughput \('; then
            current_workload=$(echo "$line" | sed 's/^\s*//' | sed 's/ Throughput (.*//')
        elif echo "$line" | grep -qP 'Average IOPS'; then
            local iops iops_fmt
            iops=$(echo "$line" | grep -oP '[0-9]+ ops/sec' | grep -oP '^[0-9]+')
            if [[ -n "$iops" ]] && [[ "$iops" -gt 0 ]]; then
                iops_fmt=$(awk "BEGIN { printf \"%.1fK\", $iops / 1000 }")
            else
                iops_fmt="$iops"
            fi
            # Read next line for bandwidth (if present)
            local bw="N/A"
            read -r next_line || true
            if echo "$next_line" | grep -qP 'GPU Data Rate'; then
                bw=$(echo "$next_line" | grep -oP '[0-9.]+\s*GB/s' | head -1)
            fi
            echo "| $current_workload | $iops_fmt | $bw |"
        fi
    done < "$f"
}

# ── Write the Report ─────────────────────────────────────────────────────

cat > "$REPORT" << HEADER
# GPU Storage Engine — Benchmark Report

**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ($(date '+%Y-%m-%d %H:%M:%S %Z') local)
**Duration**: ${BENCH_TOTAL}s total benchmark time

---

## System Configuration

| Component | Details |
|-----------|---------|
| **GPU** | $GPU_NAME ($GPU_MEM) |
| **Driver** | $GPU_DRIVER |
| **PCIe** | Gen$GPU_PCIE_GEN x$GPU_PCIE_W |
| **GPU Temp** | ${GPU_TEMP}°C |
| **GPU Power** | $GPU_POWER |
| **CUDA** | $CUDA_VER |
| **CPU** | $CPU_MODEL |
| **Kernel** | $KERNEL_VER |
| **LZ4 Mode** | $COMP_MODE |

## Test Sanity Check

$TEST_RESULT

---

## 1. CRC32C Throughput (Standalone Kernel)

Measures CRC32C hash throughput at various data sizes, comparing GPU kernel vs CPU.

HEADER

if [[ -f "$RAW_DIR/csum.txt" ]] && ! grep -q "SKIPPED" "$RAW_DIR/csum.txt"; then
    extract_csum_table >> "$REPORT"
    echo "" >> "$REPORT"
    # Extract elapsed
    local_elapsed=$(cat "$RAW_DIR/csum.elapsed" 2>/dev/null || echo "?")
    echo "*Runtime: ${local_elapsed}s*" >> "$REPORT"
else
    echo "*Skipped*" >> "$REPORT"
fi

cat >> "$REPORT" << 'SECT2'

---

## 2. EC Parity Generation (Standalone Kernel)

XOR P-parity generation at various stripe counts and cell sizes.

SECT2

if [[ -f "$RAW_DIR/ec.txt" ]] && ! grep -q "SKIPPED" "$RAW_DIR/ec.txt"; then
    extract_ec_table >> "$REPORT"
    echo "" >> "$REPORT"
    local_elapsed=$(cat "$RAW_DIR/ec.elapsed" 2>/dev/null || echo "?")
    echo "*Runtime: ${local_elapsed}s*" >> "$REPORT"
else
    echo "*Skipped*" >> "$REPORT"
fi

cat >> "$REPORT" << 'SECT3'

---

## 3. LZ4 Compression Throughput (Standalone Kernel)

SECT3

if [[ -f "$RAW_DIR/compress.txt" ]] && ! grep -q "SKIPPED" "$RAW_DIR/compress.txt"; then
    for pattern in zeros repeating random; do
        echo "" >> "$REPORT"
        echo "### Pattern: \`$pattern\`" >> "$REPORT"
        echo "" >> "$REPORT"
        extract_compress_section "$pattern" >> "$REPORT"
    done
    echo "" >> "$REPORT"
    local_elapsed=$(cat "$RAW_DIR/compress.elapsed" 2>/dev/null || echo "?")
    echo "*Runtime: ${local_elapsed}s*" >> "$REPORT"
else
    echo "*Skipped*" >> "$REPORT"
fi

cat >> "$REPORT" << 'SECT4'

---

## 4. Persistent Kernel Dispatch (128 Threads)

End-to-end latency and throughput via the lock-free work queue with 128 concurrent CPU threads.

> **Note**: GPU Data Rate = `IOPS × data_size`. Measures GPU-internal (VRAM-to-VRAM) processing
> throughput, **not** PCIe transfer bandwidth. PCIe 3.0 x16 is ~15.75 GB/s.

SECT4

if [[ -f "$RAW_DIR/dispatch.txt" ]] && ! grep -q "SKIPPED" "$RAW_DIR/dispatch.txt"; then
    echo "### Latency Percentiles" >> "$REPORT"
    echo "" >> "$REPORT"
    extract_dispatch_latency >> "$REPORT"
    echo "" >> "$REPORT"

    echo "### Throughput (Steady-State)" >> "$REPORT"
    echo "" >> "$REPORT"
    extract_dispatch_throughput >> "$REPORT"
    echo "" >> "$REPORT"

    echo "### Per-Second Histogram" >> "$REPORT"
    echo "" >> "$REPORT"
    echo '```' >> "$REPORT"
    grep -E "^\s+\[" "$RAW_DIR/dispatch.txt" >> "$REPORT" || true
    echo '```' >> "$REPORT"
    echo "" >> "$REPORT"
    local_elapsed=$(cat "$RAW_DIR/dispatch.elapsed" 2>/dev/null || echo "?")
    echo "*Runtime: ${local_elapsed}s*" >> "$REPORT"
else
    echo "*Skipped (--quick mode or binary not found)*" >> "$REPORT"
fi

cat >> "$REPORT" << 'SECT5'

---

## 5. End-to-End Full Path (H2D + Compute + D2H)

Measures real-world performance including PCIe transfers (pinned host memory).
This reflects actual DAOS integration latency where data originates from host RAM.

SECT5

if [[ -f "$RAW_DIR/e2e.txt" ]] && ! grep -q "SKIPPED" "$RAW_DIR/e2e.txt"; then
    # Disable errexit for parsing — grep returns 1 on no-match
    set +e
    set +o pipefail

    echo "### Single-Thread Latency" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Workload | p50 (µs) | p99 (µs) | avg (µs) | IOPS | Verify |" >> "$REPORT"
    echo "|----------|----------|----------|----------|------|--------|" >> "$REPORT"
    # Parse each benchmark section
    current_label=""
    p50="" ; p99="" ; avg="" ; iops="" ; verify=""
    while IFS= read -r line; do
        if echo "$line" | grep -qP '^\s+\S.+\(\d+ samples'; then
            # Flush previous
            if [[ -n "$current_label" && -n "$p50" ]]; then
                echo "| $current_label | $p50 | $p99 | $avg | $iops | $verify |" >> "$REPORT"
            fi
            current_label=$(echo "$line" | sed 's/^\s*//' | sed 's/ ([0-9].*//')
            p50=""; p99=""; avg=""; iops=""; verify=""
        elif echo "$line" | grep -qP '^\s+p50'; then
            p50=$(echo "$line" | grep -oP '[0-9.]+(?= µs)')
        elif echo "$line" | grep -qP '^\s+p99\s'; then
            p99=$(echo "$line" | grep -oP '[0-9.]+(?= µs)')
        elif echo "$line" | grep -qP '^\s+avg'; then
            avg=$(echo "$line" | grep -oP '[0-9.]+(?= µs)')
        elif echo "$line" | grep -qP '^\s+IOPS\s+='; then
            iops=$(echo "$line" | grep -oP '[0-9]+ ops' | grep -oP '^[0-9]+' || true)
        elif echo "$line" | grep -qP '^\s+verify'; then
            verify=$(echo "$line" | grep -oP '(MATCH|MISMATCH)')
        fi
    done < "$RAW_DIR/e2e.txt"
    # Flush last
    if [[ -n "$current_label" && -n "$p50" ]]; then
        echo "| $current_label | $p50 | $p99 | $avg | $iops | $verify |" >> "$REPORT"
    fi
    echo "" >> "$REPORT"

    echo "### Multi-Thread Throughput" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Workload | Threads | IOPS | Input Data Rate |" >> "$REPORT"
    echo "|----------|---------|------|-----------------|" >> "$REPORT"
    tp_label="" ; tp_threads="" ; tp_iops="" ; tp_bw=""
    while IFS= read -r line; do
        if echo "$line" | grep -qP 'Throughput \(\d+ threads'; then
            tp_label=$(echo "$line" | sed 's/^\s*//' | sed 's/ Throughput (.*//')
            tp_threads=$(echo "$line" | grep -oP '[0-9]+ threads' | grep -oP '^[0-9]+' || true)
        elif echo "$line" | grep -qP '^\s+IOPS:'; then
            tp_iops=$(echo "$line" | grep -oP '[0-9]+ ops' | grep -oP '^[0-9]+' || true)
            if [[ -n "$tp_iops" ]] && [[ "$tp_iops" -gt 0 ]]; then
                tp_iops=$(awk "BEGIN { printf \"%.1fK\", $tp_iops / 1000 }")
            fi
        elif echo "$line" | grep -qP '^\s+H2D\+Compute:'; then
            tp_bw=$(echo "$line" | grep -oP '[0-9.]+\s*GB/s' | head -1)
            echo "| $tp_label | $tp_threads | $tp_iops | $tp_bw |" >> "$REPORT"
        fi
    done < "$RAW_DIR/e2e.txt"
    echo "" >> "$REPORT"

    # Restore errexit
    set -e
    set -o pipefail

    local_elapsed=$(cat "$RAW_DIR/e2e.elapsed" 2>/dev/null || echo "?")
    echo "*Runtime: ${local_elapsed}s*" >> "$REPORT"
else
    echo "*Skipped*" >> "$REPORT"
fi

cat >> "$REPORT" << SECT6

---

## 6. Raw Output

Full raw output from each benchmark is saved in:
\`$RAW_DIR/\`

| File | Description |
|------|-------------|
| \`csum.txt\` | CRC32C throughput (GPU vs CPU) |
| \`ec.txt\` | EC parity generation throughput |
| \`compress.txt\` | LZ4 compress/decompress throughput |
| \`dispatch.txt\` | Persistent kernel dispatch latency + throughput |
| \`e2e.txt\` | End-to-end full path (H2D + Compute + D2H) |
| \`sysinfo.txt\` | System configuration snapshot |
| \`test_results.txt\` | Unit test results (pre-benchmark sanity check) |
SECT6

log "Report written to: $REPORT"
log "Raw data saved to:  $RAW_DIR/"
echo ""
echo "════════════════════════════════════════════════"
echo "  Benchmark report: $REPORT"
echo "════════════════════════════════════════════════"
