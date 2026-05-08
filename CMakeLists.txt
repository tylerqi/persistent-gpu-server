cmake_minimum_required(VERSION 3.18)
project(persistent_gpu_server LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Detect GPU architecture ──────────────────────────────────────────────
# Default to sm_75 (Turing, RTX 2060) for dev; override with -DCUDA_ARCH=sm_100 for Blackwell
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "75")
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
message(STATUS "CUDA architecture: sm_${CUDA_ARCH}")

find_package(CUDAToolkit REQUIRED)

# ── CUDA flags ───────────────────────────────────────────────────────────
# --allow-unsupported-compiler: needed for GCC 15+ with nvcc 12.9
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -allow-unsupported-compiler")
# Use LTO code generation for nvCOMPDx compatibility:
# code=lto_<arch> generates NVVM/LTO-IR instead of SASS, which is required
# to link against nvCOMPDx's LTO-IR .a library. nvlink performs final
# optimization across all LTO-IR objects.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=lto_${CUDA_ARCH}")
# Also generate native SASS for direct execution (needed for non-LTO paths)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
# Disable CMake's auto-gencode since we handle it manually
set(CMAKE_CUDA_ARCHITECTURES OFF)

# ── Dependencies ─────────────────────────────────────────────────────────
# Search for MathDx (nvCOMPDx) for device-side compression
# Try multiple candidate paths for MathDx installation
set(MATHDX_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/mathdx"
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06"
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06"
    "${CMAKE_CURRENT_SOURCE_DIR}/_deps/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12"
)
set(MATHDX_DIR "" CACHE PATH "Path to NVIDIA MathDx (auto-detected if empty)")
set(MATHDX_FOUND FALSE)
foreach(CANDIDATE ${MATHDX_CANDIDATES})
    if(EXISTS "${CANDIDATE}/include/nvcompdx.hpp")
        set(MATHDX_DIR "${CANDIDATE}")
        set(MATHDX_FOUND TRUE)
        break()
    endif()
endforeach()
if(MATHDX_FOUND)
    message(STATUS "Found MathDx (nvCOMPDx) at ${MATHDX_DIR}")
    add_compile_definitions(USE_NVCOMPDX)
    add_compile_definitions(NVCOMPDX_OLDER_CTK_SUPPORT)  # Allow CUDA 12.4 (officially requires 12.6+)
    include_directories(${MATHDX_DIR}/include)
    # CUTLASS and commonDx are bundled as external dependencies
    if(EXISTS "${MATHDX_DIR}/external/cutlass/include")
        include_directories(${MATHDX_DIR}/external/cutlass/include)
    endif()
    # nvCOMPDx is distributed as LTO-IR (.a / .fatbin); prefer .fatbin for cross-version compat
    set(NVCOMPDX_FATBIN "${MATHDX_DIR}/lib/libnvcompdx.fatbin")
    find_library(NVCOMPDX_LIB NAMES nvcompdx PATHS "${MATHDX_DIR}/lib" "${MATHDX_DIR}/lib64")
    if(EXISTS "${NVCOMPDX_FATBIN}")
        list(APPEND EXTRA_LIBS ${NVCOMPDX_FATBIN})
        message(STATUS "  libnvcompdx.fatbin: ${NVCOMPDX_FATBIN}")
    elseif(NVCOMPDX_LIB)
        list(APPEND EXTRA_LIBS ${NVCOMPDX_LIB})
        message(STATUS "  libnvcompdx: ${NVCOMPDX_LIB}")
    endif()
else()
    message(WARNING "MathDx (nvCOMPDx) not found. Falling back to uncompressed stubs. Run scripts/download_nvcompdx.sh or set MATHDX_DIR.")
endif()

# ── Library: libdaos_gpu_engine ──────────────────────────────────────────
# Built as STATIC to avoid CUDA fatbin registration conflicts between the
# library and test executables. With a shared library, each .cu executable
# registers its own (empty) fatbin which overrides the library's kernel
# registration, causing "named symbol not found" on kernel launch.
add_library(daos_gpu_engine STATIC
    src/gpu_engine.cu
    src/gpu_csum.cu
    src/gpu_ec.cu
    src/gpu_mem.cu
    src/gpu_mempool.cu
    src/gpu_comp.cu
)
target_include_directories(daos_gpu_engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(daos_gpu_engine PRIVATE CUDA::cudart pthread ${EXTRA_LIBS})
set_target_properties(daos_gpu_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# Device link must use -dlto to resolve LTO-IR objects from nvCOMPDx
target_link_options(daos_gpu_engine PRIVATE $<DEVICE_LINK:-dlto>)

# ── Tests ────────────────────────────────────────────────────────────────
enable_testing()

# Simple test runner (no external dependency like gtest — keeps it lean)
function(add_gpu_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE daos_gpu_engine CUDA::cudart)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(${TEST_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 60)
endfunction()

add_gpu_test(test_engine    tests/test_engine.cu)
add_gpu_test(test_csum      tests/test_csum.cu)
add_gpu_test(test_ec        tests/test_ec.cu)
add_gpu_test(test_error     tests/test_error.cu)
add_gpu_test(test_latency   tests/test_latency.cu)
set_tests_properties(test_latency PROPERTIES TIMEOUT 180)
add_gpu_test(test_compress  tests/test_compress.cu)
add_gpu_test(test_verify_all tests/test_verify_all.cu)
add_gpu_test(test_mempool    tests/test_mempool.cu)

# ── Benchmarks ───────────────────────────────────────────────────────────
add_executable(bench_csum     benchmarks/bench_csum.cu)
target_link_libraries(bench_csum PRIVATE daos_gpu_engine CUDA::cudart)
target_include_directories(bench_csum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(bench_dispatch benchmarks/bench_dispatch.cu)
target_link_libraries(bench_dispatch PRIVATE daos_gpu_engine CUDA::cudart)
target_include_directories(bench_dispatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Example ──────────────────────────────────────────────────────────────
add_executable(simple_server  examples/simple_server.cu)
target_link_libraries(simple_server PRIVATE daos_gpu_engine CUDA::cudart)
target_include_directories(simple_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── Install ──────────────────────────────────────────────────────────────
install(TARGETS daos_gpu_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
