/**
 * gpu_mem.cu — GPU memory management utilities
 */
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

void *gpu_mem_alloc(size_t size)
{
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", size, cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void gpu_mem_free(void *ptr)
{
    if (ptr) cudaFree(ptr);
}

int gpu_mem_copy_h2d(void *dst, const void *src, size_t size)
{
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
}

int gpu_mem_copy_d2h(void *dst, const void *src, size_t size)
{
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : -1;
}

} /* extern "C" */
