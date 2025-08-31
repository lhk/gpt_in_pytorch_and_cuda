#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

// Simple CUDA kernel for row-wise softmax on a 2D matrix
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float max_val = input[row * cols];
        for (int j = 1; j < cols; ++j) {
            float v = input[row * cols + j];
            if (v > max_val) max_val = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float e = expf(input[row * cols + j] - max_val);
            sum += e;
            output[row * cols + j] = e;
        }
        for (int j = 0; j < cols; ++j) {
            output[row * cols + j] /= sum;
        }
    }
}

// Parallel reduction softmax kernel (one block per row, multiple threads per row)
__global__ void softmax_kernel_reduction(const float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];
    float* smax = sdata;
    float* sexp = sdata + blockDim.x;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row < rows) {
        float local_max = -FLT_MAX;
        for (int j = tid; j < cols; j += blockDim.x) {
            float v = input[row * cols + j];
            if (v > local_max) local_max = v;
        }
        smax[tid] = local_max;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
            }
            __syncthreads();
        }
        float max_val = smax[0];
        float local_sum = 0.0f;
        for (int j = tid; j < cols; j += blockDim.x) {
            float e = expf(input[row * cols + j] - max_val);
            sexp[tid] = e;
            local_sum += e;
            output[row * cols + j] = e;
        }
        smax[tid] = local_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smax[tid] += smax[tid + s];
            }
            __syncthreads();
        }
        float sum = smax[0];
        for (int j = tid; j < cols; j += blockDim.x) {
            output[row * cols + j] /= sum;
        }
    }
}

// Host wrapper for original kernel
void launch_softmax_kernel(const float* d_input, float* d_output, int rows, int cols, int threads, size_t shmem) {
    softmax_kernel<<<rows, 1>>>(d_input, d_output, rows, cols);
}

// Host wrapper for reduction kernel
void launch_softmax_kernel_reduction(const float* d_input, float* d_output, int rows, int cols, int threads, size_t shmem) {
    softmax_kernel_reduction<<<rows, threads, shmem>>>(d_input, d_output, rows, cols);
}

// Benchmarking harness
void runSoftmaxKernel(
    void (*kernel_launcher)(const float*, float*, int, int, int, size_t),
    const char* kernel_name,
    int rows, int cols, int warmup, int runs, int threads, size_t shmem
) {
    printf("\n--- Benchmarking %s ---\n", kernel_name);
    size_t bytes = rows * cols * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = (float)(i % 100) / 10.0f;
    }
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up runs
    for (int i = 0; i < warmup; ++i) {
        kernel_launcher(d_input, d_output, rows, cols, threads, shmem);
    }
    cudaDeviceSynchronize();

    // Benchmark runs
    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        kernel_launcher(d_input, d_output, rows, cols, threads, shmem);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_ms += elapsed_ms;
    }
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    printf("Average time over %d runs: %f ms\n", runs, total_ms / runs);
    printf("Sample output: softmax[0]=%f, softmax[last]=%f\n", h_output[0], h_output[rows*cols-1]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int rows = 1024;
    int cols = 512;
    int warmup = 3;
    int runs = 5;
    int threads = 128;
    size_t shmem = 2 * threads * sizeof(float);
    runSoftmaxKernel(launch_softmax_kernel, "Original", rows, cols, warmup, runs, 1, 0);
    runSoftmaxKernel(launch_softmax_kernel_reduction, "Reduction", rows, cols, warmup, runs, threads, shmem);
    return 0;
}
