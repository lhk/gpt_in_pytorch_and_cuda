#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

// Simple CUDA kernel for row-wise softmax on a 2D matrix
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        // Find max for numerical stability
        float max_val = input[row * cols];
        for (int j = 1; j < cols; ++j) {
            float v = input[row * cols + j];
            if (v > max_val) max_val = v;
        }
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float e = expf(input[row * cols + j] - max_val);
            sum += e;
            output[row * cols + j] = e; // store temporarily
        }
        // Normalize
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
        // 1. Find max (parallel reduction)
        float local_max = -FLT_MAX;
        for (int j = tid; j < cols; j += blockDim.x) {
            float v = input[row * cols + j];
            if (v > local_max) local_max = v;
        }
        smax[tid] = local_max;
        __syncthreads();
        // Reduce within block
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (smax[tid + s] > smax[tid]) smax[tid] = smax[tid + s];
            }
            __syncthreads();
        }
        float max_val = smax[0];

        // 2. Compute exp and sum (parallel reduction)
        float local_sum = 0.0f;
        for (int j = tid; j < cols; j += blockDim.x) {
            float e = expf(input[row * cols + j] - max_val);
            sexp[tid] = e; // store for normalization
            local_sum += e;
            output[row * cols + j] = e; // store temporarily
        }
        smax[tid] = local_sum; // reuse smax for sum
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smax[tid] += smax[tid + s];
            }
            __syncthreads();
        }
        float sum = smax[0];

        // 3. Normalize
        for (int j = tid; j < cols; j += blockDim.x) {
            output[row * cols + j] /= sum;
        }
    }
}

int main() {
    const int rows = 2, cols = 4;
    float h_input[rows * cols] = {1, 2, 3, 4, 2, 4, 6, 8};
    float h_output[rows * cols] = {0};
    float h_output_reduction[rows * cols] = {0};
    float *d_input, *d_output, *d_output_reduction;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output_reduction, rows * cols * sizeof(float));
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Original kernel (1 thread per row)
    softmax_kernel<<<rows, 1>>>(d_input, d_output, rows, cols);
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Reduction kernel (multiple threads per row)
    int threads = 32; // can be tuned
    size_t shmem = 2 * threads * sizeof(float);
    softmax_kernel_reduction<<<rows, threads, shmem>>>(d_input, d_output_reduction, rows, cols);
    cudaMemcpy(h_output_reduction, d_output_reduction, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Softmax output (original):\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", h_output[i * cols + j]);
        }
        printf("\n");
    }

    printf("Softmax output (reduction):\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", h_output_reduction[i * cols + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_reduction);
    return 0;
}
