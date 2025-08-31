#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_SIZE 16

// Original kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N) {
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Tiled kernel (shared memory, square tiles)
__global__ void matmul_kernel_tiled(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (tiledRow < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Host wrapper for original kernel
void launch_matmul_kernel(const float* d_A, const float* d_B, float* d_C, int N, dim3 grid, dim3 block) {
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
}

// Host wrapper for tiled kernel
void launch_matmul_kernel_tiled(const float* d_A, const float* d_B, float* d_C, int N, dim3 grid, dim3 block) {
    matmul_kernel_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
}

// Benchmarking harness
void runMatrixMultiplicationKernel(
    void (*kernel_launcher)(const float*, const float*, float*, int, dim3, dim3),
    const char* kernel_name,
    int N, int warmup, int runs
) {
    printf("\n--- Benchmarking %s ---\n", kernel_name);
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Kernel launch config
    dim3 threadsPerBlock, numBlocks;
    // Both kernels now use 2D launch config
    threadsPerBlock = dim3(TILE_SIZE, TILE_SIZE);
    numBlocks = dim3((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    // Warm-up runs
    for (int i = 0; i < warmup; ++i) {
        kernel_launcher(d_A, d_B, d_C, N, numBlocks, threadsPerBlock);
    }
    cudaDeviceSynchronize();

    // Benchmark runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.0f;
    for (int i = 0; i < runs; ++i) {
        cudaEventRecord(start);
        kernel_launcher(d_A, d_B, d_C, N, numBlocks, threadsPerBlock);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_ms += elapsed_ms;
    }
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Average time over %d runs: %f ms\n", runs, total_ms / runs);
    printf("Sample output: C[0]=%f, C[N*N-1]=%f\n", h_C[0], h_C[N*N-1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int N = 1028;
    int warmup = 3;
    int runs = 5;
    runMatrixMultiplicationKernel(launch_matmul_kernel, "Original", N, warmup, runs);
    runMatrixMultiplicationKernel(launch_matmul_kernel_tiled, "Tiled", N, warmup, runs);
    return 0;
}
