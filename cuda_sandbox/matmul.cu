#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_SIZE 16

__global__ void matmul_kernel_tiled(const float* A, const float* B, float* C, int N) {
    // Each block computes one row of C (blockIdx.y = row)
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_A_row[TILE_SIZE];
    float sum = 0.0f;
    if (row < N && col < N) {
        // Each thread loads one element of the row of A into shared memory
        if (threadIdx.x < N) {
            shared_A_row[threadIdx.x] = A[row * N + threadIdx.x];
        }
        __syncthreads();
        for (int k = 0; k < N; ++k) {
            sum += shared_A_row[k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

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

int main() {
    cudaEvent_t start, stop;
    float elapsed_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int N = 2;
    float h_A[N*N] = {1, 2, 3, 4}; // 2x2 matrix
    float h_B[N*N] = {5, 6, 7, 8}; // 2x2 matrix
    float h_C[N*N] = {0};
    float h_C_tiled[N*N] = {0};

    float *d_A, *d_B, *d_C, *d_C_tiled;
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float));
    cudaMalloc((void**)&d_C_tiled, N*N*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);


    // For original kernel (2D launch)
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    // Time original kernel
    cudaEventRecord(start);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Original kernel time: %f ms\n", elapsed_ms);

    // For tiled kernel: 1 block per row, 1D threads for columns
    dim3 threadsPerBlockTiled(TILE_SIZE);
    dim3 numBlocksTiled( (N+TILE_SIZE-1)/TILE_SIZE, N ); // N blocks in y, each for a row

    // Time tiled kernel
    cudaEventRecord(start);
    matmul_kernel_tiled<<<numBlocksTiled, threadsPerBlockTiled>>>(d_A, d_B, d_C_tiled, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaMemcpy(h_C_tiled, d_C_tiled, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Tiled kernel time: %f ms\n", elapsed_ms);

    printf("Result (original):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%f ", h_C[i*N + j]);
        printf("\n");
    }

    printf("\nResult (tiled):\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            printf("%f ", h_C_tiled[i*N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_tiled);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}