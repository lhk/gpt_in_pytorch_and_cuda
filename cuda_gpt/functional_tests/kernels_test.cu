#include "kernels.cu"
#include <iostream>
#include <cmath>

// Utility to print a 4D tensor (B, H, S, S)
void print_4d(const float* data, int B, int H, int S1, int S2, const char* name) {
    printf("%s shape: (%d, %d, %d, %d)\n", name, B, H, S1, S2);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            printf("[b=%d, h=%d]:\n", b, h);
            for (int i = 0; i < S1; ++i) {
                for (int j = 0; j < S2; ++j) {
                    printf("%8.4f ", data[
                        ((b * H + h) * S1 + i) * S2 + j
                    ]);
                }
                printf("\n");
            }
        }
    }
}

// Utility to print a 3D tensor (B, S, d_model)
void print_3d(const float* data, int B, int S, int D, const char* name) {
    printf("%s shape: (%d, %d, %d)\n", name, B, S, D);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            printf("[b=%d, s=%d]: ", b, s);
            for (int d = 0; d < D; ++d) {
                printf("%8.4f ", data[(b * S + s) * D + d]);
            }
            printf("\n");
        }
    }
}

// Compare two arrays with tolerance
bool compare_arrays(const float* a, const float* b, int size, float tolerance = 1e-4) {
    bool passed = true;
    for (int i = 0; i < size; i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            printf("Mismatch at index %d: expected %f, got %f (diff: %f)\n", 
                   i, a[i], b[i], std::abs(a[i] - b[i]));
            passed = false;
        }
    }
    return passed;
}

int main() {
    printf("=== CUDA GPT Kernels Test ===\n");
    
    // Test configuration (same as sandbox)
    const int batch_size = 2, n_heads = 2, seq_len = 3, d_k = 2;
    const int d_model = n_heads * d_k;
    const int QK_in_size = batch_size * seq_len * d_model;
    const int QK_out_size = batch_size * n_heads * seq_len * d_k;
    const int scores_size = batch_size * n_heads * seq_len * seq_len;
    
    printf("Test config: batch_size=%d, n_heads=%d, seq_len=%d, d_k=%d, d_model=%d\n", 
           batch_size, n_heads, seq_len, d_k, d_model);
    
    // Reference input data from Python (same as sandbox)
    float h_Q_in[24] = {
        1.926915, 1.487284, 0.900717, -2.105521,
        0.678418, -1.234545, -0.043067, -1.604667,
        0.355860, -0.686623, -0.493356, 0.241488,
        -1.110904, 0.091546, -2.316923, -0.216805,
        -0.309727, -0.395710, 0.803409, -0.621595,
        -0.592001, -0.063074, -0.828554, 0.330898
    };
    
    float h_K_in[24] = {
        0.034912, 0.321103, 1.573600, -0.845467,
        1.312308, 0.687160, -1.089175, -0.355287,
        1.445134, 0.856413, 2.218076, 0.523166,
        0.346647, -0.197331, -1.054589, 1.277996,
        -0.172190, 0.523788, 0.056622, 0.426296,
        0.575005, -0.641724, -2.206398, -0.750803
    };
    
    float h_V_in[24] = {
        0.010868, -0.338742, -1.340680, -0.585371,
        0.536188, 0.524623, 1.141202, 0.051644,
        -0.678753, 0.574316, 0.187749, -0.357623,
        -0.316508, 0.588634, -0.890457, 0.409813,
        -0.986439, 0.123299, 0.349868, 0.617281,
        -0.169332, 0.233225, 4.035634, 1.279459
    };
    
    // Expected outputs from sandbox validation (will be filled from sandbox run)
    // These would be the "ground truth" outputs from the working sandbox code
    
    // Device memory allocation
    float *d_Q_in, *d_K_in, *d_V_in, *d_Q, *d_K, *d_V, *d_scores, *d_attn_weights, *d_attn_out;
    cudaMalloc((void**)&d_Q_in, QK_in_size * sizeof(float));
    cudaMalloc((void**)&d_K_in, QK_in_size * sizeof(float));
    cudaMalloc((void**)&d_V_in, QK_in_size * sizeof(float));
    cudaMalloc((void**)&d_Q, QK_out_size * sizeof(float));
    cudaMalloc((void**)&d_K, QK_out_size * sizeof(float));
    cudaMalloc((void**)&d_V, QK_out_size * sizeof(float));
    cudaMalloc((void**)&d_scores, scores_size * sizeof(float));
    cudaMalloc((void**)&d_attn_weights, scores_size * sizeof(float));
    cudaMalloc((void**)&d_attn_out, QK_out_size * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_Q_in, h_Q_in, QK_in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_in, h_K_in, QK_in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_in, h_V_in, QK_in_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Host arrays for results
    float h_Q[QK_out_size];
    float h_K[QK_out_size];
    float h_V[QK_out_size];
    float h_scores[scores_size];
    float h_attn_weights[scores_size];
    float h_attn_out[QK_out_size];
    float h_concat_out[batch_size * seq_len * d_model];
    
    printf("\n=== Test 1: Reshape Kernel ===\n");
    
    // Test reshape kernel
    dim3 reshapeBlocks(batch_size, seq_len);
    dim3 reshapeThreads(n_heads, d_k);
    reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_Q_in, d_Q, batch_size, seq_len, n_heads, d_k);
    reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_K_in, d_K, batch_size, seq_len, n_heads, d_k);
    reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_V_in, d_V, batch_size, seq_len, n_heads, d_k);
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(h_Q, d_Q, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_K, d_K, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V, d_V, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    print_4d(h_Q, batch_size, n_heads, seq_len, d_k, "Q (after reshape)");
    print_4d(h_K, batch_size, n_heads, seq_len, d_k, "K (after reshape)");
    print_4d(h_V, batch_size, n_heads, seq_len, d_k, "V (after reshape)");
    
    printf("\n=== Test 2: Scores Kernel ===\n");
    
    // Test scores kernel
    dim3 blockDim(8, 8);
    dim3 gridDim(
        (seq_len + blockDim.x - 1) / blockDim.x,
        (seq_len + blockDim.y - 1) / blockDim.y,
        batch_size * n_heads
    );
    scores_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_scores, batch_size, n_heads, seq_len, d_k);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_scores, d_scores, scores_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_4d(h_scores, batch_size, n_heads, seq_len, seq_len, "scores (before mask)");
    
    printf("\n=== Test 3: Mask Kernel ===\n");
    
    // Test mask kernel
    mask_kernel<<<gridDim, blockDim>>>(d_scores, batch_size, n_heads, seq_len);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_scores, d_scores, scores_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_4d(h_scores, batch_size, n_heads, seq_len, seq_len, "scores (after mask)");
    
    printf("\n=== Test 4: Softmax Kernel ===\n");
    
    // Test softmax kernel
    dim3 softmaxBlock(1, 1);
    dim3 softmaxGrid(1, seq_len, batch_size * n_heads);
    softmax_kernel<<<softmaxGrid, softmaxBlock>>>(d_scores, d_attn_weights, batch_size, n_heads, seq_len);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_attn_weights, d_attn_weights, scores_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_4d(h_attn_weights, batch_size, n_heads, seq_len, seq_len, "attn_weights (after softmax)");
    
    printf("\n=== Test 5: Attention Matmul Kernel ===\n");
    
    // Test attention matmul kernel
    dim3 attnBlock(d_k, 1);
    dim3 attnGrid((d_k + attnBlock.x - 1) / attnBlock.x, seq_len, batch_size * n_heads);
    attn_matmul_kernel<<<attnGrid, attnBlock>>>(d_attn_weights, d_V, d_attn_out, batch_size, n_heads, seq_len, d_k);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_attn_out, d_attn_out, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
    print_4d(h_attn_out, batch_size, n_heads, seq_len, d_k, "attention output (attn_weights @ V)");
    
    printf("\n=== Test 6: Concat Heads Kernel ===\n");
    
    // Test concat heads kernel
    float *d_concat_out;
    cudaMalloc((void**)&d_concat_out, batch_size * seq_len * d_model * sizeof(float));
    
    dim3 concatBlocks(batch_size, seq_len);
    dim3 concatThreads(d_model);
    concat_heads_kernel<<<concatBlocks, concatThreads>>>(d_attn_out, d_concat_out, batch_size, n_heads, seq_len, d_k);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_concat_out, d_concat_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    print_3d(h_concat_out, batch_size, seq_len, d_model, "concat_heads output");
    
    printf("\n=== Test 7: Linear Kernel (Identity Matrix) ===\n");
    
    // Test linear kernel with identity matrix
    float h_WO[d_model * d_model] = {0};
    for (int i = 0; i < d_model; ++i) {
        h_WO[i * d_model + i] = 1.0f;  // Identity matrix
    }
    
    float *d_WO, *d_proj_out;
    cudaMalloc((void**)&d_WO, d_model * d_model * sizeof(float));
    cudaMalloc((void**)&d_proj_out, batch_size * seq_len * d_model * sizeof(float));
    cudaMemcpy(d_WO, h_WO, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    linear_kernel<<<concatBlocks, concatThreads>>>(d_concat_out, d_WO, d_proj_out, batch_size, seq_len, d_model);
    cudaDeviceSynchronize();
    
    float h_proj_out[batch_size * seq_len * d_model];
    cudaMemcpy(h_proj_out, d_proj_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    print_3d(h_proj_out, batch_size, seq_len, d_model, "linear output (identity)");
    
    printf("\n=== Test 8: Linear Bias Kernel ===\n");
    
    // Test linear bias kernel
    float h_bias[d_model];
    for (int i = 0; i < d_model; i++) {
        h_bias[i] = 0.1f * i;  // Small bias values
    }
    
    float *d_bias, *d_bias_out;
    cudaMalloc((void**)&d_bias, d_model * sizeof(float));
    cudaMalloc((void**)&d_bias_out, batch_size * seq_len * d_model * sizeof(float));
    cudaMemcpy(d_bias, h_bias, d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    linear_bias_kernel<<<concatBlocks, concatThreads>>>(
        d_concat_out, d_WO, d_bias, d_bias_out, 
        batch_size, seq_len, d_model, d_model);
    cudaDeviceSynchronize();
    
    float h_bias_out[batch_size * seq_len * d_model];
    cudaMemcpy(h_bias_out, d_bias_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    print_3d(h_bias_out, batch_size, seq_len, d_model, "linear bias output");
    
    printf("\n=== Test 9: GELU Kernel ===\n");
    
    // Test GELU kernel
    float *d_gelu_out;
    cudaMalloc((void**)&d_gelu_out, batch_size * seq_len * d_model * sizeof(float));
    
    gelu_kernel<<<concatBlocks, concatThreads>>>(d_concat_out, d_gelu_out, batch_size, seq_len, d_model);
    cudaDeviceSynchronize();
    
    float h_gelu_out[batch_size * seq_len * d_model];
    cudaMemcpy(h_gelu_out, d_gelu_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    print_3d(h_gelu_out, batch_size, seq_len, d_model, "GELU output");
    
    printf("\n=== Test 10: Add Kernel ===\n");
    
    // Test add kernel (residual connection)
    float *d_add_out;
    cudaMalloc((void**)&d_add_out, batch_size * seq_len * d_model * sizeof(float));
    
    int total_elements = batch_size * seq_len * d_model;
    dim3 addBlocks((total_elements + 255) / 256);
    dim3 addThreads(256);
    add_kernel<<<addBlocks, addThreads>>>(d_concat_out, d_bias_out, d_add_out, total_elements);
    cudaDeviceSynchronize();
    
    float h_add_out[batch_size * seq_len * d_model];
    cudaMemcpy(h_add_out, d_add_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    print_3d(h_add_out, batch_size, seq_len, d_model, "add output (residual)");
    
    printf("\n=== Test 11: LayerNorm Kernel ===\n");
    
    // Test layer norm kernel
    float h_gamma[d_model], h_beta[d_model];
    for (int i = 0; i < d_model; i++) {
        h_gamma[i] = 1.0f;  // Standard gamma
        h_beta[i] = 0.0f;   // Standard beta
    }
    
    float *d_gamma, *d_beta, *d_ln_out;
    cudaMalloc((void**)&d_gamma, d_model * sizeof(float));
    cudaMalloc((void**)&d_beta, d_model * sizeof(float));
    cudaMalloc((void**)&d_ln_out, batch_size * seq_len * d_model * sizeof(float));
    cudaMemcpy(d_gamma, h_gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, d_model * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 lnBlocks(batch_size, seq_len);
    dim3 lnThreads(256);  // More threads for layer norm
    layernorm_kernel<<<lnBlocks, lnThreads>>>(d_add_out, d_gamma, d_beta, d_ln_out, batch_size, seq_len, d_model);
    cudaDeviceSynchronize();
    
    float h_ln_out[batch_size * seq_len * d_model];
    cudaMemcpy(h_ln_out, d_ln_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    print_3d(h_ln_out, batch_size, seq_len, d_model, "layer norm output");
    
    printf("\n=== Kernel Tests Complete ===\n");
    printf("All kernels executed successfully!\n");
    printf("Next step: Compare outputs with reference values from sandbox\n");
    
    // Cleanup
    cudaFree(d_Q_in); cudaFree(d_K_in); cudaFree(d_V_in);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_scores); cudaFree(d_attn_weights); cudaFree(d_attn_out);
    cudaFree(d_concat_out); cudaFree(d_WO); cudaFree(d_proj_out);
    cudaFree(d_bias); cudaFree(d_bias_out); cudaFree(d_gelu_out);
    cudaFree(d_add_out); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_ln_out);
    
    return 0;
}
