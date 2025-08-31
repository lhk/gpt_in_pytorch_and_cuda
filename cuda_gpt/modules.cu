#include "kernels.cu"
#include <iostream>
#include <vector>
#include <fstream>

// Forward declarations
__global__ void reshape_for_mhsa_kernel(const float* src, float* dst, int batch_size, int seq_len, int n_heads, int d_k);
__global__ void scores_kernel(const float* Q, const float* K, float* scores, int batch_size, int n_heads, int seq_len, int d_k);
__global__ void attn_matmul_kernel(const float* attn_weights, const float* V, float* out, int batch_size, int n_heads, int seq_len, int d_k);
__global__ void mask_kernel(float* scores, int batch_size, int n_heads, int seq_len);
__global__ void softmax_kernel(const float* scores, float* attn_weights, int batch_size, int n_heads, int seq_len);
__global__ void concat_heads_kernel(const float* src, float* dst, int batch_size, int n_heads, int seq_len, int d_k);
__global__ void linear_kernel(const float* src, const float* W, float* dst, int batch_size, int seq_len, int d_model);
__global__ void linear_bias_kernel(const float* src, const float* W, const float* bias, float* dst, int batch_size, int seq_len, int d_in, int d_out);
__global__ void gelu_kernel(const float* src, float* dst, int batch_size, int seq_len, int d_model);
__global__ void add_kernel(const float* src1, const float* src2, float* dst, int total_elements);
__global__ void layernorm_kernel(const float* src, const float* gamma, const float* beta, float* dst, int batch_size, int seq_len, int d_model, float eps);

// Layer Normalization Class
class LayerNorm {
private:
    float* d_gamma;
    float* d_beta;
    int d_model;
    float eps;

public:
    LayerNorm(int d_model, float eps = 1e-5) : d_model(d_model), eps(eps) {
        cudaMalloc(&d_gamma, d_model * sizeof(float));
        cudaMalloc(&d_beta, d_model * sizeof(float));
        
        // Initialize gamma to 1.0 and beta to 0.0
        std::vector<float> gamma(d_model, 1.0f);
        std::vector<float> beta(d_model, 0.0f);
        cudaMemcpy(d_gamma, gamma.data(), d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta.data(), d_model * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ~LayerNorm() {
        cudaFree(d_gamma);
        cudaFree(d_beta);
    }
    
    void forward(const float* src, float* dst, int batch_size, int seq_len) {
        dim3 block(d_model);  // One thread per feature dimension
        dim3 grid(batch_size, seq_len);  // 2D grid: batch x sequence
        layernorm_kernel<<<grid, block>>>(src, d_gamma, d_beta, dst, batch_size, seq_len, d_model, eps);
        cudaDeviceSynchronize();
    }
    
    void load_weights(const float* gamma_weights, const float* beta_weights) {
        cudaMemcpy(d_gamma, gamma_weights, d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, beta_weights, d_model * sizeof(float), cudaMemcpyHostToDevice);
    }
};

class MultiHeadSelfAttention {
private:
    int d_model, n_heads, d_k, max_seq_len;
    
    // Device weight pointers
    float *d_WQ, *d_WK, *d_WV, *d_WO;
    float *d_bias_WO;  // Only WO has bias in PyTorch
    
    // Temporary device buffers (allocated once, reused)
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_attn_weights;
    float *d_attn_out, *d_concat_out;
    
public:
    MultiHeadSelfAttention(int d_model, int n_heads, int max_seq_len, 
                          const float* WQ_weights, const float* WK_weights, 
                          const float* WV_weights, const float* WO_weights,
                          const float* WO_bias) 
        : d_model(d_model), n_heads(n_heads), d_k(d_model / n_heads), max_seq_len(max_seq_len) {
        
        // Allocate device memory for weights
        cudaMalloc(&d_WQ, d_model * d_model * sizeof(float));
        cudaMalloc(&d_WK, d_model * d_model * sizeof(float));
        cudaMalloc(&d_WV, d_model * d_model * sizeof(float));
        cudaMalloc(&d_WO, d_model * d_model * sizeof(float));
        cudaMalloc(&d_bias_WO, d_model * sizeof(float));
        
        // Copy weights to device
        cudaMemcpy(d_WQ, WQ_weights, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_WK, WK_weights, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_WV, WV_weights, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_WO, WO_weights, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias_WO, WO_bias, d_model * sizeof(float), cudaMemcpyHostToDevice);
        
        // Allocate temporary buffers (sized for max batch and sequence length)
        int max_batch = 16; // Configurable
        cudaMalloc(&d_Q, max_batch * n_heads * max_seq_len * d_k * sizeof(float));
        cudaMalloc(&d_K, max_batch * n_heads * max_seq_len * d_k * sizeof(float));
        cudaMalloc(&d_V, max_batch * n_heads * max_seq_len * d_k * sizeof(float));
        cudaMalloc(&d_scores, max_batch * n_heads * max_seq_len * max_seq_len * sizeof(float));
        cudaMalloc(&d_attn_weights, max_batch * n_heads * max_seq_len * max_seq_len * sizeof(float));
        cudaMalloc(&d_attn_out, max_batch * n_heads * max_seq_len * d_k * sizeof(float));
        cudaMalloc(&d_concat_out, max_batch * max_seq_len * d_model * sizeof(float));
    }
    
    ~MultiHeadSelfAttention() {
        // Free all device memory
        cudaFree(d_WQ); cudaFree(d_WK); cudaFree(d_WV); cudaFree(d_WO); cudaFree(d_bias_WO);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        cudaFree(d_scores); cudaFree(d_attn_weights); cudaFree(d_attn_out); cudaFree(d_concat_out);
    }
    
    // Forward pass: input [B, S, d_model] -> output [B, S, d_model]
    void forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
        // Linear projections Q, K, V
        dim3 linearBlocks(batch_size, seq_len);
        dim3 linearThreads(d_model);
        
        // Apply WQ, WK, WV (no bias for Q, K, V in standard transformer)
        linear_kernel<<<linearBlocks, linearThreads>>>(d_input, d_WQ, d_Q, batch_size, seq_len, d_model);
        linear_kernel<<<linearBlocks, linearThreads>>>(d_input, d_WK, d_K, batch_size, seq_len, d_model);
        linear_kernel<<<linearBlocks, linearThreads>>>(d_input, d_WV, d_V, batch_size, seq_len, d_model);
        cudaDeviceSynchronize();
        
        // Reshape for multi-head attention: (B, S, d_model) -> (B, n_heads, S, d_k)
        dim3 reshapeBlocks(batch_size, seq_len);
        dim3 reshapeThreads(n_heads, d_k);
        reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_Q, d_Q, batch_size, seq_len, n_heads, d_k);
        reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_K, d_K, batch_size, seq_len, n_heads, d_k);
        reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_V, d_V, batch_size, seq_len, n_heads, d_k);
        cudaDeviceSynchronize();
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        dim3 scoresBlocks((seq_len + 7) / 8, (seq_len + 7) / 8, batch_size * n_heads);
        dim3 scoresThreads(8, 8);
        scores_kernel<<<scoresBlocks, scoresThreads>>>(d_Q, d_K, d_scores, batch_size, n_heads, seq_len, d_k);
        cudaDeviceSynchronize();
        
        // Apply causal mask
        mask_kernel<<<scoresBlocks, scoresThreads>>>(d_scores, batch_size, n_heads, seq_len);
        cudaDeviceSynchronize();
        
        // Softmax
        dim3 softmaxBlocks(1, seq_len, batch_size * n_heads);
        dim3 softmaxThreads(1, 1);
        softmax_kernel<<<softmaxBlocks, softmaxThreads>>>(d_scores, d_attn_weights, batch_size, n_heads, seq_len);
        cudaDeviceSynchronize();
        
        // Attention output: attn_weights @ V
        dim3 attnBlocks((d_k + 1) / 2, seq_len, batch_size * n_heads);
        dim3 attnThreads(2, 1);
        attn_matmul_kernel<<<attnBlocks, attnThreads>>>(d_attn_weights, d_V, d_attn_out, batch_size, n_heads, seq_len, d_k);
        cudaDeviceSynchronize();
        
        // Concatenate heads: (B, n_heads, S, d_k) -> (B, S, d_model)
        dim3 concatBlocks(batch_size, seq_len);
        dim3 concatThreads(d_model);
        concat_heads_kernel<<<concatBlocks, concatThreads>>>(d_attn_out, d_concat_out, batch_size, n_heads, seq_len, d_k);
        cudaDeviceSynchronize();
        
        // Output projection with bias
        linear_bias_kernel<<<linearBlocks, linearThreads>>>(d_concat_out, d_WO, d_bias_WO, d_output, batch_size, seq_len, d_model, d_model);
        cudaDeviceSynchronize();
    }
};

class MLP {
private:
    int d_model, d_ff;
    
    // Device weight pointers
    float *d_W1, *d_bias1;  // First linear layer
    float *d_W2, *d_bias2;  // Second linear layer
    
    // Temporary device buffer
    float *d_hidden;
    
public:
    MLP(int d_model, int d_ff, int max_seq_len,
        const float* W1_weights, const float* bias1,
        const float* W2_weights, const float* bias2)
        : d_model(d_model), d_ff(d_ff) {
        
        // Allocate device memory for weights
        cudaMalloc(&d_W1, d_model * d_ff * sizeof(float));
        cudaMalloc(&d_bias1, d_ff * sizeof(float));
        cudaMalloc(&d_W2, d_ff * d_model * sizeof(float));
        cudaMalloc(&d_bias2, d_model * sizeof(float));
        
        // Copy weights to device
        cudaMemcpy(d_W1, W1_weights, d_model * d_ff * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias1, bias1, d_ff * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, W2_weights, d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias2, bias2, d_model * sizeof(float), cudaMemcpyHostToDevice);
        
        // Allocate temporary buffer
        int max_batch = 16;
        cudaMalloc(&d_hidden, max_batch * max_seq_len * d_ff * sizeof(float));
    }
    
    ~MLP() {
        cudaFree(d_W1); cudaFree(d_bias1);
        cudaFree(d_W2); cudaFree(d_bias2);
        cudaFree(d_hidden);
    }
    
    // Forward pass: input [B, S, d_model] -> output [B, S, d_model]
    void forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
        // First linear layer + bias
        dim3 blocks1(batch_size, seq_len);
        dim3 threads1(d_ff);
        linear_bias_kernel<<<blocks1, threads1>>>(d_input, d_W1, d_bias1, d_hidden, batch_size, seq_len, d_model, d_ff);
        cudaDeviceSynchronize();
        
        // GELU activation
        gelu_kernel<<<blocks1, threads1>>>(d_hidden, d_hidden, batch_size, seq_len, d_ff);
        cudaDeviceSynchronize();
        
        // Second linear layer + bias
        dim3 blocks2(batch_size, seq_len);
        dim3 threads2(d_model);
        linear_bias_kernel<<<blocks2, threads2>>>(d_hidden, d_W2, d_bias2, d_output, batch_size, seq_len, d_ff, d_model);
        cudaDeviceSynchronize();
    }
};

// Transformer Block: combines MHSA + MLP with layer norm and residual connections
class TransformerBlock {
private:
    MultiHeadSelfAttention* mhsa;
    MLP* mlp;
    LayerNorm* ln1;  // Layer norm before MHSA
    LayerNorm* ln2;  // Layer norm before MLP
    
    // Temporary buffers for residual connections
    float *d_temp1, *d_temp2, *d_norm1, *d_norm2;
    int max_batch, max_seq_len, d_model;
    
public:
    TransformerBlock(int d_model_param, int n_heads, int d_ff, int max_seq_len_param,
                    const float* mhsa_weights[], const float* mhsa_bias,
                    const float* mlp_weights[], const float* mlp_biases[],
                    const float* ln1_gamma, const float* ln1_beta,
                    const float* ln2_gamma, const float* ln2_beta) {
        
        // Initialize member variables
        max_batch = 16;
        max_seq_len = max_seq_len_param;
        d_model = d_model_param;
        
        // Create MHSA module
        mhsa = new MultiHeadSelfAttention(
            d_model, n_heads, max_seq_len,
            mhsa_weights[0], mhsa_weights[1], mhsa_weights[2], mhsa_weights[3],
            mhsa_bias
        );
        
        // Create MLP module
        mlp = new MLP(
            d_model, d_ff, max_seq_len,
            mlp_weights[0], mlp_biases[0], mlp_weights[1], mlp_biases[1]
        );
        
        // Create layer norm modules
        ln1 = new LayerNorm(d_model);
        ln2 = new LayerNorm(d_model);
        
        // Load layer norm weights
        ln1->load_weights(ln1_gamma, ln1_beta);
        ln2->load_weights(ln2_gamma, ln2_beta);
        
        // Allocate temporary buffers
        cudaMalloc(&d_temp1, max_batch * max_seq_len * d_model * sizeof(float));
        cudaMalloc(&d_temp2, max_batch * max_seq_len * d_model * sizeof(float));
        cudaMalloc(&d_norm1, max_batch * max_seq_len * d_model * sizeof(float));
        cudaMalloc(&d_norm2, max_batch * max_seq_len * d_model * sizeof(float));
    }
    
    ~TransformerBlock() {
        delete mhsa;
        delete mlp;
        delete ln1;
        delete ln2;
        cudaFree(d_temp1);
        cudaFree(d_temp2);
        cudaFree(d_norm1);
        cudaFree(d_norm2);
    }
    
    // Forward pass: x = x + mhsa(ln1(x)), x = x + mlp(ln2(x))  [Pre-norm architecture]
    void forward(const float* d_input, float* d_output, int batch_size, int seq_len) {
        int total_elements = batch_size * seq_len * d_model;
        
        // First residual block: x = x + mhsa(ln1(x))
        ln1->forward(d_input, d_norm1, batch_size, seq_len);  // ln1(x)
        mhsa->forward(d_norm1, d_temp1, batch_size, seq_len); // mhsa(ln1(x))
        
        // Residual connection: output = input + temp1
        dim3 addBlocks((total_elements + 255) / 256);
        dim3 addThreads(256);
        add_kernel<<<addBlocks, addThreads>>>(d_input, d_temp1, d_output, total_elements);
        cudaDeviceSynchronize();
        
        // Second residual block: x = x + mlp(ln2(x))
        ln2->forward(d_output, d_norm2, batch_size, seq_len); // ln2(x)
        mlp->forward(d_norm2, d_temp2, batch_size, seq_len);  // mlp(ln2(x))
        
        // Residual connection: output = output + temp2
        add_kernel<<<addBlocks, addThreads>>>(d_output, d_temp2, d_output, total_elements);
        cudaDeviceSynchronize();
    }
};
