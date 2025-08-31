#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel to reshape (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
// Input: src [B, S, d_model], Output: dst [B, n_heads, S, d_k]
__global__ void reshape_for_mhsa_kernel(
    const float* src, float* dst,
    int batch_size, int seq_len, int n_heads, int d_k
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = threadIdx.x;
    int d = threadIdx.y;
    if (b >= batch_size || s >= seq_len || h >= n_heads || d >= d_k) return;
    int d_model = n_heads * d_k;
    // src index: [b, s, h * d_k + d]
    int src_idx = (b * seq_len + s) * d_model + h * d_k + d;
    // dst index: [b, h, s, d]
    int dst_idx = ((b * n_heads + h) * seq_len + s) * d_k + d;
    dst[dst_idx] = src[src_idx];
}

// Kernel to compute attention scores: scores = Q @ K^T / sqrt(d_k)
// Q: (batch_size, n_heads, seq_len, d_k)
// K: (batch_size, n_heads, seq_len, d_k)
// scores: (batch_size, n_heads, seq_len, seq_len)
__global__ void scores_kernel(
    const float* Q, // [B, H, S, D]
    const float* K, // [B, H, S, D]
    float* scores,   // [B, H, S, S]
    int batch_size, int n_heads, int seq_len, int d_k
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.z;
    int b = bh / n_heads;
    int h = bh % n_heads;
    if (b >= batch_size || h >= n_heads || i >= seq_len || j >= seq_len) return;

    float sum = 0.0f;
    for (int d = 0; d < d_k; d++) {
        int q_idx = ((b * n_heads + h) * seq_len + i) * d_k + d;
        int k_idx = ((b * n_heads + h) * seq_len + j) * d_k + d;
        sum += Q[q_idx] * K[k_idx];
    }
    int scores_idx = ((b * n_heads + h) * seq_len + i) * seq_len + j;
    scores[scores_idx] = sum / sqrtf((float)d_k);
}

// Kernel to compute attention output: out = attn_weights @ V
// attn_weights: (batch_size, n_heads, seq_len, seq_len)
// V:           (batch_size, n_heads, seq_len, d_k)
// out:         (batch_size, n_heads, seq_len, d_k)
__global__ void attn_matmul_kernel(
    const float* attn_weights, // [B, H, S, S]
    const float* V,           // [B, H, S, d_k]
    float* out,               // [B, H, S, d_k]
    int batch_size, int n_heads, int seq_len, int d_k
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // sequence position (row)
    int d = blockIdx.x * blockDim.x + threadIdx.x; // d_k (column)
    int bh = blockIdx.z;
    int b = bh / n_heads;
    int h = bh % n_heads;
    if (b >= batch_size || h >= n_heads || i >= seq_len || d >= d_k) return;

    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        int attn_idx = ((b * n_heads + h) * seq_len + i) * seq_len + j;
        int v_idx = ((b * n_heads + h) * seq_len + j) * d_k + d;
        sum += attn_weights[attn_idx] * V[v_idx];
    }
    int out_idx = ((b * n_heads + h) * seq_len + i) * d_k + d;
    out[out_idx] = sum;
}

__global__ void mask_kernel(
    float* scores,
    int batch_size, int n_heads, int seq_len
){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int bh = blockIdx.z;
    int b = bh / n_heads;
    int h = bh % n_heads;
    if (b >= batch_size || h >= n_heads || i >= seq_len || j >= seq_len) return;

    if(i < j){
        int scores_idx = ((b * n_heads + h) * seq_len + i) * seq_len + j;
        scores[scores_idx] = -INFINITY;
    }
}

// Softmax kernel for (batch, n_heads, seq_len, seq_len) tensor
// Input: scores [B, H, S, S], Output: attn_weights [B, H, S, S]
__global__ void softmax_kernel(
    const float* scores, float* attn_weights,
    int batch_size, int n_heads, int seq_len
) {
    // Each thread computes softmax for one (b, h, i, :)
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int bh = blockIdx.z;
    int b = bh / n_heads;
    int h = bh % n_heads;
    if (b >= batch_size || h >= n_heads || i >= seq_len) return;

    // Pointer to the start of the row: scores[b, h, i, :]
    int row_start = ((b * n_heads + h) * seq_len + i) * seq_len;

    float max_val = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        max_val = fmaxf(max_val, scores[row_start + j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        sum += expf(scores[row_start + j] - max_val);
    }
    for (int j = 0; j < seq_len; ++j) {
        attn_weights[row_start + j] = expf(scores[row_start + j] - max_val) / sum;
    }
}

// Concatenate heads: (B, n_heads, S, d_k) -> (B, S, d_model)
__global__ void concat_heads_kernel(
    const float* src, // [B, n_heads, S, d_k]
    float* dst,       // [B, S, d_model]
    int batch_size, int n_heads, int seq_len, int d_k
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;
    int d_model = n_heads * d_k;
    if (b >= batch_size || s >= seq_len || d >= d_model) return;

    int h = d / d_k;
    int dk = d % d_k;
    int src_idx = ((b * n_heads + h) * seq_len + s) * d_k + dk;
    int dst_idx = (b * seq_len + s) * d_model + d;
    dst[dst_idx] = src[src_idx];
}

// Linear projection: (B, S, d_model) x (d_model, d_model) -> (B, S, d_model)
// W is row-major: W[out, in]
__global__ void linear_kernel(
    const float* src,   // [B, S, d_model]
    const float* W,     // [d_model, d_model]
    float* dst,         // [B, S, d_model]
    int batch_size, int seq_len, int d_model
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int out = threadIdx.x;
    if (b >= batch_size || s >= seq_len || out >= d_model) return;

    float sum = 0.0f;
    for (int in = 0; in < d_model; ++in) {
        int src_idx = (b * seq_len + s) * d_model + in;
        int w_idx = out * d_model + in;
        sum += src[src_idx] * W[w_idx];
    }
    int dst_idx = (b * seq_len + s) * d_model + out;
    dst[dst_idx] = sum;
}

// Linear projection with bias: (B, S, d_in) x (d_out, d_in) + bias -> (B, S, d_out)
__global__ void linear_bias_kernel(
    const float* src,   // [B, S, d_in]
    const float* W,     // [d_out, d_in]
    const float* bias,  // [d_out]
    float* dst,         // [B, S, d_out]
    int batch_size, int seq_len, int d_in, int d_out
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int out = threadIdx.x;
    if (b >= batch_size || s >= seq_len || out >= d_out) return;

    float sum = bias[out];
    for (int in = 0; in < d_in; ++in) {
        int src_idx = (b * seq_len + s) * d_in + in;
        int w_idx = out * d_in + in;
        sum += src[src_idx] * W[w_idx];
    }
    int dst_idx = (b * seq_len + s) * d_out + out;
    dst[dst_idx] = sum;
}

// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void gelu_kernel(
    const float* src, float* dst,
    int batch_size, int seq_len, int d_model
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = threadIdx.x;
    if (b >= batch_size || s >= seq_len || d >= d_model) return;
    
    int idx = (b * seq_len + s) * d_model + d;
    float x = src[idx];
    float x3 = x * x * x;
    float tanh_arg = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
    dst[idx] = x * 0.5f * (1.0f + tanhf(tanh_arg));
}

// Element-wise addition: dst = src1 + src2
__global__ void add_kernel(
    const float* src1, const float* src2, float* dst,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    dst[idx] = src1[idx] + src2[idx];
}

// Layer normalization: (x - mean) / sqrt(variance + eps) * gamma + beta
__global__ void layernorm_kernel(
    const float* src, const float* gamma, const float* beta, float* dst,
    int batch_size, int seq_len, int d_model, float eps = 1e-5f
) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    if (b >= batch_size || s >= seq_len) return;
    
    // Compute mean
    float mean = 0.0f;
    for (int d = 0; d < d_model; ++d) {
        int idx = (b * seq_len + s) * d_model + d;
        mean += src[idx];
    }
    mean /= d_model;
    
    // Compute variance
    float variance = 0.0f;
    for (int d = 0; d < d_model; ++d) {
        int idx = (b * seq_len + s) * d_model + d;
        float diff = src[idx] - mean;
        variance += diff * diff;
    }
    variance /= d_model;
    
    // Normalize and scale
    float inv_std = 1.0f / sqrtf(variance + eps);
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        int idx = (b * seq_len + s) * d_model + d;
        dst[idx] = (src[idx] - mean) * inv_std * gamma[d] + beta[d];
    }
}
