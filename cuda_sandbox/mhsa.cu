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

    if(i<j){
        int scores_idx = ((b * n_heads + h) * seq_len + i) * seq_len + j;
        scores[scores_idx]=-INFINITY;
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

	// TODO: User should implement the softmax over scores[row_start + j], j in 0..seq_len-1
	// Write result to attn_weights[row_start + j]
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

int main() {

	// Small debug sizes
	const int batch_size = 2, n_heads = 2, seq_len = 3, d_k = 2;
	const int d_model = n_heads * d_k;
	const int QK_in_size = batch_size * seq_len * d_model;
	const int QK_out_size = batch_size * n_heads * seq_len * d_k;
	const int scores_size = batch_size * n_heads * seq_len * seq_len;
	float h_attn_weights[scores_size] = {0};

	// Host allocations
	float h_Q[QK_out_size];
	float h_K[QK_out_size];
	float h_V[QK_out_size];
	float h_scores[scores_size] = {0};
	float h_attn_out[QK_out_size] = {0};
	float h_concat_out[batch_size * seq_len * d_model];
	float h_proj_out[batch_size * seq_len * d_model];
	// Identity matrices for projections (WQ, WK, WV, WO)
	float h_WQ[d_model * d_model] = {0};
	float h_WK[d_model * d_model] = {0};
	float h_WV[d_model * d_model] = {0};
	float h_WO[d_model * d_model] = {0};
	for (int i = 0; i < d_model; ++i) {
		h_WQ[i * d_model + i] = 1.0f;
		h_WK[i * d_model + i] = 1.0f;
		h_WV[i * d_model + i] = 1.0f;
		h_WO[i * d_model + i] = 1.0f;
	}

	// Device allocations for projection weights
	float *d_WQ, *d_WK, *d_WV, *d_WO;
	cudaMalloc((void**)&d_WQ, d_model * d_model * sizeof(float));
	cudaMalloc((void**)&d_WK, d_model * d_model * sizeof(float));
	cudaMalloc((void**)&d_WV, d_model * d_model * sizeof(float));
	cudaMalloc((void**)&d_WO, d_model * d_model * sizeof(float));
	cudaMemcpy(d_WQ, h_WQ, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_WK, h_WK, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_WV, h_WV, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_WO, h_WO, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice);

	// now using random values (same as in python)
	// Q: (2, 3, 4) flattened
	float h_Q_in[24] = {
		1.926915, 1.487284, 0.900717, -2.105521,
		0.678418, -1.234545, -0.043067, -1.604667,
		0.355860, -0.686623, -0.493356, 0.241488,
		-1.110904, 0.091546, -2.316923, -0.216805,
		-0.309727, -0.395710, 0.803409, -0.621595,
		-0.592001, -0.063074, -0.828554, 0.330898
	};

	// K: (2, 3, 4) flattened
	float h_K_in[24] = {
		0.034912, 0.321103, 1.573600, -0.845467,
		1.312308, 0.687160, -1.089175, -0.355287,
		1.445134, 0.856413, 2.218076, 0.523166,
		0.346647, -0.197331, -1.054589, 1.277996,
		-0.172190, 0.523788, 0.056622, 0.426296,
		0.575005, -0.641724, -2.206398, -0.750803
	};

	// V: (2, 3, 4) flattened (from Python V_flat)
	float h_V_in[24] = {
		0.010868, -0.338742, -1.340680, -0.585371,
		0.536188, 0.524623, 1.141202, 0.051644,
		-0.678753, 0.574316, 0.187749, -0.357623,
		-0.316508, 0.588634, -0.890457, 0.409813,
		-0.986439, 0.123299, 0.349868, 0.617281,
		-0.169332, 0.233225, 4.035634, 1.279459
	};

	// Device allocations
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

	cudaMemcpy(d_Q_in, h_Q_in, QK_in_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_K_in, h_K_in, QK_in_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V_in, h_V_in, QK_in_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_Q, 0, QK_out_size * sizeof(float));
	cudaMemset(d_K, 0, QK_out_size * sizeof(float));
	cudaMemset(d_V, 0, QK_out_size * sizeof(float));
	cudaMemset(d_scores, 0, scores_size * sizeof(float));
	cudaMemset(d_attn_weights, 0, scores_size * sizeof(float));
	cudaMemset(d_attn_out, 0, QK_out_size * sizeof(float));

	// Launch reshape kernel for Q, K, V
	dim3 reshapeBlocks(batch_size, seq_len);
	dim3 reshapeThreads(n_heads, d_k);
	reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_Q_in, d_Q, batch_size, seq_len, n_heads, d_k);
	reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_K_in, d_K, batch_size, seq_len, n_heads, d_k);
	reshape_for_mhsa_kernel<<<reshapeBlocks, reshapeThreads>>>(d_V_in, d_V, batch_size, seq_len, n_heads, d_k);
	cudaDeviceSynchronize();

	cudaMemcpy(h_Q, d_Q, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_K, d_K, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_V, d_V, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);


	// --- Linear projections for Q, K, V (identity, so no-op, but shown for completeness) ---
	// If you want to apply: linear_kernel<<<...>>>(d_Q, d_WQ, d_Q, ...); etc.

	// Print host Q, K, V in CUDA style for debugging (after reshape)
	print_4d(h_Q, batch_size, n_heads, seq_len, d_k, "Q");
	print_4d(h_K, batch_size, n_heads, seq_len, d_k, "K");
	print_4d(h_V, batch_size, n_heads, seq_len, d_k, "V");

	// Launch scores kernel: one thread per (b, h, i, j)
	dim3 blockDim(8, 8);
	dim3 gridDim(
		(seq_len + blockDim.x - 1) / blockDim.x,
		(seq_len + blockDim.y - 1) / blockDim.y,
		batch_size * n_heads
	);
	scores_kernel<<<gridDim, blockDim>>>(d_Q, d_K, d_scores, batch_size, n_heads, seq_len, d_k);
	cudaDeviceSynchronize();

	// Launch mask kernel: zero out scores where i < j (causal mask)
	mask_kernel<<<gridDim, blockDim>>>(d_scores, batch_size, n_heads, seq_len);
	cudaDeviceSynchronize();

	cudaMemcpy(h_scores, d_scores, scores_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_4d(h_scores, batch_size, n_heads, seq_len, seq_len, "scores (after mask)");

	// Launch softmax kernel: one thread per (b, h, i, :)
	dim3 softmaxBlock(1, 1);
	dim3 softmaxGrid(1, seq_len, batch_size * n_heads);
	softmax_kernel<<<softmaxGrid, softmaxBlock>>>(d_scores, d_attn_weights, batch_size, n_heads, seq_len);
	cudaDeviceSynchronize();

	cudaMemcpy(h_attn_weights, d_attn_weights, scores_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_4d(h_attn_weights, batch_size, n_heads, seq_len, seq_len, "attn_weights (after softmax)");

	// Launch attention output kernel: attn_weights @ V
	dim3 attnBlock(d_k, 1);
	dim3 attnGrid((d_k + attnBlock.x - 1) / attnBlock.x, seq_len, batch_size * n_heads);
	attn_matmul_kernel<<<attnGrid, attnBlock>>>(d_attn_weights, d_V, d_attn_out, batch_size, n_heads, seq_len, d_k);
	cudaDeviceSynchronize();

	cudaMemcpy(h_attn_out, d_attn_out, QK_out_size * sizeof(float), cudaMemcpyDeviceToHost);
	print_4d(h_attn_out, batch_size, n_heads, seq_len, d_k, "attention output (attn_weights @ V)");

	// --- Concatenate heads: (B, n_heads, S, d_k) -> (B, S, d_model) ---
	float *d_concat_out;
	cudaMalloc((void**)&d_concat_out, batch_size * seq_len * d_model * sizeof(float));
	dim3 concatBlocks(batch_size, seq_len);
	dim3 concatThreads(d_model);
	concat_heads_kernel<<<concatBlocks, concatThreads>>>(d_attn_out, d_concat_out, batch_size, n_heads, seq_len, d_k);
	cudaDeviceSynchronize();
	cudaMemcpy(h_concat_out, d_concat_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
	print_3d(h_concat_out, batch_size, seq_len, d_model, "concat_heads output");

	// --- Output linear projection: (B, S, d_model) x (d_model, d_model) ---
	float *d_proj_out;
	cudaMalloc((void**)&d_proj_out, batch_size * seq_len * d_model * sizeof(float));
	linear_kernel<<<concatBlocks, concatThreads>>>(d_concat_out, d_WO, d_proj_out, batch_size, seq_len, d_model);
	cudaDeviceSynchronize();
	cudaMemcpy(h_proj_out, d_proj_out, batch_size * seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
	print_3d(h_proj_out, batch_size, seq_len, d_model, "final output after WO");

	// Free new allocations
	cudaFree(d_WQ);
	cudaFree(d_WK);
	cudaFree(d_WV);
	cudaFree(d_WO);
	cudaFree(d_concat_out);
	cudaFree(d_proj_out);

	cudaFree(d_Q_in);
	cudaFree(d_K_in);
	cudaFree(d_V_in);
	cudaFree(d_Q);
	cudaFree(d_K);
	cudaFree(d_V);
	cudaFree(d_scores);
	cudaFree(d_attn_weights);
	cudaFree(d_attn_out);
	return 0;
}
