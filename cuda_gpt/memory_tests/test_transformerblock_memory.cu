#include "../modules.cu"
#include <iostream>
#include <memory>
#include <stdexcept>

size_t get_gpu_memory_used() {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get GPU memory info: " + std::string(cudaGetErrorString(err)));
    }
    return total_bytes - free_bytes;
}

int main() {
    std::cout << "=== TransformerBlock Memory Test ===" << std::endl;
    
    try {
        // Initialize CUDA context and get baseline memory
        cudaSetDevice(0);
        size_t initial_memory = get_gpu_memory_used();
        std::cout << "Initial GPU memory: " << (initial_memory / 1024) << " KB" << std::endl;
        
        // Setup parameters and dummy weights
        int d_model = 64, n_heads = 2, d_ff = 128, max_seq_len = 32;
        
        // MHSA weights
        std::vector<float> mhsa_wq(d_model * d_model, 0.01f);
        std::vector<float> mhsa_wk(d_model * d_model, 0.01f);
        std::vector<float> mhsa_wv(d_model * d_model, 0.01f);
        std::vector<float> mhsa_wo(d_model * d_model, 0.01f);
        std::vector<float> mhsa_bias(d_model, 0.0f);
        
        // MLP weights
        std::vector<float> mlp_w1(d_model * d_ff, 0.01f);
        std::vector<float> mlp_b1(d_ff, 0.0f);
        std::vector<float> mlp_w2(d_ff * d_model, 0.01f);
        std::vector<float> mlp_b2(d_model, 0.0f);
        
        // LayerNorm weights
        std::vector<float> ln1_gamma(d_model, 1.0f);
        std::vector<float> ln1_beta(d_model, 0.0f);
        std::vector<float> ln2_gamma(d_model, 1.0f);
        std::vector<float> ln2_beta(d_model, 0.0f);
        
        // Package weights for constructor
        const float* mhsa_weights[] = {mhsa_wq.data(), mhsa_wk.data(), mhsa_wv.data(), mhsa_wo.data()};
        const float* mlp_weights[] = {mlp_w1.data(), mlp_w2.data()};
        const float* mlp_biases[] = {mlp_b1.data(), mlp_b2.data()};
        
        // Test 1: Manual allocation/deallocation
        std::cout << "Testing manual allocation..." << std::endl;
        TransformerBlock* block_manual = new TransformerBlock(
            d_model, n_heads, d_ff, max_seq_len,
            mhsa_weights, mhsa_bias.data(),
            mlp_weights, mlp_biases,
            ln1_gamma.data(), ln1_beta.data(),
            ln2_gamma.data(), ln2_beta.data()
        );
        delete block_manual;
        
        // Test 2: Smart pointer (RAII)
        std::cout << "Testing smart pointer allocation..." << std::endl;
        {
            auto block_smart = std::make_unique<TransformerBlock>(
                d_model, n_heads, d_ff, max_seq_len,
                mhsa_weights, mhsa_bias.data(),
                mlp_weights, mlp_biases,
                ln1_gamma.data(), ln1_beta.data(),
                ln2_gamma.data(), ln2_beta.data()
            );
        }
        
        // Test 3: Multiple allocations
        std::cout << "Testing multiple allocations..." << std::endl;
        std::vector<std::unique_ptr<TransformerBlock>> block_instances;
        for (int i = 0; i < 3; i++) {
            block_instances.push_back(std::make_unique<TransformerBlock>(
                d_model, n_heads, d_ff, max_seq_len,
                mhsa_weights, mhsa_bias.data(),
                mlp_weights, mlp_biases,
                ln1_gamma.data(), ln1_beta.data(),
                ln2_gamma.data(), ln2_beta.data()
            ));
        }
        block_instances.clear();
        
        // Verify memory has returned to baseline
        size_t final_memory = get_gpu_memory_used();
        std::cout << "Final GPU memory: " << (final_memory / 1024) << " KB" << std::endl;
        
        if (final_memory != initial_memory) {
            throw std::runtime_error("MEMORY LEAK DETECTED: Initial=" + 
                std::to_string(initial_memory / 1024) + "KB, Final=" + 
                std::to_string(final_memory / 1024) + "KB, Difference=" + 
                std::to_string((long long)final_memory - (long long)initial_memory) + " bytes");
        }
        
        std::cout << "TransformerBlock memory test PASSED - no leaks detected" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
