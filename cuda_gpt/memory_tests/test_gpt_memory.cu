#include "../gpt.cu"
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
    std::cout << "=== GPT Memory Test ===" << std::endl;
    
    try {
        // Initialize CUDA context and get baseline memory
        cudaSetDevice(0);
        size_t initial_memory = get_gpu_memory_used();
        std::cout << "Initial GPU memory: " << (initial_memory / 1024) << " KB" << std::endl;
        
        // Setup minimal GPT configuration
        GPTConfig config;
        config.vocab_size = 100;
        config.d_model = 64;
        config.n_heads = 2;
        config.d_ff = 128;
        config.n_layers = 2;
        config.max_seq_len = 32;
        
        // Create dummy weights for testing
        ExtractedWeights weights;
        
        // Token and positional embeddings
        weights.token_embedding.resize(config.vocab_size * config.d_model, 0.01f);
        weights.pos_embedding.resize(config.max_seq_len * config.d_model, 0.01f);
        
        // Final layer norm
        weights.final_ln_gamma.resize(config.d_model, 1.0f);
        weights.final_ln_beta.resize(config.d_model, 0.0f);
        
        // LM head
        weights.lm_head.resize(config.d_model * config.vocab_size, 0.01f);
        
        // Layer weights
        weights.layers.resize(config.n_layers);
        for (int layer = 0; layer < config.n_layers; layer++) {
            auto& layer_weights = weights.layers[layer];
            
            // MHSA weights
            layer_weights.w_q.resize(config.d_model * config.d_model, 0.01f);
            layer_weights.w_k.resize(config.d_model * config.d_model, 0.01f);
            layer_weights.w_v.resize(config.d_model * config.d_model, 0.01f);
            layer_weights.w_o.resize(config.d_model * config.d_model, 0.01f);
            layer_weights.w_o_bias.resize(config.d_model, 0.0f);
            
            // MLP weights
            layer_weights.mlp_w1.resize(config.d_model * config.d_ff, 0.01f);
            layer_weights.mlp_b1.resize(config.d_ff, 0.0f);
            layer_weights.mlp_w2.resize(config.d_ff * config.d_model, 0.01f);
            layer_weights.mlp_b2.resize(config.d_model, 0.0f);
            
            // Layer norm weights
            layer_weights.ln1_gamma.resize(config.d_model, 1.0f);
            layer_weights.ln1_beta.resize(config.d_model, 0.0f);
            layer_weights.ln2_gamma.resize(config.d_model, 1.0f);
            layer_weights.ln2_beta.resize(config.d_model, 0.0f);
        }
        
        // Test 1: Manual allocation/deallocation
        std::cout << "Testing manual allocation..." << std::endl;
        GPTModel* gpt_manual = new GPTModel(config, weights);
        delete gpt_manual;
        
        // Test 2: Smart pointer (RAII)
        std::cout << "Testing smart pointer allocation..." << std::endl;
        {
            auto gpt_smart = std::make_unique<GPTModel>(config, weights);
        }
        
        // Test 3: Multiple allocations
        std::cout << "Testing multiple allocations..." << std::endl;
        std::vector<std::unique_ptr<GPTModel>> gpt_instances;
        for (int i = 0; i < 3; i++) {
            gpt_instances.push_back(std::make_unique<GPTModel>(config, weights));
        }
        gpt_instances.clear();
        
        // Verify memory has returned to baseline
        size_t final_memory = get_gpu_memory_used();
        std::cout << "Final GPU memory: " << (final_memory / 1024) << " KB" << std::endl;
        
        if (final_memory != initial_memory) {
            throw std::runtime_error("MEMORY LEAK DETECTED: Initial=" + 
                std::to_string(initial_memory / 1024) + "KB, Final=" + 
                std::to_string(final_memory / 1024) + "KB, Difference=" + 
                std::to_string((long long)final_memory - (long long)initial_memory) + " bytes");
        }
        
        std::cout << "GPT memory test PASSED - no leaks detected" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
