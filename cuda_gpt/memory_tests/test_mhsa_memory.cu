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
    std::cout << "=== MultiHeadSelfAttention Memory Test ===" << std::endl;
    
    try {
        // Initialize CUDA context and get baseline memory
        cudaSetDevice(0);
        size_t initial_memory = get_gpu_memory_used();
        std::cout << "Initial GPU memory: " << (initial_memory / 1024) << " KB" << std::endl;
        
        // Setup dummy weights
        int d_model = 64, n_heads = 2, max_seq_len = 32;
        
        std::vector<float> wq(d_model * d_model, 0.01f);
        std::vector<float> wk(d_model * d_model, 0.01f);
        std::vector<float> wv(d_model * d_model, 0.01f);
        std::vector<float> wo(d_model * d_model, 0.01f);
        std::vector<float> wo_bias(d_model, 0.0f);
        
        // Test 1: Manual allocation/deallocation
        std::cout << "Testing manual allocation..." << std::endl;
        MultiHeadSelfAttention* mhsa_manual = new MultiHeadSelfAttention(
            d_model, n_heads, max_seq_len,
            wq.data(), wk.data(), wv.data(), wo.data(), wo_bias.data()
        );
        delete mhsa_manual;
        
        // Test 2: Smart pointer (RAII)
        std::cout << "Testing smart pointer allocation..." << std::endl;
        {
            auto mhsa_smart = std::make_unique<MultiHeadSelfAttention>(
                d_model, n_heads, max_seq_len,
                wq.data(), wk.data(), wv.data(), wo.data(), wo_bias.data()
            );
        }
        
        // Test 3: Multiple allocations
        std::cout << "Testing multiple allocations..." << std::endl;
        std::vector<std::unique_ptr<MultiHeadSelfAttention>> mhsa_instances;
        for (int i = 0; i < 5; i++) {
            mhsa_instances.push_back(std::make_unique<MultiHeadSelfAttention>(
                d_model, n_heads, max_seq_len,
                wq.data(), wk.data(), wv.data(), wo.data(), wo_bias.data()
            ));
        }
        mhsa_instances.clear();
        
        // Verify memory has returned to baseline
        size_t final_memory = get_gpu_memory_used();
        std::cout << "Final GPU memory: " << (final_memory / 1024) << " KB" << std::endl;
        
        if (final_memory != initial_memory) {
            throw std::runtime_error("MEMORY LEAK DETECTED: Initial=" + 
                std::to_string(initial_memory / 1024) + "KB, Final=" + 
                std::to_string(final_memory / 1024) + "KB, Difference=" + 
                std::to_string((long long)final_memory - (long long)initial_memory) + " bytes");
        }
        
        std::cout << "MHSA memory test PASSED - no leaks detected" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
