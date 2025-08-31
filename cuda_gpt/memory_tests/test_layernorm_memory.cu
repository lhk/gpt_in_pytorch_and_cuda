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
    std::cout << "=== LayerNorm Memory Test ===" << std::endl;
    
    try {
        // Initialize CUDA context and get baseline memory
        cudaSetDevice(0);
        size_t initial_memory = get_gpu_memory_used();
        std::cout << "Initial GPU memory: " << (initial_memory / 1024) << " KB" << std::endl;
        // Test 1: Manual allocation/deallocation
        std::cout << "Testing manual allocation..." << std::endl;
        LayerNorm* ln_manual = new LayerNorm(64);
        delete ln_manual;

        // this leaks both device and host memory, uncomment to sanity check that the CUDA leak detection is working
        //LayerNorm* ln_manual_leak = new LayerNorm(64);
    
        // this leaks host memory, uncomment to check whether valgrind is set up correctly here
        //tint* host_memory_leak = new int[1024];

        
        // Test 2: Smart pointer (RAII)
        std::cout << "Testing smart pointer allocation..." << std::endl;
        {
            auto ln_smart = std::make_unique<LayerNorm>(128);
        }
        
        // Test 3: Multiple allocations
        std::cout << "Testing multiple allocations..." << std::endl;
        std::vector<std::unique_ptr<LayerNorm>> layer_norms;
        for (int i = 0; i < 10; i++) {
            layer_norms.push_back(std::make_unique<LayerNorm>(64));
        }
        layer_norms.clear();
        
        // Verify memory has returned to baseline
        size_t final_memory = get_gpu_memory_used();
        std::cout << "Final GPU memory: " << (final_memory / 1024) << " KB" << std::endl;
        
        if (final_memory != initial_memory) {
            throw std::runtime_error("MEMORY LEAK DETECTED: Initial=" + 
                std::to_string(initial_memory / 1024) + "KB, Final=" + 
                std::to_string(final_memory / 1024) + "KB, Difference=" + 
                std::to_string((long long)final_memory - (long long)initial_memory) + " bytes");
        }
        
        std::cout << "LayerNorm memory test PASSED - no leaks detected" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
