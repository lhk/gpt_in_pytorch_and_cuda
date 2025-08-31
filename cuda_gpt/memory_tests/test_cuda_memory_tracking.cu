#include "../modules.cu"
#include <iostream>
#include <memory>

void print_gpu_memory(const std::string& label) {
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        size_t used_bytes = total_bytes - free_bytes;
        std::cout << label << ": Used=" << (used_bytes / 1024) << "KB, Free=" << (free_bytes / 1024) << "KB" << std::endl;
    } else {
        std::cout << label << ": Error getting memory info" << std::endl;
    }
}

int main() {
    std::cout << "=== CUDA Device Memory Tracking Test ===" << std::endl;
    
    // Initialize CUDA context
    cudaSetDevice(0);
    
    print_gpu_memory("Initial state");
    
    {
        LayerNorm* ln = new LayerNorm(64);
        print_gpu_memory("After LayerNorm allocation");
        
        delete ln;
        print_gpu_memory("After LayerNorm deletion");
    }
    
    {
        auto ln = std::make_unique<LayerNorm>(128);
        print_gpu_memory("After smart_ptr LayerNorm allocation");
    }
    print_gpu_memory("After smart_ptr LayerNorm auto-deletion");
    
    // Multiple allocations
    print_gpu_memory("Before multiple allocations");
    std::vector<std::unique_ptr<LayerNorm>> layer_norms;
    for (int i = 0; i < 10; i++) {
        layer_norms.push_back(std::make_unique<LayerNorm>(64));
    }
    print_gpu_memory("After 10 LayerNorm allocations");
    
    layer_norms.clear();
    print_gpu_memory("After clearing all LayerNorms");
    
    std::cout << "CUDA device memory tracking completed" << std::endl;
    
    return 0;
}
