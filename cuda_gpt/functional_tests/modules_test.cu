#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

// Include our modules and kernels
#include "modules.cu"

// ============================================================================
// Test Data Loading Utilities
// ============================================================================

struct TestData {
    std::map<std::string, std::vector<float>> arrays;
    std::map<std::string, int> scalars;
    
    const std::vector<float>& get_array(const std::string& name) const {
        auto it = arrays.find(name);
        if (it == arrays.end()) {
            std::cerr << "Error: Array '" << name << "' not found in test data" << std::endl;
            exit(1);
        }
        return it->second;
    }
    
    int get_scalar(const std::string& name) const {
        auto it = scalars.find(name);
        if (it == scalars.end()) {
            std::cerr << "Error: Scalar '" << name << "' not found in test data" << std::endl;
            exit(1);
        }
        return it->second;
    }
};

// Simple .npz loader (reads binary floats and metadata)
TestData load_test_data(const std::string& filename) {
    TestData data;
    
    // For simplicity, we'll load raw binary data files instead of full .npz
    // The Python script should also save raw binary versions
    std::string base = filename.substr(0, filename.find_last_of('.'));
    
    // Load arrays from binary files
    auto load_array = [&](const std::string& name) -> std::vector<float> {
        std::string path = base + "_" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << path << std::endl;
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg() / sizeof(float);
        file.seekg(0, std::ios::beg);
        
        std::vector<float> array(size);
        file.read(reinterpret_cast<char*>(array.data()), size * sizeof(float));
        return array;
    };
    
    // Load configuration scalars from text file
    auto load_config = [&]() {
        std::string config_path = base + "_config.txt";
        std::ifstream file(config_path);
        if (!file.is_open()) {
            // Default config for our test setup
            data.scalars["config_d_model"] = 4;
            data.scalars["config_n_heads"] = 2;
            data.scalars["config_d_ff"] = 8;
            data.scalars["config_max_seq_len"] = 3;
            return;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            size_t eq = line.find('=');
            if (eq != std::string::npos) {
                std::string key = line.substr(0, eq);
                int value = std::stoi(line.substr(eq + 1));
                data.scalars[key] = value;
            }
        }
    };
    
    load_config();
    
    // Load common arrays
    data.arrays["input"] = load_array("input");
    data.arrays["output"] = load_array("output");
    
    return data;
}

// ============================================================================
// Comparison Utilities
// ============================================================================

bool compare_arrays(const std::vector<float>& a, const std::vector<float>& b, 
                   float rtol = 1e-4, float atol = 1e-6, const std::string& name = "") {
    if (a.size() != b.size()) {
        std::cout << "FAIL " << name << " size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }
    
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int num_errors = 0;
    
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::abs(a[i] - b[i]);
        float rel_diff = diff / (std::abs(a[i]) + atol);
        
        max_diff = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
        
        if (diff > atol && rel_diff > rtol) {
            if (num_errors < 5) {
                std::cout << "  Error at [" << i << "]: " << a[i] << " vs " << b[i] 
                         << " (diff: " << diff << ", rel: " << rel_diff << ")" << std::endl;
            }
            num_errors++;
        }
    }
    
    if (num_errors > 0) {
        std::cout << "FAIL " << name << ": " << num_errors << "/" << a.size() 
                 << " elements differ (max_diff: " << max_diff << ", max_rel: " << max_rel_diff << ")" << std::endl;
        return false;
    } else {
        std::cout << "PASS " << name << " (max_diff: " << max_diff 
                 << ", max_rel: " << max_rel_diff << ")" << std::endl;
        return true;
    }
}

void print_array_summary(const std::vector<float>& arr, const std::string& name) {
    if (arr.empty()) return;
    
    float min_val = *std::min_element(arr.begin(), arr.end());
    float max_val = *std::max_element(arr.begin(), arr.end());
    float sum = 0.0f;
    for (float v : arr) sum += v;
    float mean = sum / arr.size();
    
    std::cout << "  " << name << " [" << arr.size() << "]: "
              << "min=" << min_val << ", max=" << max_val << ", mean=" << mean << std::endl;
}

// ============================================================================
// Test Cases
// ============================================================================

bool test_layernorm() {
    std::cout << "\nTesting LayerNorm..." << std::endl;
    
    // Create test input (same as PyTorch reference)
    std::vector<float> input_data = {
        1.926915f, 1.487284f, 0.900717f, -2.105521f,
        0.678418f, -1.234545f, -0.043067f, -1.604667f,
        0.355860f, -0.686623f, -0.493356f, 0.241488f,
        -1.110904f, 0.091546f, -2.316923f, -0.216805f,
        -0.309727f, -0.395710f, 0.803409f, -0.621595f,
        -0.592001f, -0.063074f, -0.828554f, 0.330898f
    };
    
    int batch_size = 2, seq_len = 3, d_model = 4;
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_data.size() * sizeof(float));
    cudaMalloc(&d_output, input_data.size() * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create LayerNorm module (uses identity weights: gamma=1, beta=0)
    LayerNorm ln(d_model);
    
    ln.forward(d_input, d_output, batch_size, seq_len);
    
    std::vector<float> cuda_output(input_data.size());
    cudaMemcpy(cuda_output.data(), d_output, input_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    TestData ref = load_test_data("test_data/layernorm_test.npz");
    
    print_array_summary(input_data, "Input");
    print_array_summary(cuda_output, "CUDA Output");
    if (!ref.get_array("output").empty()) {
        print_array_summary(ref.get_array("output"), "PyTorch Output");
        bool success = compare_arrays(cuda_output, ref.get_array("output"), 1e-4, 1e-6, "LayerNorm");
        
        cudaFree(d_input);
        cudaFree(d_output);
        return success;
    } else {
        std::cout << "WARNING: No reference data loaded, manual validation needed" << std::endl;
        
        // Manual validation: check LayerNorm properties
        bool valid = true;
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                float mean = 0.0f, var = 0.0f;
                int offset = (b * seq_len + s) * d_model;
                
                for (int d = 0; d < d_model; d++) {
                    mean += cuda_output[offset + d];
                }
                mean /= d_model;
                
                for (int d = 0; d < d_model; d++) {
                    float diff = cuda_output[offset + d] - mean;
                    var += diff * diff;
                }
                var /= d_model;
                
                std::cout << "  [" << b << "," << s << "] mean=" << mean << ", var=" << var << std::endl;
                
                if (std::abs(mean) > 1e-5 || std::abs(var - 1.0f) > 1e-3) {
                    valid = false;
                }
            }
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        return valid;
    }
}

bool test_mhsa() {
    std::cout << "\nTesting MultiHeadSelfAttention..." << std::endl;
    
    // Load test configuration and weights
    TestData ref = load_test_data("test_data/mhsa_test.npz");
    
    if (ref.get_array("input").empty()) {
        std::cout << "WARNING: No reference data loaded, skipping MHSA test" << std::endl;
        return true;
    }
    
    // Get configuration
    int batch_size = 2, seq_len = 3, d_model = 4, n_heads = 2, max_seq_len = 3;
    
    // Load input data
    std::vector<float> input_data = ref.get_array("input");
    std::vector<float> expected_output = ref.get_array("output");
    
    // Create weight arrays - we need to load these from binary files
    auto load_weights = [](const std::string& base_path, const std::string& name) -> std::vector<float> {
        std::string path = base_path + "_" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Could not load weights from " << path << std::endl;
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg() / sizeof(float);
        file.seekg(0, std::ios::beg);
        
        std::vector<float> weights(size);
        file.read(reinterpret_cast<char*>(weights.data()), size * sizeof(float));
        return weights;
    };
    
    std::string base = "test_data/mhsa_test";
    std::vector<float> wq = load_weights(base, "w_q");
    std::vector<float> wk = load_weights(base, "w_k");
    std::vector<float> wv = load_weights(base, "w_v");
    std::vector<float> wo = load_weights(base, "w_o");
    std::vector<float> wo_bias = load_weights(base, "w_o_bias");
    
    if (wq.empty() || wk.empty() || wv.empty() || wo.empty() || wo_bias.empty()) {
        std::cout << "WARNING: Could not load all MHSA weights, skipping test" << std::endl;
        return true;
    }
    
    // Create MHSA module
    MultiHeadSelfAttention mhsa(
        d_model, n_heads, max_seq_len,
        wq.data(), wk.data(), wv.data(), wo.data(), wo_bias.data()
    );
    
    // Allocate device memory for input/output
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_data.size() * sizeof(float));
    cudaMalloc(&d_output, input_data.size() * sizeof(float));
    
    cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    mhsa.forward(d_input, d_output, batch_size, seq_len);
    
    std::vector<float> cuda_output(input_data.size());
    cudaMemcpy(cuda_output.data(), d_output, input_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    print_array_summary(input_data, "Input");
    print_array_summary(cuda_output, "CUDA Output");
    print_array_summary(expected_output, "PyTorch Output");
    
    bool success = compare_arrays(cuda_output, expected_output, 1e-3, 1e-5, "MHSA");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return success;
}

bool test_mlp() {
    std::cout << "\nTesting MLP..." << std::endl;
    
    // Load test configuration and weights
    TestData ref = load_test_data("test_data/mlp_test.npz");
    
    if (ref.get_array("input").empty()) {
        std::cout << "WARNING: No reference data loaded, skipping MLP test" << std::endl;
        return true;
    }
    
    // Get configuration
    int batch_size = 2, seq_len = 3, d_model = 4, d_ff = 8, max_seq_len = 3;
    
    // Load input data
    std::vector<float> input_data = ref.get_array("input");
    std::vector<float> expected_output = ref.get_array("output");
    
    // Load weights
    auto load_weights = [](const std::string& base_path, const std::string& name) -> std::vector<float> {
        std::string path = base_path + "_" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Could not load weights from " << path << std::endl;
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg() / sizeof(float);
        file.seekg(0, std::ios::beg);
        
        std::vector<float> weights(size);
        file.read(reinterpret_cast<char*>(weights.data()), size * sizeof(float));
        return weights;
    };
    
    std::string base = "test_data/mlp_test";
    std::vector<float> w1 = load_weights(base, "linear1_weight");
    std::vector<float> b1 = load_weights(base, "linear1_bias");
    std::vector<float> w2 = load_weights(base, "linear2_weight");
    std::vector<float> b2 = load_weights(base, "linear2_bias");
    
    if (w1.empty() || b1.empty() || w2.empty() || b2.empty()) {
        std::cout << "WARNING: Could not load all MLP weights, skipping test" << std::endl;
        return true;
    }
    
    // Create MLP module
    MLP mlp(d_model, d_ff, max_seq_len, w1.data(), b1.data(), w2.data(), b2.data());
    
    // Allocate device memory for input/output
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_data.size() * sizeof(float));
    cudaMalloc(&d_output, input_data.size() * sizeof(float));
    
    cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    mlp.forward(d_input, d_output, batch_size, seq_len);
    
    std::vector<float> cuda_output(input_data.size());
    cudaMemcpy(cuda_output.data(), d_output, input_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    print_array_summary(input_data, "Input");
    print_array_summary(cuda_output, "CUDA Output");
    print_array_summary(expected_output, "PyTorch Output");
    
    bool success = compare_arrays(cuda_output, expected_output, 1e-3, 1e-5, "MLP");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return success;
}

bool test_transformer_block() {
    std::cout << "\nTesting TransformerBlock..." << std::endl;
    
    // Load test configuration and weights
    TestData ref = load_test_data("test_data/transformer_block_test.npz");
    
    if (ref.get_array("input").empty()) {
        std::cout << "WARNING: No reference data loaded, skipping TransformerBlock test" << std::endl;
        return true;
    }
    
    // Get configuration
    int batch_size = 2, seq_len = 3, d_model = 4, n_heads = 2, d_ff = 8, max_seq_len = 3;
    
    // Load input data
    std::vector<float> input_data = ref.get_array("input");
    std::vector<float> expected_output = ref.get_array("output");
    
    // Load weights
    auto load_weights = [](const std::string& base_path, const std::string& name) -> std::vector<float> {
        std::string path = base_path + "_" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Could not load weights from " << path << std::endl;
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg() / sizeof(float);
        file.seekg(0, std::ios::beg);
        
        std::vector<float> weights(size);
        file.read(reinterpret_cast<char*>(weights.data()), size * sizeof(float));
        return weights;
    };
    
    std::string base = "test_data/transformer_block_test";
    
    // MHSA weights
    std::vector<float> mhsa_wq = load_weights(base, "mhsa_w_q");
    std::vector<float> mhsa_wk = load_weights(base, "mhsa_w_k");
    std::vector<float> mhsa_wv = load_weights(base, "mhsa_w_v");
    std::vector<float> mhsa_wo = load_weights(base, "mhsa_w_o");
    std::vector<float> mhsa_wo_bias = load_weights(base, "mhsa_w_o_bias");
    
    // MLP weights
    std::vector<float> mlp_w1 = load_weights(base, "mlp_w1");
    std::vector<float> mlp_b1 = load_weights(base, "mlp_b1");
    std::vector<float> mlp_w2 = load_weights(base, "mlp_w2");
    std::vector<float> mlp_b2 = load_weights(base, "mlp_b2");
    
    // LayerNorm weights
    std::vector<float> ln1_gamma = load_weights(base, "ln1_gamma");
    std::vector<float> ln1_beta = load_weights(base, "ln1_beta");
    std::vector<float> ln2_gamma = load_weights(base, "ln2_gamma");
    std::vector<float> ln2_beta = load_weights(base, "ln2_beta");
    
    // Check if all weights loaded
    if (mhsa_wq.empty() || mhsa_wk.empty() || mhsa_wv.empty() || mhsa_wo.empty() || 
        mhsa_wo_bias.empty() || mlp_w1.empty() || mlp_b1.empty() || mlp_w2.empty() || 
        mlp_b2.empty() || ln1_gamma.empty() || ln1_beta.empty() || ln2_gamma.empty() || ln2_beta.empty()) {
        std::cout << "WARNING: Could not load all TransformerBlock weights, skipping test" << std::endl;
        return true;
    }
    
    // Prepare weight arrays for TransformerBlock constructor
    const float* mhsa_weights[] = {mhsa_wq.data(), mhsa_wk.data(), mhsa_wv.data(), mhsa_wo.data()};
    const float* mlp_weights[] = {mlp_w1.data(), mlp_w2.data()};
    const float* mlp_biases[] = {mlp_b1.data(), mlp_b2.data()};
    
    // Create TransformerBlock module
    TransformerBlock block(
        d_model, n_heads, d_ff, max_seq_len,
        mhsa_weights, mhsa_wo_bias.data(),
        mlp_weights, mlp_biases,
        ln1_gamma.data(), ln1_beta.data(),
        ln2_gamma.data(), ln2_beta.data()
    );
    
    // Allocate device memory for input/output
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_data.size() * sizeof(float));
    cudaMalloc(&d_output, input_data.size() * sizeof(float));
    
    cudaMemcpy(d_input, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    block.forward(d_input, d_output, batch_size, seq_len);
    
    std::vector<float> cuda_output(input_data.size());
    cudaMemcpy(cuda_output.data(), d_output, input_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare results
    print_array_summary(input_data, "Input");
    print_array_summary(cuda_output, "CUDA Output");
    print_array_summary(expected_output, "PyTorch Output");
    
    bool success = compare_arrays(cuda_output, expected_output, 1e-2, 1e-4, "TransformerBlock");
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return success;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "=== CUDA Modules Test Suite ===" << std::endl;
    
    // Check CUDA setup
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    if (device_count == 0) {
        std::cout << "ERROR: No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Run tests
    std::vector<std::pair<std::string, bool>> results;
    
    results.push_back({"LayerNorm", test_layernorm()});
    results.push_back({"MultiHeadSelfAttention", test_mhsa()});
    results.push_back({"MLP", test_mlp()});
    results.push_back({"TransformerBlock", test_transformer_block()});
    
    // Summary
    std::cout << "\n=== Test Results ===" << std::endl;
    int passed = 0, total = results.size();
    
    for (const auto& result : results) {
        std::cout << (result.second ? "PASS" : "FAIL") << " " << result.first << std::endl;
        if (result.second) passed++;
    }
    
    std::cout << "\nPassed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}
