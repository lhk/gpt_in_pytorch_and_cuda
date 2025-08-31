#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>

// Include our GPT implementation
#include "gpt.cu"

// ============================================================================
// Test Data Loading Utilities
// ============================================================================

// Load GPT test data from binary files
GPTTestData load_gpt_test_data(const std::string& base_path) {
    GPTTestData data;
    
    // Load arrays from binary files
    auto load_float_array = [&](const std::string& name) -> std::vector<float> {
        std::string path = base_path + "_" + name + ".bin";
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
    
    auto load_int_array = [&](const std::string& name) -> std::vector<int> {
        std::string path = base_path + "_" + name + ".bin";
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << path << std::endl;
            return {};
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg() / sizeof(int);
        file.seekg(0, std::ios::beg);
        
        std::vector<int> array(size);
        file.read(reinterpret_cast<char*>(array.data()), size * sizeof(int));
        return array;
    };
    
    // Load configuration scalars from text file
    auto load_config = [&]() {
        std::string config_path = base_path + "_config.txt";
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file " << config_path << std::endl;
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
    data.int_arrays["input_ids"] = load_int_array("input_ids");
    data.float_arrays["logits"] = load_float_array("logits");
    
    // Load all weight arrays
    data.float_arrays["token_embedding"] = load_float_array("token_embedding");
    data.float_arrays["pos_embedding"] = load_float_array("pos_embedding");
    data.float_arrays["final_ln_gamma"] = load_float_array("final_ln_gamma");
    data.float_arrays["final_ln_beta"] = load_float_array("final_ln_beta");
    data.float_arrays["lm_head_weight"] = load_float_array("lm_head_weight");
    
    // Load layer weights (assuming 2 layers from our test data)
    for (int layer = 0; layer < 2; layer++) {
        std::string layer_prefix = "layer_" + std::to_string(layer) + "_";
        
        // MHSA weights
        data.float_arrays[layer_prefix + "mhsa_w_q"] = load_float_array(layer_prefix + "mhsa_w_q");
        data.float_arrays[layer_prefix + "mhsa_w_k"] = load_float_array(layer_prefix + "mhsa_w_k");
        data.float_arrays[layer_prefix + "mhsa_w_v"] = load_float_array(layer_prefix + "mhsa_w_v");
        data.float_arrays[layer_prefix + "mhsa_w_o"] = load_float_array(layer_prefix + "mhsa_w_o");
        data.float_arrays[layer_prefix + "mhsa_w_o_bias"] = load_float_array(layer_prefix + "mhsa_w_o_bias");
        
        // MLP weights
        data.float_arrays[layer_prefix + "mlp_w1"] = load_float_array(layer_prefix + "mlp_w1");
        data.float_arrays[layer_prefix + "mlp_b1"] = load_float_array(layer_prefix + "mlp_b1");
        data.float_arrays[layer_prefix + "mlp_w2"] = load_float_array(layer_prefix + "mlp_w2");
        data.float_arrays[layer_prefix + "mlp_b2"] = load_float_array(layer_prefix + "mlp_b2");
        
        // LayerNorm weights
        data.float_arrays[layer_prefix + "ln1_gamma"] = load_float_array(layer_prefix + "ln1_gamma");
        data.float_arrays[layer_prefix + "ln1_beta"] = load_float_array(layer_prefix + "ln1_beta");
        data.float_arrays[layer_prefix + "ln2_gamma"] = load_float_array(layer_prefix + "ln2_gamma");
        data.float_arrays[layer_prefix + "ln2_beta"] = load_float_array(layer_prefix + "ln2_beta");
    }
    
    return data;
}

// ============================================================================
// Comparison Utilities
// ============================================================================

bool compare_arrays(const std::vector<float>& a, const std::vector<float>& b, 
                   float rtol = 1e-4, float atol = 1e-6, const std::string& name = "") {
    if (a.size() != b.size()) {
        std::cout << "Size mismatch in " << name << ": " << a.size() << " vs " << b.size() << std::endl;
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
            if (num_errors < 5) {  // Show first 5 errors
                std::cout << "  Error at [" << i << "]: " << a[i] << " vs " << b[i] 
                         << " (diff: " << diff << ", rel: " << rel_diff << ")" << std::endl;
            }
            num_errors++;
        }
    }
    
    if (num_errors > 0) {
        std::cout << name << " failed: " << num_errors << "/" << a.size() 
                 << " elements differ (max_diff: " << max_diff << ", max_rel: " << max_rel_diff << ")" << std::endl;
        return false;
    } else {
        std::cout << name << " passed (max_diff: " << max_diff 
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

void print_int_array_summary(const std::vector<int>& arr, const std::string& name) {
    if (arr.empty()) return;
    
    std::cout << "  " << name << " [" << arr.size() << "]: ";
    for (size_t i = 0; i < std::min(arr.size(), size_t(10)); i++) {
        std::cout << arr[i] << " ";
    }
    if (arr.size() > 10) std::cout << "...";
    std::cout << std::endl;
}

// ============================================================================
// Test Cases
// ============================================================================

bool test_gpt_forward_single_token() {
    std::cout << "\nTesting GPT Forward Pass (Single Token)..." << std::endl;
    
    // Load test data
    GPTTestData test_data = load_gpt_test_data("test_data/gpt_forward_single_token_test");
    
    if (test_data.get_int_array("input_ids").empty() || test_data.get_float_array("logits").empty()) {
        std::cout << "Could not load single token test data, skipping test" << std::endl;
        return true;
    }
    
    // Get test configuration
    std::vector<int> input_ids = test_data.get_int_array("input_ids");
    std::vector<float> expected_logits = test_data.get_float_array("logits");
    
    int batch_size = 1;
    int seq_len = input_ids.size() / batch_size;
    int vocab_size = test_data.get_scalar("config_vocab_size");
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << ", Seq len: " << seq_len << ", Vocab size: " << vocab_size << std::endl;
    
    print_int_array_summary(input_ids, "Input tokens");
    print_array_summary(expected_logits, "Expected logits");
    
    // Create test model using the unified GPTModel
    auto model = GPTModel::create_from_test_data(test_data);
    
    // Run forward pass
    std::vector<float> cuda_logits(expected_logits.size());
    model->forward(input_ids.data(), cuda_logits.data(), batch_size, seq_len);
    
    print_array_summary(cuda_logits, "CUDA logits");
    
    // Compare results
    bool success = compare_arrays(cuda_logits, expected_logits, 1e-3, 1e-5, "Single Token GPT Forward");
    
    return success;
}

bool test_gpt_forward_multi_token() {
    std::cout << "\nTesting GPT Forward Pass (Multi Token)..." << std::endl;
    
    // Load test data
    GPTTestData test_data = load_gpt_test_data("test_data/gpt_forward_multi_token_test");
    
    if (test_data.get_int_array("input_ids").empty() || test_data.get_float_array("logits").empty()) {
        std::cout << "Could not load multi token test data, skipping test" << std::endl;
        return true;
    }
    
    // Get test configuration
    std::vector<int> input_ids = test_data.get_int_array("input_ids");
    std::vector<float> expected_logits = test_data.get_float_array("logits");
    
    int batch_size = 1;
    int seq_len = input_ids.size() / batch_size;
    int vocab_size = test_data.get_scalar("config_vocab_size");
    
    std::cout << "Test configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << ", Seq len: " << seq_len << ", Vocab size: " << vocab_size << std::endl;
    
    print_int_array_summary(input_ids, "Input tokens");
    print_array_summary(expected_logits, "Expected logits");
    
    // Create test model using the unified GPTModel
    auto model = GPTModel::create_from_test_data(test_data);
    
    // Run forward pass
    std::vector<float> cuda_logits(expected_logits.size());
    model->forward(input_ids.data(), cuda_logits.data(), batch_size, seq_len);
    
    print_array_summary(cuda_logits, "CUDA logits");
    
    // Compare results
    bool success = compare_arrays(cuda_logits, expected_logits, 1e-3, 1e-5, "Multi Token GPT Forward");
    
    return success;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "=== CUDA GPT Test Suite ===" << std::endl;
    
    // Check CUDA setup
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    if (device_count == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    // Run tests
    std::vector<std::pair<std::string, bool>> results;
    
    results.push_back({"GPT Forward (Single Token)", test_gpt_forward_single_token()});
    results.push_back({"GPT Forward (Multi Token)", test_gpt_forward_multi_token()});
    
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
