#include "gpt.cu"
#include <iostream>
#include <string>
#include <chrono>

int main() {
    try {
        // Load model
        std::string weights_dir = "/home/timely/lklein/shakespeare_gpt/cuda_gpt/weights";
        auto model = GPTModel::create_from_weights(weights_dir);
        
        // Load tokenizer
        std::string tokenizer_path = "functional_tests/test_data/reference_tokenizer.txt";
        CharacterTokenizer tokenizer(tokenizer_path);
        
        // Get user input
        std::cout << "Enter prompt: ";
        std::string prompt;
        std::getline(std::cin, prompt);
        
        // Generate with timing
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        std::vector<int> generated_tokens = model->generate_tokens(prompt_tokens, 100, 0.8f, 40);
        std::string generated_text = tokenizer.decode(generated_tokens);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate timing
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        float duration_ms = duration.count() / 1000.0f;
        
        // Output
        std::cout << "Generated in " << duration_ms << " ms: " << prompt << generated_text << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
