#include "gpt.cu"
#include <iostream>
#include <vector>
#include <exception>

// Example usage and text generation demo
int main() {
    try {
        // Initialize random seed for sampling
        srand(static_cast<unsigned int>(time(nullptr)));
        
        std::cout << "=== CUDA GPT Text Generation ===" << std::endl;
        
        // Create model from extracted weights
        std::string weights_dir = "/home/timely/lklein/shakespeare_gpt/cuda_gpt/weights";
        auto model = GPTModel::create_from_weights(weights_dir);
        
        // Load tokenizer from reference file (matches Python training tokenizer exactly)
        std::string tokenizer_path = "functional_tests/test_data/reference_tokenizer.txt";
        CharacterTokenizer tokenizer(tokenizer_path);
        
        // Verify tokenizer matches model vocab size
        if (tokenizer.vocab_size != model->config.vocab_size) {
            std::cerr << "Warning: Tokenizer vocab size (" << tokenizer.vocab_size 
                      << ") doesn't match model (" << model->config.vocab_size << ")" << std::endl;
        }
        
        std::cout << "\n=== Simple Tokenizer Test ===" << std::endl;
        
        // Test basic characters
        std::vector<std::string> test_chars = {"A", " ", "a", "H"};
        for (const std::string& ch : test_chars) {
            std::vector<int> tokens = tokenizer.encode(ch);
            std::string decoded = tokenizer.decode(tokens);
            std::cout << "'" << ch << "' -> " << (tokens.empty() ? -1 : tokens[0]) << " -> '" << decoded << "'" << std::endl;
        }
        
        std::cout << "\n=== Single Character Generation Test ===" << std::endl;
        
        // Test with a simple single character
        std::string simple_prompt = "H";
        std::vector<int> simple_tokens = tokenizer.encode(simple_prompt);
        
        if (!simple_tokens.empty()) {
            std::cout << "Testing with single character 'H' -> token " << simple_tokens[0] << std::endl;
            
            // Generate just one token to debug
            int next_token = model->generate_next_token(simple_tokens);
            std::string next_char = tokenizer.decode({next_token});
            
            std::cout << "Generated token: " << next_token << " -> '" << next_char << "'" << std::endl;
        }
        
        std::cout << "\n=== Text Generation Demo ===" << std::endl;
        
        // Test prompts
        std::vector<std::string> prompts = {
            "HAMLET:",
            "To be or not to be",
            "The king",
            "What is"
        };
        
        // Test different sampling strategies
        struct SamplingConfig {
            float temperature;
            int top_k;
            std::string name;
        };
        
        std::vector<SamplingConfig> sampling_configs = {
            {0.0f, -1, "Greedy"},
            {0.8f, -1, "Temperature 0.8"},
            {1.0f, 40, "Top-k 40"},
            {0.8f, 40, "Temperature 0.8 + Top-k 40"}
        };
        
        for (const std::string& prompt : prompts) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "PROMPT: \"" << prompt << "\"" << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            
            // Encode prompt
            std::vector<int> prompt_tokens = tokenizer.encode(prompt);
            std::cout << "Prompt tokens (" << prompt_tokens.size() << "): ";
            for (int i = 0; i < std::min(10, (int)prompt_tokens.size()); i++) {
                std::cout << prompt_tokens[i] << " ";
            }
            std::cout << std::endl;
            
            for (const auto& config : sampling_configs) {
                std::cout << "\n[" << config.name << "]" << std::endl;
                
                try {
                    // Generate continuation  
                    int max_new_tokens = 50;
                    std::vector<int> generated_tokens = model->generate_tokens(
                        prompt_tokens, max_new_tokens, config.temperature, config.top_k);
                    
                    // Decode generated text
                    std::string generated_text = tokenizer.decode(generated_tokens);
                    
                    // Print results
                    std::cout << "Generated: \"" << prompt << generated_text << "\"" << std::endl;
                    std::cout << "Length: " << generated_text.length() << " chars" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cout << "Error with " << config.name << ": " << e.what() << std::endl;
                }
            }
        }
        
        std::cout << "\n=== Testing Forward Pass Consistency ===" << std::endl;
        
        // Quick test: forward pass with known tokens
        std::vector<int> test_tokens = {1, 2, 3, 4, 5};
        int batch_size = 1;
        int seq_len = test_tokens.size();
        
        std::vector<float> logits(batch_size * seq_len * model->config.vocab_size);
        model->forward(test_tokens.data(), logits.data(), batch_size, seq_len);
        
        std::cout << "Forward pass completed successfully!" << std::endl;
        std::cout << "Output shape: [" << batch_size << ", " << seq_len << ", " << model->config.vocab_size << "]" << std::endl;
        
        // Print sample logits for last position (used for generation)
        std::cout << "\nSample logits for last position (first 10 values):" << std::endl;
        int last_pos_offset = (seq_len - 1) * model->config.vocab_size;
        for (int i = 0; i < 10; i++) {
            std::cout << "  logits[" << i << "] = " << logits[last_pos_offset + i] << std::endl;
        }
        
        std::cout << "\nCUDA GPT text generation working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
