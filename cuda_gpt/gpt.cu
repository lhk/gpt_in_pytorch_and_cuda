#include "modules.cu"
#include "weight_loader.h"
#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Test data structure (used for loading test weights)
struct GPTTestData {
    std::map<std::string, std::vector<float>> float_arrays;
    std::map<std::string, std::vector<int>> int_arrays;
    std::map<std::string, int> scalars;
    
    const std::vector<float>& get_float_array(const std::string& name) const {
        auto it = float_arrays.find(name);
        if (it == float_arrays.end()) {
            std::cerr << "Error: Float array '" << name << "' not found in test data" << std::endl;
            exit(1);
        }
        return it->second;
    }
    
    const std::vector<int>& get_int_array(const std::string& name) const {
        auto it = int_arrays.find(name);
        if (it == int_arrays.end()) {
            std::cerr << "Error: Int array '" << name << "' not found in test data" << std::endl;
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

// Utility functions to convert test data to unified format
class TestDataConverter {
public:
    static GPTConfig extract_config_from_test_data(const GPTTestData& test_data) {
        GPTConfig config;
        config.d_model = test_data.get_scalar("config_d_model");
        config.n_heads = test_data.get_scalar("config_n_heads");
        config.d_ff = test_data.get_scalar("config_d_ff");
        config.n_layers = test_data.get_scalar("config_n_layers");
        config.max_seq_len = test_data.get_scalar("config_max_seq_len");
        config.vocab_size = test_data.get_scalar("config_vocab_size");
        config.eps = 1e-5f;
        return config;
    }
    
    static ExtractedWeights convert_test_data_to_weights(const GPTTestData& test_data) {
        ExtractedWeights weights;
        
        // Extract config
        weights.d_model = test_data.get_scalar("config_d_model");
        weights.n_heads = test_data.get_scalar("config_n_heads");
        weights.d_ff = test_data.get_scalar("config_d_ff");
        weights.n_layers = test_data.get_scalar("config_n_layers");
        weights.max_seq_len = test_data.get_scalar("config_max_seq_len");
        weights.vocab_size = test_data.get_scalar("config_vocab_size");
        
        // Copy embeddings
        weights.token_embedding = test_data.get_float_array("token_embedding");
        weights.pos_embedding = test_data.get_float_array("pos_embedding");
        
        // Copy final layer weights
        weights.final_ln_gamma = test_data.get_float_array("final_ln_gamma");
        weights.final_ln_beta = test_data.get_float_array("final_ln_beta");
        weights.lm_head = test_data.get_float_array("lm_head_weight");
        
        // Copy layer weights
        weights.layers.resize(weights.n_layers);
        for (int layer = 0; layer < weights.n_layers; layer++) {
            std::string layer_prefix = "layer_" + std::to_string(layer) + "_";
            
            // MHSA weights
            weights.layers[layer].w_q = test_data.get_float_array(layer_prefix + "mhsa_w_q");
            weights.layers[layer].w_k = test_data.get_float_array(layer_prefix + "mhsa_w_k");
            weights.layers[layer].w_v = test_data.get_float_array(layer_prefix + "mhsa_w_v");
            weights.layers[layer].w_o = test_data.get_float_array(layer_prefix + "mhsa_w_o");
            weights.layers[layer].w_o_bias = test_data.get_float_array(layer_prefix + "mhsa_w_o_bias");
            
            // MLP weights
            weights.layers[layer].mlp_w1 = test_data.get_float_array(layer_prefix + "mlp_w1");
            weights.layers[layer].mlp_b1 = test_data.get_float_array(layer_prefix + "mlp_b1");
            weights.layers[layer].mlp_w2 = test_data.get_float_array(layer_prefix + "mlp_w2");
            weights.layers[layer].mlp_b2 = test_data.get_float_array(layer_prefix + "mlp_b2");
            
            // LayerNorm weights
            weights.layers[layer].ln1_gamma = test_data.get_float_array(layer_prefix + "ln1_gamma");
            weights.layers[layer].ln1_beta = test_data.get_float_array(layer_prefix + "ln1_beta");
            weights.layers[layer].ln2_gamma = test_data.get_float_array(layer_prefix + "ln2_gamma");
            weights.layers[layer].ln2_beta = test_data.get_float_array(layer_prefix + "ln2_beta");
        }
        
        return weights;
    }
};

// Main GPT Model
class GPTModel {
public:
    GPTConfig config;
    
private:
    std::vector<std::unique_ptr<TransformerBlock>> blocks;
    std::unique_ptr<LayerNorm> final_ln;  // Final layer normalization
    
    // Embedding lookup table (stored on host for now)
    float *h_token_embeddings;  // [vocab_size, d_model]
    float *h_pos_embeddings;    // [max_seq_len, d_model]
    
    // Final output projection weights
    float *d_lm_head_weights;   // [d_model, vocab_size]
    float *d_lm_head_bias;      // [vocab_size]
    
    // Device buffers
    float *d_input_embeddings;
    float *d_final_hidden;
    float *d_final_norm;
    float *d_logits;
    
public:
    // Single unified constructor
    GPTModel(const GPTConfig& cfg, const ExtractedWeights& weights) : config(cfg) {
        // Allocate embedding tables
        h_token_embeddings = new float[config.vocab_size * config.d_model];
        h_pos_embeddings = new float[config.max_seq_len * config.d_model];
        
        // Copy embeddings
        memcpy(h_token_embeddings, weights.token_embedding.data(), 
               config.vocab_size * config.d_model * sizeof(float));
        memcpy(h_pos_embeddings, weights.pos_embedding.data(), 
               config.max_seq_len * config.d_model * sizeof(float));
        
        init_model();
        load_weights(weights);
    }

private:
    void init_model() {
        // Allocate final layer norm using smart pointer
        final_ln = std::make_unique<LayerNorm>(config.d_model);
        
        // Allocate output projection weights
        cudaMalloc(&d_lm_head_weights, config.d_model * config.vocab_size * sizeof(float));
        cudaMalloc(&d_lm_head_bias, config.vocab_size * sizeof(float));
        
        // Initialize output projection with dummy weights (will be overridden if real weights provided)
        std::vector<float> lm_head_weights(config.d_model * config.vocab_size, 0.01f);
        std::vector<float> lm_head_bias(config.vocab_size, 0.0f);
        cudaMemcpy(d_lm_head_weights, lm_head_weights.data(), 
                  config.d_model * config.vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lm_head_bias, lm_head_bias.data(), 
                  config.vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Allocate device buffers
        int max_batch = 16;
        cudaMalloc(&d_input_embeddings, max_batch * config.max_seq_len * config.d_model * sizeof(float));
        cudaMalloc(&d_final_hidden, max_batch * config.max_seq_len * config.d_model * sizeof(float));
        cudaMalloc(&d_final_norm, max_batch * config.max_seq_len * config.d_model * sizeof(float));
        cudaMalloc(&d_logits, max_batch * config.max_seq_len * config.vocab_size * sizeof(float));
    }
    
    void load_weights(const ExtractedWeights& weights) {
        std::cout << "Loading weights into CUDA model..." << std::endl;
        
        // Load final layer norm weights
        final_ln->load_weights(weights.final_ln_gamma.data(), weights.final_ln_beta.data());
        
        // Load output projection weights (note: lm_head has no bias in the checkpoint, keep bias as zero)
        cudaMemcpy(d_lm_head_weights, weights.lm_head.data(), 
                  config.d_model * config.vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        // d_lm_head_bias remains zero from init_model()
        
        // Create transformer blocks with weights
        for (int layer = 0; layer < config.n_layers; layer++) {
            const ExtractedWeights::LayerWeights& layer_weights = weights.layers[layer];
            
            const float* mhsa_weights[4] = {
                layer_weights.w_q.data(),
                layer_weights.w_k.data(),
                layer_weights.w_v.data(),
                layer_weights.w_o.data()
            };
            
            const float* mlp_weights[2] = {
                layer_weights.mlp_w1.data(),
                layer_weights.mlp_w2.data()
            };
            
            const float* mlp_biases[2] = {
                layer_weights.mlp_b1.data(),
                layer_weights.mlp_b2.data()
            };
            
            blocks.push_back(std::make_unique<TransformerBlock>(
                config.d_model, config.n_heads, config.d_ff, config.max_seq_len,
                mhsa_weights, layer_weights.w_o_bias.data(),
                mlp_weights, mlp_biases,
                layer_weights.ln1_gamma.data(), layer_weights.ln1_beta.data(),
                layer_weights.ln2_gamma.data(), layer_weights.ln2_beta.data()
            ));
            
            std::cout << "  Loaded layer " << layer << " weights" << std::endl;
        }
        
        std::cout << "✓ All weights loaded into CUDA model!" << std::endl;
    }

public:
    
    ~GPTModel() {
        // Clean up host memory arrays
        delete[] h_token_embeddings;
        delete[] h_pos_embeddings;
        
        // Clean up CUDA device memory
        cudaFree(d_lm_head_weights);
        cudaFree(d_lm_head_bias);
        cudaFree(d_input_embeddings);
        cudaFree(d_final_hidden);
        cudaFree(d_final_norm);
        cudaFree(d_logits);
        
        // Note: blocks and final_ln are automatically cleaned up by smart pointers
    }
    
    // Host-side embedding lookup: token_ids -> embeddings
    void embed_tokens(const int* token_ids, float* embeddings, int batch_size, int seq_len) {
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                int token_id = token_ids[b * seq_len + s];
                for (int d = 0; d < config.d_model; d++) {
                    // Token embedding + positional embedding
                    float token_emb = h_token_embeddings[token_id * config.d_model + d];
                    float pos_emb = h_pos_embeddings[s * config.d_model + d];
                    embeddings[(b * seq_len + s) * config.d_model + d] = token_emb + pos_emb;
                }
            }
        }
    }
    
    // Forward pass: token_ids -> logits
    void forward(const int* token_ids, float* logits, int batch_size, int seq_len) {
        // Host-side embedding lookup
        float* h_embeddings = new float[batch_size * seq_len * config.d_model];
        embed_tokens(token_ids, h_embeddings, batch_size, seq_len);
        
        // Copy embeddings to device
        cudaMemcpy(d_input_embeddings, h_embeddings, 
                  batch_size * seq_len * config.d_model * sizeof(float), 
                  cudaMemcpyHostToDevice);
        delete[] h_embeddings;
        
        // Pass through transformer blocks
        float* current_input = d_input_embeddings;
        float* current_output = d_final_hidden;
        
        for (size_t i = 0; i < blocks.size(); i++) {
            blocks[i]->forward(current_input, current_output, batch_size, seq_len);
            // Swap buffers for next layer
            std::swap(current_input, current_output);
        }
        
        // Final layer normalization
        final_ln->forward(current_input, d_final_norm, batch_size, seq_len);
        
        // Output projection: [batch, seq_len, d_model] @ [d_model, vocab_size] -> [batch, seq_len, vocab_size]
        // Note: PyTorch lm_head has no bias, so we use zero bias
        dim3 blocks_lm(batch_size, seq_len);
        dim3 threads_lm(config.vocab_size);
        linear_bias_kernel<<<blocks_lm, threads_lm>>>(
            d_final_norm, d_lm_head_weights, d_lm_head_bias, d_logits, 
            batch_size, seq_len, config.d_model, config.vocab_size);
        cudaDeviceSynchronize();
        
        // Copy logits back to host
        cudaMemcpy(logits, d_logits, 
                  batch_size * seq_len * config.vocab_size * sizeof(float), 
                  cudaMemcpyDeviceToHost);
    }
    
    // Load weights from checkpoint file
    void load_checkpoint(const std::string& checkpoint_path) {
        // TODO: Implement checkpoint loading
        std::cout << "Loading checkpoint from: " << checkpoint_path << std::endl;
        // This would read binary files and populate all weight arrays
    }
    
    // Generate next token with optional temperature sampling
    int generate_next_token(const std::vector<int>& sequence, float temperature = 1.0, int top_k = -1) {
        if (sequence.empty() || sequence.size() > config.max_seq_len) {
            throw std::runtime_error("Invalid sequence length for generation");
        }
        
        // Run forward pass on current sequence
        int batch_size = 1;
        int seq_len = sequence.size();
        std::vector<float> logits(batch_size * seq_len * config.vocab_size);
        
        forward(sequence.data(), logits.data(), batch_size, seq_len);
        
        // Get logits for the last token (most recent position)
        float* last_token_logits = &logits[(seq_len - 1) * config.vocab_size];
        
        // Copy logits to work with them
        std::vector<float> working_logits(last_token_logits, last_token_logits + config.vocab_size);
        
        // Apply temperature scaling
        if (temperature > 0.0f && temperature != 1.0f) {
            for (int i = 0; i < config.vocab_size; i++) {
                working_logits[i] /= temperature;
            }
        }
        
        // Apply top-k filtering if specified
        if (top_k > 0 && top_k < config.vocab_size) {
            // Create vector of (logit, index) pairs
            std::vector<std::pair<float, int>> logit_pairs;
            for (int i = 0; i < config.vocab_size; i++) {
                logit_pairs.push_back({working_logits[i], i});
            }
            
            // Sort by logit value (highest first)
            std::sort(logit_pairs.begin(), logit_pairs.end(), 
                     [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                         return a.first > b.first;
                     });
            
            // Set all but top-k logits to -infinity
            float threshold = logit_pairs[top_k - 1].first;
            for (int i = 0; i < config.vocab_size; i++) {
                if (working_logits[i] < threshold) {
                    working_logits[i] = -INFINITY;
                }
            }
        }
        
        // Convert logits to probabilities using softmax
        std::vector<float> probs(config.vocab_size);
        
        // Find max logit for numerical stability
        float max_logit = *std::max_element(working_logits.begin(), working_logits.end());
        
        // Compute softmax
        float sum_exp = 0.0f;
        for (int i = 0; i < config.vocab_size; i++) {
            if (working_logits[i] == -INFINITY) {
                probs[i] = 0.0f;
            } else {
                probs[i] = expf(working_logits[i] - max_logit);
                sum_exp += probs[i];
            }
        }
        
        // Normalize probabilities
        for (int i = 0; i < config.vocab_size; i++) {
            probs[i] /= sum_exp;
        }
        
        // Sample from the probability distribution
        int selected_token;
        if (temperature == 0.0f) {
            // Greedy sampling: find token with highest probability
            selected_token = std::max_element(probs.begin(), probs.end()) - probs.begin();
        } else {
            // Multinomial sampling
            float random_val = static_cast<float>(rand()) / RAND_MAX;
            float cumulative_prob = 0.0f;
            selected_token = config.vocab_size - 1; // fallback
            
            for (int i = 0; i < config.vocab_size; i++) {
                cumulative_prob += probs[i];
                if (random_val <= cumulative_prob) {
                    selected_token = i;
                    break;
                }
            }
        }
        
        return selected_token;
    }
    
    // Autoregressive text generation with configurable sampling
    std::vector<int> generate_tokens(const std::vector<int>& prompt_tokens, int max_new_tokens, 
                                   float temperature = 1.0, int top_k = -1) {
        if (prompt_tokens.empty()) {
            throw std::runtime_error("Empty prompt provided");
        }
        
        if (prompt_tokens.size() + max_new_tokens > config.max_seq_len) {
            throw std::runtime_error("Prompt + generation would exceed max sequence length");
        }
        
        std::cout << "Generating " << max_new_tokens << " tokens with temperature=" 
                  << temperature << ", top_k=" << top_k << std::endl;
        
        // Start with the prompt
        std::vector<int> sequence = prompt_tokens;
        
        // Generate tokens one by one
        for (int i = 0; i < max_new_tokens; i++) {
            int next_token = generate_next_token(sequence, temperature, top_k);
            sequence.push_back(next_token);
        }
        
        // Return only the newly generated tokens (excluding prompt)
        std::vector<int> generated_tokens(sequence.begin() + prompt_tokens.size(), sequence.end());
        return generated_tokens;
    }
    
    // Convenience method: generate from prompt tokens and return full sequence
    std::vector<int> generate_sequence(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                     float temperature = 1.0, int top_k = -1) {
        std::vector<int> generated = generate_tokens(prompt_tokens, max_new_tokens, temperature, top_k);
        
        // Combine prompt + generated
        std::vector<int> full_sequence = prompt_tokens;
        full_sequence.insert(full_sequence.end(), generated.begin(), generated.end());
        
        return full_sequence;
    }
    
    // Static factory method to create model with real weights from checkpoint
    static std::unique_ptr<GPTModel> create_from_weights(const std::string& weights_dir) {
        std::cout << "Creating GPT model from extracted weights..." << std::endl;
        
        // Load weights using WeightLoader
        ExtractedWeights weights = WeightLoader::load_weights(weights_dir);
        
        // Extract config from weights
        GPTConfig config = WeightLoader::extract_config(weights);
        
        // Create model with unified constructor
        return std::unique_ptr<GPTModel>(new GPTModel(config, weights));
    }
    
    // Static factory method to create model from test data
    static std::unique_ptr<GPTModel> create_from_test_data(const GPTTestData& test_data) {
        std::cout << "Creating GPT model from test data..." << std::endl;
        
        // Convert test data to unified format
        GPTConfig config = TestDataConverter::extract_config_from_test_data(test_data);
        ExtractedWeights weights = TestDataConverter::convert_test_data_to_weights(test_data);
        
        // Create model with unified constructor
        return std::unique_ptr<GPTModel>(new GPTModel(config, weights));
    }
};

// Simple character tokenizer (mirrors the Python implementation)
class CharacterTokenizer {
private:
    std::vector<char> chars;
    std::map<char, int> char_to_idx;
    std::map<int, char> idx_to_char;
    
public:
    int vocab_size;
    
    // Constructor that loads from reference tokenizer file (preferred)
    CharacterTokenizer(const std::string& reference_file) {
        std::ifstream file(reference_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open reference tokenizer file: " + reference_file);
        }
        
        std::string line;
        // Read vocab size
        if (!std::getline(file, line) || line.substr(0, 11) != "vocab_size=") {
            throw std::runtime_error("Invalid reference tokenizer format");
        }
        vocab_size = std::stoi(line.substr(11));
        
        chars.resize(vocab_size);
        
        // Read character mappings: index:ascii_code
        while (std::getline(file, line)) {
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) continue;
            
            int index = std::stoi(line.substr(0, colon_pos));
            int ascii_code = std::stoi(line.substr(colon_pos + 1));
            
            if (index >= 0 && index < vocab_size) {
                char c = static_cast<char>(ascii_code);
                chars[index] = c;
                char_to_idx[c] = index;
                idx_to_char[index] = c;
            }
        }
        
        std::cout << "Loaded reference tokenizer with " << vocab_size << " characters" << std::endl;
        std::cout << "First 20 chars: ";
        for (int i = 0; i < std::min(20, vocab_size); i++) {
            char c = chars[i];
            if (c == '\n') std::cout << "\\n";
            else if (c == '\t') std::cout << "\\t";
            else if (c == ' ') std::cout << "_";
            else std::cout << c;
        }
        std::cout << (vocab_size > 20 ? "..." : "") << std::endl;
    }
    
    // Legacy constructor that creates tokenizer from text (for backwards compatibility)
    CharacterTokenizer(const std::string& text, bool is_text_content) {
        if (!is_text_content) {
            // This is a file path, redirect to the reference file constructor
            *this = CharacterTokenizer(text);
            return;
        }
        
        // Get all unique characters and sort them (matches Python sorted(list(set(text))))
        std::set<char> unique_chars;
        for (char c : text) {
            unique_chars.insert(c);
        }
        
        chars = std::vector<char>(unique_chars.begin(), unique_chars.end());
        vocab_size = chars.size();
        
        // Create mappings
        for (int i = 0; i < vocab_size; i++) {
            char_to_idx[chars[i]] = i;
            idx_to_char[i] = chars[i];
        }
        
        std::cout << "Tokenizer created from text with " << vocab_size << " characters" << std::endl;
        std::cout << "First 20 chars: ";
        for (int i = 0; i < std::min(20, vocab_size); i++) {
            char c = chars[i];
            if (c == '\n') std::cout << "\\n";
            else if (c == '\t') std::cout << "\\t";
            else if (c == ' ') std::cout << "_";
            else std::cout << c;
        }
        std::cout << (vocab_size > 20 ? "..." : "") << std::endl;
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            auto it = char_to_idx.find(c);
            if (it != char_to_idx.end()) {
                tokens.push_back(it->second);
            } else {
                std::cerr << "Warning: Unknown character '" << c << "' (skipping)" << std::endl;
            }
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for (int token : tokens) {
            auto it = idx_to_char.find(token);
            if (it != idx_to_char.end()) {
                text += it->second;
            } else {
                std::cerr << "Warning: Unknown token " << token << " (skipping)" << std::endl;
            }
        }
        return text;
    }
};

// Load a small sample of text for tokenizer creation and testing
std::string load_sample_text(const std::string& file_path, int max_chars = -1) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }
    
    std::string text;
    if (max_chars <= 0) {
        // Load entire file
        file.seekg(0, std::ios::end);
        text.reserve(file.tellg());
        file.seekg(0, std::ios::beg);
        text.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    } else {
        // Load limited characters
        char c;
        int count = 0;
        while (file.get(c) && count < max_chars) {
            text += c;
            count++;
        }
    }
    
    std::cout << "Loaded " << text.length() << " characters for tokenizer" << std::endl;
    return text;
}
