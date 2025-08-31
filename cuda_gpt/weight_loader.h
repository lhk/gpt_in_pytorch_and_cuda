#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

// Configuration struct (matches the one in gpt.cu)
struct GPTConfig {
    int d_model = 256;
    int n_heads = 4;
    int d_ff = 512;
    int n_layers = 4;
    int max_seq_len = 256;
    int vocab_size = 107;
    float eps = 1e-5f;
};

// Structure to hold all extracted weights
struct ExtractedWeights {
    // Config
    int d_model, n_heads, d_ff, n_layers, max_seq_len, vocab_size;
    
    // Embeddings
    std::vector<float> token_embedding;
    std::vector<float> pos_embedding;
    
    // Per-layer weights
    struct LayerWeights {
        std::vector<float> w_q, w_k, w_v, w_o, w_o_bias;
        std::vector<float> ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
        std::vector<float> mlp_w1, mlp_b1, mlp_w2, mlp_b2;
    };
    std::vector<LayerWeights> layers;
    
    // Final weights
    std::vector<float> final_ln_gamma, final_ln_beta;
    std::vector<float> lm_head;
};

class WeightLoader {
public:
    // Load both config and weights from checkpoint directory
    static ExtractedWeights load_weights(const std::string& weights_dir) {
        ExtractedWeights weights;
        
        // Load config
        load_config(weights_dir + "/config.txt", weights);
        
        std::cout << "Loading weights for config:" << std::endl;
        std::cout << "  d_model: " << weights.d_model << std::endl;
        std::cout << "  n_heads: " << weights.n_heads << std::endl;
        std::cout << "  d_ff: " << weights.d_ff << std::endl;
        std::cout << "  n_layers: " << weights.n_layers << std::endl;
        std::cout << "  vocab_size: " << weights.vocab_size << std::endl;
        
        // Load embeddings
        weights.token_embedding = load_binary_float(weights_dir + "/token_embedding.bin");
        weights.pos_embedding = load_binary_float(weights_dir + "/pos_embedding.bin");
        
        std::cout << "Loaded embeddings: token(" << weights.token_embedding.size() 
                  << "), pos(" << weights.pos_embedding.size() << ")" << std::endl;
        
        // Load layer weights
        weights.layers.resize(weights.n_layers);
        for (int i = 0; i < weights.n_layers; i++) {
            std::string layer_dir = weights_dir + "/layer_" + std::to_string(i);
            
            // MHSA weights
            weights.layers[i].w_q = load_binary_float(layer_dir + "/w_q.bin");
            weights.layers[i].w_k = load_binary_float(layer_dir + "/w_k.bin");
            weights.layers[i].w_v = load_binary_float(layer_dir + "/w_v.bin");
            weights.layers[i].w_o = load_binary_float(layer_dir + "/w_o.bin");
            weights.layers[i].w_o_bias = load_binary_float(layer_dir + "/w_o_bias.bin");
            
            // Layer norms
            weights.layers[i].ln1_gamma = load_binary_float(layer_dir + "/ln1_gamma.bin");
            weights.layers[i].ln1_beta = load_binary_float(layer_dir + "/ln1_beta.bin");
            weights.layers[i].ln2_gamma = load_binary_float(layer_dir + "/ln2_gamma.bin");
            weights.layers[i].ln2_beta = load_binary_float(layer_dir + "/ln2_beta.bin");
            
            // MLP weights
            weights.layers[i].mlp_w1 = load_binary_float(layer_dir + "/mlp_w1.bin");
            weights.layers[i].mlp_b1 = load_binary_float(layer_dir + "/mlp_b1.bin");
            weights.layers[i].mlp_w2 = load_binary_float(layer_dir + "/mlp_w2.bin");
            weights.layers[i].mlp_b2 = load_binary_float(layer_dir + "/mlp_b2.bin");
            
            std::cout << "Loaded layer " << i << " weights" << std::endl;
        }
        
        // Load final weights
        weights.final_ln_gamma = load_binary_float(weights_dir + "/final_ln_gamma.bin");
        weights.final_ln_beta = load_binary_float(weights_dir + "/final_ln_beta.bin");
        weights.lm_head = load_binary_float(weights_dir + "/lm_head.bin");
        
        std::cout << "✓ All weights loaded successfully!" << std::endl;
        
        return weights;
    }
    
    // Extract GPTConfig from ExtractedWeights
    static GPTConfig extract_config(const ExtractedWeights& weights) {
        GPTConfig config;
        config.d_model = weights.d_model;
        config.n_heads = weights.n_heads;
        config.d_ff = weights.d_ff;
        config.n_layers = weights.n_layers;
        config.max_seq_len = weights.max_seq_len;
        config.vocab_size = weights.vocab_size;
        config.eps = 1e-5f;  // Default value
        return config;
    }

private:
    static void load_config(const std::string& config_path, ExtractedWeights& weights) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + config_path);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            auto pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            std::string key = line.substr(0, pos);
            int value = std::stoi(line.substr(pos + 1));
            
            if (key == "d_model") weights.d_model = value;
            else if (key == "n_heads") weights.n_heads = value;
            else if (key == "d_ff") weights.d_ff = value;
            else if (key == "n_layers") weights.n_layers = value;
            else if (key == "max_seq_len") weights.max_seq_len = value;
            else if (key == "vocab_size") weights.vocab_size = value;
        }
    }
    
    static std::vector<float> load_binary_float(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Read all floats
        std::vector<float> data(size / sizeof(float));
        file.read(reinterpret_cast<char*>(data.data()), size);
        
        return data;
    }
};
