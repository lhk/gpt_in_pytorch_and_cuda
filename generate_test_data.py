#!/usr/bin/env python3
"""
Generate reference test data from PyTorch modules for CUDA validation.
Creates .npz files with input/output pairs and metadata.
"""

import sys
import os
sys.path.append('/home/timely/lklein/shakespeare_gpt/src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path

# Import the PyTorch modules
from models.gpt import GPTConfig, MultiHeadSelfAttention, MLP, TransformerBlock

def create_test_config():
    """Create test configuration matching our CUDA tests."""
    return GPTConfig(
        d_model=4,      # Small for debugging
        n_heads=2,      # 2 heads, d_k = 2
        d_ff=8,         # Small FFN
        n_layers=1,     # Single layer for now
        max_seq_len=3,  # Short sequences
        dropout=0.0     # No dropout for deterministic testing
    )

def get_test_input():
    """Create the same test input as used in sandbox/kernels."""
    torch.manual_seed(42)
    batch_size, seq_len, d_model = 2, 3, 4
    
    # Use the exact same test data as sandbox
    x_data = [
        1.926915, 1.487284, 0.900717, -2.105521,
        0.678418, -1.234545, -0.043067, -1.604667,
        0.355860, -0.686623, -0.493356, 0.241488,
        -1.110904, 0.091546, -2.316923, -0.216805,
        -0.309727, -0.395710, 0.803409, -0.621595,
        -0.592001, -0.063074, -0.828554, 0.330898
    ]
    
    x = torch.tensor(x_data, dtype=torch.float32).view(batch_size, seq_len, d_model)
    return x

def set_identity_weights(module):
    """Set weights to identity matrices for consistent testing."""
    with torch.no_grad():
        if hasattr(module, 'weight'):
            if len(module.weight.shape) == 2:
                # Linear layer - set to identity if square, otherwise keep small random
                out_dim, in_dim = module.weight.shape
                if out_dim == in_dim:
                    module.weight.copy_(torch.eye(out_dim))
                else:
                    # For non-square matrices, use small random values
                    module.weight.normal_(0, 0.02)
        
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.zero_()

def generate_mhsa_test_data():
    """Generate MultiHeadSelfAttention test data."""
    print("Generating MHSA test data...")
    
    config = create_test_config()
    x = get_test_input()
    
    # Create MHSA module
    mhsa = MultiHeadSelfAttention(config)
    
    # Set to eval mode (no dropout)
    mhsa.eval()
    
    # Set identity weights for reproducible testing
    set_identity_weights(mhsa.w_q)
    set_identity_weights(mhsa.w_k)  
    set_identity_weights(mhsa.w_v)
    set_identity_weights(mhsa.w_o)
    
    # Forward pass
    with torch.no_grad():
        output = mhsa(x)
    
    # Extract weights for CUDA
    weights = {
        'w_q': mhsa.w_q.weight.detach().numpy(),
        'w_k': mhsa.w_k.weight.detach().numpy(), 
        'w_v': mhsa.w_v.weight.detach().numpy(),
        'w_o': mhsa.w_o.weight.detach().numpy(),
        'w_o_bias': mhsa.w_o.bias.detach().numpy()
    }
    
    # Save test data
    base_path = 'cuda_gpt/test_data/mhsa_test'
    np.savez(base_path + '.npz',
             input=x.numpy(),
             output=output.numpy(),
             config_d_model=config.d_model,
             config_n_heads=config.n_heads,
             config_d_ff=config.d_ff,
             config_max_seq_len=config.max_seq_len,
             **weights)
    
    # Save as binary files for CUDA
    x.numpy().astype(np.float32).tofile(base_path + '_input.bin')
    output.numpy().astype(np.float32).tofile(base_path + '_output.bin')
    for name, weight in weights.items():
        weight.astype(np.float32).tofile(base_path + f'_{name}.bin')
    
    # Save config
    with open(base_path + '_config.txt', 'w') as f:
        f.write(f'config_d_model={config.d_model}\n')
        f.write(f'config_n_heads={config.n_heads}\n')
        f.write(f'config_d_ff={config.d_ff}\n')
        f.write(f'config_max_seq_len={config.max_seq_len}\n')
    
    print(f"✓ MHSA test data saved")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    return x, output

def generate_mlp_test_data():
    """Generate MLP test data."""
    print("\nGenerating MLP test data...")
    
    config = create_test_config()
    x = get_test_input()
    
    # Create MLP module  
    mlp = MLP(config)
    mlp.eval()
    
    # Set small random weights (identity doesn't work for non-square)
    with torch.no_grad():
        mlp.linear1.weight.normal_(0, 0.02)
        mlp.linear1.bias.zero_()
        mlp.linear2.weight.normal_(0, 0.02) 
        mlp.linear2.bias.zero_()
    
    # Forward pass
    with torch.no_grad():
        output = mlp(x)
    
    # Extract weights
    weights = {
        'linear1_weight': mlp.linear1.weight.detach().numpy(),
        'linear1_bias': mlp.linear1.bias.detach().numpy(),
        'linear2_weight': mlp.linear2.weight.detach().numpy(),
        'linear2_bias': mlp.linear2.bias.detach().numpy()
    }
    
    # Save test data
    base_path = 'cuda_gpt/test_data/mlp_test'
    np.savez(base_path + '.npz',
             input=x.numpy(),
             output=output.numpy(), 
             config_d_model=config.d_model,
             config_d_ff=config.d_ff,
             **weights)
    
    # Save as binary files for CUDA
    x.numpy().astype(np.float32).tofile(base_path + '_input.bin')
    output.numpy().astype(np.float32).tofile(base_path + '_output.bin')
    for name, weight in weights.items():
        weight.astype(np.float32).tofile(base_path + f'_{name}.bin')
    
    # Save config
    with open(base_path + '_config.txt', 'w') as f:
        f.write(f'config_d_model={config.d_model}\n')
        f.write(f'config_d_ff={config.d_ff}\n')
    
    print(f"✓ MLP test data saved")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    return x, output

def generate_layernorm_test_data():
    """Generate LayerNorm test data."""
    print("\nGenerating LayerNorm test data...")
    
    config = create_test_config()
    x = get_test_input()
    
    # Create LayerNorm
    ln = nn.LayerNorm(config.d_model)
    ln.eval()
    
    # Use standard initialization
    with torch.no_grad():
        ln.weight.fill_(1.0)
        ln.bias.zero_()
    
    # Forward pass
    with torch.no_grad():
        output = ln(x)
    
    # Save test data
    base_path = 'cuda_gpt/test_data/layernorm_test'
    np.savez(base_path + '.npz',
             input=x.numpy(),
             output=output.numpy(),
             gamma=ln.weight.detach().numpy(),
             beta=ln.bias.detach().numpy(),
             config_d_model=config.d_model)
    
    # Also save as binary files for CUDA loader
    x.numpy().astype(np.float32).tofile(base_path + '_input.bin')
    output.numpy().astype(np.float32).tofile(base_path + '_output.bin')
    ln.weight.detach().numpy().astype(np.float32).tofile(base_path + '_gamma.bin')
    ln.bias.detach().numpy().astype(np.float32).tofile(base_path + '_beta.bin')
    
    # Save config as text file
    with open(base_path + '_config.txt', 'w') as f:
        f.write(f'config_d_model={config.d_model}\n')
    
    print(f"✓ LayerNorm test data saved")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    return x, output

def generate_transformer_block_test_data():
    """Generate TransformerBlock test data."""
    print("\nGenerating TransformerBlock test data...")
    
    config = create_test_config()
    x = get_test_input()
    
    # Create TransformerBlock
    block = TransformerBlock(config)
    block.eval()
    
    # Set deterministic weights
    set_identity_weights(block.attention.w_q)
    set_identity_weights(block.attention.w_k)
    set_identity_weights(block.attention.w_v) 
    set_identity_weights(block.attention.w_o)
    
    with torch.no_grad():
        # MLP weights - small random
        block.mlp.linear1.weight.normal_(0, 0.02)
        block.mlp.linear1.bias.zero_()
        block.mlp.linear2.weight.normal_(0, 0.02)
        block.mlp.linear2.bias.zero_()
        
        # LayerNorm weights
        block.ln1.weight.fill_(1.0)
        block.ln1.bias.zero_()
        block.ln2.weight.fill_(1.0) 
        block.ln2.bias.zero_()
    
    # Forward pass
    with torch.no_grad():
        output = block(x)
    
    # Extract all weights
    weights = {
        # MHSA weights
        'mhsa_w_q': block.attention.w_q.weight.detach().numpy(),
        'mhsa_w_k': block.attention.w_k.weight.detach().numpy(),
        'mhsa_w_v': block.attention.w_v.weight.detach().numpy(),
        'mhsa_w_o': block.attention.w_o.weight.detach().numpy(),
        'mhsa_w_o_bias': block.attention.w_o.bias.detach().numpy(),
        
        # MLP weights  
        'mlp_w1': block.mlp.linear1.weight.detach().numpy(),
        'mlp_b1': block.mlp.linear1.bias.detach().numpy(),
        'mlp_w2': block.mlp.linear2.weight.detach().numpy(),
        'mlp_b2': block.mlp.linear2.bias.detach().numpy(),
        
        # LayerNorm weights
        'ln1_gamma': block.ln1.weight.detach().numpy(),
        'ln1_beta': block.ln1.bias.detach().numpy(),
        'ln2_gamma': block.ln2.weight.detach().numpy(),
        'ln2_beta': block.ln2.bias.detach().numpy()
    }
    
    # Save test data
    base_path = 'cuda_gpt/test_data/transformer_block_test'
    np.savez(base_path + '.npz',
             input=x.numpy(),
             output=output.numpy(),
             config_d_model=config.d_model,
             config_n_heads=config.n_heads, 
             config_d_ff=config.d_ff,
             config_max_seq_len=config.max_seq_len,
             **weights)
    
    # Save as binary files for CUDA
    x.numpy().astype(np.float32).tofile(base_path + '_input.bin')
    output.numpy().astype(np.float32).tofile(base_path + '_output.bin')
    for name, weight in weights.items():
        weight.astype(np.float32).tofile(base_path + f'_{name}.bin')
    
    # Save config
    with open(base_path + '_config.txt', 'w') as f:
        f.write(f'config_d_model={config.d_model}\n')
        f.write(f'config_n_heads={config.n_heads}\n')
        f.write(f'config_d_ff={config.d_ff}\n')
        f.write(f'config_max_seq_len={config.max_seq_len}\n')
    
    print(f"TransformerBlock test data saved")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    return x, output

def generate_gpt_forward_test_data():
    """Generate full GPT model forward pass test data."""
    print("\nGenerating GPT forward pass test data...")
    
    # Create test configuration matching our CUDA model
    config = GPTConfig(
        d_model=4,      # Small for debugging
        n_heads=2,      # 2 heads, d_k = 2
        d_ff=8,         # Small FFN
        n_layers=2,     # 2 layers for meaningful testing
        max_seq_len=5,  # Longer sequences for testing
        dropout=0.0     # No dropout for deterministic testing
    )
    
    # Create a small vocabulary for testing
    vocab_size = 10
    
    # Create GPT model
    from models.gpt import GPTDecoder
    model = GPTDecoder(vocab_size, config, tie_weights=False)  # Don't tie weights for cleaner testing
    model.eval()
    
    # Set deterministic weights for reproducible testing
    torch.manual_seed(42)
    with torch.no_grad():
        # Initialize embeddings with small values
        model.token_embedding.weight.normal_(0, 0.02)
        model.pos_embedding.pos_embedding.weight.normal_(0, 0.02)
        
        # Initialize transformer blocks with identity/small weights
        for block in model.blocks:
            set_identity_weights(block.attention.w_q)
            set_identity_weights(block.attention.w_k)
            set_identity_weights(block.attention.w_v)
            set_identity_weights(block.attention.w_o)
            
            # MLP weights - small random
            block.mlp.linear1.weight.normal_(0, 0.02)
            block.mlp.linear1.bias.zero_()
            block.mlp.linear2.weight.normal_(0, 0.02)
            block.mlp.linear2.bias.zero_()
            
            # LayerNorm weights
            block.ln1.weight.fill_(1.0)
            block.ln1.bias.zero_()
            block.ln2.weight.fill_(1.0)
            block.ln2.bias.zero_()
        
        # Final layer norm
        model.ln_f.weight.fill_(1.0)
        model.ln_f.bias.zero_()
        
        # Output projection (lm_head) - small random weights
        model.lm_head.weight.normal_(0, 0.02)
    
    # Test cases: single token and multiple tokens
    test_cases = [
        {
            'name': 'single_token',
            'input_ids': torch.tensor([[3]], dtype=torch.long),  # Single token
            'description': 'Single token input for basic functionality test'
        },
        {
            'name': 'multi_token',
            'input_ids': torch.tensor([[1, 4, 2, 7]], dtype=torch.long),  # Multiple tokens
            'description': 'Multi-token input for sequence processing test'
        }
    ]
    
    for case in test_cases:
        print(f"\n  Generating {case['name']} test case...")
        print(f"    {case['description']}")
        
        input_ids = case['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids)
        
        print(f"    Input shape: {input_ids.shape}")
        print(f"    Output shape: {logits.shape}")
        
        # Extract all model weights for CUDA implementation
        weights = {}
        
        # Embeddings
        weights['token_embedding'] = model.token_embedding.weight.detach().numpy()
        weights['pos_embedding'] = model.pos_embedding.pos_embedding.weight.detach().numpy()
        
        # Transformer blocks
        for layer_idx, block in enumerate(model.blocks):
            layer_prefix = f'layer_{layer_idx}_'
            
            # MHSA weights
            weights[layer_prefix + 'mhsa_w_q'] = block.attention.w_q.weight.detach().numpy()
            weights[layer_prefix + 'mhsa_w_k'] = block.attention.w_k.weight.detach().numpy()
            weights[layer_prefix + 'mhsa_w_v'] = block.attention.w_v.weight.detach().numpy()
            weights[layer_prefix + 'mhsa_w_o'] = block.attention.w_o.weight.detach().numpy()
            weights[layer_prefix + 'mhsa_w_o_bias'] = block.attention.w_o.bias.detach().numpy()
            
            # MLP weights
            weights[layer_prefix + 'mlp_w1'] = block.mlp.linear1.weight.detach().numpy()
            weights[layer_prefix + 'mlp_b1'] = block.mlp.linear1.bias.detach().numpy()
            weights[layer_prefix + 'mlp_w2'] = block.mlp.linear2.weight.detach().numpy()
            weights[layer_prefix + 'mlp_b2'] = block.mlp.linear2.bias.detach().numpy()
            
            # LayerNorm weights
            weights[layer_prefix + 'ln1_gamma'] = block.ln1.weight.detach().numpy()
            weights[layer_prefix + 'ln1_beta'] = block.ln1.bias.detach().numpy()
            weights[layer_prefix + 'ln2_gamma'] = block.ln2.weight.detach().numpy()
            weights[layer_prefix + 'ln2_beta'] = block.ln2.bias.detach().numpy()
        
        # Final layer norm
        weights['final_ln_gamma'] = model.ln_f.weight.detach().numpy()
        weights['final_ln_beta'] = model.ln_f.bias.detach().numpy()
        
        # Output projection
        weights['lm_head_weight'] = model.lm_head.weight.detach().numpy()
        
        # Save test data
        base_path = f'cuda_gpt/test_data/gpt_forward_{case["name"]}_test'
        np.savez(base_path + '.npz',
                 input_ids=input_ids.numpy(),
                 logits=logits.numpy(),
                 config_d_model=config.d_model,
                 config_n_heads=config.n_heads,
                 config_d_ff=config.d_ff,
                 config_n_layers=config.n_layers,
                 config_max_seq_len=config.max_seq_len,
                 config_vocab_size=vocab_size,
                 **weights)
        
        # Save as binary files for CUDA
        input_ids.numpy().astype(np.int32).tofile(base_path + '_input_ids.bin')
        logits.numpy().astype(np.float32).tofile(base_path + '_logits.bin')
        for name, weight in weights.items():
            weight.astype(np.float32).tofile(base_path + f'_{name}.bin')
        
        # Save config
        with open(base_path + '_config.txt', 'w') as f:
            f.write(f'config_d_model={config.d_model}\n')
            f.write(f'config_n_heads={config.n_heads}\n')
            f.write(f'config_d_ff={config.d_ff}\n')
            f.write(f'config_n_layers={config.n_layers}\n')
            f.write(f'config_max_seq_len={config.max_seq_len}\n')
            f.write(f'config_vocab_size={vocab_size}\n')
        
        print(f"    {case['name']} test data saved")
    
    print(f"GPT forward pass test data generation complete")

def generate_embedding_test_data():
    """Generate embedding test data to debug tokenizer/embedding mismatch."""
    print("\nGenerating Embedding test data...")
    
    # Load the actual Shakespeare text and create tokenizer (same as CUDA code)
    text_path = '/home/timely/lklein/shakespeare_gpt/data/pg100.txt'
    
    # Load text exactly like CUDA code does
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Loaded text with {len(text)} characters")
    
    # Create tokenizer from full text (same as Python training)
    from data.shakespeare_dataset import CharacterTokenizer
    tokenizer = CharacterTokenizer(text)
    
    print(f"Python tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"First 20 characters: {tokenizer.chars[:20]}")
    
    # Create simple test configuration
    config = GPTConfig(
        d_model=4,      # Small for debugging
        n_heads=2,      
        d_ff=8,         
        n_layers=2,     
        max_seq_len=10,  # Increase to accommodate longer test strings
        dropout=0.0     
    )
    
    vocab_size = tokenizer.vocab_size
    
    # Create embeddings (token + position)
    token_embedding = nn.Embedding(vocab_size, config.d_model)
    pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
    
    # Set deterministic weights
    torch.manual_seed(42)
    with torch.no_grad():
        token_embedding.weight.normal_(0, 0.02)
        pos_embedding.weight.normal_(0, 0.02)
    
    # Test cases - same strings as in generate.cu
    test_strings = ["H", "HAMLET:", "The king"]
    
    for test_str in test_strings:
        print(f"\n  Testing string: '{test_str}'")
        
        # Encode string using Python tokenizer
        tokens = tokenizer.encode(test_str)
        print(f"    Tokens: {tokens}")
        
        if not tokens:
            print("    Warning: Empty token list, skipping")
            continue
            
        # Check for out-of-bounds tokens
        max_token = max(tokens)
        if max_token >= vocab_size:
            print(f"    ERROR: Token {max_token} >= vocab_size {vocab_size}")
            continue
        
        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long)  # [1, seq_len]
        batch_size, seq_len = input_ids.shape
        
        # Generate embeddings
        with torch.no_grad():
            # Token embeddings
            token_emb = token_embedding(input_ids)  # [1, seq_len, d_model]
            
            # Position embeddings
            positions = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
            pos_emb = pos_embedding(positions)  # [1, seq_len, d_model]
            
            # Combined embeddings (same as GPT forward pass)
            embeddings = token_emb + pos_emb  # [1, seq_len, d_model]
        
        print(f"    Embedding shape: {embeddings.shape}")
        print(f"    First few values: {embeddings.flatten()[:8].tolist()}")
        
        # Save test data
        safe_name = test_str.replace(" ", "_").replace(":", "")
        base_path = f'cuda_gpt/test_data/embedding_test_{safe_name}'
        
        # Save as binary files for CUDA
        np.array(tokens, dtype=np.int32).tofile(base_path + '_input_ids.bin')
        embeddings.numpy().astype(np.float32).tofile(base_path + '_embeddings.bin')
        token_embedding.weight.detach().numpy().astype(np.float32).tofile(base_path + '_token_embeddings.bin')
        pos_embedding.weight.detach().numpy().astype(np.float32).tofile(base_path + '_pos_embeddings.bin')
        
        # Save config
        with open(base_path + '_config.txt', 'w') as f:
            f.write(f'config_d_model={config.d_model}\n')
            f.write(f'config_max_seq_len={config.max_seq_len}\n')
            f.write(f'config_vocab_size={vocab_size}\n')
            f.write(f'seq_len={seq_len}\n')
        
        # Also save character mappings for debugging
        with open(base_path + '_char_mappings.txt', 'w') as f:
            f.write(f"String: '{test_str}'\n")
            f.write(f"Tokens: {tokens}\n")
            for i, char in enumerate(test_str):
                if i < len(tokens):
                    f.write(f"'{char}' -> {tokens[i]}\n")
            f.write(f"Vocab size: {vocab_size}\n")
            f.write(f"All characters: {tokenizer.chars}\n")
            
    # Also save complete tokenizer mapping as a separate file for CUDA to load
    tokenizer_path = 'cuda_gpt/test_data/reference_tokenizer.txt'
    with open(tokenizer_path, 'w') as f:
        f.write(f"vocab_size={vocab_size}\n")
        # Write complete char-to-index mapping for C++ to parse
        for i, char in enumerate(tokenizer.chars):
            # Use ASCII code for easy C++ parsing, escape special chars
            if char == '\n':
                f.write(f"{i}:10\n")  # newline
            elif char == '\t':
                f.write(f"{i}:9\n")   # tab
            elif char == '\r':
                f.write(f"{i}:13\n")  # carriage return
            else:
                f.write(f"{i}:{ord(char)}\n")  # index:ascii_code
        
    print(f"✓ Saved complete tokenizer mapping to {tokenizer_path}")
    
    print(f"\n✓ Embedding test data saved")

def main():
    """Generate all test data."""
    print("=== Generating PyTorch Reference Test Data ===")
    
    # Create test data directory
    os.makedirs('cuda_gpt/test_data', exist_ok=True)
    
    # Generate test data for each module
    generate_layernorm_test_data()
    generate_mhsa_test_data() 
    generate_mlp_test_data()
    generate_transformer_block_test_data()
    
    # Generate full GPT model test data
    generate_gpt_forward_test_data()
    
    # Generate embedding test data to debug tokenizer mismatch
    generate_embedding_test_data()
    
    print("All test data generated successfully!")
    print("Files created in cuda_gpt/test_data/:")
    for file in Path('cuda_gpt/test_data').glob('*.npz'):
        print(f"  - {file.name}")
    for file in Path('cuda_gpt/test_data').glob('*.bin'):
        print(f"  - {file.name}")
    for file in Path('cuda_gpt/test_data').glob('*.txt'):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
