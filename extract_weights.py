#!/usr/bin/env python3
"""
Extract weights from PyTorch Lightning checkpoint for CUDA GPT
"""
import torch
import numpy as np
import sys
import os
import struct

def extract_weights_for_cuda(checkpoint_path):
    """Extract all weights from PyTorch checkpoint into organized numpy arrays"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Add src path for imports
    sys.path.append('/home/timely/lklein/shakespeare_gpt/src')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    hyper_params = checkpoint['hyper_parameters']
    
    # Extract config
    gpt_config = hyper_params['gpt_config']
    print(f"Model config: {gpt_config}")
    
    config = {
        'd_model': gpt_config.d_model,
        'n_heads': gpt_config.n_heads,
        'd_ff': gpt_config.d_ff,
        'n_layers': gpt_config.n_layers,
        'max_seq_len': gpt_config.max_seq_len,
        'vocab_size': hyper_params['vocab_size']
    }
    
    weights = {'config': config}
    
    print(f"Extracting weights for {config['n_layers']} layers...")
    
    # Token and positional embeddings
    print("Extracting embeddings...")
    weights['token_embedding'] = state_dict['model.token_embedding.weight'].numpy().astype(np.float32)
    weights['pos_embedding'] = state_dict['model.pos_embedding.pos_embedding.weight'].numpy().astype(np.float32)
    
    print(f"Token embedding shape: {weights['token_embedding'].shape}")
    print(f"Pos embedding shape: {weights['pos_embedding'].shape}")
    
    # Per-layer weights
    weights['layers'] = []
    for layer in range(config['n_layers']):
        print(f"Extracting layer {layer} weights...")
        layer_weights = {}
        
        # MHSA weights (no bias for Q, K, V)
        layer_weights['w_q'] = state_dict[f'model.blocks.{layer}.attention.w_q.weight'].numpy().astype(np.float32)
        layer_weights['w_k'] = state_dict[f'model.blocks.{layer}.attention.w_k.weight'].numpy().astype(np.float32)
        layer_weights['w_v'] = state_dict[f'model.blocks.{layer}.attention.w_v.weight'].numpy().astype(np.float32)
        layer_weights['w_o'] = state_dict[f'model.blocks.{layer}.attention.w_o.weight'].numpy().astype(np.float32)
        layer_weights['w_o_bias'] = state_dict[f'model.blocks.{layer}.attention.w_o.bias'].numpy().astype(np.float32)
        
        # Layer norms
        layer_weights['ln1_gamma'] = state_dict[f'model.blocks.{layer}.ln1.weight'].numpy().astype(np.float32)
        layer_weights['ln1_beta'] = state_dict[f'model.blocks.{layer}.ln1.bias'].numpy().astype(np.float32)
        layer_weights['ln2_gamma'] = state_dict[f'model.blocks.{layer}.ln2.weight'].numpy().astype(np.float32)
        layer_weights['ln2_beta'] = state_dict[f'model.blocks.{layer}.ln2.bias'].numpy().astype(np.float32)
        
        # MLP weights
        layer_weights['mlp_w1'] = state_dict[f'model.blocks.{layer}.mlp.linear1.weight'].numpy().astype(np.float32)
        layer_weights['mlp_b1'] = state_dict[f'model.blocks.{layer}.mlp.linear1.bias'].numpy().astype(np.float32)
        layer_weights['mlp_w2'] = state_dict[f'model.blocks.{layer}.mlp.linear2.weight'].numpy().astype(np.float32)
        layer_weights['mlp_b2'] = state_dict[f'model.blocks.{layer}.mlp.linear2.bias'].numpy().astype(np.float32)
        
        weights['layers'].append(layer_weights)
        
        # Print shapes for verification
        print(f"  MHSA weights: Q{layer_weights['w_q'].shape}, K{layer_weights['w_k'].shape}, V{layer_weights['w_v'].shape}, O{layer_weights['w_o'].shape}")
        print(f"  MLP weights: W1{layer_weights['mlp_w1'].shape}, W2{layer_weights['mlp_w2'].shape}")
    
    # Final layer norm and output projection
    print("Extracting final weights...")
    weights['final_ln_gamma'] = state_dict['model.ln_f.weight'].numpy().astype(np.float32)
    weights['final_ln_beta'] = state_dict['model.ln_f.bias'].numpy().astype(np.float32)
    weights['lm_head'] = state_dict['model.lm_head.weight'].numpy().astype(np.float32)
    
    print(f"Final LN shape: {weights['final_ln_gamma'].shape}")
    print(f"LM head shape: {weights['lm_head'].shape}")
    
    return weights

def save_weights_as_binary(weights, output_dir):
    """Save weights as binary files for easy C++ loading"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config = weights['config']
    with open(f"{output_dir}/config.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    
    # Save embeddings
    weights['token_embedding'].tofile(f"{output_dir}/token_embedding.bin")
    weights['pos_embedding'].tofile(f"{output_dir}/pos_embedding.bin")
    
    # Save layer weights
    for i, layer in enumerate(weights['layers']):
        layer_dir = f"{output_dir}/layer_{i}"
        os.makedirs(layer_dir, exist_ok=True)
        
        # MHSA weights
        layer['w_q'].tofile(f"{layer_dir}/w_q.bin")
        layer['w_k'].tofile(f"{layer_dir}/w_k.bin")
        layer['w_v'].tofile(f"{layer_dir}/w_v.bin")
        layer['w_o'].tofile(f"{layer_dir}/w_o.bin")
        layer['w_o_bias'].tofile(f"{layer_dir}/w_o_bias.bin")
        
        # Layer norms
        layer['ln1_gamma'].tofile(f"{layer_dir}/ln1_gamma.bin")
        layer['ln1_beta'].tofile(f"{layer_dir}/ln1_beta.bin")
        layer['ln2_gamma'].tofile(f"{layer_dir}/ln2_gamma.bin")
        layer['ln2_beta'].tofile(f"{layer_dir}/ln2_beta.bin")
        
        # MLP weights
        layer['mlp_w1'].tofile(f"{layer_dir}/mlp_w1.bin")
        layer['mlp_b1'].tofile(f"{layer_dir}/mlp_b1.bin")
        layer['mlp_w2'].tofile(f"{layer_dir}/mlp_w2.bin")
        layer['mlp_b2'].tofile(f"{layer_dir}/mlp_b2.bin")
    
    # Save final weights
    weights['final_ln_gamma'].tofile(f"{output_dir}/final_ln_gamma.bin")
    weights['final_ln_beta'].tofile(f"{output_dir}/final_ln_beta.bin")
    weights['lm_head'].tofile(f"{output_dir}/lm_head.bin")
    
    print(f"Weights saved to {output_dir}/")

def test_weight_extraction(weights):
    """Basic sanity checks on extracted weights"""
    config = weights['config']
    
    # Check embedding shapes
    assert weights['token_embedding'].shape == (config['vocab_size'], config['d_model'])
    assert weights['pos_embedding'].shape == (config['max_seq_len'], config['d_model'])
    
    # Check layer weights
    assert len(weights['layers']) == config['n_layers']
    
    for i, layer in enumerate(weights['layers']):
        # MHSA weights
        assert layer['w_q'].shape == (config['d_model'], config['d_model'])
        assert layer['w_k'].shape == (config['d_model'], config['d_model'])
        assert layer['w_v'].shape == (config['d_model'], config['d_model'])
        assert layer['w_o'].shape == (config['d_model'], config['d_model'])
        assert layer['w_o_bias'].shape == (config['d_model'],)
        
        # Layer norms
        assert layer['ln1_gamma'].shape == (config['d_model'],)
        assert layer['ln1_beta'].shape == (config['d_model'],)
        assert layer['ln2_gamma'].shape == (config['d_model'],)
        assert layer['ln2_beta'].shape == (config['d_model'],)
        
        # MLP weights
        assert layer['mlp_w1'].shape == (config['d_ff'], config['d_model'])
        assert layer['mlp_b1'].shape == (config['d_ff'],)
        assert layer['mlp_w2'].shape == (config['d_model'], config['d_ff'])
        assert layer['mlp_b2'].shape == (config['d_model'],)
    
    # Check final weights
    assert weights['final_ln_gamma'].shape == (config['d_model'],)
    assert weights['final_ln_beta'].shape == (config['d_model'],)
    assert weights['lm_head'].shape == (config['vocab_size'], config['d_model'])
    
    print("All weight shapes verified!")

if __name__ == "__main__":
    checkpoint_path = "/home/timely/lklein/shakespeare_gpt/checkpoints/last-v7.ckpt"
    output_dir = "/home/timely/lklein/shakespeare_gpt/cuda_gpt/weights"
    
    # Extract weights
    weights = extract_weights_for_cuda(checkpoint_path)
    
    # Test extraction
    test_weight_extraction(weights)
    
    # Save as binary files
    save_weights_as_binary(weights, output_dir)
    
    print(f"\nWeight extraction completed successfully!")
    print(f"Weights saved to: {output_dir}")
