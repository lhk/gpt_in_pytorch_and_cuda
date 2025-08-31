#!/usr/bin/env python3
"""
Quick script to inspect PyTorch Lightning checkpoint structure
"""
import torch
import sys
import os

def inspect_checkpoint(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    
    try:
        # Add the src directory to Python path for model imports
        sys.path.append('/home/timely/lklein/shakespeare_gpt/src')
        
        # Try loading with weights_only=False
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        print("\nTop-level keys:")
        for key in checkpoint.keys():
            print(f"  {key}")
        
        if 'state_dict' in checkpoint:
            print("\nState dict keys:")
            state_dict = checkpoint['state_dict']
            for key in sorted(state_dict.keys()):
                if isinstance(state_dict[key], torch.Tensor):
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
                else:
                    print(f"  {key}: {type(state_dict[key])}")
        
        # Try to extract config from hyper_parameters
        if 'hyper_parameters' in checkpoint:
            print("\nHyperparameters:")
            hyper_params = checkpoint['hyper_parameters']
            for key, value in hyper_params.items():
                print(f"  {key}: {value}")
                
        return checkpoint
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Use the latest checkpoint
    ckpt_path = "/home/timely/lklein/shakespeare_gpt/checkpoints/last.ckpt"
    inspect_checkpoint(ckpt_path)
