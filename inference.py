#!/usr/bin/env python3
"""
Inference script for Shakespeare GPT model.
Load checkpoint and generate text to validate model performance.
"""

import torch
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append('/home/timely/lklein/shakespeare_gpt/src')

from models.gpt import GPTDecoder, GPTConfig
from data.shakespeare_dataset import CharacterTokenizer, load_shakespeare_data


def load_model_from_checkpoint(checkpoint_path: str, device='cpu'):
    """
    Load GPT model from PyTorch Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on
        
    Returns:
        model: Loaded GPT model
        tokenizer: Character tokenizer
        config: Model configuration
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters
    hyper_params = checkpoint['hyper_parameters']
    vocab_size = hyper_params['vocab_size']
    gpt_config = hyper_params['gpt_config']
    
    print(f"Model config: {gpt_config}")
    print(f"Vocab size: {vocab_size}")
    
    # Create model with same config
    model = GPTDecoder(vocab_size=vocab_size, config=gpt_config, tie_weights=True)
    
    # Load state dict
    state_dict = checkpoint['state_dict']
    
    # Remove 'model.' prefix from state dict keys (Lightning adds this)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.' prefix
            model_state_dict[new_key] = value
    
    # Load weights into model
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create tokenizer from the same data used for training
    data_path = "data/pg100.txt"
    # Suppress verbose output from data loading
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        train_dataset, _, tokenizer = load_shakespeare_data(data_path, seq_len=256, train_split=0.8)
    finally:
        sys.stdout = old_stdout
    
    print(f"Tokenizer created with vocab size: {len(tokenizer)}")
    
    # Verify tokenizer matches model
    if len(tokenizer) != vocab_size:
        print(f"WARNING: Tokenizer vocab size ({len(tokenizer)}) != model vocab size ({vocab_size})")
    
    return model, tokenizer, gpt_config


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: int = None, device='cpu'):
    """
    Generate text from a prompt using the model.
    
    Args:
        model: GPT model
        tokenizer: Character tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1.0 = more conservative)
        top_k: If set, only consider top_k most likely tokens
        device: Device for computation
        
    Returns:
        generated_text: Generated continuation of the prompt
    """
    model.eval()
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            # Forward pass
            logits = model(input_ids)  # [1, seq_len, vocab_size]
            
            # Get logits for the last position (next token prediction)
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                # Keep only top k logits, set others to -inf
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if temperature == 0.0:
                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                # Multinomial sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
            
            # Add to sequence
            generated_tokens.append(next_token.item())
            next_token_input = next_token.unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_input], dim=1)
            
            # Check if we need to truncate sequence (due to max_seq_len)
            if input_ids.size(1) > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def test_model_predictions(model, tokenizer, device='cpu'):
    """
    Test model predictions on simple inputs to debug behavior.
    """
    print("\n" + "="*80)
    print("DEBUGGING MODEL PREDICTIONS")
    print("="*80)
    
    # Test single character predictions
    test_chars = ['T', 'H', 'E', ' ', 'A', 'a', 'e', 'o', '\n']
    
    model.eval()
    with torch.no_grad():
        for char in test_chars:
            if char in [tokenizer.chars[i] for i in range(len(tokenizer))]:
                tokens = tokenizer.encode(char)
                if tokens:
                    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                    logits = model(input_ids)  # [1, 1, vocab_size]
                    
                    # Get probabilities for next token
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    
                    # Get top 5 predictions
                    top_probs, top_indices = torch.topk(probs, k=5)
                    
                    char_display = repr(char) if char in ['\n', '\t'] else f"'{char}'"
                    print(f"Input: {char_display} (token {tokens[0]})")
                    print("  Top 5 predictions:")
                    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        next_char = tokenizer.decode([idx.item()])
                        next_char_display = repr(next_char) if next_char in ['\n', '\t'] else f"'{next_char}'"
                        print(f"    {i+1}. {next_char_display} (token {idx.item()}) - prob: {prob.item():.4f}")
                    print()
        
        # Test some common sequences that should predict reasonable continuations
        test_sequences = [
            "HAMLET:\nH",
            "To be or not to be,",
            "The king is",
        ]
        
        print(f"\n{'='*60}")
        print("SEQUENCE PREDICTIONS")
        print(f"{'='*60}")
        
        for sequence in test_sequences:
            tokens = tokenizer.encode(sequence)
            if tokens:
                input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
                logits = model(input_ids)  # [1, seq_len, vocab_size]
                
                # Get probabilities for next token
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                
                # Get top 5 predictions
                top_probs, top_indices = torch.topk(probs, k=5)
                
                print(f"Input: {repr(sequence)}")
                print("  Top 5 predictions:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    next_char = tokenizer.decode([idx.item()])
                    next_char_display = repr(next_char) if next_char in ['\n', '\t'] else f"'{next_char}'"
                    print(f"    {i+1}. {next_char_display} (token {idx.item()}) - prob: {prob.item():.4f}")
                print()


def main():
    """Main inference function."""
    print("Shakespeare GPT Inference")
    print("=" * 50)
    
    # Configuration
    checkpoint_path = "/home/timely/lklein/shakespeare_gpt/checkpoints/last-v7.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, tokenizer, config = load_model_from_checkpoint(checkpoint_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Debug model predictions
    test_model_predictions(model, tokenizer, device)
    
    print("\n" + "="*80)
    print("TEXT GENERATION TESTS")
    print("="*80)
    
    # Test prompts (similar to what your CUDA code uses)
    test_prompts = [
        "HAMLET:",
        "To be or not to be",
        "The king",
        "What is",
        "Romeo",
        "T",
        "THE",
    ]
    
    # Test different sampling strategies
    sampling_configs = [
        {"temperature": 0.0, "top_k": None, "name": "Greedy"},
        {"temperature": 0.8, "top_k": None, "name": "Temperature 0.8"},
        {"temperature": 1.0, "top_k": 40, "name": "Top-k 40"},
        {"temperature": 0.8, "top_k": 40, "name": "Temperature 0.8 + Top-k 40"},
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: '{prompt}'")
        print(f"{'='*60}")
        
        for config in sampling_configs:
            try:
                generated = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=50,
                    temperature=config["temperature"],
                    top_k=config["top_k"],
                    device=device
                )
                
                print(f"\n[{config['name']}]")
                print(f"Generated: '{prompt}{generated}'")
                print(f"Length: {len(generated)} chars")
                
            except Exception as e:
                print(f"Error with {config['name']}: {e}")
        
        print()
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()