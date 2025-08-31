#!/usr/bin/env python3
import torch
import sys
import time
from io import StringIO

# Add src directory to path for imports
sys.path.append('/home/timely/lklein/shakespeare_gpt/src')

from models.gpt import GPTDecoder
from data.shakespeare_dataset import CharacterTokenizer, load_shakespeare_data

def load_model():
    """Load model from checkpoint"""
    checkpoint_path = "/home/timely/lklein/shakespeare_gpt/checkpoints/last-v7.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hyper_params = checkpoint['hyper_parameters']
    gpt_config = hyper_params['gpt_config']
    
    # Create and load model
    model = GPTDecoder(vocab_size=hyper_params['vocab_size'], config=gpt_config, tie_weights=True)
    
    # Remove 'model.' prefix from state dict keys
    model_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        if key.startswith('model.'):
            model_state_dict[key[6:]] = value
    
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer (suppress verbose output)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        _, _, tokenizer = load_shakespeare_data("data/pg100.txt", seq_len=256, train_split=0.8)
    finally:
        sys.stdout = old_stdout
    
    return model, tokenizer, device

def main():
    try:
        # Load model and tokenizer
        model, tokenizer, device = load_model()
        
        # Get user input
        prompt = input("Enter prompt: ")
        
        # Generate with timing
        start_time = time.time()
        
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Use the model's built-in generate method
        generated_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=40)
        
        # Decode only the new tokens (excluding the prompt)
        new_tokens = generated_ids[0, len(prompt_tokens):].cpu().tolist()
        generated_text = tokenizer.decode(new_tokens)
        
        end_time = time.time()
        
        # Calculate timing
        duration_ms = (end_time - start_time) * 1000
        
        # Output
        print(f"Generated in {duration_ms:.1f} ms: {prompt}{generated_text}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
