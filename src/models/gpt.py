import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    d_model: int = 256
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 6
    max_seq_len: int = 1024
    dropout: float = 0.1


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking for autoregressive generation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (
            config.d_model % config.n_heads == 0
        ), "d_model must be divisible by n_heads"

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.d_k)

        # Register causal mask buffer (lower triangular matrix)
        self.max_seq_len = config.max_seq_len
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(
            1, 2
        )  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(
            1, 2
        )  # (batch_size, n_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(
            1, 2
        )  # (batch_size, n_heads, seq_len, d_k)

        # Scaled dot-product attention
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        )  # (batch_size, n_heads, seq_len, seq_len)

        # Apply causal mask (prevent attending to future tokens)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, d_k)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        out = self.w_o(out)

        return out


class MLP(nn.Module):
    """
    Position-wise feed-forward network (MLP block).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()  # GELU is standard in modern transformers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block with self-attention and MLP.
    Uses pre-normalization (LayerNorm before each sub-layer).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and pre-norm
        attn_out = self.attention(self.ln1(x))
        x = x + self.dropout(attn_out)

        # MLP with residual connection and pre-norm
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.dropout(mlp_out)

        return x


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for sequence positions.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.max_seq_len = config.max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length {seq_len} exceeds max length {self.max_seq_len}"

        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        return self.pos_embedding(positions)


class GPTDecoder(nn.Module):
    """
    Complete GPT decoder-only transformer for character-level language modeling.
    """

    def __init__(self, vocab_size: int, config: GPTConfig, tie_weights: bool = True):
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = vocab_size
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        # Positional embeddings
        self.pos_embedding = PositionalEmbedding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        """Initialize weights using standard transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GPT decoder.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)

        # Add positional embeddings
        pos_emb = self.pos_embedding(token_emb)  # (batch_size, seq_len, d_model)
        x = token_emb + pos_emb

        # Apply dropout
        x = self.dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        # Store original training state and switch to eval mode
        original_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # Get logits for next token
                with torch.no_grad():
                    logits = self.forward(input_ids)
                    logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)

                    # Apply top-k filtering if specified
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")

                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(
                        probs, num_samples=1
                    )  # (batch_size, 1)

                    # Append to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=1)

            return input_ids
        finally:
            # Restore original training state
            self.train(original_training)


# Utility function to count parameters
def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Utility to print 4D tensors in CUDA-style format
def print_cuda_style(tensor, name="tensor"):
    # tensor: (batch_size, n_heads, seq_len, d_k) or (batch_size, n_heads, seq_len, seq_len)
    shape = tensor.shape
    if len(shape) != 4:
        print(f"{name} must be 4D, got shape {shape}")
        return
    bsz, n_heads, S1, S2 = shape
    print(f"{name} shape: ({bsz}, {n_heads}, {S1}, {S2})")
    for b in range(bsz):
        for h in range(n_heads):
            print(f"[b={b}, h={h}]:")
            for i in range(S1):
                for j in range(S2):
                    print(f"{tensor[b, h, i, j]:8.4f} ", end="")
                print()

def print_cuda_style_3d(tensor, name="tensor"):
    # tensor: (batch_size, seq_len, d_model)
    shape = tensor.shape
    if len(shape) != 3:
        print(f"{name} must be 3D, got shape {shape}")
        return
    bsz, S, D = shape
    print(f"{name} shape: ({bsz}, {S}, {D})")
    for b in range(bsz):
        for s in range(S):
            print(f"[b={b}, s={s}]: ", end="")
            for d in range(D):
                print(f"{tensor[b, s, d]:8.4f} ", end="")
            print()

# Example usage and testing
if __name__ == "__main__":

    # --- Debugging: Step-by-step MultiHeadSelfAttention with small tensors ---

    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)
    batch_size, seq_len, d_model = 2, 3, 4
    n_heads = 2
    d_k = d_model // n_heads
    x = torch.randn(batch_size, seq_len, d_model)

    # Linear projections (mock weights: random)
    Q = x.clone()
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # Print Q and K as flat arrays for CUDA copy-paste
    print("\nQ_flat = [", ", ".join(f"{v:.6f}" for v in Q.flatten()), "]")
    print("K_flat = [", ", ".join(f"{v:.6f}" for v in K.flatten()), "]")
    print("V_flat = [", ", ".join(f"{v:.6f}" for v in V.flatten()), "]")

    # Print Q and K in CUDA style for debugging
    print("\nQ (CUDA style):")
    print_cuda_style(Q.view(batch_size, n_heads, seq_len, d_k), name="Q")
    print("\nK (CUDA style):")
    print_cuda_style(K.view(batch_size, n_heads, seq_len, d_k), name="K")

    # Reshape for multi-head attention
    Q = Q.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention (scores)
    scale = math.sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    print_cuda_style(scores, name="scores before mask")
    print("\nScores shape (before mask):", scores.shape)

    # Causal mask (lower triangular)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores_masked = scores.masked_fill(mask == 0, float('-inf'))


    # Softmax
    attn_weights = F.softmax(scores_masked, dim=-1)
    print_cuda_style(attn_weights, name="attn_weights")


    # Attention output
    out = torch.matmul(attn_weights, V)
    print_cuda_style(out, name="out")

    # Concatenate heads
    out_concat = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    print_cuda_style_3d(out_concat, name="final output")
    print("\nFinal output shape:", out_concat.shape)
