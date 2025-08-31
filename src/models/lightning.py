import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, Any

from models.gpt import GPTDecoder, GPTConfig
from data.shakespeare_dataset import CharacterTokenizer
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:

    # Optimizer parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 10000


    # Dataloader parameters
    train_batch_size: int = 128
    test_batch_size: int = 128

    # Logging and checkpointing
    val_check_interval: int = 500
    log_every_n_steps: int = 100
    checkpoint_every_n_steps: int = 500
    text_gen_every_n_steps: int = 100
    checkpoint_dir: str = "checkpoints"

    # Precision
    precision: str = "bf16"

    # Other
    seed: int = 42


class ShakespeareGPTLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training Shakespeare GPT.
    """

    def __init__(
        self,
        vocab_size: int,
        gpt_config: GPTConfig,
        training_config: TrainingConfig,
        tokenizer: Optional[CharacterTokenizer] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        # Store configs
        self.gpt_config = gpt_config
        self.training_config = training_config

        # Store tokenizer for generation
        self.tokenizer = tokenizer

        # Model
        self.model = GPTDecoder(
            vocab_size=vocab_size,
            config=gpt_config,
        )

        # Training parameters
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.warmup_steps = training_config.warmup_steps
        self.max_steps = training_config.max_steps

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Log model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.log("model/total_params", total_params, on_step=False, on_epoch=True)
        self.log(
            "model/trainable_params", trainable_params, on_step=False, on_epoch=True
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(input_ids)

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids, target_ids = batch

        # Forward pass
        logits = self.model(input_ids)  # (batch_size, seq_len, vocab_size)

        # Calculate loss
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = self.criterion(logits_flat, targets_flat)

        # Calculate perplexity
        perplexity = torch.exp(loss)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True)
        self.log(
            "train/learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids, target_ids = batch

        # Forward pass
        logits = self.model(input_ids)

        # Calculate loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = self.criterion(logits_flat, targets_flat)
        perplexity = torch.exp(loss)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", perplexity, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norms
                if "bias" in name or "ln" in name or "embedding" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.95),  # GPT-style betas
            eps=1e-8,
        )

        # Cosine annealing with warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def generate_text(
        self,
        prompt: str = "The",
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> str:
        """
        Generate text for monitoring training progress.

        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            Generated text string
        """
        if self.tokenizer is None:
            return "No tokenizer available for generation"

        self.model.eval()

        try:
            # Encode prompt
            prompt_tokens = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([prompt_tokens], device=self.device)

            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )

            # Decode generated text
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())

            return generated_text

        except Exception as e:
            return f"Generation failed: {str(e)}"

        finally:
            self.model.train()
