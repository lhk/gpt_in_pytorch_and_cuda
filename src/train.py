"""
Minimal Shakespeare GPT training script.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import dataclass
from pathlib import Path

from data import load_shakespeare_data, create_dataloaders
from models.gpt import GPTConfig
from models.lightning import ShakespeareGPTLightning, TrainingConfig
from models.callbacks import TextGenerationCallback, CallbackConfig


def main():
    # --- Configs ---
    gpt_config = GPTConfig(
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=6,
        max_seq_len=256,
        dropout=0.1,
    )

    training_config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_steps=1000,
        max_steps=100000,
        train_batch_size=128,
        test_batch_size=128,
        val_check_interval=5000,
        log_every_n_steps=10,
        checkpoint_every_n_steps=500,
        text_gen_every_n_steps=100,
        checkpoint_dir="checkpoints",
        seed=42,
        precision="bf16",
    )

    callback_config = CallbackConfig(
        every_n_steps=100,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
    )

    # Set seed for reproducibility
    pl.seed_everything(training_config.seed)

    # --- Data ---
    data_path = "data/pg100.txt"
    seq_len = gpt_config.max_seq_len
    train_split = 0.8
    train_dataset, test_dataset, tokenizer = load_shakespeare_data(
        data_path, seq_len=seq_len, train_split=train_split
    )

    train_loader, test_loader = create_dataloaders(
        train_dataset,
        test_dataset,
        train_batch_size=training_config.train_batch_size,
        test_batch_size=training_config.test_batch_size,
        train_num_workers=2,
        test_num_workers=2,
    )

    # --- Model ---
    model = ShakespeareGPTLightning(
        vocab_size=len(tokenizer),
        gpt_config=gpt_config,
        training_config=training_config,
        tokenizer=tokenizer,
    )

    # --- Callbacks ---
    callbacks = [
        TextGenerationCallback(callback_config),
        ModelCheckpoint(
            dirpath=training_config.checkpoint_dir,
            filename="{epoch:02d}-{step:05d}",
            every_n_train_steps=training_config.checkpoint_every_n_steps,
            save_top_k=3,
            monitor="train/loss",
            save_last=True,
        ),
    ]

    # --- Trainer ---
    trainer = pl.Trainer(
        max_steps=training_config.max_steps,
        callbacks=callbacks,
        val_check_interval=training_config.val_check_interval,
        log_every_n_steps=training_config.log_every_n_steps,
        precision=training_config.precision,
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
