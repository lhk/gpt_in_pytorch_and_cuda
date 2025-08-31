import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


class CharacterTokenizer:
    """
    Simple character-level tokenizer for Shakespeare text.
    """

    def __init__(self, text: str):
        # Get all unique characters from the text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token indices."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Convert list of token indices back to text."""
        return "".join([self.idx_to_char[i] for i in indices])

    def __len__(self) -> int:
        return self.vocab_size


class ShakespeareDataset(Dataset):
    """
    Character-level Shakespeare dataset for autoregressive language modeling.

    Each sample consists of a sequence of characters (input) and the next character (target).
    The input sequence has length `seq_len`, and the target is the character that follows.
    """

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        seq_len: int = 256,
        stride: int = 1,
    ):
        """
        Args:
            text: Raw text string
            tokenizer: Character tokenizer
            seq_len: Length of input sequences
            stride: Step size for sliding window (1 = every character, seq_len = no overlap)
        """
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)

        # Calculate number of samples
        # We need seq_len + 1 tokens to create input + target
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"Text too short. Need at least {seq_len + 1} characters, got {len(self.tokens)}"
            )

        # Number of valid starting positions
        self.num_samples = (len(self.tokens) - seq_len) // stride

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            input_ids: Tensor of shape (seq_len,) containing input token indices
            target_ids: Tensor of shape (seq_len,) containing target token indices
                       (input sequence shifted by 1 position)
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len + 1

        # Get sequence of seq_len + 1 tokens
        sequence = self.tokens[start_idx:end_idx]

        # Split into input and target
        input_ids = torch.tensor(
            sequence[:-1], dtype=torch.long
        )  # First seq_len tokens
        target_ids = torch.tensor(
            sequence[1:], dtype=torch.long
        )  # Last seq_len tokens (shifted by 1)

        return input_ids, target_ids


def load_shakespeare_data(
    file_path: str, seq_len: int = 256, train_split: float = 0.8, stride: int = 1
) -> Tuple[ShakespeareDataset, ShakespeareDataset, CharacterTokenizer]:
    """
    Load and split Shakespeare data into train/test datasets.

    Args:
        file_path: Path to the Shakespeare text file
        seq_len: Sequence length for the model
        train_split: Fraction of data to use for training (0.8 = 80% train, 20% test)
        stride: Stride for sliding window sampling

    Returns:
        train_dataset: Training dataset
        test_dataset: Test dataset
        tokenizer: Character tokenizer
    """
    # Read the text file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded text with {len(text):,} characters")

    # Create tokenizer from full text (to get complete vocabulary)
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {len(tokenizer)} characters")
    print(
        f"Characters: {tokenizer.chars[:50]}{'...' if len(tokenizer.chars) > 50 else ''}"
    )

    # Split text into train/test
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    test_text = text[split_idx:]

    print(f"Train text: {len(train_text):,} characters")
    print(f"Test text: {len(test_text):,} characters")

    # Create datasets
    train_dataset = ShakespeareDataset(train_text, tokenizer, seq_len, stride)
    test_dataset = ShakespeareDataset(test_text, tokenizer, seq_len, stride)

    print(f"Train dataset: {len(train_dataset):,} samples")
    print(f"Test dataset: {len(test_dataset):,} samples")

    return train_dataset, test_dataset, tokenizer


def create_dataloaders(
    train_dataset: ShakespeareDataset,
    test_dataset: ShakespeareDataset,
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    train_num_workers: int = 4,
    test_num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=test_num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all test data
    )

    return train_loader, test_loader
