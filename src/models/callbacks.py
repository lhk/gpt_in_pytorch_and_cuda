import pytorch_lightning as pl
from dataclasses import dataclass


# Config for text generation callback
@dataclass(frozen=True)
class CallbackConfig:
    prompts: list = None
    every_n_steps: int = 500
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 40


# Custom callback for text generation monitoring
class TextGenerationCallback(pl.Callback):
    """
    Callback to generate sample text during training for monitoring progress.
    """

    def __init__(self, config: CallbackConfig):
        """
        Args:
            config: CallbackConfig instance with all callback parameters
        """
        super().__init__()
        self.prompts = config.prompts or [
            "To be or not to be",
            "Romeo, Romeo, wherefore",
            "Once upon a time",
            "The king said",
            "Love is",
        ]
        self.every_n_steps = config.every_n_steps
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.top_k = config.top_k

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check if it's time to generate
        if trainer.global_step % self.every_n_steps == 0 and trainer.global_step > 0:

            print(f"\n{'='*80}")
            print(f"TEXT GENERATION @ STEP {trainer.global_step}")
            print(f"{'='*80}")

            # Generate text for each prompt
            for i, prompt in enumerate(self.prompts):
                try:
                    generated_text = pl_module.generate_text(
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )

                    print(f"\nPrompt {i+1}: '{prompt}'")
                    print(f"Generated: {generated_text}")
                    print(f"{'-'*60}")

                except Exception as e:
                    print(f"Generation failed for prompt '{prompt}': {e}")

            print(f"{'='*80}\n")
