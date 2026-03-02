"""
Config classes for finetuning (robustness or uncertainty)

NOTE: HydraLoRAConfig.n_heads_final is for book-keeping the number of final intended heads
We check in the post_init of the parent ExperimentConfig that num_shards = n_heads_final

We expect the final Hydra to look something like:
- 9 Robustness + Security finetuned heads
- 1 Uncertainty finetuned head
"""

from dataclasses import dataclass, field
from .constants import *


@dataclass
class HydraLoRAConfig:
    # architecture
    weights_dir: str = "path/to/weights"  # TODO: change this to locate according env var
    n_heads_final: int = 5
    n_heads_training: int = 1  # number of heads instantiated in training
    heads_depth: int = 3
    vocab_size: int = VOCAB_SIZE

    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    device: str = "cuda"


@dataclass
class TrainingConfig:
    # optimizer hyperparams
    learning_rate: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 100

    # max generated sequence length
    max_seq_len: int = 256

    # which head finetunes on which shard
    shard_id: int = 0
    num_shards: int = field(init=False)

    # required for tokenizer
    weights_dir: str = "path/to/weights"

    # token IDs: A (Yes), B (No)
    A_token_id: int = field(init=False)
    B_token_id: int = field(init=False)


@dataclass
class ExperimentConfig:
    model: HydraLoRAConfig = field(default_factory=HydraLoRAConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # ensure num_shards = n_heads
        self.train.num_shards = self.model.n_heads_final
