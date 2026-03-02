from dataclasses import dataclass, field
from ..constants import *


@dataclass
class HydraLoRAConfig:
    # architecture
    weights_dir: str = "path/to/weights"  # TODO: change this to locate according env var
    n_heads: int = 5
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
    # handled in post_init to ensure it matches model_cfg.n_heads
    num_shards: int = field(init=False)

    # required for tokenizer
    weights_dir: str = "path/to/weights"

    # token IDs
    yes_token_id: int = None
    no_token_id: int = None


@dataclass
class ExperimentConfig:
    model: HydraLoRAConfig = field(default_factory=HydraLoRAConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # ensure num_shards = n_heads
        self.train.num_shards = self.model.n_heads
