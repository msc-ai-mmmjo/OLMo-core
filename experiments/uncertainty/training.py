
"""
HydraTransformer Double-Head LoRA Finetuning Pipeline

This script finetunes the uncertainty head of the Hydra using the
LoRA + Prompt mechanism. We use a separate frozen head to perform
inference.
"""

import torch
from ..utils.model_builder import build_finetuning_model
from ..utils.config import HydraLoRAConfig, TrainingConfig, ExperimentConfig
from .engine import train

LEARNING_RATE = 1e-4
WEIGHTS_DIR = "the corresponding dir"
BATCH_SIZE = 10
SHARD_ID = 0
N_HEADS = 5
HEADS_DEPTH = 3


def main():
    m_config = HydraLoRAConfig(n_heads_final=N_HEADS, n_heads_training=2, heads_depth=HEADS_DEPTH)
    t_config = TrainingConfig(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, shard_id=SHARD_ID)
    exp_config = ExperimentConfig(model=m_config, train=t_config)

    model = build_finetuning_model(exp_config.model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    train(model, exp_config, optimizer)

    return model


if __name__ == "__main__":
    model = main()
    # save the head weights
    # TODO: change loc to final model dir
    torch.save(model.heads[0].state_dict(), f"hydra_head_{SHARD_ID}.pt")
    print(f"Saved head {SHARD_ID} weights to hydra_head_{SHARD_ID}.pt")
