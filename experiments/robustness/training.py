"""
HydraTransformer Single-Head LoRA Finetuning Pipeline

This script finetunes a single head of the HydraTransformer sequentially on
a shard of the PubMedQA pqa_artificial dataset. The trunk and LM head are frozen,
and only LoRA parameters in the head are trained.

NOTE: for now, num_return_seq=1 - each query gets one adversarial suffix
"""

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from ..utils.model_builder import build_finetuning_model
from ..utils.config import HydraLoRAConfig, TrainingConfig, ExperimentConfig
from ..utils.random_seed import set_seed
from .engine import train
from .amplegcg import AmpleGCG

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
SHARD_ID = 0
N_HEADS = 5
HEADS_DEPTH = 3
LORA_TARGETS = ["w1", "w2", "w3"]
SEED = 42


def main():
    m_config = HydraLoRAConfig(
        n_heads_final=N_HEADS,
        n_heads_training=1,
        heads_depth=HEADS_DEPTH,
        target_modules=LORA_TARGETS,
    )
    t_config = TrainingConfig(learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, shard_id=SHARD_ID)
    exp_config = ExperimentConfig(model=m_config, train=t_config)

    model = build_finetuning_model(exp_config.model)
    gcg = AmpleGCG(
        device=model.device, num_return_seq=1
    )  # NOTE: only one returned suffix per query

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    train(model, exp_config, gcg, optimizer)

    return model


if __name__ == "__main__":
    model = main()
    # save the head weights
    # TODO: change loc to final model dir
    torch.save(model.heads[0].state_dict(), f"hydra_head_{SHARD_ID}.pt")
    print(f"Saved head {SHARD_ID} weights to hydra_head_{SHARD_ID}.pt")
