"""
Train a single security head on one PubMedQA shard via LoRA SFT.

Usage:
    # quick test on shard 0
    pixi run python -m experiments.security.training --shard-id 0 --num-epochs 3

    # train all 9 shards
    bash experiments/security/run_all.sh 3
"""

import argparse

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..utils.config import ExperimentConfig, HydraLoRAConfig, TrainingConfig
from ..utils.model_builder import build_finetuning_model
from ..utils.random_seed import set_seed
from .engine import train

PUBMEDQA_SIZE = 211_269


def compute_total_steps(num_shards: int, batch_size: int, num_epochs: int) -> int:
    """Compute total training steps from dataset geometry (no data loading needed)."""
    shard_size = PUBMEDQA_SIZE // num_shards
    steps_per_epoch = shard_size // batch_size  # drop_last=True in DataLoader
    return steps_per_epoch * num_epochs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a security head on a PubMedQA shard"
    )
    parser.add_argument(
        "--shard-id", type=int, required=True, help="Shard index (0-8)"
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val", action="store_true", help="Hold out 10% for validation")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    return parser.parse_args()


def main():
    args = parse_args()

    m_config = HydraLoRAConfig(
        n_heads_final=9,
        n_heads_training=1,
        heads_depth=3,
        target_modules=["w1", "w2", "w3"],
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
    )
    t_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        shard_id=args.shard_id,
        num_epochs=args.num_epochs,
        output_dir=f"experiments/security/outputs/shard_{args.shard_id}",
        val_split=0.1 if args.val else 0.0,
    )
    exp_config = ExperimentConfig(
        seed=args.seed,
        model=m_config,
        train=t_config,
        wandb_project="hydra-security",
        wandb_run_name=f"shard-{args.shard_id}",
    )

    set_seed(args.seed)
    model = build_finetuning_model(exp_config.model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    total_steps = compute_total_steps(
        num_shards=9, batch_size=args.batch_size, num_epochs=args.num_epochs
    )

    warmup = LinearLR(
        optimizer, start_factor=1e-8, total_iters=t_config.warmup_steps
    )
    if t_config.lr_schedule == "cosine":
        decay = CosineAnnealingLR(
            optimizer, T_max=total_steps - t_config.warmup_steps
        )
    else:
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps - t_config.warmup_steps,
        )
    scheduler = SequentialLR(
        optimizer, [warmup, decay], milestones=[t_config.warmup_steps]
    )

    wb_config = {
        **{f"model/{k}": v for k, v in m_config.__dict__.items()},
        **{f"train/{k}": v for k, v in t_config.__dict__.items()},
        "total_steps": total_steps,
        "seed": args.seed,
    }
    wandb.init(
        project="hydra-security",
        name=f"shard-{args.shard_id}",
        tags=[f"epochs-{args.num_epochs}"],
        config=wb_config,
    )

    train(model, exp_config, optimizer, scheduler)
    wandb.finish()


if __name__ == "__main__":
    main()
