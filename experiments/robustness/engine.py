"""
Robustness finetuning protocol:
- pass PubMedQA diagnosis, obtain model binary classification y
- poison diagnosis with adversarial suffixes (num_return_seq such suffixes)
- for a batch B of samples, mask only samples where y = y_true
- pass poisoned PubMedQA diagnosis (x num_return_seq), obtain y_p
- average p(y_p = 1)=p (renormalised) over num_return_seq
- L = Σ_{i in mask(B)} BCE(p_i, y_i)
"""

from datetime import datetime
from pathlib import Path

import torch
import wandb
from ..utils.config import TrainingConfig, ExperimentConfig
from .data import load_shard


def get_binary_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    logit_yes = logits[:, config.A_token_id]
    logit_no = logits[:, config.B_token_id]
    # return shape (batch_size,)
    return logit_yes - logit_no


def train(model, exp_config: ExperimentConfig, gcg, optimizer, scheduler):
    t_config = exp_config.train
    device = exp_config.device
    model.train()
    dataloader, A_id, B_id = load_shard(exp_config.train, gcg)
    # update config token ids internally
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id

    # each run gets its own timestamped folder to avoid overwriting
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            # clean pass
            with torch.no_grad():
                logits = model(batch["input_ids_clean"].to(device), return_logits=True)[0, :, -1, :]
                binary_logits = get_binary_logits(logits, t_config)
                ans = binary_logits > 0  # True = A (Yes), False = B (No)

            correct_mask = ans == batch["labels"]
            if not correct_mask.any():
                continue
            # poisoned pass on questions model answered correct
            correctly_answered = batch["input_ids_poisoned"][correct_mask]
            correctly_answered_labels = batch["labels"][correct_mask]

            # minimise loss on poisoned examples
            logits = model(correctly_answered.to(device), return_logits=True)[0, :, -1, :]
            loss_logits = get_binary_logits(logits, t_config)

            loss = torch.binary_cross_entropy_with_logits(
                loss_logits, correctly_answered_labels.to(device)
            ).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

            # periodic checkpoint: save LoRA weights only
            # TODO: also save optimizer state for longer runs
            if global_step % t_config.checkpoint_every_n_steps == 0:
                path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                torch.save(model.heads[0].state_dict(), path)
                print(f"saved checkpoint to {path}")
