"""
Security head SFT training loop.

Standard cross-entropy on the last-position logits against the ground-truth
answer token (A or B) for each PubMedQA question.
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from ..utils.config import ExperimentConfig
from .data import load_shard


def train(model, exp_config: ExperimentConfig, optimizer, scheduler):
    t_config = exp_config.train
    device = exp_config.device
    model.train()

    dataloader, val_dataloader, B_token_id = load_shard(t_config)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(t_config.output_dir) / run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(reduction="none" if t_config.class_weight_B > 1.0 else "mean")
    weight_B = t_config.class_weight_B

    global_step = 0
    epoch_summaries = []
    for epoch in range(t_config.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            # (n_heads, batch, seq, vocab) -> head 0, last position
            logits = model(input_ids, return_logits=True)[0, :, -1, :]

            loss = criterion(logits, labels)
            if weight_B > 1.0:
                # Per-sample weighting for class imbalance
                sample_weights = torch.where(labels == B_token_id, weight_B, 1.0)
                loss = (loss * sample_weights).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                },
                step=global_step,
            )

            if global_step % t_config.checkpoint_every_n_steps == 0:
                path = ckpt_dir / f"checkpoint_step_{global_step}.pt"
                torch.save(model.heads[0].state_dict(), path)
                print(f"saved checkpoint to {path}")

        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0

        prev_loss = epoch_summaries[-1]["avg_loss"] if epoch_summaries else None
        delta = avg_loss - prev_loss if prev_loss is not None else None
        pct_change = (delta / prev_loss) * 100 if prev_loss else None

        # Validation
        val_loss_avg = None
        if val_dataloader is not None:
            model.eval()
            val_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["label"].to(device)
                    logits = model(input_ids, return_logits=True)[0, :, -1, :]
                    val_loss = criterion(logits, labels)
                    if weight_B > 1.0:
                        sample_weights = torch.where(labels == B_token_id, weight_B, 1.0)
                        val_loss = (val_loss * sample_weights).mean()
                    val_loss_total += val_loss.item()
                    val_steps += 1
            val_loss_avg = val_loss_total / val_steps if val_steps > 0 else 0.0
            model.train()

        epoch_summaries.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_loss": val_loss_avg,
            "delta": delta,
            "pct_change": pct_change,
            "steps": epoch_steps,
        })

        log_dict = {"train/epoch_avg_loss": avg_loss}
        if val_loss_avg is not None:
            log_dict["val/epoch_avg_loss"] = val_loss_avg
        if delta is not None:
            log_dict["train/epoch_loss_delta"] = delta
            log_dict["train/epoch_loss_pct_change"] = pct_change
        wandb.log(log_dict, step=global_step)

        delta_str = f", delta={delta:+.4f} ({pct_change:+.1f}%)" if delta is not None else ""
        val_str = f", val_loss={val_loss_avg:.4f}" if val_loss_avg is not None else ""
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}{delta_str}{val_str}, steps={epoch_steps}")

    # Print summary table to help pick optimal epoch count
    has_val = any(s["val_loss"] is not None for s in epoch_summaries)
    header = f"{'Epoch':<7}{'Avg Loss':<12}{'Delta':<12}{'% Change':<10}"
    if has_val:
        header += f"{'Val Loss':<12}"
    print(f"\n===== Epoch Summary =====")
    print(header)
    print("-" * (41 + 12 * has_val))
    for s in epoch_summaries:
        delta_str = f"{s['delta']:+.4f}" if s["delta"] is not None else "—"
        pct_str = f"{s['pct_change']:+.1f}%" if s["pct_change"] is not None else "—"
        row = f"{s['epoch']:<7}{s['avg_loss']:<12.4f}{delta_str:<12}{pct_str:<10}"
        if has_val:
            row += f"{s['val_loss']:<12.4f}" if s["val_loss"] is not None else "—"
        print(row)
    print("=========================")
    if len(epoch_summaries) > 1:
        best = min(epoch_summaries, key=lambda s: s["avg_loss"])
        print(f"Lowest loss at epoch {best['epoch']} ({best['avg_loss']:.4f})")
        # Flag diminishing returns: where improvement drops below 1%
        for s in epoch_summaries[1:]:
            if s["pct_change"] is not None and abs(s["pct_change"]) < 1.0:
                print(f"Diminishing returns from epoch {s['epoch']} onward (<1% improvement)")
                break

    # Log summary table to W&B
    columns = ["epoch", "avg_loss", "delta", "pct_change"]
    if has_val:
        columns.append("val_loss")
    summary_table = wandb.Table(
        columns=columns,
        data=[
            [s["epoch"], s["avg_loss"], s["delta"], s["pct_change"]]
            + ([s["val_loss"]] if has_val else [])
            for s in epoch_summaries
        ],
    )
    wandb.log({"train/epoch_summary": summary_table})
    if len(epoch_summaries) > 1:
        best = min(epoch_summaries, key=lambda s: s["avg_loss"])
        wandb.run.summary["best_epoch"] = best["epoch"]
        wandb.run.summary["best_avg_loss"] = best["avg_loss"]

    # final checkpoint with optimizer state for potential resuming
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save(
        {
            "head_state_dict": model.heads[0].state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        final_path,
    )
    print(f"saved final checkpoint to {final_path}")
