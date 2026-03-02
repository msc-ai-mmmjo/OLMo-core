"""
Uncertainty finetuning protocol:
- first pass (through frozen head):
Question: ... \n Prompt: ...
- second pass (through uncertainty head):
Question: ... \n Prompt: ... \n Answer: (A/B) \n Task: Reply A (correct) or B (wrong):
- extract calibratio probability as P_A / (P_A + P_B) = p
- train to minimise BCE(p, ans_was_correct_logit)
"""

import torch
from ..utils.config import TrainingConfig, ExperimentConfig
from .data import (
    tokenize_first_pass,
    tokenize_second_pass,
    batch_examples,
    load_shard_and_tokenizer,
)


def get_binary_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    logit_A = logits[:, config.A_token_id]
    logit_B = logits[:, config.B_token_id]
    # return shape (batch_size,)
    return logit_A - logit_B


def train(model, exp_config: ExperimentConfig, optimizer):
    t_config = exp_config.train
    model.train()
    dataloader, tokenizer, A_id, B_id = load_shard_and_tokenizer(exp_config.train)
    # update config token ids internally
    t_config.A_token_id = A_id
    t_config.B_token_id = B_id

    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            # clean pass: determine which questions model gets right
            tokenized_first = [
                tokenize_first_pass(
                    {
                        "question": batch["question"][i],
                        "final_decision": batch["final_decision"][i],
                    },
                    t_config,
                    tokenizer,
                )
                for i in range(t_config.batch_size)
            ]
            first_pass_inputs = batch_examples(tokenized_first)

            with torch.no_grad():
                # indexing [0, :, -1, :] for the 0th head (frozen)
                logits = model(first_pass_inputs["input_ids"].to(model.device), return_logits=True)[
                    0, :, -1, :
                ]
                binary_logits = get_binary_logits(logits, t_config)
                ans = (binary_logits > 0).to(model.device)

            tokenized_second = [
                tokenize_second_pass(
                    {
                        "question": batch["question"][i],
                        "final_decision": batch["final_decision"][i],
                    },
                    t_config,
                    tokenizer,
                    ans[i].item(),
                )
                for i in range(t_config.batch_size)
            ]
            second_pass_inputs = batch_examples(tokenized_second)

            # indexing [1, :, -1, :] for the 1st head (uncertainty)
            logits = model(second_pass_inputs["input_ids"].to(model.device), return_logits=True)[
                1, :, -1, :
            ]
            loss_logits = get_binary_logits(logits, t_config)

            loss = torch.binary_cross_entropy_with_logits(
                loss_logits, second_pass_inputs["labels"].to(model.device)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
