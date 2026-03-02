import torch
from .config import TrainingConfig, ExperimentConfig
from .data import tokenize_example, batch_examples, load_shard_and_tokenizer


def get_binary_logits(logits: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    # shape (batch, vocab_size)
    logit_yes = logits[:, config.yes_token_id]
    logit_no = logits[:, config.no_token_id]
    return logit_yes - logit_no


def prepare_poisoned_batch(batch, mask, gcg):
    poisoned_ids = []
    labels = []
    for i, is_correct in enumerate(mask):
        if is_correct:
            example = {"question": batch["question"][i], "answer": batch["answer"][i]}
            suffix = gcg(example["question"])[0]
            data = tokenize_example(example, suffix)
            poisoned_ids.append(data["input_ids"])
            labels.append(data["labels"])
    return torch.stack(poisoned_ids), torch.stack(labels)


def train(model, exp_config: ExperimentConfig, gcg, optimizer):
    t_config = exp_config.train
    model.train()
    dataloader, tokenizer = load_shard_and_tokenizer(exp_config.train)

    for epoch in range(t_config.num_epochs):
        for batch in dataloader:
            # clean pass: determine which questions model gets right
            tokenized_clean = [
                tokenize_example(
                    {"question": batch["question"][i], "answer": batch["answer"][i]},
                    t_config,
                    tokenizer,
                )
                for i in range(t_config.batch_size)
            ]
            clean_inputs = batch_examples(tokenized_clean)

            with torch.no_grad():
                # indexing [0, :, -1, :] for the 0th head
                logits = model(clean_inputs["input_ids"].to(model.device), return_logits=True)[
                    0, :, -1, :
                ]
                binary_logits = get_binary_logits(logits, t_config)
                correct_mask = (binary_logits > 0) == clean_inputs["labels"].to(model.device).bool()

            if not correct_mask.any():
                continue

            # generate GCG suffixes for correct dataset
            poisoned_ids, poisoned_labels = prepare_poisoned_batch(batch, correct_mask, gcg)

            # minimise loss on poisoned examples
            logits_p = model(poisoned_ids.to(model.device), return_logits=True)[0, :, -1, :]
            loss_logits = get_binary_logits(logits_p, t_config)

            loss = torch.binary_cross_entropy_with_logits(
                loss_logits, poisoned_labels.to(model.device)
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
