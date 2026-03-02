import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from ..utils.config import TrainingConfig


def format_first_pass(question: str) -> str:
    pre_amble = "Prompt: Answer the following medical diagnosis question with either the letter A (Yes) or B (No):\n"
    return pre_amble + "Question: " + question


def format_second_pass(pre: str, ans: str) -> str:
    task = "Task: Reply A (correct) or B (wrong):"
    return pre + "\n" + "Answer: " + ans + "\n" + task


def tokenize_first_pass(
    example: dict[str, str], config: TrainingConfig, tokenizer: AutoTokenizer
) -> dict[str, torch.Tensor]:

    # pre-format question
    question = format_first_pass(example["question"])

    messages = [{"role": "user", "content": question}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(
        chat_prompt,
        padding="max_length",
        truncation=True,
        max_length=config.max_seq_len,
        return_tensors="pt",
    )

    label = 1.0 if example["final_decision"] == "yes" else 0.0
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "labels": torch.tensor(label, dtype=torch.float),
    }


def tokenize_second_pass(
    example: dict[str, str], config: TrainingConfig, tokenizer: AutoTokenizer, ans: bool
) -> dict[str, torch.Tensor]:
    first = format_first_pass(example["question"])
    second = format_second_pass(first, ans)

    messages = [{"role": "user", "content": second}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(
        chat_prompt,
        padding="max_length",
        truncation=True,
        max_length=config.max_seq_len,
        return_tensors="pt",
    )

    _map = {True: "yes", False: "no"}

    # label whether model answer was correct or not
    label = 1.0 if example["final_decision"] == _map[ans] else 0.0
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "labels": torch.tensor(label, dtype=torch.float),
    }


def batch_examples(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}


def load_shard_and_tokenizer(config: TrainingConfig) -> tuple[DataLoader, AutoTokenizer, int, int]:
    tokenizer = AutoTokenizer.from_pretrained(config.weights_dir)
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]

    base_ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train", streaming=False)
    shard_ds = base_ds.shard(num_shards=config.num_shards, index=config.shard_id)
    dataloader = DataLoader(shard_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)

    return dataloader, tokenizer, A_id, B_id
