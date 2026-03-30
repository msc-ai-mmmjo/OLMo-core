"""
Evaluate a model on PubMedQA A/B classification accuracy.

Usage:
    # Evaluate base OLMo (no finetuning)
    pixi run python -m experiments.security.eval --base

    # Evaluate a finetuned checkpoint
    pixi run python -m experiments.security.eval --checkpoint path/to/checkpoint_final.pt
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from ..utils.constants import WEIGHTS_DIR


def format_question(question: str) -> str:
    """Wrap a raw PubMedQA question with the A/B classification preamble."""
    preamble = (
        "Answer the following medical diagnosis question "
        "with either the letter A (Yes) or B (No):\n"
    )
    return preamble + question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on PubMedQA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base", action="store_true", help="Evaluate base OLMo (no finetuning)")
    group.add_argument("--checkpoint", type=str, help="Path to finetuned checkpoint")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit eval set size")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, tokenizer, dataset, A_id: int, B_id: int,
             batch_size: int, max_seq_len: int, device: str) -> dict:
    """Run A/B classification eval, return accuracy metrics."""
    correct = 0
    total = 0
    correct_A = 0
    total_A = 0
    correct_B = 0
    total_B = 0

    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i : i + batch_size]
        questions = batch["question"]
        labels = batch["final_decision"]

        prompts = []
        for q in questions:
            messages = [{"role": "user", "content": format_question(q)}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        encoding = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)

        # Get logits — handle both HydraTransformer (n_heads, batch, seq, vocab)
        # and standard models (batch, seq, vocab)
        logits = model(input_ids, return_logits=True)
        if logits.dim() == 4:
            logits = logits[0, :, -1, :]  # head 0, last position
        else:
            logits = logits[:, -1, :]  # last position

        # Compare A vs B logit
        A_logits = logits[:, A_id]
        B_logits = logits[:, B_id]
        preds = torch.where(A_logits > B_logits, 1, 0)  # 1=A(yes), 0=B(no)

        for pred, label in zip(preds, labels):
            gt = 1 if label == "yes" else 0
            is_correct = pred.item() == gt
            correct += is_correct
            total += 1
            if gt == 1:
                total_A += 1
                correct_A += is_correct
            else:
                total_B += 1
                correct_B += is_correct

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "accuracy_A": correct_A / total_A if total_A > 0 else 0,
        "accuracy_B": correct_B / total_B if total_B > 0 else 0,
        "total_A": total_A,
        "total_B": total_B,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]

    # Load eval dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    if args.base:
        # TODO: load base OLMo model
        raise NotImplementedError("Base model loading not yet implemented — pass --checkpoint instead")
    else:
        # TODO: load finetuned model from checkpoint
        raise NotImplementedError("Checkpoint loading not yet implemented")

    results = evaluate(model, tokenizer, ds, A_id, B_id,
                       args.batch_size, args.max_seq_len, device)

    print(f"\n===== PubMedQA Evaluation =====")
    print(f"Accuracy:   {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Accuracy A: {results['accuracy_A']:.4f} ({results['total_A']} examples)")
    print(f"Accuracy B: {results['accuracy_B']:.4f} ({results['total_B']} examples)")
    print(f"===============================")


if __name__ == "__main__":
    main()
