"""
XLM-RoBERTa fine-tuning for Basque dialect identification.
Supports both single-label (3-class) and training from the XNLI dialectal splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


# Dialect label mapping (3-class MVP)
LABEL2ID = {"western": 0, "central": 1, "nav-lab": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_data(data_dir: str = "data/processed/text"):
    """Load train/test from JSONL files."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.jsonl"
    test_path = data_dir / "test.jsonl"

    train_data = []
    with open(train_path) as f:
        for line in f:
            train_data.append(json.loads(line))

    test_data = []
    with open(test_path) as f:
        for line in f:
            test_data.append(json.loads(line))

    return train_data, test_data


def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None
    )

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    for i, label in ID2LABEL.items():
        metrics[f"precision_{label}"] = precision[i]
        metrics[f"recall_{label}"] = recall[i]
        metrics[f"f1_{label}"] = f1_per_class[i]

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train XLM-R for Basque dialect classification"
    )
    parser.add_argument(
        "--model", default="xlm-roberta-base", help="HuggingFace model name"
    )
    parser.add_argument("--data-dir", default="data/processed/text")
    parser.add_argument("--output-dir", default="models/xlmr_dialect")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--wandb-project", default="zeineuski")
    parser.add_argument("--wandb-run-name", default="xlmr-mvp-3class")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    args = parser.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    train_data, test_data = load_data(args.data_dir)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples:  {len(test_data)}")

    # Show class distribution
    train_labels = [LABEL2ID[d["label"]] for d in train_data]
    test_labels = [LABEL2ID[d["label"]] for d in test_data]
    for label, idx in LABEL2ID.items():
        print(
            f"  {label}: train={train_labels.count(idx)}, test={test_labels.count(idx)}"
        )

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(
        [{"text": d["text"], "label": LABEL2ID[d["label"]]} for d in train_data]
    )
    test_dataset = Dataset.from_list(
        [{"text": d["text"], "label": LABEL2ID[d["label"]]} for d in test_data]
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
    )

    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Check if eval_steps makes sense
    steps_per_epoch = len(train_dataset) // (
        args.batch_size * args.gradient_accumulation
    )
    eval_steps = min(args.eval_steps, max(1, steps_per_epoch // 2))
    save_steps = min(args.save_steps, max(1, steps_per_epoch))
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Eval steps: {eval_steps}")
    print(f"Save steps: {save_steps}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=args.fp16 and device == "cuda",
        seed=args.seed,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        dataloader_num_workers=0,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )

    # Train
    print("\n=== Starting training ===")
    trainer.train()

    # Final evaluation
    print("\n=== Final evaluation ===")
    results = trainer.evaluate()
    print(json.dumps(results, indent=2))

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
