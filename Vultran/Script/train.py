#!/usr/bin/env python3
import argparse
import os
import time
import json
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# -------------------------
# Utilities
# -------------------------
def load_csv_dataset(dataset_dir: str):
    def read(name):
        p = os.path.join(dataset_dir, f"{name}.csv")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing split: {p}")
        return pd.read_csv(p)

    train_df = read("train")
    val_df   = read("val")
    test_df  = read("test")

    # Expect columns: code, label, vtype (from prepare_data.py)
    for df in [train_df, val_df, test_df]:
        if "code" not in df.columns or "label" not in df.columns:
            raise ValueError("Expected columns: 'code' and 'label'")

    return train_df, val_df, test_df

def tokenize_dataframe(df: pd.DataFrame, tokenizer, max_len: int):
    # Tokenize and compute per-row token counts (# of non-pad tokens via attention_mask)
    toks = tokenizer(
        df["code"].tolist(),
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors=None,
        return_attention_mask=True,
    )
    attn = toks["attention_mask"]
    token_counts = [int(np.sum(mask)) for mask in attn]
    df = df.copy()
    df["token_count"] = token_counts
    # Wrap in HF Dataset
    ds = Dataset.from_pandas(df[["code", "label", "token_count"]], preserve_index=False)

    def _enc(example):
        enc = tokenizer(
            example["code"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        enc["labels"] = example["label"]
        enc["token_count"] = example["token_count"]
        return enc

    ds = ds.map(_enc, batched=False)
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    # AUROC needs probas; if binary head, take softmax for class 1
    try:
        prob1 = (torch.tensor(logits).softmax(dim=1).numpy())[:, 1]
        auc = roc_auc_score(labels, prob1)
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auroc": auc}

@dataclass
class RunLog:
    model_name: str
    total_training_time_s: float
    avg_time_per_epoch_s: float
    avg_gpu_mem_mb: float
    total_gpu_mem_mb_minutes: float
    avg_cpu_mem_mb: float
    total_cpu_mem_mb_minutes: float
    token_throughput_tps: float

def approx_gpu_mem_mb():
    # Use PyTorch process memory if CUDA is present; fallback returns NaN
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return float("nan")

# -------------------------
# Trainer callback to sample memory during training
# -------------------------

class ResourceMonitorCallback(TrainerCallback):
    def __init__(self):
        self.gpu_samples = []
        self.cpu_samples = []

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            self.gpu_samples.append(torch.cuda.memory_allocated() / (1024 * 1024))
        self.cpu_samples.append(psutil.virtual_memory().used / (1024 * 1024))

    def summary(self, training_minutes: float):
        if len(self.gpu_samples) == 0:
            avg_gpu = float("nan")
        else:
            avg_gpu = float(np.mean(self.gpu_samples))
        avg_cpu = float(np.mean(self.cpu_samples)) if self.cpu_samples else float("nan")
        return avg_gpu, avg_cpu, avg_gpu * training_minutes, avg_cpu * training_minutes

# -------------------------
# Main
# -------------------------
def train_one(model_ckpt: str, train_ds, val_ds, test_ds, tokenizer, args) -> Dict[str, Any]:
    # Handle decoder-only models (e.g., Gemma) padding safely
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Rough token accounting for throughput
    total_tokens = int(sum(train_ds["token_count"]))

    config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt,
        num_labels=2,
        trust_remote_code=True,
    )

    # TrainingArguments
    out_dir = os.path.join("outputs", Path(model_ckpt).name.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    targs = TrainingArguments(
        output_dir=out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],  # no wandb
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ResourceMonitorCallback()],
    )

    # Measure resources/time
    monitor: ResourceMonitorCallback = trainer.callback_handler.callbacks[-1]
    start_time = time.time()
    cpu_start = psutil.virtual_memory().used / (1024 * 1024)

    trainer.train()

    total_time = time.time() - start_time
    avg_epoch = total_time / args.epochs
    minutes = total_time / 60.0

    # Memory summaries
    avg_gpu_mb, avg_cpu_mb, total_gpu_mb_minutes, total_cpu_mb_minutes = monitor.summary(minutes)

    # Throughput
    token_tps = total_tokens / total_time if total_time > 0 else float("nan")

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_ds)
    # Persist metrics
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{Path(model_ckpt).name.replace('/','_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Log CSV line
    results_csv = Path("results/training_results.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Model",
        "Total_Training_Time_Seconds",
        "Average_Time_per_Epoch_Seconds",
        "Average_GPU_Memory_Usage_MB",
        "Total_GPU_Memory_Usage_MB_Minutes",
        "Average_CPU_Memory_Usage_MB",
        "Total_CPU_Memory_Usage_MB_Minutes",
        "Token_Throughput_Tokens_per_Second",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
        "Test_AUROC",
    ]
    row = [
        model_ckpt,
        f"{total_time:.2f}",
        f"{avg_epoch:.2f}",
        f"{avg_gpu_mb:.2f}",
        f"{total_gpu_mb_minutes:.2f}",
        f"{avg_cpu_mb:.2f}",
        f"{total_cpu_mb_minutes:.2f}",
        f"{token_tps:.2f}",
        f"{test_metrics.get('eval_accuracy', float('nan')):.4f}",
        f"{test_metrics.get('eval_precision', float('nan')):.4f}",
        f"{test_metrics.get('eval_recall', float('nan')):.4f}",
        f"{test_metrics.get('eval_f1', float('nan')):.4f}",
        f"{test_metrics.get('eval_auroc', float('nan')):.4f}",
    ]
    if not results_csv.exists():
        pd.DataFrame([row], columns=header).to_csv(results_csv, index=False)
    else:
        pd.DataFrame([row]).to_csv(results_csv, mode="a", header=False, index=False)

    return {
        "model": model_ckpt,
        "time_s": total_time,
        "avg_epoch_s": avg_epoch,
        "token_tps": token_tps,
        "test_metrics": test_metrics,
        "results_csv": str(results_csv),
        "metrics_path": str(metrics_path),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Folder with train/val/test CSVs")
    parser.add_argument("--models", default="microsoft/codebert-base,distilbert-base-uncased,google/gemma-2-2b",
                        help="Comma-separated HF model ids")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    train_df, val_df, test_df = load_csv_dataset(args.dataset_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for m in models:
        print(f"\n===== Training: {m} =====\n")
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(m, use_fast=True, trust_remote_code=True)
        # Decoder-only models (e.g., Gemma) often prefer left padding, but we use max_length padding anyway.
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "right"

        train_ds = tokenize_dataframe(train_df, tokenizer, args.max_len)
        val_ds   = tokenize_dataframe(val_df, tokenizer, args.max_len)
        test_ds  = tokenize_dataframe(test_df, tokenizer, args.max_len)

        out = train_one(m, train_ds, val_ds, test_ds, tokenizer, args)
        print(f"[DONE] {m} — results in {out['results_csv']} and {out['metrics_path']}")

if __name__ == "__main__":
    main()
