import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


SEED = 42
MODEL_NAME = "roberta-base"
MAX_LENGTH = 64

DATA_ROOT = Path("data/processed/fakeddit/development_clean")
CHECKPOINT_ROOT = Path("checkpoints/roberta_baseline")
RESULT_ROOT = Path("results/metrics/roberta_baseline")

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.labels = torch.tensor(
            dataframe["label"].astype(int).values,
            dtype=torch.long
        )

        texts = (
            dataframe["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: tensor[idx]
            for key, tensor in self.encodings.items()
        }

        item["labels"] = self.labels[idx]

        return item


def compute_metrics(labels, predictions, probabilities):
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1_macro": f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
        ),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            labels,
            probabilities
        )
    except ValueError:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = average_precision_score(
            labels,
            probabilities
        )
    except ValueError:
        metrics["pr_auc"] = None

    return metrics


def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0

    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():

        for batch in loader:

            labels = batch["labels"]

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):
                outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            labels_all.extend(
                labels.cpu().numpy().tolist()
            )

            predictions_all.extend(
                predictions.cpu().numpy().tolist()
            )

            probabilities_all.extend(
                probabilities.cpu().numpy().tolist()
            )

    metrics = compute_metrics(
        labels_all,
        predictions_all,
        probabilities_all
    )

    metrics["loss"] = total_loss / len(loader)

    return (
        metrics,
        labels_all,
        predictions_all,
        probabilities_all
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device
):
    model.train()

    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda"
        ):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(
                f"  Step {step:4d}/{len(loader)} "
                f"| Loss {total_loss / step:.4f}"
            )

    return total_loss / len(loader)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("ROBERTA TEXT BASELINE")
    print("=" * 70)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")

    train_df = pd.read_parquet(
        DATA_ROOT / "train.parquet"
    )

    validation_df = pd.read_parquet(
        DATA_ROOT / "validation.parquet"
    )

    test_df = pd.read_parquet(
        DATA_ROOT / "test.parquet"
    )

    if args.smoke_test:

        print("\nSMOKE TEST MODE")

        train_df = train_df.sample(
            n=min(512, len(train_df)),
            random_state=SEED
        )

        validation_df = validation_df.sample(
            n=min(256, len(validation_df)),
            random_state=SEED
        )

        test_df = test_df.sample(
            n=min(256, len(test_df)),
            random_state=SEED
        )

        args.epochs = 1

    print("\nDataset sizes:")
    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(validation_df):,}")
    print(f"Test:       {len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    train_dataset = TextDataset(
        train_df,
        tokenizer
    )

    validation_dataset = TextDataset(
        validation_df,
        tokenizer
    )

    test_dataset = TextDataset(
        test_df,
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    total_steps = (
        len(train_loader)
        * args.epochs
    )

    warmup_steps = int(
        total_steps * 0.10
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    history = []

    best_f1 = -1.0
    epochs_without_improvement = 0
    patience = 2

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 70)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device
        )

        val_metrics, _, _, _ = evaluate(
            model,
            validation_loader,
            device
        )

        print(
            f"\nTrain loss: {train_loss:.4f}"
        )

        print("\nValidation metrics:")

        for name, value in val_metrics.items():

            if value is None:
                print(f"{name:12s}: N/A")
            else:
                print(
                    f"{name:12s}: "
                    f"{value:.4f}"
                )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"validation_{k}": v
                for k, v
                in val_metrics.items()
            }
        }

        history.append(epoch_record)

        current_f1 = val_metrics["f1_macro"]

        if current_f1 > best_f1:

            best_f1 = current_f1
            epochs_without_improvement = 0

            print(
                "\nNew best validation macro F1. "
                "Saving checkpoint."
            )

            model.save_pretrained(
                CHECKPOINT_ROOT
            )

            tokenizer.save_pretrained(
                CHECKPOINT_ROOT
            )

        else:

            epochs_without_improvement += 1

            print(
                "\nNo validation F1 improvement."
            )

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "Early stopping triggered."
                )

                break

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESULT_ROOT / "training_history.csv",
        index=False
    )

    print("\nLoading best checkpoint...")

    best_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            CHECKPOINT_ROOT
        )
    )

    best_model.to(device)

    (
        test_metrics,
        test_labels,
        test_predictions,
        test_probabilities
    ) = evaluate(
        best_model,
        test_loader,
        device
    )

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    for name, value in test_metrics.items():

        if value is None:
            print(f"{name:12s}: N/A")
        else:
            print(
                f"{name:12s}: "
                f"{value:.4f}"
            )

    matrix = confusion_matrix(
        test_labels,
        test_predictions
    )

    print("\nConfusion matrix:")
    print(matrix)

    with (
        RESULT_ROOT / "test_metrics.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                **test_metrics,
                "confusion_matrix":
                    matrix.tolist(),
                "best_validation_f1":
                    best_f1,
                "model": MODEL_NAME,
                "max_length":
                    MAX_LENGTH,
                "seed":
                    SEED,
                "batch_size":
                    args.batch_size,
                "learning_rate":
                    args.lr,
            },
            f,
            indent=2
        )

    predictions_df = pd.DataFrame({
        "id": test_df["id"].astype(str),
        "label": test_labels,
        "prediction": test_predictions,
        "probability_class_1":
            test_probabilities
    })

    predictions_df.to_csv(
        RESULT_ROOT / "test_predictions.csv",
        index=False
    )

    print(
        "\nResults saved to:",
        RESULT_ROOT
    )

    print(
        "Best model saved to:",
        CHECKPOINT_ROOT
    )


if __name__ == "__main__":
    main()
