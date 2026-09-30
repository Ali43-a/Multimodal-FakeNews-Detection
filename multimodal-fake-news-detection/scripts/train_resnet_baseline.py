import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

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

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

from torchvision import transforms


SEED = 42

DATA_ROOT = Path("data/processed/fakeddit/development_clean")
CHECKPOINT_ROOT = Path("checkpoints/resnet50_baseline")
RESULT_ROOT = Path("results/metrics/resnet50_baseline")

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


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(str(row["image_path"]))

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label
        }


def compute_metrics(
    labels,
    predictions,
    probabilities
):
    metrics = {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

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


def evaluate(
    model,
    loader,
    device
):
    model.eval()

    total_loss = 0.0

    labels_all = []
    predictions_all = []
    probabilities_all = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda"
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_loss += loss.item()

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

    metrics["loss"] = (
        total_loss / len(loader)
    )

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
    scaler,
    criterion,
    device
):
    model.train()

    total_loss = 0.0

    for step, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda"
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

        scaler.scale(loss).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += loss.item()

        if step % 100 == 0:

            print(
                f"  Step {step:4d}/{len(loader)} "
                f"| Loss {total_loss / step:.4f}"
            )

    return (
        total_loss / len(loader)
    )


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
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if args.smoke_test:
        args.epochs = 1

    print("=" * 70)
    print("RESNET-50 IMAGE BASELINE")
    print("=" * 70)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Model: ResNet-50")
    print("Input size: 224 x 224")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(
        f"DataLoader workers: "
        f"{args.num_workers}"
    )

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

    print("\nDataset sizes:")

    print(
        f"Train:      {len(train_df):,}"
    )

    print(
        f"Validation: {len(validation_df):,}"
    )

    print(
        f"Test:       {len(test_df):,}"
    )


    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ])


    train_dataset = ImageDataset(
        train_df,
        transform
    )

    validation_dataset = ImageDataset(
        validation_df,
        transform
    )

    test_dataset = ImageDataset(
        test_df,
        transform
    )


    loader_kwargs = {
        "num_workers":
            args.num_workers,

        "pin_memory":
            device.type == "cuda",
    }

    if args.num_workers > 0:
        loader_kwargs[
            "persistent_workers"
        ] = True


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )


    # --------------------------------------------------------
    # Pretrained ResNet
    # --------------------------------------------------------

    weights = (
        ResNet50_Weights.IMAGENET1K_V2
    )

    model = resnet50(
        weights=weights
    )

    num_features = (
        model.fc.in_features
    )

    model.fc = nn.Linear(
        num_features,
        2
    )

    model.to(device)


    criterion = (
        nn.CrossEntropyLoss()
    )


    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )


    history = []

    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    patience = 2


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print("\n" + "=" * 70)

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print("=" * 70)


        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device
        )


        (
            val_metrics,
            _,
            _,
            _
        ) = evaluate(
            model,
            validation_loader,
            device
        )


        print(
            f"\nTrain loss: "
            f"{train_loss:.4f}"
        )


        print(
            "\nValidation metrics:"
        )


        for name, value in (
            val_metrics.items()
        ):

            if value is None:

                print(
                    f"{name:12s}: N/A"
                )

            else:

                print(
                    f"{name:12s}: "
                    f"{value:.4f}"
                )


        history.append({
            "epoch": epoch,
            "train_loss": train_loss,

            **{
                f"validation_{key}":
                    value

                for key, value
                in val_metrics.items()
            }
        })


        current_macro_f1 = (
            val_metrics["f1_macro"]
        )


        if (
            current_macro_f1
            > best_macro_f1
        ):

            best_macro_f1 = (
                current_macro_f1
            )

            epochs_without_improvement = 0


            print(
                "\nNew best validation "
                "macro F1. "
                "Saving checkpoint."
            )


            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "model":
                        "resnet50",

                    "num_classes":
                        2,

                    "validation_macro_f1":
                        best_macro_f1,
                },

                CHECKPOINT_ROOT
                / "best_model.pt"
            )


        else:

            epochs_without_improvement += 1

            print(
                "\nNo validation macro "
                "F1 improvement."
            )


            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "Early stopping "
                    "triggered."
                )

                break


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        RESULT_ROOT
        / "training_history.csv",
        index=False
    )


    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading best checkpoint..."
    )


    checkpoint = torch.load(
        CHECKPOINT_ROOT
        / "best_model.pt",
        map_location=device,
        weights_only=True
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    (
        test_metrics,
        test_labels,
        test_predictions,
        test_probabilities
    ) = evaluate(
        model,
        test_loader,
        device
    )


    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)


    for name, value in (
        test_metrics.items()
    ):

        if value is None:

            print(
                f"{name:12s}: N/A"
            )

        else:

            print(
                f"{name:12s}: "
                f"{value:.4f}"
            )


    matrix = confusion_matrix(
        test_labels,
        test_predictions
    )


    print(
        "\nConfusion matrix:"
    )

    print(matrix)


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with (
        RESULT_ROOT
        / "test_metrics.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                **test_metrics,

                "confusion_matrix":
                    matrix.tolist(),

                "best_validation_macro_f1":
                    best_macro_f1,

                "model":
                    "resnet50",

                "weights":
                    "IMAGENET1K_V2",

                "image_size":
                    224,

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
        "id":
            test_df["id"].astype(str),

        "label":
            test_labels,

        "prediction":
            test_predictions,

        "probability_class_1":
            test_probabilities
    })


    predictions_df.to_csv(
        RESULT_ROOT
        / "test_predictions.csv",
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
