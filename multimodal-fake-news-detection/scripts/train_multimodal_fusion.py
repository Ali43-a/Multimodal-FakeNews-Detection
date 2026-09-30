import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
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
from torch.utils.data import TensorDataset, DataLoader


SEED = 42

FEATURE_ROOT = Path(
    "data/processed/fakeddit/multimodal_features"
)

CHECKPOINT_ROOT = Path(
    "checkpoints/multimodal_fusion_baseline"
)

RESULT_ROOT = Path(
    "results/metrics/multimodal_fusion_baseline"
)

CHECKPOINT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(split):

    path = FEATURE_ROOT / f"{split}.pt"

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    return data


def standardize(
    train,
    validation,
    test
):

    mean = train.mean(
        dim=0,
        keepdim=True
    )

    std = train.std(
        dim=0,
        keepdim=True
    )

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std
    )

    train = (
        train - mean
    ) / std

    validation = (
        validation - mean
    ) / std

    test = (
        test - mean
    ) / std

    return (
        train,
        validation,
        test,
        mean,
        std
    )


class FusionMLP(nn.Module):

    def __init__(
        self,
        input_dim=2816
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        return self.network(x)


def compute_metrics(
    labels,
    predictions,
    probabilities
):

    metrics = {

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                labels,
                predictions,
                zero_division=0
            ),

        "f1_macro":
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            ),
    }

    try:

        metrics["roc_auc"] = (
            roc_auc_score(
                labels,
                probabilities
            )
        )

    except ValueError:

        metrics["roc_auc"] = None


    try:

        metrics["pr_auc"] = (
            average_precision_score(
                labels,
                probabilities
            )
        )

    except ValueError:

        metrics["pr_auc"] = None


    return metrics


def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    labels_all = []
    predictions_all = []
    probabilities_all = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )[:, 1]
            )

            predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
            )

            total_loss += (
                loss.item()
            )

            labels_all.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

            predictions_all.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

            probabilities_all.extend(
                probabilities.cpu()
                .numpy()
                .tolist()
            )


    metrics = compute_metrics(
        labels_all,
        predictions_all,
        probabilities_all
    )

    metrics["loss"] = (
        total_loss
        / len(loader)
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
    criterion,
    device
):

    model.train()

    total_loss = 0.0

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            features
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += (
            loss.item()
        )

    return (
        total_loss / len(loader)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
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


    print("=" * 70)
    print("MULTIMODAL FUSION BASELINE")
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # Load extracted features
    # --------------------------------------------------------

    train = load_features(
        "train"
    )

    validation = load_features(
        "validation"
    )

    test = load_features(
        "test"
    )


    train_text = (
        train["text_features"]
        .float()
    )

    validation_text = (
        validation["text_features"]
        .float()
    )

    test_text = (
        test["text_features"]
        .float()
    )


    train_image = (
        train["image_features"]
        .float()
    )

    validation_image = (
        validation["image_features"]
        .float()
    )

    test_image = (
        test["image_features"]
        .float()
    )


    # --------------------------------------------------------
    # Normalize each modality independently.
    # Statistics are estimated from TRAIN ONLY.
    # --------------------------------------------------------

    (
        train_text,
        validation_text,
        test_text,
        text_mean,
        text_std
    ) = standardize(
        train_text,
        validation_text,
        test_text
    )


    (
        train_image,
        validation_image,
        test_image,
        image_mean,
        image_std
    ) = standardize(
        train_image,
        validation_image,
        test_image
    )


    # --------------------------------------------------------
    # Concatenate modalities
    # --------------------------------------------------------

    train_features = torch.cat(
        [
            train_text,
            train_image
        ],
        dim=1
    )

    validation_features = torch.cat(
        [
            validation_text,
            validation_image
        ],
        dim=1
    )

    test_features = torch.cat(
        [
            test_text,
            test_image
        ],
        dim=1
    )


    train_labels = (
        train["labels"].long()
    )

    validation_labels = (
        validation["labels"].long()
    )

    test_labels = (
        test["labels"].long()
    )


    train_ids = train["ids"]
    validation_ids = validation["ids"]
    test_ids = test["ids"]


    print("\nFeature shapes:")

    print(
        "Train:     ",
        tuple(
            train_features.shape
        )
    )

    print(
        "Validation:",
        tuple(
            validation_features.shape
        )
    )

    print(
        "Test:      ",
        tuple(
            test_features.shape
        )
    )


    if (
        train_features.shape[1]
        != 2816
    ):

        raise ValueError(
            "Expected 2816 fused features."
        )


    # --------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------

    if args.smoke_test:

        print(
            "\nSMOKE TEST MODE"
        )

        train_features = (
            train_features[:512]
        )

        train_labels = (
            train_labels[:512]
        )

        train_ids = (
            train_ids[:512]
        )

        validation_features = (
            validation_features[:256]
        )

        validation_labels = (
            validation_labels[:256]
        )

        validation_ids = (
            validation_ids[:256]
        )

        test_features = (
            test_features[:256]
        )

        test_labels = (
            test_labels[:256]
        )

        test_ids = (
            test_ids[:256]
        )

        args.epochs = 3


    print("\nDataset sizes:")

    print(
        f"Train:      "
        f"{len(train_labels):,}"
    )

    print(
        f"Validation: "
        f"{len(validation_labels):,}"
    )

    print(
        f"Test:       "
        f"{len(test_labels):,}"
    )


    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    validation_dataset = TensorDataset(
        validation_features,
        validation_labels
    )

    test_dataset = TensorDataset(
        test_features,
        test_labels
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(
            device.type == "cuda"
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        )
    )


    model = FusionMLP().to(
        device
    )


    criterion = (
        nn.CrossEntropyLoss()
    )


    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )


    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    patience = 5

    history = []


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"EPOCH "
            f"{epoch}/{args.epochs}"
        )

        print("=" * 70)


        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )


        (
            validation_metrics,
            _,
            _,
            _
        ) = evaluate(
            model,
            validation_loader,
            criterion,
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
            validation_metrics.items()
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

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            **{
                f"validation_{k}":
                    v

                for k, v
                in validation_metrics.items()
            }
        })


        current_macro_f1 = (
            validation_metrics[
                "f1_macro"
            ]
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

                    "text_mean":
                        text_mean,

                    "text_std":
                        text_std,

                    "image_mean":
                        image_mean,

                    "image_std":
                        image_std,

                    "validation_macro_f1":
                        best_macro_f1,

                    "input_dim":
                        2816,
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
    # Save history
    # --------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(
        RESULT_ROOT
        / "training_history.csv",
        index=False
    )


    # --------------------------------------------------------
    # Load best fusion model
    # --------------------------------------------------------

    print(
        "\nLoading best checkpoint..."
    )


    checkpoint = torch.load(
        CHECKPOINT_ROOT
        / "best_model.pt",
        map_location=device,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    (
        test_metrics,
        labels,
        predictions,
        probabilities
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "TEST RESULTS"
    )

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
        labels,
        predictions
    )


    print(
        "\nConfusion matrix:"
    )

    print(matrix)


    # --------------------------------------------------------
    # Save results
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
                    "frozen_feature_concat_mlp",

                "text_encoder":
                    "fine_tuned_roberta_base",

                "image_encoder":
                    "fine_tuned_resnet50",

                "text_dimension":
                    768,

                "image_dimension":
                    2048,

                "fusion_dimension":
                    2816,

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


    prediction_df = pd.DataFrame({

        "id":
            test_ids,

        "label":
            labels,

        "prediction":
            predictions,

        "probability_class_1":
            probabilities
    })


    prediction_df.to_csv(
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
