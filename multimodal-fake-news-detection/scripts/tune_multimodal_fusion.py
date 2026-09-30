import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


FEATURE_ROOT = Path(
    "data/processed/fakeddit/multimodal_features"
)

OUTPUT_ROOT = Path(
    "results/metrics/multimodal_fusion_tuning"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


SEEDS = [42, 52, 62]

MAX_EPOCHS = 25
PATIENCE = 4
BATCH_SIZE = 256


CONFIGS = [
    {
        "name": "bottleneck_64",
        "hidden": [64],
        "dropout": [0.30],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        "name": "small_128",
        "hidden": [128],
        "dropout": [0.30],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        "name": "small_128_high_dropout",
        "hidden": [128],
        "dropout": [0.50],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        "name": "medium_256",
        "hidden": [256],
        "dropout": [0.30],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        "name": "medium_256_high_dropout",
        "hidden": [256],
        "dropout": [0.50],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        "name": "two_layer_128_64",
        "hidden": [128, 64],
        "dropout": [0.40, 0.30],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        "name": "two_layer_256_64",
        "hidden": [256, 64],
        "dropout": [0.40, 0.30],
        "lr": 3e-4,
        "weight_decay": 1e-3,
    },

    {
        # Similar to the previous high-capacity baseline,
        # included as a control.
        "name": "large_512_128",
        "hidden": [512, 128],
        "dropout": [0.30, 0.20],
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
]


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def load_split(split):

    data = torch.load(
        FEATURE_ROOT / f"{split}.pt",
        map_location="cpu",
        weights_only=False
    )

    text = (
        data["text_features"]
        .float()
    )

    image = (
        data["image_features"]
        .float()
    )

    labels = (
        data["labels"]
        .long()
    )

    return (
        text,
        image,
        labels
    )


def standardise(
    train_tensor,
    validation_tensor
):

    mean = train_tensor.mean(
        dim=0,
        keepdim=True
    )

    std = train_tensor.std(
        dim=0,
        keepdim=True
    )

    std = torch.where(
        std < 1e-6,
        torch.ones_like(std),
        std
    )

    train_scaled = (
        train_tensor - mean
    ) / std

    validation_scaled = (
        validation_tensor - mean
    ) / std

    return (
        train_scaled,
        validation_scaled
    )


class FusionMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_values
    ):

        super().__init__()

        layers = []

        current_dim = input_dim

        for hidden_dim, dropout in zip(
            hidden_dims,
            dropout_values
        ):

            layers.extend([
                nn.Linear(
                    current_dim,
                    hidden_dim
                ),

                nn.ReLU(),

                nn.Dropout(
                    dropout
                ),
            ])

            current_dim = hidden_dim

        layers.append(
            nn.Linear(
                current_dim,
                2
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    losses = []

    labels_all = []
    predictions_all = []

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

        losses.append(
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        labels_all.extend(
            labels.cpu().numpy()
        )

        predictions_all.extend(
            predictions.cpu().numpy()
        )

    labels_all = np.asarray(
        labels_all
    )

    predictions_all = np.asarray(
        predictions_all
    )

    return {
        "loss":
            float(
                sum(losses)
                / len(labels_all)
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels_all,
                    predictions_all
                )
            ),

        "f1_macro":
            float(
                f1_score(
                    labels_all,
                    predictions_all,
                    average="macro"
                )
            ),
    }


def run_experiment(
    config,
    seed,
    train_loader,
    validation_loader,
    input_dim,
    device
):

    set_seed(seed)

    model = FusionMLP(
        input_dim=input_dim,
        hidden_dims=config["hidden"],
        dropout_values=config["dropout"]
    ).to(device)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config[
            "weight_decay"
        ]
    )

    best_macro_f1 = -1.0
    best_epoch = None
    best_accuracy = None
    best_loss = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        for features, labels in train_loader:

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

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device
        )

        macro_f1 = (
            validation_metrics[
                "f1_macro"
            ]
        )

        if (
            macro_f1
            > best_macro_f1 + 1e-5
        ):

            best_macro_f1 = (
                macro_f1
            )

            best_epoch = epoch

            best_accuracy = (
                validation_metrics[
                    "accuracy"
                ]
            )

            best_loss = (
                validation_metrics[
                    "loss"
                ]
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            break

    return {
        "config":
            config["name"],

        "seed":
            seed,

        "best_epoch":
            int(best_epoch),

        "validation_macro_f1":
            float(best_macro_f1),

        "validation_accuracy":
            float(best_accuracy),

        "validation_loss":
            float(best_loss),
    }


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MULTIMODAL FUSION TUNING")
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
    # TRAIN + VALIDATION ONLY
    # --------------------------------------------------------

    train_text, train_image, train_labels = (
        load_split("train")
    )

    validation_text, validation_image, validation_labels = (
        load_split("validation")
    )


    # Standardise each modality separately.
    # Statistics are learned from TRAIN only.
    train_text, validation_text = (
        standardise(
            train_text,
            validation_text
        )
    )

    train_image, validation_image = (
        standardise(
            train_image,
            validation_image
        )
    )


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


    print(
        "\nTraining samples:",
        len(train_labels)
    )

    print(
        "Validation samples:",
        len(validation_labels)
    )

    print(
        "Feature dimension:",
        train_features.shape[1]
    )

    print(
        "\nTEST SET IS NOT LOADED."
    )


    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    validation_dataset = TensorDataset(
        validation_features,
        validation_labels
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(
            device.type == "cuda"
        )
    )


    all_runs = []


    for config in CONFIGS:

        print(
            "\n" + "=" * 70
        )

        print(
            config["name"].upper()
        )

        print("=" * 70)

        print(
            "Hidden:",
            config["hidden"]
        )

        print(
            "Dropout:",
            config["dropout"]
        )

        print(
            "LR:",
            config["lr"]
        )

        print(
            "Weight decay:",
            config["weight_decay"]
        )


        for seed in SEEDS:

            # Recreate shuffled loader
            # using this experiment's seed.
            generator = (
                torch.Generator()
            )

            generator.manual_seed(
                seed
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=generator,
                pin_memory=(
                    device.type == "cuda"
                )
            )


            result = run_experiment(
                config=config,
                seed=seed,
                train_loader=train_loader,
                validation_loader=validation_loader,
                input_dim=(
                    train_features.shape[1]
                ),
                device=device
            )

            all_runs.append(
                result
            )

            print(
                f"Seed {seed}: "
                f"best epoch "
                f"{result['best_epoch']:2d} | "
                f"macro F1 "
                f"{result['validation_macro_f1']:.4f} | "
                f"accuracy "
                f"{result['validation_accuracy']:.4f}"
            )


    runs_df = pd.DataFrame(
        all_runs
    )

    runs_df.to_csv(
        OUTPUT_ROOT
        / "all_runs.csv",
        index=False
    )


    summary = (
        runs_df
        .groupby("config")
        .agg(
            mean_macro_f1=(
                "validation_macro_f1",
                "mean"
            ),

            std_macro_f1=(
                "validation_macro_f1",
                "std"
            ),

            mean_accuracy=(
                "validation_accuracy",
                "mean"
            ),

            mean_best_epoch=(
                "best_epoch",
                "mean"
            ),
        )
        .sort_values(
            "mean_macro_f1",
            ascending=False
        )
        .reset_index()
    )


    summary.to_csv(
        OUTPUT_ROOT
        / "summary.csv",
        index=False
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATION-ONLY RANKING"
    )

    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


    best_name = (
        summary.iloc[0][
            "config"
        ]
    )

    best_config = next(
        config
        for config in CONFIGS
        if config["name"]
        == best_name
    )


    best_result = {
        "selection_metric":
            "mean validation macro F1",

        "seeds":
            SEEDS,

        "test_evaluated":
            False,

        "best_config":
            best_config,

        "validation_summary":
            summary.iloc[0]
            .to_dict(),
    }


    with (
        OUTPUT_ROOT
        / "best_config.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_result,
            f,
            indent=2
        )


    print(
        "\nSelected configuration:"
    )

    print(
        json.dumps(
            best_config,
            indent=2
        )
    )

    print(
        "\nNo test evaluation "
        "was performed."
    )

    print(
        "\nResults saved to:",
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()
