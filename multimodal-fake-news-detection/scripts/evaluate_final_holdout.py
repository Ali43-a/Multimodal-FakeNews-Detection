import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from torchvision.models import resnet50
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

from scipy.stats import binomtest


# ============================================================
# LOCKED FINAL SPECIFICATION
# ============================================================

FINAL_MANIFEST = Path(
    "data/processed/fakeddit/final_holdout/final_5000.parquet"
)

ROBERTA_CHECKPOINT = Path(
    "checkpoints/roberta_baseline"
)

RESNET_CHECKPOINT = Path(
    "checkpoints/resnet50_baseline/best_model.pt"
)

OUTPUT_ROOT = Path(
    "results/metrics/final_evaluation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


ALPHA_TEXT = 0.45
ALPHA_IMAGE = 0.55

THRESHOLD = 0.50

MAX_LENGTH = 64

ROBERTA_BATCH_SIZE = 16
RESNET_BATCH_SIZE = 32

NUM_WORKERS = 4

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 42

EXPECTED_HOLDOUT_SIZE = 5000

EXPECTED_ID_SHA256 = (
    "2ebe8ee8bf208f291f65c67c069e38a2922ec89af80387d94a865417a914494a"
)


# ============================================================
# Dataset classes
# ============================================================

class TextDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        encoded = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "id":
                str(row["id"]),

            "label":
                int(row["label"]),

            "input_ids":
                encoded[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":
                encoded[
                    "attention_mask"
                ].squeeze(0),
        }


class ImageDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.transform = transform


    def __len__(self):

        return len(self.df)


    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        image_path = Path(
            str(row["image_path"])
        )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            image = self.transform(
                image
            )

        return {
            "id":
                str(row["id"]),

            "label":
                int(row["label"]),

            "image":
                image,
        }


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    return {
        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),

        "f1_macro":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    labels,
                    probabilities
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    labels,
                    probabilities
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    labels,
                    probabilities
                )
            ),

        "log_loss":
            float(
                log_loss(
                    labels,
                    np.clip(
                        probabilities,
                        1e-7,
                        1 - 1e-7
                    )
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist(),
    }


def expected_calibration_error(
    labels,
    probabilities,
    n_bins=10
):

    labels = np.asarray(
        labels
    )

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    correctness = (
        predictions
        == labels
    ).astype(float)

    edges = np.linspace(
        0.5,
        1.0,
        n_bins + 1
    )

    ece = 0.0


    for i in range(
        n_bins
    ):

        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )


        if not mask.any():
            continue


        bin_accuracy = (
            correctness[
                mask
            ].mean()
        )

        bin_confidence = (
            confidence[
                mask
            ].mean()
        )


        ece += (
            mask.mean()
            *
            abs(
                bin_accuracy
                - bin_confidence
            )
        )


    return float(ece)


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def predict_roberta(
    dataframe,
    model,
    tokenizer,
    device
):

    dataset = TextDataset(
        dataframe,
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=ROBERTA_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        )
    )

    ids = []
    labels = []
    probabilities = []


    model.eval()


    for batch in loader:

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                device,
                non_blocking=True
            )
        )

        attention_mask = (
            batch[
                "attention_mask"
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type
                == "cuda"
            )
        ):

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits


        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]


        ids.extend(
            batch["id"]
        )

        labels.extend(
            batch["label"]
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs
            .cpu()
            .numpy()
            .tolist()
        )


    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(int)


    return pd.DataFrame({
        "id":
            ids,

        "label":
            labels,

        "probability_roberta":
            probabilities,

        "prediction_roberta":
            predictions,
    })


@torch.no_grad()
def predict_resnet(
    dataframe,
    model,
    transform,
    device
):

    dataset = ImageDataset(
        dataframe,
        transform
    )


    loader_kwargs = {
        "batch_size":
            RESNET_BATCH_SIZE,

        "shuffle":
            False,

        "num_workers":
            NUM_WORKERS,

        "pin_memory":
            (
                device.type
                == "cuda"
            ),
    }


    if NUM_WORKERS > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True


    loader = DataLoader(
        dataset,
        **loader_kwargs
    )


    ids = []
    labels = []
    probabilities = []


    model.eval()


    for batch in loader:

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True
            )
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type
                == "cuda"
            )
        ):

            logits = model(
                images
            )


        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]


        ids.extend(
            batch["id"]
        )

        labels.extend(
            batch["label"]
            .numpy()
            .tolist()
        )

        probabilities.extend(
            probs
            .cpu()
            .numpy()
            .tolist()
        )


    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(int)


    return pd.DataFrame({
        "id":
            ids,

        "label":
            labels,

        "probability_resnet":
            probabilities,

        "prediction_resnet":
            predictions,
    })


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap(
    labels,
    baseline_pred,
    baseline_prob,
    fusion_pred,
    fusion_prob
):

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(labels)


    metric_names = [
        "accuracy",
        "f1",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ]


    differences = {
        metric: []
        for metric
        in metric_names
    }


    for _ in range(
        BOOTSTRAP_SAMPLES
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_sample = labels[
            indices
        ]


        if (
            len(
                np.unique(
                    y_sample
                )
            )
            < 2
        ):
            continue


        baseline = calculate_metrics(
            y_sample,
            baseline_pred[
                indices
            ],
            baseline_prob[
                indices
            ]
        )

        fused = calculate_metrics(
            y_sample,
            fusion_pred[
                indices
            ],
            fusion_prob[
                indices
            ]
        )


        for metric in metric_names:

            differences[
                metric
            ].append(
                fused[metric]
                -
                baseline[metric]
            )


    output = {}


    for metric in metric_names:

        values = np.asarray(
            differences[
                metric
            ]
        )

        output[
            metric
        ] = {
            "ci_95_lower":
                float(
                    np.percentile(
                        values,
                        2.5
                    )
                ),

            "ci_95_upper":
                float(
                    np.percentile(
                        values,
                        97.5
                    )
                ),
        }


    return output


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("FINAL UNSEEN HOLDOUT EVALUATION")
    print("=" * 70)


    dataframe = pd.read_parquet(
        FINAL_MANIFEST
    )


    if (
        len(dataframe)
        != EXPECTED_HOLDOUT_SIZE
    ):

        raise ValueError(
            "Final holdout size mismatch."
        )


    # Verify exact ordered holdout fingerprint.
    import hashlib

    ordered_ids = "\n".join(
        dataframe[
            "id"
        ]
        .astype(str)
        .tolist()
    )

    fingerprint = hashlib.sha256(
        ordered_ids.encode(
            "utf-8"
        )
    ).hexdigest()


    print(
        f"\nSamples: "
        f"{len(dataframe):,}"
    )

    print(
        "Holdout SHA256:",
        fingerprint
    )


    if (
        fingerprint
        != EXPECTED_ID_SHA256
    ):

        raise ValueError(
            "FINAL HOLDOUT FINGERPRINT "
            "DOES NOT MATCH LOCKED SET."
        )


    print(
        "\nHoldout fingerprint verified."
    )


    print(
        "\nLabel counts:"
    )

    print(
        dataframe[
            "label"
        ]
        .value_counts()
        .sort_index()
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"\nDevice: {device}"
    )

    if (
        device.type
        == "cuda"
    ):

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    # ========================================================
    # Load locked RoBERTa
    # ========================================================

    print(
        "\nLoading locked RoBERTa..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            ROBERTA_CHECKPOINT
        )
    )

    roberta = (
        AutoModelForSequenceClassification
        .from_pretrained(
            ROBERTA_CHECKPOINT
        )
    )

    roberta.to(
        device
    )


    # ========================================================
    # Load locked ResNet
    # ========================================================

    print(
        "Loading locked ResNet..."
    )

    checkpoint = torch.load(
        RESNET_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    resnet = resnet50(
        weights=None
    )

    features = (
        resnet.fc.in_features
    )

    resnet.fc = nn.Linear(
        features,
        2
    )

    resnet.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    resnet.to(
        device
    )


    image_transform = transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ]
        ),
    ])


    # ========================================================
    # FINAL inference
    # ========================================================

    print(
        "\nRunning RoBERTa inference..."
    )

    text_results = predict_roberta(
        dataframe,
        roberta,
        tokenizer,
        device
    )


    print(
        "Running ResNet inference..."
    )

    image_results = predict_resnet(
        dataframe,
        resnet,
        image_transform,
        device
    )


    results = text_results.merge(
        image_results[
            [
                "id",
                "probability_resnet",
                "prediction_resnet",
            ]
        ],
        on="id",
        validate="one_to_one"
    )


    # ========================================================
    # LOCKED late fusion
    # ========================================================

    results[
        "probability_fusion"
    ] = (
        ALPHA_TEXT
        *
        results[
            "probability_roberta"
        ]
        +
        ALPHA_IMAGE
        *
        results[
            "probability_resnet"
        ]
    )


    results[
        "prediction_fusion"
    ] = (
        results[
            "probability_fusion"
        ]
        >= THRESHOLD
    ).astype(int)


    labels = (
        results[
            "label"
        ].to_numpy()
    )


    # ========================================================
    # Metrics
    # ========================================================

    systems = {
        "roberta": (
            results[
                "prediction_roberta"
            ].to_numpy(),

            results[
                "probability_roberta"
            ].to_numpy(),
        ),

        "resnet": (
            results[
                "prediction_resnet"
            ].to_numpy(),

            results[
                "probability_resnet"
            ].to_numpy(),
        ),

        "late_fusion": (
            results[
                "prediction_fusion"
            ].to_numpy(),

            results[
                "probability_fusion"
            ].to_numpy(),
        ),
    }


    metrics = {}


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL PERFORMANCE"
    )

    print("=" * 70)


    for name, (
        predictions,
        probabilities
    ) in systems.items():

        model_metrics = (
            calculate_metrics(
                labels,
                predictions,
                probabilities
            )
        )

        model_metrics[
            "ece"
        ] = (
            expected_calibration_error(
                labels,
                probabilities
            )
        )

        metrics[
            name
        ] = model_metrics


        print(
            "\n" + name.upper()
        )

        print(
            "-" * len(name)
        )


        for key in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1_macro",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
            "ece",
        ]:

            print(
                f"{key:12s}: "
                f"{model_metrics[key]:.4f}"
            )


        print(
            "Confusion matrix:"
        )

        print(
            np.array(
                model_metrics[
                    "confusion_matrix"
                ]
            )
        )


    # ========================================================
    # Final paired significance:
    # late fusion vs RoBERTa
    # ========================================================

    roberta_pred = (
        systems[
            "roberta"
        ][0]
    )

    roberta_prob = (
        systems[
            "roberta"
        ][1]
    )

    fusion_pred = (
        systems[
            "late_fusion"
        ][0]
    )

    fusion_prob = (
        systems[
            "late_fusion"
        ][1]
    )


    roberta_correct = (
        roberta_pred
        == labels
    )

    fusion_correct = (
        fusion_pred
        == labels
    )


    roberta_only = int(
        np.sum(
            roberta_correct
            &
            ~fusion_correct
        )
    )

    fusion_only = int(
        np.sum(
            ~roberta_correct
            &
            fusion_correct
        )
    )


    discordant = (
        roberta_only
        +
        fusion_only
    )


    if discordant > 0:

        mcnemar_p = float(
            binomtest(
                min(
                    roberta_only,
                    fusion_only
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue
        )

    else:

        mcnemar_p = 1.0


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL FUSION VS ROBERTA "
        "PAIRED COMPARISON"
    )

    print("=" * 70)


    print(
        "\nRoBERTa correct / "
        "Fusion wrong:",
        roberta_only
    )

    print(
        "RoBERTa wrong / "
        "Fusion correct:",
        fusion_only
    )

    print(
        "Discordant:",
        discordant
    )

    print(
        f"McNemar exact p-value: "
        f"{mcnemar_p:.6f}"
    )


    observed_differences = {}


    print(
        "\nObserved differences "
        "(Fusion - RoBERTa):"
    )


    for metric in [
        "accuracy",
        "f1",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ]:

        difference = (
            metrics[
                "late_fusion"
            ][metric]
            -
            metrics[
                "roberta"
            ][metric]
        )

        observed_differences[
            metric
        ] = float(
            difference
        )

        print(
            f"{metric:12s}: "
            f"{difference:+.4f}"
        )


    print(
        f"\nRunning "
        f"{BOOTSTRAP_SAMPLES:,} "
        "paired bootstrap samples..."
    )


    bootstrap = paired_bootstrap(
        labels,
        roberta_pred,
        roberta_prob,
        fusion_pred,
        fusion_prob
    )


    print(
        "\n95% paired bootstrap "
        "confidence intervals:"
    )


    for metric in [
        "accuracy",
        "f1",
        "f1_macro",
        "roc_auc",
        "pr_auc",
    ]:

        lower = (
            bootstrap[
                metric
            ][
                "ci_95_lower"
            ]
        )

        upper = (
            bootstrap[
                metric
            ][
                "ci_95_upper"
            ]
        )

        observed = (
            observed_differences[
                metric
            ]
        )


        print(
            f"{metric:12s}: "
            f"{observed:+.4f} "
            f"[{lower:+.4f}, "
            f"{upper:+.4f}]"
        )


    # ========================================================
    # Save final outputs
    # ========================================================

    results.to_csv(
        OUTPUT_ROOT
        / "final_predictions.csv",
        index=False
    )


    final_output = {
        "holdout": {
            "samples":
                int(
                    len(results)
                ),

            "ordered_id_sha256":
                fingerprint,

            "label_counts": {
                str(key):
                    int(value)

                for key, value
                in dataframe[
                    "label"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
        },

        "locked_specification": {
            "alpha_text":
                ALPHA_TEXT,

            "alpha_image":
                ALPHA_IMAGE,

            "threshold":
                THRESHOLD,

            "max_length":
                MAX_LENGTH,

            "image_size":
                224,
        },

        "metrics":
            metrics,

        "fusion_vs_roberta": {
            "roberta_correct_fusion_wrong":
                roberta_only,

            "roberta_wrong_fusion_correct":
                fusion_only,

            "discordant":
                discordant,

            "mcnemar_exact_p":
                mcnemar_p,

            "observed_differences":
                observed_differences,

            "bootstrap_samples":
                BOOTSTRAP_SAMPLES,

            "bootstrap_ci_95":
                bootstrap,
        },
    }


    with (
        OUTPUT_ROOT
        / "final_results.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_output,
            f,
            indent=2
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL EVALUATION COMPLETE"
    )

    print("=" * 70)

    print(
        "\nResults saved to:",
        OUTPUT_ROOT
    )

    print(
        "\nThese are the locked final "
        "holdout results."
    )


if __name__ == "__main__":
    main()
