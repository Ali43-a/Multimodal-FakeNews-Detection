import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


PROB_ROOT = Path(
    "results/metrics/baseline_probabilities"
)

FUSION_ROOT = Path(
    "results/metrics/late_probability_fusion"
)

OUTPUT_ROOT = Path(
    "results/metrics/calibration_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


N_BINS = 10


def expected_calibration_error(
    labels,
    probabilities,
    n_bins=10
):

    probabilities = np.asarray(
        probabilities
    )

    labels = np.asarray(
        labels
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    correctness = (
        predictions == labels
    ).astype(float)

    bin_edges = np.linspace(
        0.5,
        1.0,
        n_bins + 1
    )

    ece = 0.0
    rows = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append({
                "bin_lower": lower,
                "bin_upper": upper,
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
                "gap": None,
            })

            continue

        mean_confidence = float(
            confidence[mask].mean()
        )

        accuracy = float(
            correctness[mask].mean()
        )

        gap = abs(
            accuracy - mean_confidence
        )

        ece += (
            count / len(labels)
        ) * gap

        rows.append({
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "count": count,
            "mean_confidence":
                mean_confidence,
            "accuracy":
                accuracy,
            "gap":
                float(gap),
        })

    return float(ece), rows


def model_calibration(
    name,
    labels,
    probabilities
):

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-7,
        1 - 1e-7
    )

    ece, bins = (
        expected_calibration_error(
            labels,
            probabilities,
            N_BINS
        )
    )

    result = {
        "model": name,

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
                    probabilities
                )
            ),

        "ece":
            ece,

        "mean_probability_class_1":
            float(
                probabilities.mean()
            ),

        "mean_confidence":
            float(
                np.maximum(
                    probabilities,
                    1 - probabilities
                ).mean()
            ),
    }

    return result, bins


roberta = pd.read_csv(
    PROB_ROOT / "roberta_test.csv"
)

resnet = pd.read_csv(
    PROB_ROOT / "resnet_test.csv"
)

fusion = pd.read_csv(
    FUSION_ROOT / "test_predictions.csv"
)


df = (
    roberta[
        [
            "id",
            "label",
            "probability_class_1"
        ]
    ]
    .rename(
        columns={
            "probability_class_1":
                "probability_roberta"
        }
    )
    .merge(
        resnet[
            [
                "id",
                "probability_class_1"
            ]
        ].rename(
            columns={
                "probability_class_1":
                    "probability_resnet"
            }
        ),
        on="id",
        validate="one_to_one"
    )
    .merge(
        fusion[
            [
                "id",
                "probability_fused"
            ]
        ],
        on="id",
        validate="one_to_one"
    )
)


labels = (
    df["label"]
    .to_numpy()
)


models = {
    "roberta":
        df[
            "probability_roberta"
        ].to_numpy(),

    "resnet":
        df[
            "probability_resnet"
        ].to_numpy(),

    "late_fusion":
        df[
            "probability_fused"
        ].to_numpy(),
}


summary = []
all_bins = {}


print("=" * 70)
print("MODEL CALIBRATION ANALYSIS")
print("=" * 70)

print(
    f"\nTest samples: {len(df):,}"
)


for name, probabilities in (
    models.items()
):

    result, bins = (
        model_calibration(
            name,
            labels,
            probabilities
        )
    )

    summary.append(
        result
    )

    all_bins[name] = bins

    print(
        "\n" + name.upper()
    )

    print(
        "-" * len(name)
    )

    print(
        f"Brier score : "
        f"{result['brier_score']:.4f}"
    )

    print(
        f"Log loss    : "
        f"{result['log_loss']:.4f}"
    )

    print(
        f"ECE         : "
        f"{result['ece']:.4f}"
    )

    print(
        f"Mean conf.  : "
        f"{result['mean_confidence']:.4f}"
    )


summary_df = pd.DataFrame(
    summary
)

summary_df.to_csv(
    OUTPUT_ROOT
    / "calibration_summary.csv",
    index=False
)


for model, bins in (
    all_bins.items()
):

    pd.DataFrame(
        bins
    ).to_csv(
        OUTPUT_ROOT
        / f"{model}_calibration_bins.csv",
        index=False
    )


with (
    OUTPUT_ROOT
    / "results.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "n_bins":
                N_BINS,

            "test_samples":
                int(len(df)),

            "models":
                summary,
        },
        f,
        indent=2
    )


print(
    "\n" + "=" * 70
)

print(
    "SUMMARY"
)

print("=" * 70)

print(
    summary_df[
        [
            "model",
            "brier_score",
            "log_loss",
            "ece",
            "mean_confidence",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)

print(
    "\nLower Brier, log loss, "
    "and ECE indicate better calibration."
)

print(
    "\nResults saved to:",
    OUTPUT_ROOT
)
