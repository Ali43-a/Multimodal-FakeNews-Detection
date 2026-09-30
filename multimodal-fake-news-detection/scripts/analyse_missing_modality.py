import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


INPUT_ROOT = Path(
    "results/metrics/baseline_probabilities"
)

OUTPUT_ROOT = Path(
    "results/metrics/missing_modality_robustness"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


ALPHA_TEXT = 0.45
ALPHA_IMAGE = 0.55

THRESHOLD = 0.5


def metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

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
    }


# ============================================================
# VALIDATION ONLY
# ============================================================

text = pd.read_csv(
    INPUT_ROOT
    / "roberta_validation.csv"
)

image = pd.read_csv(
    INPUT_ROOT
    / "resnet_validation.csv"
)


df = text.merge(
    image,
    on="id",
    suffixes=(
        "_text",
        "_image"
    ),
    validate="one_to_one"
)


if not np.array_equal(
    df["label_text"],
    df["label_image"]
):

    raise ValueError(
        "Validation labels do not match."
    )


labels = (
    df["label_text"]
    .to_numpy()
)

p_text = (
    df[
        "probability_class_1_text"
    ]
    .to_numpy()
)

p_image = (
    df[
        "probability_class_1_image"
    ]
    .to_numpy()
)


# ============================================================
# Scenarios
# ============================================================

scenarios = {}


# Full multimodal model
scenarios["normal_late_fusion"] = (
    ALPHA_TEXT
    * p_text
    +
    ALPHA_IMAGE
    * p_image
)


# Explicit fallback strategies
scenarios["missing_image_text_fallback"] = (
    p_text
)

scenarios["missing_text_image_fallback"] = (
    p_image
)


# Naive neutral-imputation strategies
scenarios["missing_image_neutral"] = (
    ALPHA_TEXT
    * p_text
    +
    ALPHA_IMAGE
    * 0.5
)

scenarios["missing_text_neutral"] = (
    ALPHA_TEXT
    * 0.5
    +
    ALPHA_IMAGE
    * p_image
)


# ============================================================
# Evaluate
# ============================================================

results = []


print("=" * 70)
print("MISSING-MODALITY ROBUSTNESS")
print("=" * 70)

print(
    f"\nValidation samples: "
    f"{len(df):,}"
)

print(
    "\nTEST SET IS NOT USED."
)


for name, probability in (
    scenarios.items()
):

    result = metrics(
        labels,
        probability
    )

    result["scenario"] = name

    results.append(
        result
    )


    print(
        "\n" + name.upper()
    )

    print(
        "-" * len(name)
    )

    for metric, value in (
        result.items()
    ):

        if metric == "scenario":
            continue

        print(
            f"{metric:12s}: "
            f"{value:.4f}"
        )


results_df = pd.DataFrame(
    results
)


baseline_macro = float(
    results_df.loc[
        results_df["scenario"]
        == "normal_late_fusion",
        "f1_macro"
    ].iloc[0]
)


results_df[
    "macro_f1_change_vs_full"
] = (
    results_df["f1_macro"]
    - baseline_macro
)


results_df.to_csv(
    OUTPUT_ROOT
    / "robustness_results.csv",
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
            "dataset_split":
                "validation",

            "test_used":
                False,

            "alpha_text":
                ALPHA_TEXT,

            "alpha_image":
                ALPHA_IMAGE,

            "results":
                results_df
                .to_dict(
                    orient="records"
                ),
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
    results_df[
        [
            "scenario",
            "accuracy",
            "f1_macro",
            "roc_auc",
            "pr_auc",
            "macro_f1_change_vs_full",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print(
    "\nResults saved to:",
    OUTPUT_ROOT
)
