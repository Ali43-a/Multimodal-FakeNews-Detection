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
    confusion_matrix,
)


INPUT_ROOT = Path(
    "results/metrics/baseline_probabilities"
)

OUTPUT_ROOT = Path(
    "results/metrics/late_probability_fusion"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


def compute_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist(),
    }


def load_pair(split):

    roberta = pd.read_csv(
        INPUT_ROOT
        / f"roberta_{split}.csv"
    )

    resnet = pd.read_csv(
        INPUT_ROOT
        / f"resnet_{split}.csv"
    )

    merged = roberta.merge(
        resnet,
        on="id",
        suffixes=(
            "_text",
            "_image"
        ),
        validate="one_to_one"
    )

    if not np.array_equal(
        merged["label_text"],
        merged["label_image"]
    ):

        raise ValueError(
            f"{split}: label mismatch."
        )

    return merged


validation = load_pair(
    "validation"
)

test = load_pair(
    "test"
)


print("=" * 70)
print("LATE PROBABILITY FUSION")
print("=" * 70)

print(
    f"\nValidation samples: "
    f"{len(validation):,}"
)

print(
    f"Test samples:       "
    f"{len(test):,}"
)


validation_labels = (
    validation["label_text"]
    .to_numpy()
)

validation_text = (
    validation[
        "probability_class_1_text"
    ]
    .to_numpy()
)

validation_image = (
    validation[
        "probability_class_1_image"
    ]
    .to_numpy()
)


# ------------------------------------------------------------
# Tune alpha ONLY on validation data.
#
# alpha = 1.0 -> text only
# alpha = 0.0 -> image only
# ------------------------------------------------------------

grid_results = []

best_alpha = None
best_macro_f1 = -1.0


for alpha in np.arange(
    0.0,
    1.0001,
    0.05
):

    fused_probability = (
        alpha
        * validation_text
        +
        (1.0 - alpha)
        * validation_image
    )

    metrics = compute_metrics(
        validation_labels,
        fused_probability
    )

    grid_results.append({
        "alpha_text":
            float(round(alpha, 2)),

        "alpha_image":
            float(
                round(
                    1.0 - alpha,
                    2
                )
            ),

        **{
            key: value
            for key, value
            in metrics.items()
            if key !=
            "confusion_matrix"
        }
    })


    current_macro_f1 = (
        metrics["f1_macro"]
    )


    if (
        current_macro_f1
        > best_macro_f1
    ):

        best_macro_f1 = (
            current_macro_f1
        )

        best_alpha = float(
            round(alpha, 2)
        )


grid_df = pd.DataFrame(
    grid_results
)

grid_df.to_csv(
    OUTPUT_ROOT
    / "validation_alpha_search.csv",
    index=False
)


print(
    f"\nBest validation alpha "
    f"(text): {best_alpha:.2f}"
)

print(
    f"Image weight: "
    f"{1.0 - best_alpha:.2f}"
)

print(
    f"Best validation macro F1: "
    f"{best_macro_f1:.4f}"
)


best_validation_probability = (
    best_alpha
    * validation_text
    +
    (1.0 - best_alpha)
    * validation_image
)


validation_metrics = (
    compute_metrics(
        validation_labels,
        best_validation_probability
    )
)


print(
    "\nValidation metrics "
    "at selected alpha:"
)

for key, value in (
    validation_metrics.items()
):

    if key == "confusion_matrix":
        continue

    print(
        f"{key:12s}: "
        f"{value:.4f}"
    )


print(
    "Confusion matrix:"
)

print(
    np.array(
        validation_metrics[
            "confusion_matrix"
        ]
    )
)


# ------------------------------------------------------------
# Freeze alpha.
# Evaluate TEST exactly once.
# ------------------------------------------------------------

test_labels = (
    test["label_text"]
    .to_numpy()
)

test_text = (
    test[
        "probability_class_1_text"
    ]
    .to_numpy()
)

test_image = (
    test[
        "probability_class_1_image"
    ]
    .to_numpy()
)


test_probability = (
    best_alpha
    * test_text
    +
    (1.0 - best_alpha)
    * test_image
)


test_metrics = compute_metrics(
    test_labels,
    test_probability
)


print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)


for key, value in (
    test_metrics.items()
):

    if key == "confusion_matrix":
        continue

    print(
        f"{key:12s}: "
        f"{value:.4f}"
    )


print(
    "\nConfusion matrix:"
)

print(
    np.array(
        test_metrics[
            "confusion_matrix"
        ]
    )
)


test_predictions = (
    test_probability >= 0.5
).astype(int)


pd.DataFrame({
    "id":
        test["id"],

    "label":
        test_labels,

    "probability_text":
        test_text,

    "probability_image":
        test_image,

    "probability_fused":
        test_probability,

    "prediction":
        test_predictions,
}).to_csv(
    OUTPUT_ROOT
    / "test_predictions.csv",
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
            "selected_alpha_text":
                best_alpha,

            "selected_alpha_image":
                1.0 - best_alpha,

            "selection_metric":
                "validation_macro_f1",

            "validation":
                validation_metrics,

            "test":
                test_metrics,
        },

        f,
        indent=2
    )


print(
    "\nResults saved to:",
    OUTPUT_ROOT
)
