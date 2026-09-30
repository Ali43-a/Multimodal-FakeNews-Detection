import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


SEED = 42
N_BOOTSTRAPS = 5000

BASELINE_ROOT = Path(
    "results/metrics/baseline_probabilities"
)

FUSION_ROOT = Path(
    "results/metrics/late_probability_fusion"
)

OUTPUT_ROOT = Path(
    "results/metrics/statistical_comparison"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load paired predictions
# ============================================================

roberta = pd.read_csv(
    BASELINE_ROOT / "roberta_test.csv"
)

fusion = pd.read_csv(
    FUSION_ROOT / "test_predictions.csv"
)


df = roberta.merge(
    fusion,
    on="id",
    suffixes=("_roberta", "_fusion"),
    validate="one_to_one"
)


if not np.array_equal(
    df["label_roberta"],
    df["label_fusion"]
):
    raise ValueError(
        "Label mismatch between prediction files."
    )


y = df["label_roberta"].to_numpy()

pred_roberta = (
    df["prediction_roberta"]
    .to_numpy()
)

pred_fusion = (
    df["prediction_fusion"]
    .to_numpy()
)

prob_roberta = (
    df["probability_class_1"]
    .to_numpy()
)

prob_fusion = (
    df["probability_fused"]
    .to_numpy()
)


print("=" * 70)
print("LATE FUSION VS ROBERTA")
print("=" * 70)

print(f"\nPaired test samples: {len(df):,}")


# ============================================================
# McNemar exact test
# ============================================================

correct_roberta = (
    pred_roberta == y
)

correct_fusion = (
    pred_fusion == y
)


# RoBERTa correct, fusion wrong
b = int(
    np.sum(
        correct_roberta
        & ~correct_fusion
    )
)

# RoBERTa wrong, fusion correct
c = int(
    np.sum(
        ~correct_roberta
        & correct_fusion
    )
)


discordant = b + c


if discordant > 0:

    mcnemar = binomtest(
        min(b, c),
        n=discordant,
        p=0.5,
        alternative="two-sided"
    )

    mcnemar_p = float(
        mcnemar.pvalue
    )

else:
    mcnemar_p = 1.0


print("\nMcNemar paired comparison:")
print(
    "RoBERTa correct / Fusion wrong:",
    b
)

print(
    "RoBERTa wrong / Fusion correct:",
    c
)

print(
    "Discordant predictions:",
    discordant
)

print(
    f"Exact p-value: {mcnemar_p:.6f}"
)


# ============================================================
# Metric helper
# ============================================================

def calculate_metrics(
    labels,
    predictions,
    probabilities
):

    return {
        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "f1":
            f1_score(
                labels,
                predictions
            ),

        "f1_macro":
            f1_score(
                labels,
                predictions,
                average="macro"
            ),

        "roc_auc":
            roc_auc_score(
                labels,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                labels,
                probabilities
            ),
    }


roberta_metrics = calculate_metrics(
    y,
    pred_roberta,
    prob_roberta
)

fusion_metrics = calculate_metrics(
    y,
    pred_fusion,
    prob_fusion
)


print("\nObserved metric differences:")
print("(Fusion - RoBERTa)")

for metric in roberta_metrics:

    delta = (
        fusion_metrics[metric]
        - roberta_metrics[metric]
    )

    print(
        f"{metric:12s}: "
        f"{delta:+.4f}"
    )


# ============================================================
# Paired bootstrap
# ============================================================

rng = np.random.default_rng(
    SEED
)

n = len(df)

bootstrap_differences = {
    metric: []
    for metric in roberta_metrics
}


print(
    f"\nRunning {N_BOOTSTRAPS:,} "
    "paired bootstrap samples..."
)


for i in range(N_BOOTSTRAPS):

    indices = rng.integers(
        0,
        n,
        size=n
    )

    y_sample = y[indices]

    # ROC/PR require both labels to appear.
    if len(np.unique(y_sample)) < 2:
        continue

    roberta_sample = (
        calculate_metrics(
            y_sample,
            pred_roberta[indices],
            prob_roberta[indices]
        )
    )

    fusion_sample = (
        calculate_metrics(
            y_sample,
            pred_fusion[indices],
            prob_fusion[indices]
        )
    )

    for metric in (
        bootstrap_differences
    ):

        bootstrap_differences[
            metric
        ].append(
            fusion_sample[metric]
            - roberta_sample[metric]
        )


# ============================================================
# Confidence intervals
# ============================================================

confidence_intervals = {}


print("\n95% bootstrap confidence intervals:")
print("(Fusion - RoBERTa)")


for metric, differences in (
    bootstrap_differences.items()
):

    values = np.asarray(
        differences
    )

    lower = float(
        np.percentile(
            values,
            2.5
        )
    )

    upper = float(
        np.percentile(
            values,
            97.5
        )
    )

    observed = float(
        fusion_metrics[metric]
        - roberta_metrics[metric]
    )

    confidence_intervals[
        metric
    ] = {
        "observed_difference":
            observed,

        "ci_95_lower":
            lower,

        "ci_95_upper":
            upper,
    }

    print(
        f"{metric:12s}: "
        f"{observed:+.4f} "
        f"[{lower:+.4f}, {upper:+.4f}]"
    )


# ============================================================
# Save results
# ============================================================

results = {
    "comparison":
        "late_probability_fusion_vs_roberta",

    "test_samples":
        int(n),

    "mcnemar": {
        "roberta_correct_fusion_wrong":
            b,

        "roberta_wrong_fusion_correct":
            c,

        "discordant":
            discordant,

        "exact_p_value":
            mcnemar_p,
    },

    "roberta_metrics": {
        key: float(value)
        for key, value
        in roberta_metrics.items()
    },

    "fusion_metrics": {
        key: float(value)
        for key, value
        in fusion_metrics.items()
    },

    "bootstrap_samples":
        N_BOOTSTRAPS,

    "confidence_intervals":
        confidence_intervals,
}


with (
    OUTPUT_ROOT / "results.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


print(
    "\nResults saved to:",
    OUTPUT_ROOT
)
