import json
from pathlib import Path

import numpy as np
import pandas as pd


PROB_ROOT = Path(
    "results/metrics/baseline_probabilities"
)

FUSION_ROOT = Path(
    "results/metrics/late_probability_fusion"
)

DATA_PATH = Path(
    "data/processed/fakeddit/development_clean/test.parquet"
)

OUTPUT_ROOT = Path(
    "results/metrics/error_analysis"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


roberta = pd.read_csv(
    PROB_ROOT / "roberta_test.csv"
)

resnet = pd.read_csv(
    PROB_ROOT / "resnet_test.csv"
)

fusion = pd.read_csv(
    FUSION_ROOT / "test_predictions.csv"
)

data = pd.read_parquet(
    DATA_PATH
)


# ------------------------------------------------------------
# Standardise IDs
# ------------------------------------------------------------

for frame in [
    roberta,
    resnet,
    fusion,
    data
]:
    frame["id"] = (
        frame["id"]
        .astype(str)
    )


# ------------------------------------------------------------
# Rename model outputs
# ------------------------------------------------------------

roberta = roberta.rename(
    columns={
        "prediction":
            "prediction_roberta",

        "probability_class_1":
            "probability_roberta",
    }
)

resnet = resnet.rename(
    columns={
        "prediction":
            "prediction_resnet",

        "probability_class_1":
            "probability_resnet",
    }
)

fusion = fusion.rename(
    columns={
        "prediction":
            "prediction_fusion",
    }
)


# ------------------------------------------------------------
# Merge paired results
# ------------------------------------------------------------

df = (
    data
    .merge(
        roberta[
            [
                "id",
                "prediction_roberta",
                "probability_roberta",
            ]
        ],
        on="id",
        validate="one_to_one"
    )
    .merge(
        resnet[
            [
                "id",
                "prediction_resnet",
                "probability_resnet",
            ]
        ],
        on="id",
        validate="one_to_one"
    )
    .merge(
        fusion[
            [
                "id",
                "prediction_fusion",
                "probability_fused",
            ]
        ],
        on="id",
        validate="one_to_one"
    )
)


df["roberta_correct"] = (
    df["prediction_roberta"]
    == df["label"]
)

df["resnet_correct"] = (
    df["prediction_resnet"]
    == df["label"]
)

df["fusion_correct"] = (
    df["prediction_fusion"]
    == df["label"]
)


# ------------------------------------------------------------
# Pairwise categories
# ------------------------------------------------------------

conditions = [
    (
        df["roberta_correct"]
        & df["fusion_correct"]
    ),

    (
        ~df["roberta_correct"]
        & df["fusion_correct"]
    ),

    (
        df["roberta_correct"]
        & ~df["fusion_correct"]
    ),

    (
        ~df["roberta_correct"]
        & ~df["fusion_correct"]
    ),
]

categories = [
    "both_correct",
    "fusion_only_correct",
    "roberta_only_correct",
    "both_wrong",
]

df["comparison_group"] = np.select(
    conditions,
    categories,
    default="unknown"
)


# ------------------------------------------------------------
# Basic counts
# ------------------------------------------------------------

counts = (
    df["comparison_group"]
    .value_counts()
    .to_dict()
)

print("=" * 70)
print("ERROR / COMPLEMENTARITY ANALYSIS")
print("=" * 70)

print("\nRoBERTa vs Fusion groups:")

for group in categories:
    print(
        f"{group:22s}: "
        f"{counts.get(group, 0):,}"
    )


# ------------------------------------------------------------
# Recovered cases
# ------------------------------------------------------------

recovered = df[
    df["comparison_group"]
    == "fusion_only_correct"
].copy()

lost = df[
    df["comparison_group"]
    == "roberta_only_correct"
].copy()


print("\nFusion recovered cases:")
print(
    f"Total: {len(recovered):,}"
)

print("\nRecovered cases by true label:")
print(
    recovered["label"]
    .value_counts()
    .sort_index()
)


resnet_correct_recovered = int(
    recovered[
        "resnet_correct"
    ].sum()
)


print(
    "\nAmong recovered cases, "
    "ResNet was independently correct:"
)

print(
    f"{resnet_correct_recovered:,} "
    f"/ {len(recovered):,} "
    f"("
    f"{resnet_correct_recovered / len(recovered):.1%}"
    f")"
)


print("\nFusion-lost cases:")
print(
    f"Total: {len(lost):,}"
)

print("\nLost cases by true label:")
print(
    lost["label"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Model agreement
# ------------------------------------------------------------

agreement = (
    df["prediction_roberta"]
    == df["prediction_resnet"]
)

print(
    "\nRoBERTa / ResNet prediction agreement:"
)

print(
    f"{agreement.mean():.1%}"
)


disagreement = df[
    ~agreement
].copy()

print(
    "Samples where modalities disagree:",
    len(disagreement)
)

print(
    "\nWhen RoBERTa and ResNet disagree:"
)

print(
    "RoBERTa correct:",
    int(
        disagreement[
            "roberta_correct"
        ].sum()
    )
)

print(
    "ResNet correct:",
    int(
        disagreement[
            "resnet_correct"
        ].sum()
    )
)

print(
    "Fusion correct:",
    int(
        disagreement[
            "fusion_correct"
        ].sum()
    )
)


# ------------------------------------------------------------
# Text length
# ------------------------------------------------------------

if "text_length_words" in df.columns:
    text_length = "text_length_words"
else:
    df["derived_text_words"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    text_length = (
        "derived_text_words"
    )


print("\nMean text length by group:")

print(
    df.groupby(
        "comparison_group"
    )[text_length]
    .mean()
    .reindex(categories)
)


# ------------------------------------------------------------
# Probability shifts
# ------------------------------------------------------------

df[
    "fusion_minus_roberta_probability"
] = (
    df["probability_fused"]
    - df["probability_roberta"]
)


print(
    "\nMean probability shift "
    "(Fusion - RoBERTa) by group:"
)

print(
    df.groupby(
        "comparison_group"
    )[
        "fusion_minus_roberta_probability"
    ]
    .mean()
    .reindex(categories)
)


# ------------------------------------------------------------
# Save detailed data
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_ROOT
    / "paired_test_error_analysis.csv",
    index=False
)

recovered.to_csv(
    OUTPUT_ROOT
    / "fusion_recovered_cases.csv",
    index=False
)

lost.to_csv(
    OUTPUT_ROOT
    / "fusion_lost_cases.csv",
    index=False
)


summary = {
    "test_samples":
        int(len(df)),

    "groups": {
        key: int(value)
        for key, value
        in counts.items()
    },

    "fusion_recovered":
        int(len(recovered)),

    "fusion_lost":
        int(len(lost)),

    "resnet_correct_among_recovered":
        resnet_correct_recovered,

    "resnet_correct_among_recovered_rate":
        float(
            resnet_correct_recovered
            / len(recovered)
        ),

    "roberta_resnet_agreement":
        float(
            agreement.mean()
        ),

    "modality_disagreement_samples":
        int(len(disagreement)),

    "when_modalities_disagree": {
        "roberta_correct":
            int(
                disagreement[
                    "roberta_correct"
                ].sum()
            ),

        "resnet_correct":
            int(
                disagreement[
                    "resnet_correct"
                ].sum()
            ),

        "fusion_correct":
            int(
                disagreement[
                    "fusion_correct"
                ].sum()
            ),
    },
}


with (
    OUTPUT_ROOT / "summary.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print(
    "\nResults saved to:",
    OUTPUT_ROOT
)
