import json
from pathlib import Path

import numpy as np
import pandas as pd


GRADCAM_SUMMARY = Path(
    "results/gradcam/gradcam_case_summary.csv"
)

SHAP_TOKENS = Path(
    "results/shap/all_token_attributions.csv"
)

OUTPUT_ROOT = Path(
    "results/case_studies"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

N_PER_GROUP = 2

GROUPS = [
    "fusion_only_correct",
    "roberta_only_correct",
    "both_correct",
    "both_wrong",
]


# ============================================================
# Load Grad-CAM cases and SHAP token attributions
# ============================================================

gradcam = pd.read_csv(
    GRADCAM_SUMMARY
)

tokens = pd.read_csv(
    SHAP_TOKENS
)

gradcam["id"] = (
    gradcam["id"]
    .astype(str)
)

tokens["id"] = (
    tokens["id"]
    .astype(str)
)


# ============================================================
# Build corrected SHAP summaries
#
# Positive SHAP = supports explained RoBERTa class
# Negative SHAP = opposes explained RoBERTa class
# ============================================================

token_summaries = []


for sample_id, sample in tokens.groupby("id"):

        sample = sample[
        sample["token"].notna()
    ].copy()

        sample["token"] = (
        sample["token"]
        .astype(str)
    )

sample = sample[
        sample["token"]
        .str.strip()
        .ne("")
    ]


positive = (
        sample[
            sample["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(5)
    )

negative = (
        sample[
            sample["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(5)
    )


supportive = [
        {
            "token": str(row["token"]),
            "shap": float(
                row["shap_value"]
            ),
        }

        for _, row
        in positive.iterrows()
    ]


opposing = [
        {
            "token": str(row["token"]),
            "shap": float(
                row["shap_value"]
            ),
        }

        for _, row
        in negative.iterrows()
    ]


token_summaries.append({
        "id":
            sample_id,

        "supportive_tokens":
            json.dumps(
                supportive
            ),

        "opposing_tokens":
            json.dumps(
                opposing
            ),
    })


token_summary = pd.DataFrame(
    token_summaries
)


# ============================================================
# Merge SHAP + Grad-CAM case information
# ============================================================

df = gradcam.merge(
    token_summary,
    on="id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# Calculate useful case-study characteristics
# ============================================================

df["fusion_probability_shift"] = (
    df["probability_fused"]
    - df["probability_roberta"]
)

df["absolute_fusion_shift"] = (
    df["fusion_probability_shift"]
    .abs()
)


df["roberta_confidence"] = np.maximum(
    df["probability_roberta"],
    1.0 - df["probability_roberta"]
)

df["resnet_confidence"] = np.maximum(
    df["probability_resnet"],
    1.0 - df["probability_resnet"]
)

df["mean_model_confidence"] = (
    df["roberta_confidence"]
    + df["resnet_confidence"]
) / 2.0


# ============================================================
# Select two representative cases from each category
# ============================================================

selected_parts = []


for group in GROUPS:

    subset = df[
        df["comparison_group"]
        == group
    ].copy()


    if group in [
        "fusion_only_correct",
        "roberta_only_correct",
    ]:

        # For modality-conflict cases, select examples where
        # fusion moved the probability most strongly.
        subset = subset.sort_values(
            "absolute_fusion_shift",
            ascending=False
        )

    else:

        # For agreement / failure cases, select examples
        # where the unimodal models were most confident.
        subset = subset.sort_values(
            "mean_model_confidence",
            ascending=False
        )


    selected_parts.append(
        subset.head(
            N_PER_GROUP
        )
    )


selected = pd.concat(
    selected_parts,
    ignore_index=True
)


# ============================================================
# Save outputs
# ============================================================

df.to_csv(
    OUTPUT_ROOT
    / "all_32_multimodal_cases.csv",
    index=False
)

selected.to_csv(
    OUTPUT_ROOT
    / "selected_8_case_studies.csv",
    index=False
)


# ============================================================
# Print selected cases
# ============================================================

print("=" * 70)
print("MULTIMODAL CASE-STUDY CONSOLIDATION")
print("=" * 70)

print(
    f"\nAvailable explained cases: "
    f"{len(df)}"
)

print(
    f"Selected case studies: "
    f"{len(selected)}"
)


for group in GROUPS:

    group_df = selected[
        selected["comparison_group"]
        == group
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        group.upper()
    )

    print("=" * 70)


    for _, row in group_df.iterrows():

        print(
            f"\nID: {row['id']}"
        )

        print(
            f"Text: {row['text']}"
        )

        print(
            f"True label: "
            f"{int(row['label'])}"
        )

        print(
            f"RoBERTa: "
            f"{int(row['prediction_roberta'])} "
            f"(P1={row['probability_roberta']:.3f})"
        )

        print(
            f"ResNet: "
            f"{int(row['prediction_resnet'])} "
            f"(P1={row['probability_resnet']:.3f})"
        )

        print(
            f"Fusion: "
            f"{int(row['prediction_fusion'])} "
            f"(P1={row['probability_fused']:.3f})"
        )

        print(
            "SHAP supporting:",
            row["supportive_tokens"]
        )

        print(
            "SHAP opposing:",
            row["opposing_tokens"]
        )

        print(
            "Grad-CAM:",
            row["gradcam_path"]
        )


print(
    "\n" + "=" * 70
)

print(
    "CASE-STUDY CONSOLIDATION COMPLETE"
)

print("=" * 70)

print(
    "\nResults saved to:",
    OUTPUT_ROOT
)
