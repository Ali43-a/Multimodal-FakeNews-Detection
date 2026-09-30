from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import shap

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


CHECKPOINT_PATH = Path(
    "checkpoints/roberta_baseline"
)

CASE_PATH = Path(
    "results/gradcam/gradcam_case_summary.csv"
)

OUTPUT_ROOT = Path(
    "results/shap"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


MAX_LENGTH = 64
MAX_EVALS = 128
BATCH_SIZE = 32


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("ROBERTA SHAP TEXT ATTRIBUTION")
print("=" * 70)

print(
    f"\nDevice: {device}"
)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load fine-tuned RoBERTa
# ============================================================

print(
    "\nLoading RoBERTa checkpoint..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        CHECKPOINT_PATH
    )
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        CHECKPOINT_PATH
    )
)

model.to(device)
model.eval()


# ============================================================
# Prediction function used by SHAP
# ============================================================

def predict_proba(texts):

    texts = [
        str(text)
        for text in texts
    ]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value
        in encoded.items()
    }

    with torch.no_grad():

        logits = model(
            **encoded
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    return (
        probabilities
        .cpu()
        .numpy()
    )


# ============================================================
# SHAP explainer
# ============================================================

masker = shap.maskers.Text(
    tokenizer
)

explainer = shap.Explainer(
    predict_proba,
    masker,
    algorithm="partition",
    output_names=[
        "class_0",
        "class_1",
    ]
)


# ============================================================
# Load same 32 cases as Grad-CAM
# ============================================================

cases = pd.read_csv(
    CASE_PATH
)

print(
    f"\nCases: {len(cases):,}"
)


all_token_rows = []
case_summaries = []


for index, row in cases.iterrows():

    sample_id = str(
        row["id"]
    )

    text = str(
        row["text"]
    )

    expected_prediction = int(
        row["prediction_roberta"]
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"{index + 1}/{len(cases)} "
        f"{sample_id}"
    )

    print(
        f"Group: "
        f"{row['comparison_group']}"
    )

    print(
        f"Text: {text}"
    )


    # --------------------------------------------------------
    # Verify current model prediction
    # --------------------------------------------------------

    probability = predict_proba(
        [text]
    )[0]

    predicted_class = int(
        np.argmax(
            probability
        )
    )


    print(
        "Probability class 0:",
        f"{probability[0]:.4f}"
    )

    print(
        "Probability class 1:",
        f"{probability[1]:.4f}"
    )

    print(
        "Predicted class:",
        predicted_class
    )


    if (
        predicted_class
        != expected_prediction
    ):

        print(
            "WARNING: prediction differs "
            "from stored evaluation output."
        )


    # --------------------------------------------------------
    # SHAP explanation
    #
    # We explain the class actually predicted
    # by RoBERTa.
    # --------------------------------------------------------

    explanation = explainer(
        [text],
        max_evals=MAX_EVALS,
        batch_size=BATCH_SIZE
    )


    tokens = list(
        explanation.data[0]
    )

    values = (
        explanation.values[
            0,
            :,
            predicted_class
        ]
    )


    # --------------------------------------------------------
    # Save token attributions
    # --------------------------------------------------------

    token_rows = []


    for token, value in zip(
        tokens,
        values
    ):

        cleaned_token = str(
            token
        )

        attribution = float(
            value
        )


        token_row = {
            "id":
                sample_id,

            "comparison_group":
                row[
                    "comparison_group"
                ],

            "label":
                int(
                    row["label"]
                ),

            "predicted_class":
                predicted_class,

            "token":
                cleaned_token,

            "shap_value":
                attribution,

            "absolute_shap":
                abs(attribution),
        }


        token_rows.append(
            token_row
        )

        all_token_rows.append(
            token_row
        )


    token_df = pd.DataFrame(
        token_rows
    )


    token_df.to_csv(
        OUTPUT_ROOT
        / f"{sample_id}_tokens.csv",
        index=False
    )


    # --------------------------------------------------------
    # Top supportive/opposing tokens
    # --------------------------------------------------------

    meaningful = token_df[
        token_df["token"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()


    supportive = (
        meaningful
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(5)
    )


    opposing = (
        meaningful
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(5)
    )


    supportive_text = [
        {
            "token":
                str(r["token"]),

            "value":
                float(
                    r["shap_value"]
                ),
        }

        for _, r
        in supportive.iterrows()
    ]


    opposing_text = [
        {
            "token":
                str(r["token"]),

            "value":
                float(
                    r["shap_value"]
                ),
        }

        for _, r
        in opposing.iterrows()
    ]


    print(
        "\nTop tokens supporting "
        f"class {predicted_class}:"
    )

    for item in supportive_text:

        print(
            f"  {item['token']!r}: "
            f"{item['value']:+.4f}"
        )


    print(
        "\nTop tokens opposing "
        f"class {predicted_class}:"
    )

    for item in opposing_text:

        print(
            f"  {item['token']!r}: "
            f"{item['value']:+.4f}"
        )


    case_summaries.append({
        "id":
            sample_id,

        "comparison_group":
            row[
                "comparison_group"
            ],

        "text":
            text,

        "label":
            int(
                row["label"]
            ),

        "prediction_roberta":
            expected_prediction,

        "shap_predicted_class":
            predicted_class,

        "probability_class_0":
            float(
                probability[0]
            ),

        "probability_class_1":
            float(
                probability[1]
            ),

        "top_supportive_tokens":
            json.dumps(
                supportive_text
            ),

        "top_opposing_tokens":
            json.dumps(
                opposing_text
            ),
    })


# ============================================================
# Save consolidated outputs
# ============================================================

pd.DataFrame(
    all_token_rows
).to_csv(
    OUTPUT_ROOT
    / "all_token_attributions.csv",
    index=False
)


pd.DataFrame(
    case_summaries
).to_csv(
    OUTPUT_ROOT
    / "shap_case_summary.csv",
    index=False
)


with (
    OUTPUT_ROOT
    / "metadata.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "method":
                "SHAP Partition Explainer",

            "model":
                "fine-tuned roberta-base",

            "cases":
                int(len(cases)),

            "max_length":
                MAX_LENGTH,

            "max_evals":
                MAX_EVALS,

            "explained_output":
                "RoBERTa predicted class",

            "interpretation":
                (
                    "Positive SHAP values support "
                    "the predicted class; negative "
                    "values oppose the predicted class."
                ),
        },
        f,
        indent=2
    )


print(
    "\n" + "=" * 70
)

print(
    "ROBERTA SHAP COMPLETE"
)

print("=" * 70)

print(
    f"\nExplained {len(cases):,} cases."
)

print(
    "Saved to:",
    OUTPUT_ROOT
)


if __name__ == "__main__":
    pass
