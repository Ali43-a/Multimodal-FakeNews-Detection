import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


FEATURE_ROOT = Path(
    "data/processed/fakeddit/multimodal_features"
)

RESULT_ROOT = Path(
    "results/metrics/linear_fusion_ablation"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42


def load_split(split):

    data = torch.load(
        FEATURE_ROOT / f"{split}.pt",
        map_location="cpu",
        weights_only=False
    )

    return {
        "ids": list(data["ids"]),
        "labels": data["labels"].numpy(),
        "text": data["text_features"].numpy(),
        "image": data["image_features"].numpy(),
    }


def metrics(
    y_true,
    y_pred,
    y_prob
):

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_prob
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            ).tolist(),
    }


train = load_split("train")
validation = load_split("validation")
test = load_split("test")


experiments = {
    "text_only": (
        train["text"],
        validation["text"],
        test["text"]
    ),

    "image_only": (
        train["image"],
        validation["image"],
        test["image"]
    ),

    "text_image_concat": (
        np.concatenate(
            [
                train["text"],
                train["image"]
            ],
            axis=1
        ),

        np.concatenate(
            [
                validation["text"],
                validation["image"]
            ],
            axis=1
        ),

        np.concatenate(
            [
                test["text"],
                test["image"]
            ],
            axis=1
        ),
    )
}


all_results = {}


for name, (
    X_train,
    X_validation,
    X_test
) in experiments.items():

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    print(
        "Dimensions:",
        X_train.shape[1]
    )

    # Fit scaling using TRAIN ONLY.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_validation = scaler.transform(
        X_validation
    )

    X_test = scaler.transform(
        X_test
    )


    model = LogisticRegression(
        C=1.0,
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED
    )


    print("Training...")

    model.fit(
        X_train,
        train["labels"]
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_prob = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_pred = (
        model.predict(
            X_validation
        )
    )

    validation_metrics = metrics(
        validation["labels"],
        validation_pred,
        validation_prob
    )


    print(
        "\nValidation metrics:"
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


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_prob = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_pred = (
        model.predict(
            X_test
        )
    )

    test_metrics = metrics(
        test["labels"],
        test_pred,
        test_prob
    )


    print(
        "\nTest metrics:"
    )

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
        "Confusion matrix:"
    )

    print(
        np.array(
            test_metrics[
                "confusion_matrix"
            ]
        )
    )


    all_results[name] = {
        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "feature_dimension":
            int(X_train.shape[1]),

        "classifier":
            "logistic_regression",

        "C":
            1.0
    }


    predictions = pd.DataFrame({
        "id":
            test["ids"],

        "label":
            test["labels"],

        "prediction":
            test_pred,

        "probability_class_1":
            test_prob
    })


    predictions.to_csv(
        RESULT_ROOT
        / f"{name}_test_predictions.csv",
        index=False
    )


with (
    RESULT_ROOT
    / "results.json"
).open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("LINEAR FUSION ABLATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:",
    RESULT_ROOT
)
