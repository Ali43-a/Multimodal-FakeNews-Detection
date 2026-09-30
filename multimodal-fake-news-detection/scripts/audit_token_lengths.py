from pathlib import Path
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


DATA_DIR = Path("data/processed/fakeddit")
OUTPUT = Path("results/metrics/fakeddit_token_lengths.json")

FILES = {
    "train": DATA_DIR / "train.parquet",
    "validation": DATA_DIR / "validation.parquet",
    "test": DATA_DIR / "test.parquet",
}

MODEL_NAME = "roberta-base"
BATCH_SIZE = 10000

print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# Prevent warnings for long samples during the audit.
tokenizer.model_max_length = 1_000_000

report = {}

for split_name, path in FILES.items():

    print("\n" + "=" * 70)
    print(split_name.upper())
    print("=" * 70)

    df = pd.read_parquet(
        path,
        columns=["id", "text"]
    )

    lengths = []

    texts = df["text"].fillna("").astype(str).tolist()

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True
        )

        lengths.extend(encoded["length"])

        if start % 100000 == 0:
            print(
                f"Processed "
                f"{min(start + BATCH_SIZE, len(texts)):,}"
                f"/{len(texts):,}"
            )

    lengths = np.asarray(lengths)

    percentiles = {
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "p99_5": float(np.percentile(lengths, 99.5)),
        "p99_9": float(np.percentile(lengths, 99.9)),
    }

    thresholds = {}

    for threshold in [32, 64, 128, 256, 512]:
        percentage = (
            (lengths <= threshold).mean() * 100
        )

        thresholds[str(threshold)] = float(percentage)

    split_report = {
        "samples": int(len(lengths)),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "percentiles": percentiles,
        "percentage_within_length": thresholds
    }

    report[split_name] = split_report

    print(f"Samples: {len(lengths):,}")
    print(f"Mean tokens: {lengths.mean():.2f}")
    print(f"Median tokens: {np.median(lengths):.0f}")
    print(f"Maximum tokens: {lengths.max()}")

    print("\nPercentiles:")

    for name, value in percentiles.items():
        print(f"  {name}: {value:.0f}")

    print("\nCoverage:")

    for threshold, percentage in thresholds.items():
        print(
            f"  <= {threshold:>3} tokens: "
            f"{percentage:.3f}%"
        )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )

print("\n" + "=" * 70)
print("Token-length audit complete.")
print(f"Saved to: {OUTPUT}")
print("=" * 70)
