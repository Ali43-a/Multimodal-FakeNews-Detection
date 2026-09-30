from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42

INPUT_DIR = Path("data/processed/fakeddit")
OUTPUT_DIR = Path("data/processed/fakeddit/development")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_SIZES = {
    "train": 20_000,
    "validation": 3_000,
    "test": 3_000,
}

FILES = {
    "train": INPUT_DIR / "train.parquet",
    "validation": INPUT_DIR / "validation.parquet",
    "test": INPUT_DIR / "test.parquet",
}


def stratified_sample(df, sample_size):
    """
    Select a reproducible sample while preserving
    the binary label distribution.
    """

    if sample_size >= len(df):
        return df.copy()

    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=SEED,
        stratify=df["label"]
    )

    return (
        sampled
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )


for split_name, path in FILES.items():

    print("=" * 70)
    print(split_name.upper())
    print("=" * 70)

    df = pd.read_parquet(path)

    target_size = TARGET_SIZES[split_name]

    sampled = stratified_sample(
        df,
        target_size
    )

    original_distribution = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    sampled_distribution = (
        sampled["label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print(f"Original size: {len(df):,}")
    print(f"Development size: {len(sampled):,}")

    print("\nOriginal label distribution:")
    print(original_distribution)

    print("\nDevelopment label distribution:")
    print(sampled_distribution)

    output_path = (
        OUTPUT_DIR /
        f"{split_name}.parquet"
    )

    sampled.to_parquet(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

print("\nDevelopment subset complete.")
