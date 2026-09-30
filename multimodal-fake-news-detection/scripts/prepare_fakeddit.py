from pathlib import Path
import pandas as pd

RAW = Path("data/raw/fakeddit/multimodal_only_samples")
OUT = Path("data/processed/fakeddit")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": RAW / "multimodal_train.tsv",
    "validation": RAW / "multimodal_validate.tsv",
    "test": RAW / "multimodal_test_public.tsv",
}

USECOLS = [
    "id",
    "clean_title",
    "image_url",
    "2_way_label",
]

processed = {}

for split, path in FILES.items():
    print(f"Processing {split}...")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=USECOLS,
        low_memory=False,
    )

    df = df.rename(
        columns={
            "clean_title": "text",
            "2_way_label": "label",
        }
    )

    df["split"] = split
    df["source_dataset"] = "fakeddit"

    df["text_length_chars"] = df["text"].str.len()
    df["text_length_words"] = (
        df["text"]
        .str.split()
        .str.len()
    )

    df["image_url_available"] = df["image_url"].notna()

    # Ensure expected types.
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype("int8")

    processed[split] = df

    print(f"  Rows: {len(df):,}")
    print(
        "  Labels:",
        df["label"].value_counts().sort_index().to_dict()
    )
    print(
        "  Missing image URLs:",
        int(df["image_url"].isna().sum())
    )

    df.to_parquet(
        OUT / f"{split}.parquet",
        index=False
    )

print("\nCreating leakage-reduced evaluation splits...")

train = processed["train"]
val = processed["validation"]
test = processed["test"]

train_titles = set(train["text"])
train_images = set(train["image_url"].dropna())

train_pairs = set(
    zip(
        train["text"],
        train["image_url"].fillna("")
    )
)

def remove_training_duplicates(df):
    pair_series = list(
        zip(
            df["text"],
            df["image_url"].fillna("")
        )
    )

    exact_pair_duplicate = pd.Series(
        [pair in train_pairs for pair in pair_series],
        index=df.index
    )

    cleaned = df.loc[~exact_pair_duplicate].copy()

    return cleaned, int(exact_pair_duplicate.sum())

val_clean, val_removed = remove_training_duplicates(val)
test_clean, test_removed = remove_training_duplicates(test)

val_clean.to_parquet(
    OUT / "validation_deduplicated.parquet",
    index=False
)

test_clean.to_parquet(
    OUT / "test_deduplicated.parquet",
    index=False
)

print(f"Validation exact pairs removed: {val_removed}")
print(f"Test exact pairs removed: {test_removed}")

print("\nProcessed files saved to:")
print(OUT)
