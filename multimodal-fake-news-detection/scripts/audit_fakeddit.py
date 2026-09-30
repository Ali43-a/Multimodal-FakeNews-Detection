from pathlib import Path
import json
import pandas as pd

BASE = Path("data/raw/fakeddit/multimodal_only_samples")

FILES = {
    "train": BASE / "multimodal_train.tsv",
    "validation": BASE / "multimodal_validate.tsv",
    "test": BASE / "multimodal_test_public.tsv",
}

OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

audit = {}

for split_name, path in FILES.items():
    print("=" * 80)
    print(f"{split_name.upper()}: {path}")
    print("=" * 80)

    df = pd.read_csv(path, sep="\t", low_memory=False)

    print(f"\nShape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nTop missing-value counts:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(20).to_string())

    print("\nLabel distributions:")
    label_info = {}

    for label_col in ["2_way_label", "3_way_label", "6_way_label"]:
        if label_col in df.columns:
            counts = df[label_col].value_counts(dropna=False).sort_index()
            print(f"\n{label_col}:")
            print(counts.to_string())

            label_info[label_col] = {
                str(k): int(v) for k, v in counts.items()
            }

    duplicate_info = {}

    for id_col in ["id", "submission_id"]:
        if id_col in df.columns:
            duplicate_info[id_col] = int(df[id_col].duplicated().sum())
            print(
                f"\nDuplicate {id_col} values: "
                f"{duplicate_info[id_col]}"
            )

    text_info = {}

    for text_col in ["clean_title", "title"]:
        if text_col in df.columns:
            lengths = df[text_col].fillna("").astype(str).str.len()

            text_info[text_col] = {
                "missing": int(df[text_col].isna().sum()),
                "empty": int((lengths == 0).sum()),
                "mean_length": float(lengths.mean()),
                "median_length": float(lengths.median()),
                "max_length": int(lengths.max()),
                "duplicates": int(df[text_col].duplicated().sum()),
            }

            print(f"\n{text_col} statistics:")
            for key, value in text_info[text_col].items():
                print(f"  {key}: {value}")

    image_info = {}

    for image_col in [
        "image_url",
        "hasImage",
        "image_id"
    ]:
        if image_col in df.columns:
            image_info[image_col] = {
                "missing": int(df[image_col].isna().sum()),
                "unique": int(df[image_col].nunique(dropna=True)),
            }

            print(f"\n{image_col}:")
            print(f"  missing: {image_info[image_col]['missing']}")
            print(f"  unique: {image_info[image_col]['unique']}")

    audit[split_name] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "memory_mb": float(
            df.memory_usage(deep=True).sum() / 1024**2
        ),
        "missing_values": {
            col: int(value)
            for col, value in missing.items()
        },
        "labels": label_info,
        "duplicates": duplicate_info,
        "text": text_info,
        "image": image_info,
    }

with open(
    OUTPUT_DIR / "fakeddit_audit.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(audit, f, indent=2)

print("\n" + "=" * 80)
print("Audit complete.")
print("Saved to results/metrics/fakeddit_audit.json")
print("=" * 80)
