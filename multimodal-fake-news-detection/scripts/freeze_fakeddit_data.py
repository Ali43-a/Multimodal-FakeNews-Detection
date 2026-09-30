from pathlib import Path
import hashlib
import json

import pandas as pd


DATA_ROOT = Path("data/processed/fakeddit/development_clean")
OUTPUT = Path("results/metrics/fakeddit_data_freeze.json")

SPLITS = {
    "train": 13129,
    "validation": 1962,
    "test": 1965,
}


def hash_ids(ids):
    values = sorted(str(x) for x in ids)
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


summary = {}
id_sets = {}


for split, expected_count in SPLITS.items():

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    path = DATA_ROOT / f"{split}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_parquet(path)

    if len(df) != expected_count:
        raise ValueError(
            f"{split}: expected {expected_count:,} rows, "
            f"found {len(df):,}"
        )

    if df["id"].duplicated().any():
        raise ValueError(f"{split}: duplicate IDs detected.")

    labels = sorted(df["label"].unique().tolist())

    if labels != [0, 1]:
        raise ValueError(f"{split}: unexpected labels {labels}")

    if "image_usable" in df.columns:
        if not df["image_usable"].all():
            raise ValueError(f"{split}: unusable images remain.")

    missing_paths = 0

    if "image_path" in df.columns:
        missing_paths = sum(
            not Path(str(p)).exists()
            for p in df["image_path"]
        )

        if missing_paths:
            raise ValueError(
                f"{split}: {missing_paths} image paths do not exist."
            )

    ids = set(df["id"].astype(str))
    id_sets[split] = ids

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    label_proportions = (
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )

    split_hash = hash_ids(ids)

    summary[split] = {
        "rows": int(len(df)),
        "unique_ids": int(df["id"].nunique()),
        "label_counts": {
            str(k): int(v)
            for k, v in label_counts.items()
        },
        "label_proportions": {
            str(k): float(v)
            for k, v in label_proportions.items()
        },
        "sample_id_sha256": split_hash,
        "missing_image_paths": int(missing_paths),
    }

    print(f"Rows: {len(df):,}")
    print(f"Unique IDs: {df['id'].nunique():,}")
    print(f"Labels: {label_counts}")
    print(f"ID SHA256: {split_hash}")


train_val = id_sets["train"] & id_sets["validation"]
train_test = id_sets["train"] & id_sets["test"]
val_test = id_sets["validation"] & id_sets["test"]

summary["cross_split_id_overlap"] = {
    "train_validation": len(train_val),
    "train_test": len(train_test),
    "validation_test": len(val_test),
}

if train_val or train_test or val_test:
    raise ValueError("Cross-split ID overlap detected.")


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("DATA FREEZE COMPLETE")
print("=" * 70)

print("\nCross-split ID overlap:")
print("Train / Validation:", len(train_val))
print("Train / Test:", len(train_test))
print("Validation / Test:", len(val_test))

print(f"\nFreeze record saved to: {OUTPUT}")
