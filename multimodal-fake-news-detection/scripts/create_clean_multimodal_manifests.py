from pathlib import Path
import pandas as pd

MANIFEST_ROOT = Path("data/processed/fakeddit/development")
INTEGRITY_ROOT = Path("results/metrics")
OUTPUT_ROOT = Path("data/processed/fakeddit/development_clean")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

PLACEHOLDER_HASH = "faa24ec881e6040655c187a681d6dc496eb8aa41e1bd0652a180b3a40b457187"

for split in ["train", "validation", "test"]:
    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    manifest = pd.read_parquet(MANIFEST_ROOT / f"{split}.parquet")
    integrity = pd.read_csv(
        INTEGRITY_ROOT / f"fakeddit_{split}_image_integrity.csv"
    )

    integrity["id"] = integrity["id"].astype(str)
    manifest["id"] = manifest["id"].astype(str)

    integrity["is_placeholder"] = (
        integrity["sha256"] == PLACEHOLDER_HASH
    )

    integrity["image_usable"] = (
        integrity["exists"].eq(True)
        & integrity["valid"].eq(True)
        & ~integrity["is_placeholder"]
    )

    keep_cols = [
        "id",
        "exists",
        "valid",
        "image_usable",
        "is_placeholder",
        "width",
        "height",
        "size_bytes",
        "sha256",
        "error",
    ]

    merged = manifest.merge(
        integrity[keep_cols],
        on="id",
        how="left",
        validate="one_to_one"
    )

    merged["image_path"] = merged["id"].apply(
        lambda x: str(
            Path("data/raw/fakeddit/development_images")
            / split
            / f"{x}.jpg"
        )
    )

    usable = merged[merged["image_usable"] == True].copy()

    print(f"Original rows: {len(merged):,}")
    print(f"Usable multimodal rows: {len(usable):,}")
    print(f"Removed: {len(merged) - len(usable):,}")

    print("\nOriginal class distribution:")
    print(merged["label"].value_counts(normalize=True).sort_index())

    print("\nUsable class distribution:")
    print(usable["label"].value_counts(normalize=True).sort_index())

    merged.to_parquet(
        OUTPUT_ROOT / f"{split}_with_image_status.parquet",
        index=False
    )

    usable.to_parquet(
        OUTPUT_ROOT / f"{split}.parquet",
        index=False
    )

print("\nClean multimodal manifests complete.")
